"""
Enhanced regime detection: Hidden Markov Models, macro nowcasting, multi-asset regimes.

dim_050 — Regime detection (HMM / macro nowcast) (target: 9)

This is a substantially enhanced version of sentinel/sbx/regime_detector.py.
The existing module provides 8-signal voting and Baum-Welch HMM for vol regime.
This module adds:

  HiddenMarkovRegimeModel   — GaussianHMM (hmmlearn) on multi-feature data:
                               SPY returns, VIX, 2s10s, IG spread, DXY.
                               2–4 state Gaussian HMM, Viterbi decoding, regime labels.

  MacroNowcaster            — Weighted Z-score nowcast from FRED macro series.
                               Growth + inflation quadrant → goldilocks/overheating/
                               deflation_risk/stagflation.

  MultiAssetRegimeEngine    — Per-asset-class regime (equity, fixed income, FX,
                               commodity). Returns 4-digit regime code.

  RegimeTransitionPredictor — Forward-looking transition probability from HMM
                               transition matrix. Asset allocation per regime.

  EarlyWarningSystem        — Composite 0–100 risk score: VIX spike, credit spread
                               widening, curve inversion, momentum breakdown.

  FastAPI router            — 7 endpoints mounted at /api/regime-enhanced

Uses: requests, pandas, numpy, yfinance, fastapi, pydantic, sqlite3.
hmmlearn is optional — falls back to a Gaussian mixture manual EM if unavailable.
"""
from __future__ import annotations

import sqlite3
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# hmmlearn is optional
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _GaussianHMM = None  # type: ignore[assignment,misc]
    _HMMLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/csv,text/html,*/*",
}
_TIMEOUT = 30

_DB_PATH = Path(__file__).parent.parent.parent / ".danteforge" / "regime_enhanced_cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Regime labels for 4-state HMM
REGIME_LABELS_4 = ["BULL_LOW_VOL", "BULL_HIGH_VOL", "BEAR_HIGH_VOL", "RISK_OFF"]
REGIME_LABELS_3 = ["BULL_LOW_VOL", "BEAR_HIGH_VOL", "RISK_OFF"]
REGIME_LABELS_2 = ["LOW_VOL", "HIGH_VOL"]

# Asset allocation per regime
REGIME_ALLOCATIONS: dict[str, dict[str, float]] = {
    "BULL_LOW_VOL": {
        "equities": 0.75,
        "bonds": 0.10,
        "credit": 0.10,
        "cash": 0.05,
    },
    "BULL_HIGH_VOL": {
        "equities": 0.55,
        "bonds": 0.20,
        "credit": 0.10,
        "gold": 0.05,
        "cash": 0.10,
    },
    "BEAR_HIGH_VOL": {
        "equities": 0.25,
        "bonds": 0.35,
        "gold": 0.15,
        "cash": 0.25,
    },
    "RISK_OFF": {
        "equities": 0.15,
        "bonds": 0.45,
        "gold": 0.20,
        "cash": 0.20,
    },
    "GOLDILOCKS": {
        "equities": 0.70,
        "bonds": 0.15,
        "credit": 0.10,
        "cash": 0.05,
    },
    "OVERHEATING": {
        "equities": 0.45,
        "commodities": 0.20,
        "tips": 0.15,
        "cash": 0.20,
    },
    "STAGFLATION": {
        "commodities": 0.30,
        "tips": 0.25,
        "equities": 0.20,
        "short_duration": 0.15,
        "cash": 0.10,
    },
    "DEFLATION_RISK": {
        "long_bonds": 0.45,
        "equities": 0.25,
        "gold": 0.15,
        "cash": 0.15,
    },
}

# ---------------------------------------------------------------------------
# SQLite cache helpers (shared pattern with inflation_vix_analytics)
# ---------------------------------------------------------------------------


def _init_cache_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fred_cache (
            series_id  TEXT,
            fetched_at INTEGER,
            csv_data   TEXT,
            PRIMARY KEY (series_id)
        )
        """
    )
    conn.commit()
    return conn


def _cache_get(series_id: str, max_age_seconds: int = 3600) -> Optional[str]:
    try:
        conn = _init_cache_db()
        row = conn.execute(
            "SELECT csv_data, fetched_at FROM fred_cache WHERE series_id=?",
            (series_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        csv_data, fetched_at = row
        if time.time() - fetched_at > max_age_seconds:
            return None
        return csv_data
    except Exception:
        return None


def _cache_set(series_id: str, csv_data: str) -> None:
    try:
        conn = _init_cache_db()
        conn.execute(
            "INSERT OR REPLACE INTO fred_cache (series_id, fetched_at, csv_data) VALUES (?,?,?)",
            (series_id, int(time.time()), csv_data),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FRED / yfinance data helpers
# ---------------------------------------------------------------------------


def _fetch_fred(series_id: str, lookback_days: int = 1260) -> pd.Series:
    """
    Fetch a FRED series as a float pandas Series with a DatetimeIndex.
    Uses SQLite cache to avoid redundant requests.
    """
    cached = _cache_get(series_id, max_age_seconds=3600)
    if cached:
        try:
            df = pd.read_csv(
                pd.io.common.StringIO(cached), index_col=0, parse_dates=True
            )
            s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
            return s[s.index >= cutoff].sort_index()
        except Exception:
            pass

    end = date.today()
    start = end - timedelta(days=lookback_days + 30)
    url = (
        f"{FRED_BASE}?id={series_id}"
        f"&observation_start={start.strftime('%Y-%m-%d')}"
        f"&observation_end={end.strftime('%Y-%m-%d')}"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        _cache_set(series_id, resp.text)
        df = pd.read_csv(
            pd.io.common.StringIO(resp.text), index_col=0, parse_dates=True
        )
        s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
        return s[s.index >= cutoff].sort_index()
    except Exception as exc:
        logger.warning("fred_fetch_failed", series_id=series_id, error=str(exc))
        return pd.Series(dtype=float)


def _yf_close(ticker: str, lookback_days: int = 1260) -> pd.Series:
    """Fetch adjusted close from yfinance."""
    end = date.today()
    start = end - timedelta(days=lookback_days + 30)
    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.Series(dtype=float)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].squeeze()
        else:
            closes = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        closes.index = pd.to_datetime(closes.index)
        return closes.dropna().sort_index()
    except Exception as exc:
        logger.warning("yfinance_failed", ticker=ticker, error=str(exc))
        return pd.Series(dtype=float)


def _latest(s: pd.Series) -> Optional[float]:
    """Return most recent non-NaN value."""
    clean = s.dropna()
    if clean.empty:
        return None
    return float(round(clean.iloc[-1], 6))


def _zscore(s: pd.Series, lookback: int = 252) -> Optional[float]:
    """Z-score of most recent value vs prior `lookback` observations."""
    clean = s.dropna()
    if len(clean) < 10:
        return None
    window = clean.iloc[-min(lookback, len(clean)):]
    std = window.std()
    if std == 0:
        return 0.0
    return float(round((window.iloc[-1] - window.mean()) / std, 4))


def _build_feature_matrix(
    lookback_days: int = 756,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Build a 5-feature daily matrix for HMM training:
      spy_ret   — SPY log daily return
      vix       — CBOE VIX level
      spread_2s10s — 10Y-2Y Treasury spread (T10Y2Y)
      ig_oas    — IG credit spread (BAMLC0A0CM)
      dxy_ret   — USD index daily log return (DTWEXBGS)

    Returns (feature_df, valid_dates).
    """
    spy_c = _yf_close("SPY", lookback_days + 30)
    spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()

    vix_s = _fetch_fred("VIXCLS", lookback_days)
    spread_s = _fetch_fred("T10Y2Y", lookback_days)
    ig_s = _fetch_fred("BAMLC0A0CM", lookback_days)
    dxy_s = _fetch_fred("DTWEXBGS", lookback_days + 30)
    dxy_ret = np.log(dxy_s / dxy_s.shift(1)).dropna()

    df = pd.DataFrame(
        {
            "spy_ret": spy_ret,
            "vix": vix_s,
            "spread_2s10s": spread_s,
            "ig_oas": ig_s,
            "dxy_ret": dxy_ret,
        }
    ).dropna()

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff].sort_index()
    return df, df.index


def _standardize(df: pd.DataFrame) -> np.ndarray:
    """
    Z-score standardize each column. Returns numpy array.
    Handles zero-std columns gracefully (fills with 0).
    """
    arr = df.values.astype(float)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds[stds == 0] = 1.0
    return (arr - means) / stds


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HMMRegimeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_states: int
    current_regime: str
    current_regime_index: int
    state_probabilities: list[float]
    regime_labels: list[str]
    transition_matrix: list[list[float]]
    expected_durations_days: list[float]
    regime_sequence_tail_20d: list[str]
    regime_means: list[dict[str, float]]
    n_training_obs: int
    hmmlearn_used: bool
    as_of: str = ""


class NowcastResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    growth_nowcast: float
    inflation_nowcast: float
    regime_quadrant: str
    regime_label: str
    signal_contributions: dict[str, dict[str, Any]]
    recommended_allocation: dict[str, float]
    as_of: str = ""


class MultiAssetRegimeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    equity_regime: str
    fixed_income_regime: str
    fx_regime: str
    commodity_regime: str
    regime_code: str
    equity_details: dict[str, Any] = Field(default_factory=dict)
    fixed_income_details: dict[str, Any] = Field(default_factory=dict)
    fx_details: dict[str, Any] = Field(default_factory=dict)
    commodity_details: dict[str, Any] = Field(default_factory=dict)
    as_of: str = ""


class TransitionProbResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_regime: str
    transition_probs_30d: dict[str, float]
    transition_probs_60d: dict[str, float]
    transition_probs_90d: dict[str, float]
    most_likely_regime_30d: str
    recommended_allocation: dict[str, float]
    as_of: str = ""


class EarlyWarningResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    composite_score: float
    risk_level: str
    triggered_signals: list[dict[str, Any]]
    signal_scores: dict[str, float]
    interpretation: str
    as_of: str = ""


# ---------------------------------------------------------------------------
# HiddenMarkovRegimeModel
# ---------------------------------------------------------------------------


class HiddenMarkovRegimeModel:
    """
    Multi-feature Gaussian HMM for market regime detection.

    Features: SPY returns, VIX, 2s10s yield spread, IG credit OAS, DXY returns.

    Uses hmmlearn.GaussianHMM if available, otherwise falls back to a
    manual Baum-Welch EM implementation.

    Regime labels (4-state default):
      BULL_LOW_VOL  — positive returns, low VIX, steep curve, tight spreads
      BULL_HIGH_VOL — positive returns, elevated VIX (transitional)
      BEAR_HIGH_VOL — negative returns, high VIX
      RISK_OFF      — extreme VIX, wide spreads, flight-to-safety
    """

    def __init__(
        self,
        n_states: int = 4,
        lookback_days: int = 756,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.random_state = random_state
        self._model: Any = None
        self._feature_df: Optional[pd.DataFrame] = None
        self._X_std: Optional[np.ndarray] = None
        self._fitted = False
        self._hmmlearn_used = False

    def _assign_labels(
        self, means: np.ndarray, feature_names: list[str]
    ) -> list[str]:
        """
        Assign human-readable regime labels based on state means.

        Heuristic: sort states by SPY return mean descending.
        Among positive-return states, sort by VIX mean ascending.
        States with highly negative returns and high VIX are BEAR_HIGH_VOL / RISK_OFF.
        """
        if self.n_states == 4:
            labels_pool = REGIME_LABELS_4.copy()
        elif self.n_states == 3:
            labels_pool = REGIME_LABELS_3.copy()
        else:
            labels_pool = REGIME_LABELS_2.copy()

        try:
            spy_idx = feature_names.index("spy_ret") if "spy_ret" in feature_names else 0
            vix_idx = feature_names.index("vix") if "vix" in feature_names else 1
        except (ValueError, IndexError):
            spy_idx, vix_idx = 0, 1

        # Score each state: higher SPY return = better, lower VIX = better
        spy_means = means[:, spy_idx]
        vix_means = means[:, vix_idx]

        # Normalize spy and vix means to [0,1]
        def _norm(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            if rng == 0:
                return np.zeros_like(arr)
            return (arr - arr.min()) / rng

        spy_norm = _norm(spy_means)
        vix_norm = _norm(vix_means)

        # Score = high spy - high vix
        scores = spy_norm - vix_norm
        sorted_indices = np.argsort(-scores)  # descending

        assigned: list[str] = [""] * self.n_states
        for rank, state_idx in enumerate(sorted_indices):
            if rank < len(labels_pool):
                assigned[state_idx] = labels_pool[rank]
            else:
                assigned[state_idx] = f"REGIME_{rank}"
        return assigned

    def _fit_hmmlearn(self, X: np.ndarray) -> None:
        """Fit using hmmlearn.GaussianHMM."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                tol=1e-4,
                random_state=self.random_state,
                verbose=False,
            )
            model.fit(X)
        self._model = model
        self._hmmlearn_used = True

    def _fit_manual_em(self, X: np.ndarray) -> None:
        """
        Manual Baum-Welch EM for Gaussian HMM.
        Fallback when hmmlearn is not installed.
        """
        n, d = X.shape
        K = self.n_states
        rng = np.random.default_rng(self.random_state)

        # Initialize with k-means-like partition
        idx = rng.choice(n, K, replace=False)
        mu = X[idx].copy()  # (K, d)
        sigma = np.array([np.eye(d) for _ in range(K)])  # (K, d, d)
        A = np.full((K, K), 1.0 / K)
        pi = np.full(K, 1.0 / K)

        def emission_log_prob(t: int) -> np.ndarray:
            """Log Gaussian emission probabilities."""
            log_probs = np.zeros(K)
            for k in range(K):
                diff = X[t] - mu[k]
                try:
                    L = np.linalg.cholesky(sigma[k] + 1e-6 * np.eye(d))
                    sol = np.linalg.solve(L, diff)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    log_probs[k] = -0.5 * (d * np.log(2 * np.pi) + log_det + sol @ sol)
                except np.linalg.LinAlgError:
                    log_probs[k] = -1e10
            return log_probs

        max_iter = 80
        prev_log_lik = -np.inf

        for _ in range(max_iter):
            # --- Forward pass (log-scale) ---
            log_alpha = np.zeros((n, K))
            log_alpha[0] = np.log(pi + 1e-300) + emission_log_prob(0)
            for t in range(1, n):
                log_emis = emission_log_prob(t)
                for k in range(K):
                    log_alpha[t, k] = (
                        _log_sum_exp(log_alpha[t - 1] + np.log(A[:, k] + 1e-300))
                        + log_emis[k]
                    )

            log_lik = _log_sum_exp(log_alpha[-1])

            # --- Backward pass ---
            log_beta = np.zeros((n, K))
            for t in range(n - 2, -1, -1):
                log_emis_next = emission_log_prob(t + 1)
                for k in range(K):
                    log_beta[t, k] = _log_sum_exp(
                        np.log(A[k, :] + 1e-300) + log_emis_next + log_beta[t + 1]
                    )

            # --- Smoothed probabilities (gamma) ---
            log_gamma = log_alpha + log_beta
            log_gamma -= _log_sum_exp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # --- Xi ---
            xi_sum = np.zeros((K, K))
            for t in range(n - 1):
                log_emis_next = emission_log_prob(t + 1)
                for k in range(K):
                    for j in range(K):
                        xi_sum[k, j] += np.exp(
                            log_alpha[t, k]
                            + np.log(A[k, j] + 1e-300)
                            + log_emis_next[j]
                            + log_beta[t + 1, j]
                            - log_lik
                        )

            # --- M-step ---
            pi = gamma[0]
            pi /= pi.sum() + 1e-300

            for k in range(K):
                denom = gamma[:, k].sum() + 1e-300
                xi_row = xi_sum[k]
                A[k] = xi_row / (xi_row.sum() + 1e-300)
                mu[k] = (gamma[:, k:k+1] * X).sum(axis=0) / denom
                diff = X - mu[k]
                sigma[k] = (gamma[:, k:k+1, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / denom
                sigma[k] += 1e-4 * np.eye(d)  # regularization

            if abs(log_lik - prev_log_lik) < 1e-5:
                break
            prev_log_lik = log_lik

        # Store as a dict-like object for unified predict interface
        self._model = _ManualHMM(
            mu=mu, sigma=sigma, A=A, pi=pi, n_states=K, log_lik=log_lik
        )
        self._hmmlearn_used = False

    def fit(self, lookback_days: Optional[int] = None) -> "HiddenMarkovRegimeModel":
        """
        Build feature matrix, standardize, and fit the HMM.

        Parameters
        ----------
        lookback_days : int, optional
            Override default training window.

        Returns self for chaining.
        """
        lb = lookback_days or self.lookback_days
        feature_df, _ = _build_feature_matrix(lb)

        if len(feature_df) < 50:
            raise ValueError(
                f"Insufficient data for HMM: only {len(feature_df)} observations. "
                "Need at least 50."
            )

        self._feature_df = feature_df
        X_std = _standardize(feature_df)
        self._X_std = X_std

        if _HMMLEARN_AVAILABLE:
            try:
                self._fit_hmmlearn(X_std)
            except Exception as exc:
                logger.warning("hmmlearn_fit_failed_fallback", error=str(exc))
                self._fit_manual_em(X_std)
        else:
            logger.info("hmmlearn_not_available_using_manual_em")
            self._fit_manual_em(X_std)

        self._fitted = True
        return self

    def detect_regime(self, lookback_days: int = 252) -> HMMRegimeResult:
        """
        Detect the current market regime using the fitted HMM.

        Parameters
        ----------
        lookback_days : int
            How many days of history to display in the result.

        Returns
        -------
        HMMRegimeResult with current regime, probabilities, and transition matrix.
        """
        if not self._fitted:
            self.fit()

        feature_df = self._feature_df
        X_std = self._X_std
        feature_names = list(feature_df.columns)

        model = self._model

        # Get state sequence and posterior probabilities
        if self._hmmlearn_used:
            state_seq = model.predict(X_std)
            posteriors = model.predict_proba(X_std)
            trans_matrix = model.transmat_.tolist()
            means = model.means_  # (K, d)
        else:
            # Manual HMM
            state_seq = model.predict(X_std)
            posteriors = model.predict_proba(X_std)
            trans_matrix = model.A.tolist()
            means = model.mu

        labels = self._assign_labels(means, feature_names)

        # Current regime
        current_state_idx = int(state_seq[-1])
        current_regime = labels[current_state_idx]
        current_probs = [round(float(p), 4) for p in posteriors[-1]]

        # Expected duration: 1 / (1 - P_ii)
        expected_durations = []
        for k in range(self.n_states):
            p_stay = float(trans_matrix[k][k])
            if p_stay >= 1.0:
                expected_durations.append(999.0)
            else:
                expected_durations.append(round(1.0 / (1.0 - p_stay + 1e-10), 1))

        # Recent 20-day regime sequence
        n_tail = min(20, len(state_seq))
        tail_seq = [labels[int(s)] for s in state_seq[-n_tail:]]

        # State means in feature space (for interpretation)
        regime_means_list = []
        for k in range(self.n_states):
            regime_means_list.append(
                {fn: round(float(means[k][i]), 4) for i, fn in enumerate(feature_names)}
            )

        return HMMRegimeResult(
            n_states=self.n_states,
            current_regime=current_regime,
            current_regime_index=current_state_idx,
            state_probabilities=current_probs,
            regime_labels=labels,
            transition_matrix=[[round(v, 4) for v in row] for row in trans_matrix],
            expected_durations_days=expected_durations,
            regime_sequence_tail_20d=tail_seq,
            regime_means=regime_means_list,
            n_training_obs=len(X_std),
            hmmlearn_used=self._hmmlearn_used,
            as_of=date.today().isoformat(),
        )


# ---------------------------------------------------------------------------
# Manual HMM wrapper (fallback)
# ---------------------------------------------------------------------------


class _ManualHMM:
    """Lightweight wrapper around manually trained HMM parameters."""

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        A: np.ndarray,
        pi: np.ndarray,
        n_states: int,
        log_lik: float,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.A = A
        self.pi = pi
        self.n_states = n_states
        self.log_lik = log_lik

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding."""
        n, d = X.shape
        K = self.n_states

        # Viterbi in log space
        log_delta = np.zeros((n, K))
        psi = np.zeros((n, K), dtype=int)

        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        for k in range(K):
            log_delta[0, k] = log_pi[k] + self._log_emis(X[0], k, d)

        for t in range(1, n):
            for k in range(K):
                scores = log_delta[t - 1] + log_A[:, k]
                psi[t, k] = int(np.argmax(scores))
                log_delta[t, k] = scores[psi[t, k]] + self._log_emis(X[t], k, d)

        # Backtrack
        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Forward-backward smoothed posteriors."""
        n, d = X.shape
        K = self.n_states
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # Forward
        log_alpha = np.zeros((n, K))
        for k in range(K):
            log_alpha[0, k] = log_pi[k] + self._log_emis(X[0], k, d)
        for t in range(1, n):
            for k in range(K):
                log_alpha[t, k] = (
                    _log_sum_exp(log_alpha[t - 1] + log_A[:, k])
                    + self._log_emis(X[t], k, d)
                )

        # Backward
        log_beta = np.zeros((n, K))
        for t in range(n - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = _log_sum_exp(
                    log_A[k, :]
                    + np.array([self._log_emis(X[t + 1], j, d) for j in range(K)])
                    + log_beta[t + 1]
                )

        log_gamma = log_alpha + log_beta
        log_norm = _log_sum_exp(log_gamma, axis=1, keepdims=True)
        posteriors = np.exp(log_gamma - log_norm)
        return posteriors

    def _log_emis(self, x: np.ndarray, k: int, d: int) -> float:
        """Log Gaussian emission probability for state k."""
        diff = x - self.mu[k]
        try:
            S = self.sigma[k] + 1e-6 * np.eye(d)
            L = np.linalg.cholesky(S)
            sol = np.linalg.solve(L, diff)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            return float(-0.5 * (d * np.log(2 * np.pi) + log_det + sol @ sol))
        except np.linalg.LinAlgError:
            return -1e10


def _log_sum_exp(
    arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False
) -> Any:
    """Numerically stable log-sum-exp."""
    if axis is None:
        c = arr.max()
        return c + np.log(np.sum(np.exp(arr - c)) + 1e-300)
    c = arr.max(axis=axis, keepdims=True)
    out = c + np.log(np.sum(np.exp(arr - c), axis=axis, keepdims=keepdims) + 1e-300)
    if not keepdims and axis is not None:
        out = out.squeeze(axis=axis)
    return out


# ---------------------------------------------------------------------------
# MacroNowcaster
# ---------------------------------------------------------------------------


class MacroNowcaster:
    """
    Real-time macro regime estimation via weighted Z-score composite.

    Growth signals (FRED):
      ICSA      — Initial unemployment claims (inverted: fewer claims = growth)
      INDPRO    — Industrial Production Index
      BAMLC0A0CM — IG credit spreads (inverted: tight = growth)
      T10Y2Y    — 2s10s yield spread (steeper = growth)

    Inflation signals (FRED):
      CPIAUCSL  — CPI All Urban (YoY computed internally)
      T5YIE     — 5Y Breakeven Inflation Rate
      VIXCLS    — VIX (inverted for inflation: not a direct signal, but used
                  as risk-off complement)

    Nowcast formula:
      growth_nowcast    = weighted Z-score composite of growth signals, range [-3, +3]
      inflation_nowcast = weighted Z-score composite of inflation signals, range [-3, +3]

    Quadrant mapping:
      high growth / low inflation   → GOLDILOCKS
      high growth / high inflation  → OVERHEATING
      low growth  / high inflation  → STAGFLATION
      low growth  / low inflation   → DEFLATION_RISK
    """

    GROWTH_SIGNALS: dict[str, dict] = {
        "ICSA": {
            "description": "Initial unemployment claims (inverted)",
            "weight": 0.30,
            "invert": True,
        },
        "INDPRO": {
            "description": "Industrial production",
            "weight": 0.25,
            "invert": False,
        },
        "BAMLC0A0CM": {
            "description": "IG credit spread (inverted: tight = growth)",
            "weight": 0.25,
            "invert": True,
        },
        "T10Y2Y": {
            "description": "2s10s yield curve slope (steep = growth)",
            "weight": 0.20,
            "invert": False,
        },
    }

    INFLATION_SIGNALS: dict[str, dict] = {
        "T5YIE": {
            "description": "5Y Breakeven Inflation Rate",
            "weight": 0.40,
            "invert": False,
        },
        "CPIAUCSL": {
            "description": "CPI YoY rate",
            "weight": 0.40,
            "invert": False,
            "yoy": True,
        },
        "VIXCLS": {
            "description": "VIX (inverted: low VIX = reflationary)",
            "weight": 0.20,
            "invert": True,
        },
    }

    def __init__(self, lookback_days: int = 756) -> None:
        self._lookback = lookback_days

    def _compute_signal_zscore(
        self,
        series_id: str,
        cfg: dict,
        lookback_days: int = 252,
    ) -> dict[str, Any]:
        """
        Fetch a signal, compute its Z-score, apply weighting and inversion.
        Returns a dict with zscore, weighted_contribution, description.
        """
        s = _fetch_fred(series_id, max(self._lookback, lookback_days + 60))
        if s.empty:
            return {
                "series_id": series_id,
                "description": cfg["description"],
                "value": None,
                "zscore": None,
                "weighted_contribution": None,
                "available": False,
            }

        # YoY transformation for CPI
        if cfg.get("yoy"):
            s = s.pct_change(periods=12) * 100  # approximate monthly YoY
            s = s.dropna()

        val = _latest(s)
        zs = _zscore(s, lookback_days)

        if zs is None:
            return {
                "series_id": series_id,
                "description": cfg["description"],
                "value": val,
                "zscore": None,
                "weighted_contribution": None,
                "available": False,
            }

        if cfg.get("invert"):
            zs = -zs

        weighted = zs * cfg["weight"]

        return {
            "series_id": series_id,
            "description": cfg["description"],
            "value": val,
            "zscore": round(zs, 4),
            "weight": cfg["weight"],
            "weighted_contribution": round(weighted, 4),
            "available": True,
        }

    def growth_nowcast(self, lookback_days: int = 252) -> float:
        """
        Composite growth nowcast scalar.
        Range: approximately -3 to +3 (sub-trend to above-trend).
        """
        total = 0.0
        total_weight = 0.0
        for sid, cfg in self.GROWTH_SIGNALS.items():
            res = self._compute_signal_zscore(sid, cfg, lookback_days)
            if res["available"] and res["weighted_contribution"] is not None:
                total += res["weighted_contribution"]
                total_weight += cfg["weight"]

        if total_weight == 0:
            return 0.0
        # Normalize by available weight
        raw = total / total_weight
        # Clip to [-3, +3]
        return float(round(np.clip(raw, -3.0, 3.0), 4))

    def inflation_nowcast(self, lookback_days: int = 252) -> float:
        """
        Composite inflation nowcast scalar.
        Range: approximately -3 to +3 (deflationary to highly inflationary).
        """
        total = 0.0
        total_weight = 0.0
        for sid, cfg in self.INFLATION_SIGNALS.items():
            res = self._compute_signal_zscore(sid, cfg, lookback_days)
            if res["available"] and res["weighted_contribution"] is not None:
                total += res["weighted_contribution"]
                total_weight += cfg["weight"]

        if total_weight == 0:
            return 0.0
        raw = total / total_weight
        return float(round(np.clip(raw, -3.0, 3.0), 4))

    def nowcast(self, lookback_days: int = 252) -> NowcastResult:
        """
        Full nowcast result with quadrant classification and signal contributions.
        """
        # Compute all signal Z-scores for attribution
        growth_contribs: dict[str, Any] = {}
        for sid, cfg in self.GROWTH_SIGNALS.items():
            growth_contribs[sid] = self._compute_signal_zscore(sid, cfg, lookback_days)

        inflation_contribs: dict[str, Any] = {}
        for sid, cfg in self.INFLATION_SIGNALS.items():
            inflation_contribs[sid] = self._compute_signal_zscore(sid, cfg, lookback_days)

        # Aggregate
        def _aggregate(contribs: dict[str, Any], weights: dict[str, dict]) -> float:
            total = 0.0
            total_w = 0.0
            for sid, res in contribs.items():
                if res["available"] and res["weighted_contribution"] is not None:
                    total += res["weighted_contribution"]
                    total_w += weights[sid]["weight"]
            if total_w == 0:
                return 0.0
            return float(round(np.clip(total / total_w, -3.0, 3.0), 4))

        g = _aggregate(growth_contribs, self.GROWTH_SIGNALS)
        inf = _aggregate(inflation_contribs, self.INFLATION_SIGNALS)

        # Quadrant classification (threshold: 0 = neutral border)
        high_growth = g > 0.3
        high_inflation = inf > 0.3

        if high_growth and not high_inflation:
            quadrant = "high_growth_low_inflation"
            label = "GOLDILOCKS"
        elif high_growth and high_inflation:
            quadrant = "high_growth_high_inflation"
            label = "OVERHEATING"
        elif not high_growth and high_inflation:
            quadrant = "low_growth_high_inflation"
            label = "STAGFLATION"
        else:
            quadrant = "low_growth_low_inflation"
            label = "DEFLATION_RISK"

        allocation = REGIME_ALLOCATIONS.get(label, REGIME_ALLOCATIONS["GOLDILOCKS"])

        all_contribs: dict[str, dict[str, Any]] = {}
        all_contribs.update({"growth_" + k: v for k, v in growth_contribs.items()})
        all_contribs.update({"inflation_" + k: v for k, v in inflation_contribs.items()})

        return NowcastResult(
            growth_nowcast=g,
            inflation_nowcast=inf,
            regime_quadrant=quadrant,
            regime_label=label,
            signal_contributions=all_contribs,
            recommended_allocation=allocation,
            as_of=date.today().isoformat(),
        )


# ---------------------------------------------------------------------------
# MultiAssetRegimeEngine
# ---------------------------------------------------------------------------


class MultiAssetRegimeEngine:
    """
    Classify current regime across four asset classes: equity, fixed income, FX, commodity.

    Equity   : SPY trend (200D MA), VIX level, 50D vs 200D crossover
    Fixed inc: 2s10s shape, IG spread level, duration risk (10Y yield level)
    FX       : DXY 50D trend, DXY vs 200D MA, EM stress proxy
    Commodity: PPIACO (PPI all commodities) YoY change, trend direction

    Returns a 4-digit regime code combining the first letter of each:
      e.g. "BNFA" = Bull / Normal / Favorable / Advancing
    """

    def snapshot(self) -> MultiAssetRegimeResult:
        """
        Classify all four asset-class regimes and return composite result.
        """
        eq = self._equity_regime()
        fi = self._fixed_income_regime()
        fx = self._fx_regime()
        cm = self._commodity_regime()

        def _code(r: str) -> str:
            return r[0].upper() if r else "U"

        code = f"{_code(eq['regime'])}{_code(fi['regime'])}{_code(fx['regime'])}{_code(cm['regime'])}"

        return MultiAssetRegimeResult(
            equity_regime=eq["regime"],
            fixed_income_regime=fi["regime"],
            fx_regime=fx["regime"],
            commodity_regime=cm["regime"],
            regime_code=code,
            equity_details=eq,
            fixed_income_details=fi,
            fx_details=fx,
            commodity_details=cm,
            as_of=date.today().isoformat(),
        )

    def _equity_regime(self) -> dict[str, Any]:
        """
        Classify equity regime from SPY price action and VIX.

        BULL_TRENDING   — price above 200D MA by >2%, VIX < 20
        BULL_VOLATILE   — price above 200D MA, VIX 20–30
        BEAR_CORRECTING — price 0–10% below 200D MA
        BEAR_TRENDING   — price > 10% below 200D MA
        """
        spy = _yf_close("SPY", 300)
        vix_s = _fetch_fred("VIXCLS", 30)
        vix_val = _latest(vix_s)

        if spy.empty or len(spy) < 50:
            return {"regime": "UNKNOWN", "details": "insufficient_data"}

        spy_current = float(spy.iloc[-1])
        ma200 = float(spy.iloc[-200:].mean()) if len(spy) >= 200 else float(spy.mean())
        ma50 = float(spy.iloc[-50:].mean())
        pct_vs_200d = round((spy_current - ma200) / ma200 * 100, 4)
        pct_vs_50d = round((spy_current - ma50) / ma50 * 100, 4)

        # 3-month return
        ret_3m: Optional[float] = None
        if len(spy) >= 63:
            ret_3m = round(float((spy.iloc[-1] / spy.iloc[-63] - 1) * 100), 4)

        # Classify
        if pct_vs_200d > 2 and (vix_val is None or vix_val < 20):
            regime = "BULL_TRENDING"
        elif pct_vs_200d > -5 and (vix_val is None or vix_val < 30):
            regime = "BULL_VOLATILE"
        elif -15 <= pct_vs_200d <= -5:
            regime = "BEAR_CORRECTING"
        else:
            regime = "BEAR_TRENDING"

        return {
            "regime": regime,
            "spy_current": round(spy_current, 2),
            "ma200": round(ma200, 2),
            "ma50": round(ma50, 2),
            "pct_vs_200d": pct_vs_200d,
            "pct_vs_50d": pct_vs_50d,
            "ret_3m_pct": ret_3m,
            "vix": vix_val,
        }

    def _fixed_income_regime(self) -> dict[str, Any]:
        """
        Classify fixed income regime.

        NORMAL_CURVE    — 2s10s > 0.50, IG OAS < 1.50
        FLAT_CURVE      — 2s10s 0–0.50
        INVERTED        — 2s10s < 0
        CREDIT_STRESS   — IG OAS > 2.50
        DURATION_RISK   — 10Y yield > 5%
        """
        spread_s = _fetch_fred("T10Y2Y", 30)
        ig_s = _fetch_fred("BAMLC0A0CM", 30)
        dgs10_s = _fetch_fred("DGS10", 30)

        spread = _latest(spread_s)
        ig_oas = _latest(ig_s)
        y10 = _latest(dgs10_s)

        if ig_oas is not None and ig_oas > 2.50:
            regime = "CREDIT_STRESS"
        elif y10 is not None and y10 > 5.0:
            regime = "DURATION_RISK"
        elif spread is not None and spread < 0:
            regime = "INVERTED"
        elif spread is not None and spread < 0.50:
            regime = "FLAT_CURVE"
        else:
            regime = "NORMAL_CURVE"

        return {
            "regime": regime,
            "spread_2s10s": spread,
            "ig_oas": ig_oas,
            "yield_10y": y10,
        }

    def _fx_regime(self) -> dict[str, Any]:
        """
        Classify FX (USD) regime using FRED DTWEXBGS (Broad Dollar Index).

        STRONG_TREND    — DXY > 200D MA by >3%
        STRENGTHENING   — DXY > 200D MA by 0–3%
        WEAKENING       — DXY < 200D MA by 0–3%
        WEAK_TREND      — DXY < 200D MA by >3%
        """
        dxy_s = _fetch_fred("DTWEXBGS", 300)

        if dxy_s.empty or len(dxy_s.dropna()) < 50:
            return {"regime": "UNKNOWN", "details": "insufficient_data"}

        dxy_c = dxy_s.dropna()
        dxy_current = float(dxy_c.iloc[-1])
        ma200 = float(dxy_c.iloc[-200:].mean()) if len(dxy_c) >= 200 else float(dxy_c.mean())
        pct_vs_200d = round((dxy_current - ma200) / ma200 * 100, 4)

        # 3M momentum
        ret_3m: Optional[float] = None
        if len(dxy_c) >= 63:
            ret_3m = round(float((dxy_c.iloc[-1] / dxy_c.iloc[-63] - 1) * 100), 4)

        if pct_vs_200d > 3:
            regime = "STRONG_USD"
        elif pct_vs_200d > 0:
            regime = "STRENGTHENING_USD"
        elif pct_vs_200d > -3:
            regime = "WEAKENING_USD"
        else:
            regime = "WEAK_USD"

        return {
            "regime": regime,
            "dxy_current": round(dxy_current, 4),
            "ma200": round(ma200, 4),
            "pct_vs_200d": pct_vs_200d,
            "ret_3m_pct": ret_3m,
        }

    def _commodity_regime(self) -> dict[str, Any]:
        """
        Classify commodity regime using FRED PPIACO (PPI All Commodities).

        ADVANCING       — PPI YoY > 3%
        STABLE          — PPI YoY -3% to +3%
        DECLINING       — PPI YoY < -3%
        """
        ppi_s = _fetch_fred("PPIACO", 400)

        if ppi_s.empty or len(ppi_s.dropna()) < 13:
            return {"regime": "UNKNOWN", "details": "insufficient_data"}

        ppi = ppi_s.dropna()
        ppi_current = float(ppi.iloc[-1])

        # YoY change (monthly data, 12 periods back)
        yoy: Optional[float] = None
        if len(ppi) >= 13:
            ppi_1y = float(ppi.iloc[-13])
            yoy = round((ppi_current / ppi_1y - 1) * 100, 4)

        # 3M momentum
        mom_3m: Optional[float] = None
        if len(ppi) >= 4:
            ppi_3m = float(ppi.iloc[-4])
            mom_3m = round((ppi_current / ppi_3m - 1) * 100, 4)

        if yoy is not None:
            if yoy > 5:
                regime = "ADVANCING"
            elif yoy > -3:
                regime = "STABLE"
            else:
                regime = "DECLINING"
        else:
            regime = "UNKNOWN"

        return {
            "regime": regime,
            "ppi_current": round(ppi_current, 4),
            "yoy_pct": yoy,
            "mom_3m_pct": mom_3m,
        }


# ---------------------------------------------------------------------------
# RegimeTransitionPredictor
# ---------------------------------------------------------------------------


class RegimeTransitionPredictor:
    """
    Forward-looking regime transition probabilities using HMM transition matrix.

    Given the current regime, computes P(in regime X after N days) by raising
    the transition matrix to the N-th power.

    Also provides asset allocation recommendations per regime.
    """

    def __init__(
        self,
        hmm_model: Optional[HiddenMarkovRegimeModel] = None,
    ) -> None:
        self._hmm = hmm_model or HiddenMarkovRegimeModel(n_states=4)
        if not self._hmm._fitted:
            try:
                self._hmm.fit()
            except Exception as exc:
                logger.warning("hmm_fit_failed_in_predictor", error=str(exc))

    def predict(
        self,
        current_regime: Optional[str] = None,
        horizons: tuple[int, ...] = (30, 60, 90),
    ) -> TransitionProbResult:
        """
        Predict transition probabilities over given horizons.

        Parameters
        ----------
        current_regime : str, optional
            Override current regime label. If None, detected from HMM.
        horizons : tuple of int
            Forecast horizons in trading days.

        Returns
        -------
        TransitionProbResult
        """
        if not self._hmm._fitted:
            raise RuntimeError("HMM model is not fitted. Call hmm.fit() first.")

        model = self._hmm._model
        if self._hmm._hmmlearn_used:
            A = model.transmat_
            labels = self._hmm._assign_labels(
                model.means_, list(self._hmm._feature_df.columns)
            )
        else:
            A = model.A
            labels = self._hmm._assign_labels(
                model.mu, list(self._hmm._feature_df.columns)
            )

        # Get current state index
        regime_result = self._hmm.detect_regime()
        if current_regime is None:
            current_regime = regime_result.current_regime
            current_idx = regime_result.current_regime_index
        else:
            # Look up index from label
            if current_regime in labels:
                current_idx = labels.index(current_regime)
            else:
                current_idx = regime_result.current_regime_index

        # Compute A^N for each horizon
        def _transition_probs_at_horizon(n_days: int) -> dict[str, float]:
            """P(end state | current state) after n_days transitions."""
            # Use matrix power
            An = np.linalg.matrix_power(np.array(A), n_days)
            row = An[current_idx]
            return {labels[k]: round(float(row[k]), 4) for k in range(len(labels))}

        probs_30 = _transition_probs_at_horizon(horizons[0] if len(horizons) > 0 else 30)
        probs_60 = _transition_probs_at_horizon(horizons[1] if len(horizons) > 1 else 60)
        probs_90 = _transition_probs_at_horizon(horizons[2] if len(horizons) > 2 else 90)

        # Most likely 30-day outcome
        most_likely_30 = max(probs_30, key=lambda k: probs_30[k])

        # Allocation based on current regime
        allocation = REGIME_ALLOCATIONS.get(
            current_regime, REGIME_ALLOCATIONS["BULL_LOW_VOL"]
        )

        return TransitionProbResult(
            current_regime=current_regime,
            transition_probs_30d=probs_30,
            transition_probs_60d=probs_60,
            transition_probs_90d=probs_90,
            most_likely_regime_30d=most_likely_30,
            recommended_allocation=allocation,
            as_of=date.today().isoformat(),
        )


# ---------------------------------------------------------------------------
# EarlyWarningSystem
# ---------------------------------------------------------------------------


class EarlyWarningSystem:
    """
    Composite early warning system for regime change.

    Monitors five signals and computes a 0–100 composite risk score:
      VIX spike            — VIX > 5-day MA by 20%+             (weight: 25)
      Credit spread widen  — IG OAS up > 30bp in 5 days         (weight: 25)
      Yield curve inversion— 2s10s crosses below 0              (weight: 20)
      Momentum breakdown   — SPY below 50D MA after above       (weight: 20)
      BEI expansion        — 5Y BEI up > 0.15% in 5 days        (weight: 10)

    Each signal scores 0 or 1 (triggered / not triggered), weighted, scaled to 100.
    """

    SIGNAL_WEIGHTS: dict[str, float] = {
        "vix_spike": 25.0,
        "credit_spread_widening": 25.0,
        "yield_curve_inversion": 20.0,
        "momentum_breakdown": 20.0,
        "bei_expansion": 10.0,
    }

    def assess(self) -> EarlyWarningResult:
        """
        Run all early warning checks and return composite risk score.

        Returns
        -------
        EarlyWarningResult with composite_score (0–100), risk_level, and triggered signals.
        """
        triggered: list[dict[str, Any]] = []
        signal_scores: dict[str, float] = {k: 0.0 for k in self.SIGNAL_WEIGHTS}

        # ----- 1. VIX spike: current > 5D MA by 20%+ -----
        vix_s = _fetch_fred("VIXCLS", 30)
        if not vix_s.empty and len(vix_s.dropna()) >= 6:
            s = vix_s.dropna()
            vix_current = float(s.iloc[-1])
            vix_5d_ma = float(s.iloc[-6:-1].mean())
            spike_pct = (vix_current - vix_5d_ma) / vix_5d_ma * 100 if vix_5d_ma > 0 else 0.0
            if spike_pct >= 20:
                signal_scores["vix_spike"] = self.SIGNAL_WEIGHTS["vix_spike"]
                triggered.append(
                    {
                        "signal": "vix_spike",
                        "description": f"VIX {vix_current:.1f} is {spike_pct:.1f}% above 5D MA ({vix_5d_ma:.1f})",
                        "severity": "high",
                        "value": round(vix_current, 2),
                        "threshold_pct": 20,
                    }
                )

        # ----- 2. Credit spread widening: IG OAS up >30bp in 5 days -----
        ig_s = _fetch_fred("BAMLC0A0CM", 30)
        if not ig_s.empty and len(ig_s.dropna()) >= 6:
            s = ig_s.dropna()
            ig_current = float(s.iloc[-1])
            ig_5d_ago = float(s.iloc[-6])
            ig_change_bp = (ig_current - ig_5d_ago) * 100  # OAS is in percent, convert to bp
            if ig_change_bp > 30:
                signal_scores["credit_spread_widening"] = self.SIGNAL_WEIGHTS["credit_spread_widening"]
                triggered.append(
                    {
                        "signal": "credit_spread_widening",
                        "description": f"IG OAS widened {ig_change_bp:.1f}bp in 5 days (now {ig_current:.2f}%)",
                        "severity": "high",
                        "value": round(ig_current, 4),
                        "change_bp": round(ig_change_bp, 1),
                    }
                )

        # ----- 3. Yield curve inversion: 2s10s crosses below 0 -----
        spread_s = _fetch_fred("T10Y2Y", 30)
        curve_inverted = False
        if not spread_s.empty and len(spread_s.dropna()) >= 6:
            s = spread_s.dropna()
            current_spread = float(s.iloc[-1])
            prev_spread = float(s.iloc[-6])
            if current_spread < 0:
                curve_inverted = True
                # Extra weight if just crossed (was positive 5 days ago)
                is_fresh_inversion = prev_spread > 0
                signal_scores["yield_curve_inversion"] = self.SIGNAL_WEIGHTS["yield_curve_inversion"]
                triggered.append(
                    {
                        "signal": "yield_curve_inversion",
                        "description": (
                            f"2s10s at {current_spread:.2f}%"
                            + (" — fresh inversion (was positive 5 days ago)" if is_fresh_inversion else "")
                        ),
                        "severity": "high" if is_fresh_inversion else "medium",
                        "value": round(current_spread, 4),
                        "fresh_inversion": is_fresh_inversion,
                    }
                )

        # ----- 4. Momentum breakdown: SPY below 50D MA after being above -----
        spy = _yf_close("SPY", 120)
        if not spy.empty and len(spy) >= 55:
            spy_current = float(spy.iloc[-1])
            ma50 = float(spy.iloc[-50:].mean())
            # Was SPY above 50D MA 10 days ago?
            spy_10d_ago = float(spy.iloc[-11])
            ma50_10d_ago = float(spy.iloc[-60:-10].mean()) if len(spy) >= 60 else ma50
            was_above = spy_10d_ago > ma50_10d_ago
            now_below = spy_current < ma50

            if was_above and now_below:
                signal_scores["momentum_breakdown"] = self.SIGNAL_WEIGHTS["momentum_breakdown"]
                pct_below = round((spy_current - ma50) / ma50 * 100, 4)
                triggered.append(
                    {
                        "signal": "momentum_breakdown",
                        "description": (
                            f"SPY crossed below 50D MA ({ma50:.2f}). "
                            f"Currently {abs(pct_below):.2f}% below."
                        ),
                        "severity": "high",
                        "value": round(spy_current, 2),
                        "pct_below_ma50": pct_below,
                    }
                )

        # ----- 5. BEI expansion: 5Y BEI up >0.15% in 5 days -----
        bei_s = _fetch_fred("T5YIE", 30)
        if not bei_s.empty and len(bei_s.dropna()) >= 6:
            s = bei_s.dropna()
            bei_current = float(s.iloc[-1])
            bei_5d_ago = float(s.iloc[-6])
            bei_change = bei_current - bei_5d_ago
            if bei_change > 0.15:
                signal_scores["bei_expansion"] = self.SIGNAL_WEIGHTS["bei_expansion"]
                triggered.append(
                    {
                        "signal": "bei_expansion",
                        "description": (
                            f"5Y BEI rose {bei_change:.3f}% in 5 days (now {bei_current:.2f}%). "
                            "Inflation expectations accelerating."
                        ),
                        "severity": "medium",
                        "value": round(bei_current, 4),
                        "change_5d": round(bei_change, 4),
                    }
                )

        # Composite score (0–100)
        composite = sum(signal_scores.values())
        composite = min(100.0, composite)

        # Risk level
        if composite >= 70:
            risk_level = "CRITICAL"
            interpretation = (
                "Multiple regime-change warning signals active. "
                "Consider significant defensive repositioning."
            )
        elif composite >= 45:
            risk_level = "HIGH"
            interpretation = (
                "Several early warning signals triggered. "
                "Elevated probability of regime transition. Review risk exposures."
            )
        elif composite >= 20:
            risk_level = "ELEVATED"
            interpretation = (
                "Some warning signals present. Monitor closely."
            )
        else:
            risk_level = "LOW"
            interpretation = "No significant regime-change warning signals detected."

        return EarlyWarningResult(
            composite_score=round(composite, 2),
            risk_level=risk_level,
            triggered_signals=triggered,
            signal_scores={k: round(v, 2) for k, v in signal_scores.items()},
            interpretation=interpretation,
            as_of=date.today().isoformat(),
        )


# ---------------------------------------------------------------------------
# Regime history helper
# ---------------------------------------------------------------------------


def _build_regime_history(lookback_days: int = 252) -> pd.DataFrame:
    """
    Build a simplified regime classification history using FRED series.

    Approximates regime using VIX (primary), yield curve slope (secondary),
    and IG credit spread (tertiary). Returns a DataFrame with date, regime, confidence.
    """
    vix_s = _fetch_fred("VIXCLS", lookback_days + 30)
    spread_s = _fetch_fred("T10Y2Y", lookback_days + 30)
    ig_s = _fetch_fred("BAMLC0A0CM", lookback_days + 30)

    df = pd.DataFrame(
        {"vix": vix_s, "spread": spread_s, "ig_oas": ig_s}
    ).dropna(how="all")

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff].sort_index()

    records: list[dict] = []
    for dt, row in df.iterrows():
        vix = row.get("vix")
        spread = row.get("spread")
        ig = row.get("ig_oas")

        # Simple scoring
        score = 0.0
        total_w = 0.0

        if not pd.isna(vix):
            w = 0.40
            score += (1.0 if vix < 15 else 0.5 if vix < 25 else 0.0) * w
            total_w += w

        if not pd.isna(spread):
            w = 0.30
            score += (1.0 if spread > 0.5 else 0.5 if spread > 0 else 0.0) * w
            total_w += w

        if not pd.isna(ig):
            w = 0.30
            score += (1.0 if ig < 1.0 else 0.5 if ig < 1.75 else 0.0) * w
            total_w += w

        composite = score / total_w if total_w > 0 else 0.5

        if composite > 0.75:
            regime = "BULL_LOW_VOL"
        elif composite > 0.50:
            regime = "BULL_HIGH_VOL"
        elif composite > 0.25:
            regime = "BEAR_HIGH_VOL"
        else:
            regime = "RISK_OFF"

        records.append(
            {
                "date": dt,
                "regime": regime,
                "composite_score": round(composite, 4),
                "vix": vix if not pd.isna(vix) else None,
                "spread_2s10s": spread if not pd.isna(spread) else None,
                "ig_oas": ig if not pd.isna(ig) else None,
            }
        )

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

regime_router = APIRouter(
    prefix="/api/regime-enhanced",
    tags=["regime", "hmm", "nowcast"],
)

# Module-level singletons (lazy-fitted)
_hmm_model: Optional[HiddenMarkovRegimeModel] = None
_nowcaster = MacroNowcaster()
_multi_asset = MultiAssetRegimeEngine()
_predictor: Optional[RegimeTransitionPredictor] = None
_ews = EarlyWarningSystem()


def _get_hmm() -> HiddenMarkovRegimeModel:
    global _hmm_model
    if _hmm_model is None or not _hmm_model._fitted:
        _hmm_model = HiddenMarkovRegimeModel(n_states=4)
        try:
            _hmm_model.fit()
        except Exception as exc:
            logger.error("hmm_singleton_fit_failed", error=str(exc))
    return _hmm_model


def _get_predictor() -> RegimeTransitionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RegimeTransitionPredictor(hmm_model=_get_hmm())
    return _predictor


@regime_router.get("/regime/current")
def get_current_regime() -> dict:
    """
    Current market regime: simplified 3-signal classification
    (VIX + 2s10s + IG OAS) for low-latency response.
    """
    try:
        vix_s = _fetch_fred("VIXCLS", 5)
        spread_s = _fetch_fred("T10Y2Y", 5)
        ig_s = _fetch_fred("BAMLC0A0CM", 5)

        vix = _latest(vix_s)
        spread = _latest(spread_s)
        ig = _latest(ig_s)

        # Fast classification
        risk_score = 0.0
        n = 0
        if vix is not None:
            risk_score += (1 if vix < 15 else 0.5 if vix < 25 else 0)
            n += 1
        if spread is not None:
            risk_score += (1 if spread > 0.5 else 0.5 if spread > 0 else 0)
            n += 1
        if ig is not None:
            risk_score += (1 if ig < 1.0 else 0.5 if ig < 1.75 else 0)
            n += 1

        avg = risk_score / n if n > 0 else 0.5

        if avg > 0.75:
            regime = "BULL_LOW_VOL"
        elif avg > 0.50:
            regime = "BULL_HIGH_VOL"
        elif avg > 0.25:
            regime = "BEAR_HIGH_VOL"
        else:
            regime = "RISK_OFF"

        return {
            "current_regime": regime,
            "composite_score": round(avg, 4),
            "signals": {"vix": vix, "spread_2s10s": spread, "ig_oas": ig},
            "recommended_allocation": REGIME_ALLOCATIONS.get(regime, {}),
            "as_of": date.today().isoformat(),
        }
    except Exception as exc:
        logger.error("regime_current_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/regime/hmm")
def get_hmm_regime(
    n_states: int = Query(default=4, ge=2, le=4),
    lookback_days: int = Query(default=756, ge=252, le=1260),
) -> dict:
    """
    HMM-based regime detection. Fits a {n_states}-state Gaussian HMM on
    SPY returns, VIX, 2s10s, IG OAS, and DXY. Returns current regime,
    state probabilities, transition matrix, and recent regime sequence.
    """
    try:
        hmm = HiddenMarkovRegimeModel(n_states=n_states, lookback_days=lookback_days)
        hmm.fit()
        result = hmm.detect_regime()
        return result.model_dump()
    except Exception as exc:
        logger.error("hmm_regime_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/regime/nowcast")
def get_macro_nowcast(
    lookback_days: int = Query(default=252, ge=63, le=756),
) -> dict:
    """
    Macro nowcast: growth and inflation composite Z-scores plus regime quadrant.
    Uses FRED series: ICSA, INDPRO, BAMLC0A0CM, T10Y2Y, T5YIE, CPIAUCSL, VIXCLS.
    """
    try:
        result = _nowcaster.nowcast(lookback_days=lookback_days)
        return result.model_dump()
    except Exception as exc:
        logger.error("nowcast_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/regime/multi-asset")
def get_multi_asset_regime() -> dict:
    """
    Regime classification across equity, fixed income, FX, and commodity.
    Returns individual regimes and a 4-character composite code.
    """
    try:
        result = _multi_asset.snapshot()
        return result.model_dump()
    except Exception as exc:
        logger.error("multi_asset_regime_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/regime/transition-probs")
def get_transition_probs(
    current_regime: Optional[str] = Query(default=None),
) -> dict:
    """
    Forward-looking transition probabilities (30D, 60D, 90D) from the HMM model.
    Also returns the most likely regime at 30 days and recommended asset allocation.
    """
    try:
        predictor = _get_predictor()
        result = predictor.predict(current_regime=current_regime)
        return result.model_dump()
    except Exception as exc:
        logger.error("transition_probs_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/regime/early-warning")
def get_early_warning() -> dict:
    """
    Early warning system: 0–100 composite risk score from VIX spike,
    credit spread widening, yield curve inversion, momentum breakdown, and BEI expansion.
    """
    try:
        result = _ews.assess()
        return result.model_dump()
    except Exception as exc:
        logger.error("early_warning_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/regime/history")
def get_regime_history(
    days: int = Query(default=252, ge=30, le=1260),
) -> dict:
    """
    Historical daily regime classification using FRED signals
    (VIX, 2s10s, IG OAS). Returns approximate regime labels and composite score.
    """
    try:
        df = _build_regime_history(days)
        if df.empty:
            return {"history": [], "n_days": 0}
        records = df.reset_index()
        records["date"] = records["date"].astype(str)
        return {
            "history": records.where(records.notna(), None).to_dict(orient="records"),
            "n_days": len(records),
        }
    except Exception as exc:
        logger.error("regime_history_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_current_regime_snapshot() -> dict:
    """Convenience: current regime from FRED signals (no HMM fitting required)."""
    vix_s = _fetch_fred("VIXCLS", 5)
    spread_s = _fetch_fred("T10Y2Y", 5)
    ig_s = _fetch_fred("BAMLC0A0CM", 5)
    vix = _latest(vix_s)
    spread = _latest(spread_s)
    ig = _latest(ig_s)
    return {
        "vix": vix,
        "spread_2s10s": spread,
        "ig_oas": ig,
        "as_of": date.today().isoformat(),
    }


def get_nowcast_snapshot() -> dict:
    """Convenience: macro nowcast result as dict."""
    return MacroNowcaster().nowcast().model_dump()


def get_early_warning_snapshot() -> dict:
    """Convenience: early warning assessment as dict."""
    return EarlyWarningSystem().assess().model_dump()


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "HiddenMarkovRegimeModel",
    "MacroNowcaster",
    "MultiAssetRegimeEngine",
    "RegimeTransitionPredictor",
    "EarlyWarningSystem",
    "regime_router",
    "get_current_regime_snapshot",
    "get_nowcast_snapshot",
    "get_early_warning_snapshot",
    # Pydantic models
    "HMMRegimeResult",
    "NowcastResult",
    "MultiAssetRegimeResult",
    "TransitionProbResult",
    "EarlyWarningResult",
    # Constants
    "REGIME_ALLOCATIONS",
    "REGIME_LABELS_4",
]
