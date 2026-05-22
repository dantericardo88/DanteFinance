"""
Volatility Regime Detection via Hidden Markov Models — pure numpy/scipy, zero
external HMM dependencies (hmmlearn is tried gracefully but not required).

dim_116 — Vol regime detection (HMM / Markov-switching)

Classes
-------
HMMParams
    Container for fitted 2-state Gaussian HMM parameters with derived properties.

GaussianHMM
    Gaussian HMM fitted via Baum-Welch (EM) implemented from scratch.
    .fit()          → GaussianHMM  (trains via EM)
    .predict()      → np.ndarray   (Viterbi most-likely state sequence)
    .predict_proba()→ np.ndarray   (T × n_states posterior state probabilities)
    .score()        → float        (log-likelihood)
    .decode()       → (log_prob, states)

RegimeAnalytics
    High-level analytics on top of a fitted GaussianHMM.

MarkovSwitchingVol
    2-regime Markov-switching conditional volatility (MS-GARCH simplified).

Convenience functions
---------------------
fit_hmm             Fit GaussianHMM and return model.
detect_regimes      Return 0/1 state labels for a return series.
regime_vol_ratio    Ratio high_vol_sigma / low_vol_sigma.
compute_regime_stats Mean / std by regime.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

# ──────────────────────────────────────────────────────────────────────────────
# Optional hmmlearn (graceful fallback)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM as _HmmlearnGHMM  # type: ignore[import]
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HmmlearnGHMM = None  # type: ignore[assignment,misc]
    _HMMLEARN_AVAILABLE = False

_LOG_2PI = np.log(2.0 * np.pi)
_EPS = 1e-300  # floor to avoid log(0)


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HMMParams:
    """Fitted Gaussian HMM parameters."""

    means: np.ndarray          # shape (n_states,)
    stds: np.ndarray           # shape (n_states,)
    trans_matrix: np.ndarray   # shape (n_states, n_states)
    initial_probs: np.ndarray  # shape (n_states,)
    log_likelihood: float = 0.0
    n_iter: int = 0
    converged: bool = False

    @property
    def n_states(self) -> int:
        return len(self.means)

    def regime_durations(self) -> np.ndarray:
        """Expected duration in each state = 1 / (1 - A[k,k])."""
        diag = np.diag(self.trans_matrix)
        diag_clipped = np.clip(diag, 0.0, 1.0 - 1e-10)
        return 1.0 / (1.0 - diag_clipped)

    def stationary_distribution(self) -> np.ndarray:
        """Stationary distribution — left eigenvector of trans_matrix for eigenvalue 1."""
        K = self.n_states
        # Solve pi @ A = pi with sum(pi) = 1
        # Equivalent: (A^T - I) @ pi = 0
        A = self.trans_matrix.T - np.eye(K)
        A[-1, :] = 1.0  # normalisation constraint
        b = np.zeros(K)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            pi = np.ones(K) / K
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi

    def __repr__(self) -> str:  # pragma: no cover
        dur = self.regime_durations()
        return (
            f"HMMParams(n_states={self.n_states}, "
            f"means={np.round(self.means, 6)}, "
            f"stds={np.round(self.stds, 6)}, "
            f"ll={self.log_likelihood:.2f}, "
            f"n_iter={self.n_iter}, converged={self.converged}, "
            f"durations={np.round(dur, 1)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    """Log N(x; mu, sigma^2)."""
    if sigma <= 0:
        return -1e300
    return -0.5 * (_LOG_2PI + 2.0 * np.log(sigma) + ((x - mu) / sigma) ** 2)


def _emission_log_probs(obs: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Compute log emission probabilities for all observations and states.

    Returns shape (T, K).
    """
    T = len(obs)
    K = len(means)
    log_b = np.empty((T, K))
    for k in range(K):
        log_b[:, k] = norm.logpdf(obs, loc=means[k], scale=stds[k])
    return log_b


def _forward_log(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Log-space forward pass (numerically stable).

    Parameters
    ----------
    log_b : (T, K) log emission probabilities
    log_A : (K, K) log transition matrix
    log_pi: (K,)   log initial distribution

    Returns
    -------
    log_alpha : (T, K)
    log_scales : (T,) — used for log-likelihood (sum of log normalisation constants)
    """
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_scales = np.empty(T)

    # t=0
    log_alpha[0] = log_pi + log_b[0]
    lse = _logsumexp(log_alpha[0])
    log_scales[0] = lse
    log_alpha[0] -= lse

    for t in range(1, T):
        # log_alpha[t, j] = log_b[t, j] + logsumexp_k(log_alpha[t-1, k] + log_A[k, j])
        for j in range(K):
            log_alpha[t, j] = log_b[t, j] + _logsumexp(log_alpha[t - 1] + log_A[:, j])
        lse = _logsumexp(log_alpha[t])
        log_scales[t] = lse
        log_alpha[t] -= lse

    return log_alpha, log_scales


def _backward_log(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_scales: np.ndarray,
) -> np.ndarray:
    """
    Log-space backward pass.

    Returns
    -------
    log_beta : (T, K)
    """
    T, K = log_b.shape
    log_beta = np.zeros((T, K))

    for t in range(T - 2, -1, -1):
        for k in range(K):
            log_beta[t, k] = _logsumexp(
                log_A[k, :] + log_b[t + 1, :] + log_beta[t + 1, :]
            )
        # scale symmetrically (subtract the same scale used in forward)
        log_beta[t] -= log_scales[t + 1]

    return log_beta


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable logsumexp."""
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


# ──────────────────────────────────────────────────────────────────────────────
# GaussianHMM — Baum-Welch EM from scratch
# ──────────────────────────────────────────────────────────────────────────────

class GaussianHMM:
    """
    2-state (or n-state) Gaussian Hidden Markov Model.

    Training uses Baum-Welch (EM) implemented from scratch in log-space for
    numerical stability.  Decoding uses the Viterbi algorithm.

    Parameters
    ----------
    n_states : int, default 2
    n_iter   : int, default 100  — maximum EM iterations
    tol      : float, default 1e-4 — convergence threshold on log-likelihood change
    random_state : int, optional
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self._params: Optional[HMMParams] = None

    # ── public interface ──────────────────────────────────────────────────────

    def fit(self, observations: np.ndarray) -> "GaussianHMM":
        """Fit via Baum-Welch EM.  Returns self."""
        obs = np.asarray(observations, dtype=float)
        if obs.ndim != 1:
            raise ValueError("observations must be 1-D array")
        T = len(obs)
        K = self.n_states

        rng = np.random.default_rng(self.random_state)

        # ── Initialise parameters ──────────────────────────────────────────────
        # k-means-like init: sort observations, split evenly
        sorted_obs = np.sort(obs)
        chunk = T // K
        means = np.array([sorted_obs[i * chunk + chunk // 2] for i in range(K)], dtype=float)
        # Use robust std per chunk; fallback to global
        stds = np.empty(K)
        for k in range(K):
            chunk_data = sorted_obs[k * chunk: (k + 1) * chunk]
            s = float(np.std(chunk_data))
            stds[k] = s if s > 1e-10 else float(np.std(obs)) / K

        trans_matrix = np.full((K, K), 1.0 / K)
        # Slightly higher self-transition to encourage persistence
        np.fill_diagonal(trans_matrix, 0.7)
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_matrix /= row_sums

        initial_probs = np.ones(K) / K

        prev_ll = -np.inf
        converged = False
        n_iter_done = 0

        for iteration in range(self.n_iter):
            n_iter_done = iteration + 1

            # ── Pre-compute log quantities ─────────────────────────────────────
            log_A = np.log(np.clip(trans_matrix, _EPS, None))
            log_pi = np.log(np.clip(initial_probs, _EPS, None))
            log_b = _emission_log_probs(obs, means, stds)

            # ── Forward pass ──────────────────────────────────────────────────
            log_alpha, log_scales = _forward_log(log_b, log_A, log_pi)

            # Log-likelihood = sum of log scale factors
            ll = float(np.sum(log_scales))

            # ── Backward pass ─────────────────────────────────────────────────
            log_beta = _backward_log(log_b, log_A, log_scales)

            # ── E-step: compute gamma and xi ──────────────────────────────────
            # gamma(t, k) = P(S_t = k | obs, params)
            log_gamma = log_alpha + log_beta
            # normalise row-wise in log space
            log_gamma_normaliser = np.array([_logsumexp(log_gamma[t]) for t in range(T)])
            log_gamma -= log_gamma_normaliser[:, None]
            gamma = np.exp(log_gamma)

            # xi(t, k, j) = P(S_t=k, S_{t+1}=j | obs, params)  for t < T-1
            # xi[t, k, j] ∝ alpha_t(k) * A[k,j] * b_{t+1}(j) * beta_{t+1}(j)
            log_xi = np.empty((T - 1, K, K))
            for t in range(T - 1):
                for k in range(K):
                    for j in range(K):
                        log_xi[t, k, j] = (
                            log_alpha[t, k]
                            + log_A[k, j]
                            + log_b[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                # Normalise
                denom = _logsumexp(log_xi[t].ravel())
                log_xi[t] -= denom

            xi = np.exp(log_xi)  # (T-1, K, K)

            # ── M-step: update parameters ─────────────────────────────────────
            # Initial probs
            initial_probs = gamma[0]
            initial_probs = np.clip(initial_probs, _EPS, None)
            initial_probs /= initial_probs.sum()

            # Transition matrix
            for k in range(K):
                for j in range(K):
                    trans_matrix[k, j] = xi[:, k, j].sum()
                row_sum = trans_matrix[k].sum()
                if row_sum > _EPS:
                    trans_matrix[k] /= row_sum
                else:
                    trans_matrix[k] = np.ones(K) / K

            # Means and stds
            gamma_sum = gamma.sum(axis=0)  # (K,)
            for k in range(K):
                g = gamma[:, k]
                gs = gamma_sum[k]
                if gs < _EPS:
                    continue
                means[k] = (g * obs).sum() / gs
                variance = (g * (obs - means[k]) ** 2).sum() / gs
                stds[k] = float(np.sqrt(max(variance, 1e-12)))

            # ── Convergence check ─────────────────────────────────────────────
            if abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

        # Store params
        self._params = HMMParams(
            means=means.copy(),
            stds=stds.copy(),
            trans_matrix=trans_matrix.copy(),
            initial_probs=initial_probs.copy(),
            log_likelihood=float(ll),
            n_iter=n_iter_done,
            converged=converged,
        )
        return self

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi most likely state sequence. Returns int array of shape (T,)."""
        _, states = self.decode(observations)
        return states

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Posterior state probabilities via forward-backward.

        Returns
        -------
        gamma : np.ndarray of shape (T, n_states)
            gamma[t, k] = P(S_t = k | obs, params)
        """
        self._check_fitted()
        obs = np.asarray(observations, dtype=float)
        p = self._params
        K = p.n_states
        T = len(obs)

        log_A = np.log(np.clip(p.trans_matrix, _EPS, None))
        log_pi = np.log(np.clip(p.initial_probs, _EPS, None))
        log_b = _emission_log_probs(obs, p.means, p.stds)

        log_alpha, log_scales = _forward_log(log_b, log_A, log_pi)
        log_beta = _backward_log(log_b, log_A, log_scales)

        log_gamma = log_alpha + log_beta
        log_gamma_normaliser = np.array([_logsumexp(log_gamma[t]) for t in range(T)])
        log_gamma -= log_gamma_normaliser[:, None]
        return np.exp(log_gamma)

    def score(self, observations: np.ndarray) -> float:
        """Log-likelihood of observations under the fitted model."""
        self._check_fitted()
        obs = np.asarray(observations, dtype=float)
        p = self._params
        log_A = np.log(np.clip(p.trans_matrix, _EPS, None))
        log_pi = np.log(np.clip(p.initial_probs, _EPS, None))
        log_b = _emission_log_probs(obs, p.means, p.stds)
        _, log_scales = _forward_log(log_b, log_A, log_pi)
        return float(np.sum(log_scales))

    def decode(self, observations: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Viterbi algorithm.

        Returns
        -------
        (log_prob, state_sequence)
        """
        self._check_fitted()
        obs = np.asarray(observations, dtype=float)
        p = self._params
        K = p.n_states
        T = len(obs)

        log_A = np.log(np.clip(p.trans_matrix, _EPS, None))
        log_pi = np.log(np.clip(p.initial_probs, _EPS, None))
        log_b = _emission_log_probs(obs, p.means, p.stds)

        # delta[t, k] = max log-prob of state sequence ending in state k at t
        log_delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)

        log_delta[0] = log_pi + log_b[0]

        for t in range(1, T):
            for j in range(K):
                candidates = log_delta[t - 1] + log_A[:, j]
                best = int(np.argmax(candidates))
                log_delta[t, j] = candidates[best] + log_b[t, j]
                psi[t, j] = best

        # Backtrack
        states = np.empty(T, dtype=int)
        states[T - 1] = int(np.argmax(log_delta[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        log_prob = float(np.max(log_delta[T - 1]))
        return log_prob, states

    @property
    def params(self) -> HMMParams:
        self._check_fitted()
        return self._params  # type: ignore[return-value]

    # ── private ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._params is None:
            raise RuntimeError("Model not yet fitted — call .fit() first.")


# ──────────────────────────────────────────────────────────────────────────────
# Regime Analytics
# ──────────────────────────────────────────────────────────────────────────────

class RegimeAnalytics:
    """
    High-level analytics wrapper around a fitted GaussianHMM.

    Parameters
    ----------
    hmm     : fitted GaussianHMM
    returns : 1-D return series used for fitting
    """

    def __init__(self, hmm: GaussianHMM, returns: np.ndarray) -> None:
        self._hmm = hmm
        self._returns = np.asarray(returns, dtype=float)
        self._params = hmm.params
        self._states = hmm.predict(returns)
        self._proba = hmm.predict_proba(returns)

    # ── State identification ──────────────────────────────────────────────────

    def low_vol_state(self) -> int:
        """Index of the low-volatility regime (smallest std)."""
        return int(np.argmin(self._params.stds))

    def high_vol_state(self) -> int:
        """Index of the high-volatility regime (largest std)."""
        return int(np.argmax(self._params.stds))

    def current_regime(self) -> str:
        """'low_vol' or 'high_vol' based on the last observation."""
        last_state = int(self._states[-1])
        return "low_vol" if last_state == self.low_vol_state() else "high_vol"

    # ── Regime periods ────────────────────────────────────────────────────────

    def regime_periods(self) -> List[dict]:
        """
        List of contiguous periods in each state.

        Each entry: {'state': int, 'start': int, 'end': int, 'duration': int}
        """
        states = self._states
        T = len(states)
        periods: List[dict] = []
        start = 0
        for t in range(1, T + 1):
            if t == T or states[t] != states[t - 1]:
                periods.append(
                    {
                        "state": int(states[t - 1]),
                        "start": start,
                        "end": t - 1,
                        "duration": t - start,
                    }
                )
                start = t
        return periods

    # ── Statistics ────────────────────────────────────────────────────────────

    def regime_returns(self) -> dict:
        """
        Mean and std of returns for each regime.

        Returns dict keyed by state index: {'mean': float, 'std': float, 'n': int}
        """
        result: dict = {}
        for k in range(self._params.n_states):
            mask = self._states == k
            r = self._returns[mask]
            result[k] = {
                "mean": float(np.mean(r)) if len(r) > 0 else float("nan"),
                "std": float(np.std(r)) if len(r) > 0 else float("nan"),
                "n": int(np.sum(mask)),
            }
        return result

    def transition_analysis(self) -> dict:
        """Expected durations and transition probabilities."""
        A = self._params.trans_matrix
        durations = self._params.regime_durations()
        return {
            "trans_matrix": A.tolist(),
            "expected_durations": durations.tolist(),
            "stationary": self._params.stationary_distribution().tolist(),
        }

    def regime_vol_ratio(self) -> float:
        """Ratio of high-vol sigma to low-vol sigma."""
        stds = self._params.stds
        return float(stds[self.high_vol_state()] / max(stds[self.low_vol_state()], 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# Markov-Switching Volatility (simplified MS-GARCH)
# ──────────────────────────────────────────────────────────────────────────────

class MarkovSwitchingVol:
    """
    2-regime (or n-regime) Markov-switching conditional volatility.

    Uses the same Baum-Welch EM framework as GaussianHMM but operates on
    absolute returns as the 'volatility' signal, then reconstructs a weighted
    conditional volatility path.
    """

    def __init__(self, n_states: int = 2) -> None:
        self.n_states = n_states
        self._hmm: Optional[GaussianHMM] = None
        self._returns: Optional[np.ndarray] = None
        self._proba: Optional[np.ndarray] = None

    def fit(self, returns: np.ndarray) -> "MarkovSwitchingVol":
        """
        Fit Markov-switching model to the return series.

        The emissions are modelled as Gaussian (mean and variance per state).
        """
        obs = np.asarray(returns, dtype=float)
        self._returns = obs
        hmm = GaussianHMM(n_states=self.n_states, n_iter=100, tol=1e-4, random_state=0)
        hmm.fit(obs)
        self._hmm = hmm
        self._proba = hmm.predict_proba(obs)
        return self

    def conditional_vol(self) -> np.ndarray:
        """
        Weighted conditional volatility path.

        sigma_t = sum_k P(S_t=k | obs) * sigma_k
        """
        if self._hmm is None or self._proba is None:
            raise RuntimeError("Call .fit() first.")
        stds = self._hmm.params.stds  # (K,)
        # weighted average of state stds
        return np.dot(self._proba, stds)  # (T,)

    def regime_conditional_vol(self, state: int) -> float:
        """Unconditional volatility for the given state."""
        if self._hmm is None:
            raise RuntimeError("Call .fit() first.")
        return float(self._hmm.params.stds[state])


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────────────

def fit_hmm(returns: np.ndarray, n_states: int = 2) -> GaussianHMM:
    """Fit a Gaussian HMM to ``returns`` and return the fitted model."""
    model = GaussianHMM(n_states=n_states, n_iter=200, tol=1e-5, random_state=42)
    model.fit(returns)
    return model


def detect_regimes(returns: np.ndarray) -> np.ndarray:
    """
    Fit 2-state HMM and return Viterbi state labels (0 or 1).

    The label 0 is assigned to the low-volatility state and 1 to the
    high-volatility state (labels are relabelled for consistency).
    """
    model = fit_hmm(returns, n_states=2)
    raw_states = model.predict(returns)
    stds = model.params.stds

    # Relabel: low-vol → 0, high-vol → 1
    low_idx = int(np.argmin(stds))
    out = np.where(raw_states == low_idx, 0, 1).astype(int)
    return out


def regime_vol_ratio(returns: np.ndarray) -> float:
    """
    Fit 2-state HMM and return sigma_high / sigma_low.
    """
    model = fit_hmm(returns, n_states=2)
    stds = model.params.stds
    return float(np.max(stds) / max(np.min(stds), 1e-12))


def compute_regime_stats(returns: np.ndarray, states: np.ndarray) -> dict:
    """
    Compute mean and std of ``returns`` for each unique state in ``states``.

    Returns
    -------
    dict keyed by state index: {'mean': float, 'std': float, 'n': int}
    """
    obs = np.asarray(returns, dtype=float)
    st = np.asarray(states, dtype=int)
    unique_states = np.unique(st)
    result: dict = {}
    for k in unique_states:
        mask = st == k
        r = obs[mask]
        result[int(k)] = {
            "mean": float(np.mean(r)) if len(r) > 0 else float("nan"),
            "std": float(np.std(r)) if len(r) > 0 else float("nan"),
            "n": int(np.sum(mask)),
        }
    return result
