"""
Cross-Asset Signal Fusion — pure numpy, zero external ML deps.

dim_133 — Cross-asset signal fusion (ML ensemble, correlation-weighted)

Implements multiple signal combination strategies:
  - Correlation-weighted ensemble (diversification weights)
  - IC-weighted ensemble (information-ratio weights)
  - Signal orthogonalization via Gram-Schmidt
  - Walk-forward cross-validation
  - Marginal contribution analysis

Classes
-------
SignalBundle
    Container for a (T, n_signals) matrix of signals plus forward returns.

FusionResult
    Holds the blended signal, per-signal and ensemble Sharpe, diversification ratio,
    weights, and signal correlation matrix.

CorrelationWeightedFusion
    Weights signals by (1 - avg_cross_correlation) to maximise diversification.
    .fit() / .transform() / .weights()

ICWeightedFusion
    Weights signals proportional to their rolling IC with forward returns.
    .fit() / .transform() / .rolling_ic()

SignalOrthogonalizer
    Gram-Schmidt orthogonalization of signal columns.
    .fit_transform() / .is_orthogonal()

WalkForwardValidator
    Rolling train/test validation of a fusion method.
    .validate() → per-fold Sharpe, IC, weights evolution

SignalFusionEngine
    High-level orchestrator: fuse, compare_methods, walk_forward,
    marginal_contribution.

Convenience functions
---------------------
correlation_weighted_ensemble   One-shot correlation-weighted fusion.
ic_weighted_ensemble            One-shot IC-weighted fusion.
orthogonalize_signals           Gram-Schmidt orthogonalize columns.
ensemble_sharpe                 Sharpe of an ensemble signal against returns.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SignalBundle:
    """
    Container for multiple alpha signals and their shared forward returns.

    Attributes
    ----------
    signals : np.ndarray
        Shape (T, n_signals).
    returns : np.ndarray
        Shape (T,) — forward returns aligned to signal timestamps.
    signal_names : List[str]
    """
    signals: np.ndarray
    returns: np.ndarray
    signal_names: List[str]

    def __post_init__(self) -> None:
        self.signals = np.asarray(self.signals, dtype=float)
        self.returns = np.asarray(self.returns, dtype=float)
        if self.signals.ndim == 1:
            self.signals = self.signals[:, np.newaxis]

    @property
    def n_signals(self) -> int:
        return self.signals.shape[1]

    @property
    def T(self) -> int:
        return self.signals.shape[0]


@dataclass
class FusionResult:
    """Output of a fusion method."""
    ensemble_signal: np.ndarray
    weights: np.ndarray
    individual_sharpes: np.ndarray
    ensemble_sharpe: float
    diversification_ratio: float
    correlation_matrix: np.ndarray


# ---------------------------------------------------------------------------
# Sharpe utility
# ---------------------------------------------------------------------------

def ensemble_sharpe(
    ensemble: np.ndarray,
    returns: np.ndarray,
    ann: float = 252.0,
) -> float:
    """
    Annualised Sharpe of a signal-driven strategy.

    Position = sign(ensemble); P&L = position * forward_return.

    Parameters
    ----------
    ensemble : np.ndarray
        Composite signal.
    returns : np.ndarray
        Forward returns.
    ann : float
        Annualisation factor (default 252 trading days).

    Returns
    -------
    float
    """
    ensemble = np.asarray(ensemble, dtype=float)
    returns = np.asarray(returns, dtype=float)
    n = min(len(ensemble), len(returns))
    pos = np.sign(ensemble[:n])
    pnl = pos * returns[:n]
    std_pnl = float(np.std(pnl, ddof=1))
    if std_pnl < 1e-12:
        return 0.0
    return float(np.mean(pnl) / std_pnl * np.sqrt(ann))


def _signal_sharpe(signal: np.ndarray, returns: np.ndarray, ann: float = 252.0) -> float:
    return ensemble_sharpe(signal, returns, ann)


def _pearson_ic(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with safety checks."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return 0.0
    a2, b2 = a[mask], b[mask]
    if np.std(a2) < 1e-12 or np.std(b2) < 1e-12:
        return 0.0
    return float(np.corrcoef(a2, b2)[0, 1])


# ---------------------------------------------------------------------------
# Correlation-weighted fusion
# ---------------------------------------------------------------------------

class CorrelationWeightedFusion:
    """
    Assign weights to signals based on their diversification value.

    Weight formula (from Choueifaty & Coignard 2008 inspired):
        w_i = (1 - avg_cross_corr_i) / sum_j(1 - avg_cross_corr_j)

    where avg_cross_corr_i = average |correlation| of signal i with all other signals.
    """

    def __init__(self) -> None:
        self._weights: Optional[np.ndarray] = None
        self._corr_matrix: Optional[np.ndarray] = None

    def fit(self, bundle: SignalBundle) -> "CorrelationWeightedFusion":
        S = bundle.signals  # (T, K)
        K = bundle.n_signals

        # Compute pairwise correlation matrix
        corr = np.corrcoef(S.T)  # (K, K)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        self._corr_matrix = corr

        if K == 1:
            self._weights = np.array([1.0])
            return self

        # avg cross-correlation for each signal (excluding self-correlation)
        avg_cross = np.zeros(K)
        for i in range(K):
            others = [abs(corr[i, j]) for j in range(K) if j != i]
            avg_cross[i] = float(np.mean(others)) if others else 0.0

        raw_weights = 1.0 - avg_cross
        # Clip to non-negative
        raw_weights = np.clip(raw_weights, 0.0, None)
        total = raw_weights.sum()
        if total < 1e-12:
            raw_weights = np.ones(K)
            total = float(K)
        self._weights = raw_weights / total
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Blend signals using fitted weights."""
        if self._weights is None:
            raise RuntimeError("Call .fit() first.")
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals[:, np.newaxis]
        return signals @ self._weights

    def weights(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call .fit() first.")
        return self._weights.copy()


# ---------------------------------------------------------------------------
# IC-weighted fusion
# ---------------------------------------------------------------------------

class ICWeightedFusion:
    """
    Weight signals proportional to their rolling Information Coefficient
    with forward returns.

    Falls back to equal weighting when any IC <= 0.
    """

    def __init__(self) -> None:
        self._weights: Optional[np.ndarray] = None
        self._rolling_ic: Optional[np.ndarray] = None

    def fit(
        self, bundle: SignalBundle, window: int = 60
    ) -> "ICWeightedFusion":
        S = bundle.signals  # (T, K)
        R = bundle.returns  # (T,)
        K = bundle.n_signals
        T = bundle.T

        # Compute rolling IC for each signal
        rolling_ic = np.full((T, K), np.nan)
        for t in range(window, T + 1):
            for k in range(K):
                rolling_ic[t - 1, k] = _pearson_ic(
                    S[t - window: t, k], R[t - window: t]
                )

        self._rolling_ic = rolling_ic

        # Use mean IC over valid (non-NaN) rows as weights
        mean_ic = np.nanmean(rolling_ic, axis=0)  # (K,)

        if np.any(mean_ic <= 0):
            # Fallback: equal weight
            self._weights = np.ones(K) / K
        else:
            total = mean_ic.sum()
            self._weights = mean_ic / total
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call .fit() first.")
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals[:, np.newaxis]
        return signals @ self._weights

    def weights(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call .fit() first.")
        return self._weights.copy()

    def rolling_ic(self) -> np.ndarray:
        if self._rolling_ic is None:
            raise RuntimeError("Call .fit() first.")
        return self._rolling_ic.copy()


# ---------------------------------------------------------------------------
# Signal Orthogonalizer (Gram-Schmidt)
# ---------------------------------------------------------------------------

class SignalOrthogonalizer:
    """
    Gram-Schmidt orthogonalization of signal columns.

    Each signal s_i has removed from it the projections onto all previous
    orthogonalized signals:

        s_i_orth = s_i - sum_{j<i} proj_{e_j}(s_i)

    where proj_{e_j}(s_i) = (s_i · e_j / |e_j|^2) * e_j
    """

    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Orthogonalize columns of signals via Gram-Schmidt.

        Parameters
        ----------
        signals : np.ndarray, shape (T, K)

        Returns
        -------
        orth : np.ndarray, shape (T, K)
        """
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals[:, np.newaxis]
        T, K = signals.shape
        orth = np.zeros_like(signals)

        for k in range(K):
            v = signals[:, k].copy()
            for j in range(k):
                e_j = orth[:, j]
                norm_sq = float(np.dot(e_j, e_j))
                if norm_sq > 1e-12:
                    v = v - (np.dot(v, e_j) / norm_sq) * e_j
            orth[:, k] = v

        return orth

    def is_orthogonal(self, signals: np.ndarray, tol: float = 1e-6) -> bool:
        """Check whether signal columns are mutually orthogonal."""
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            return True
        K = signals.shape[1]
        for i in range(K):
            for j in range(i + 1, K):
                dot = abs(float(np.dot(signals[:, i], signals[:, j])))
                if dot > tol:
                    return False
        return True


# ---------------------------------------------------------------------------
# Walk-forward Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Rolling walk-forward validation of signal fusion strategies.

    Parameters
    ----------
    train_window : int
        Number of time steps used for training in each fold (default 252).
    test_window : int
        Number of time steps used for evaluation in each fold (default 63).
    """

    def __init__(self, train_window: int = 252, test_window: int = 63) -> None:
        self.train_window = train_window
        self.test_window = test_window

    def validate(
        self,
        bundle: SignalBundle,
        fusion_method: str = "correlation",
    ) -> dict:
        """
        Run walk-forward folds and collect per-fold metrics.

        Returns
        -------
        dict with keys:
          fold_sharpes : List[float]
          fold_ics     : List[float]
          fold_weights : List[np.ndarray]
          n_folds      : int
        """
        T = bundle.T
        fold_sharpes: List[float] = []
        fold_ics: List[float] = []
        fold_weights: List[np.ndarray] = []

        t = self.train_window
        while t + self.test_window <= T:
            train_bundle = SignalBundle(
                signals=bundle.signals[t - self.train_window: t],
                returns=bundle.returns[t - self.train_window: t],
                signal_names=bundle.signal_names,
            )
            test_signals = bundle.signals[t: t + self.test_window]
            test_returns = bundle.returns[t: t + self.test_window]

            if fusion_method == "ic":
                fuser = ICWeightedFusion()
                fuser.fit(train_bundle, window=min(60, self.train_window // 4))
            else:  # default: correlation
                fuser = CorrelationWeightedFusion()
                fuser.fit(train_bundle)

            w = fuser.weights()
            ens = fuser.transform(test_signals)
            sharpe = _signal_sharpe(ens, test_returns)
            ic = _pearson_ic(ens, test_returns)

            fold_sharpes.append(sharpe)
            fold_ics.append(ic)
            fold_weights.append(w)
            t += self.test_window

        return {
            "fold_sharpes": fold_sharpes,
            "fold_ics": fold_ics,
            "fold_weights": fold_weights,
            "n_folds": len(fold_sharpes),
        }


# ---------------------------------------------------------------------------
# Signal Fusion Engine
# ---------------------------------------------------------------------------

class SignalFusionEngine:
    """
    High-level orchestrator for signal fusion strategies.

    Parameters
    ----------
    bundle : SignalBundle
        Signal data container.
    """

    def __init__(self, bundle: SignalBundle) -> None:
        self.bundle = bundle
        self._corr_fusion = CorrelationWeightedFusion()
        self._ic_fusion = ICWeightedFusion()
        self._orth = SignalOrthogonalizer()

    def _build_fusion_result(
        self,
        fuser,
        name: str = "fusion",
    ) -> FusionResult:
        """Fit fuser and build a FusionResult."""
        if isinstance(fuser, CorrelationWeightedFusion):
            fuser.fit(self.bundle)
        else:
            fuser.fit(self.bundle, window=min(60, self.bundle.T // 4 or 20))

        ens = fuser.transform(self.bundle.signals)
        w = fuser.weights()
        K = self.bundle.n_signals
        ind_sharpes = np.array([
            _signal_sharpe(self.bundle.signals[:, k], self.bundle.returns)
            for k in range(K)
        ])
        ens_sharpe = _signal_sharpe(ens, self.bundle.returns)
        max_ind = float(np.max(np.abs(ind_sharpes))) if len(ind_sharpes) else 1.0
        div_ratio = abs(ens_sharpe) / max(max_ind, 1e-12)

        corr = np.corrcoef(self.bundle.signals.T)
        if corr.ndim == 0:
            corr = np.array([[1.0]])

        return FusionResult(
            ensemble_signal=ens,
            weights=w,
            individual_sharpes=ind_sharpes,
            ensemble_sharpe=ens_sharpe,
            diversification_ratio=div_ratio,
            correlation_matrix=corr,
        )

    def fuse(self, method: str = "correlation") -> FusionResult:
        """
        Fuse signals using the specified method.

        Parameters
        ----------
        method : str
            'correlation' | 'ic' | 'orthogonal'
        """
        if method == "ic":
            return self._build_fusion_result(ICWeightedFusion(), method)
        if method == "orthogonal":
            orth_signals = self._orth.fit_transform(self.bundle.signals)
            orth_bundle = SignalBundle(
                signals=orth_signals,
                returns=self.bundle.returns,
                signal_names=self.bundle.signal_names,
            )
            fuser = CorrelationWeightedFusion()
            fuser.fit(orth_bundle)
            ens = fuser.transform(orth_signals)
            w = fuser.weights()
            K = self.bundle.n_signals
            ind_sharpes = np.array([
                _signal_sharpe(orth_signals[:, k], self.bundle.returns)
                for k in range(K)
            ])
            ens_sharpe = _signal_sharpe(ens, self.bundle.returns)
            max_ind = float(np.max(np.abs(ind_sharpes))) if len(ind_sharpes) else 1.0
            div_ratio = abs(ens_sharpe) / max(max_ind, 1e-12)
            corr = np.corrcoef(self.bundle.signals.T)
            if corr.ndim == 0:
                corr = np.array([[1.0]])
            return FusionResult(
                ensemble_signal=ens,
                weights=w,
                individual_sharpes=ind_sharpes,
                ensemble_sharpe=ens_sharpe,
                diversification_ratio=div_ratio,
                correlation_matrix=corr,
            )
        # Default: correlation
        return self._build_fusion_result(CorrelationWeightedFusion(), method)

    def compare_methods(self) -> Dict[str, FusionResult]:
        """Compare correlation, IC, and orthogonal fusion methods."""
        results: Dict[str, FusionResult] = {}
        for method in ("correlation", "ic", "orthogonal"):
            try:
                results[method] = self.fuse(method)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Fusion method '{method}' failed: {exc}")
        return results

    def walk_forward(self) -> dict:
        """Run walk-forward validation with correlation weighting."""
        validator = WalkForwardValidator(
            train_window=min(252, self.bundle.T // 3 or 20),
            test_window=min(63, self.bundle.T // 6 or 10),
        )
        return validator.validate(self.bundle, fusion_method="correlation")

    def marginal_contribution(self) -> np.ndarray:
        """
        Marginal Sharpe contribution of each signal.

        Defined as:
            Sharpe(ensemble) - Sharpe(ensemble without signal i)
        using equal-weight ensemble as baseline.
        """
        K = self.bundle.n_signals
        S = self.bundle.signals
        R = self.bundle.returns

        # Full equal-weight ensemble
        ens_full = S.mean(axis=1)
        sharpe_full = _signal_sharpe(ens_full, R)

        mc = np.zeros(K)
        for i in range(K):
            cols = [j for j in range(K) if j != i]
            if not cols:
                mc[i] = 0.0
                continue
            ens_excl = S[:, cols].mean(axis=1)
            mc[i] = sharpe_full - _signal_sharpe(ens_excl, R)
        return mc


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def correlation_weighted_ensemble(
    signals: np.ndarray, returns: np.ndarray
) -> FusionResult:
    """
    One-shot correlation-weighted signal fusion.

    Parameters
    ----------
    signals : np.ndarray, shape (T, K)
    returns : np.ndarray, shape (T,)

    Returns
    -------
    FusionResult
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    K = signals.shape[1]
    names = [f"signal_{i}" for i in range(K)]
    bundle = SignalBundle(signals=signals, returns=returns, signal_names=names)
    engine = SignalFusionEngine(bundle)
    return engine.fuse("correlation")


def ic_weighted_ensemble(
    signals: np.ndarray, returns: np.ndarray
) -> FusionResult:
    """
    One-shot IC-weighted signal fusion.

    Parameters
    ----------
    signals : np.ndarray, shape (T, K)
    returns : np.ndarray, shape (T,)

    Returns
    -------
    FusionResult
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    K = signals.shape[1]
    names = [f"signal_{i}" for i in range(K)]
    bundle = SignalBundle(signals=signals, returns=returns, signal_names=names)
    engine = SignalFusionEngine(bundle)
    return engine.fuse("ic")


def orthogonalize_signals(signals: np.ndarray) -> np.ndarray:
    """
    Gram-Schmidt orthogonalize the columns of a signal matrix.

    Parameters
    ----------
    signals : np.ndarray, shape (T, K)

    Returns
    -------
    orth : np.ndarray, shape (T, K)
    """
    return SignalOrthogonalizer().fit_transform(signals)
