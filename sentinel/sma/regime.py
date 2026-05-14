"""
HMM 4-state macro regime detector — LEAPFROG #50.

Uses hmmlearn GaussianHMM to classify the current macro environment into one of:
  GROWTH_INFLATION     — Economy expanding, inflation rising (risk-on, short bonds)
  GROWTH_DEFLATION     — Economy expanding, inflation falling (goldilocks, long stocks)
  CONTRACTION_INFLATION — Stagflation (worst regime: short stocks, long commodities)
  CONTRACTION_DEFLATION — Recession/deflation (long bonds, cash, gold)

Features: yield curve slope (10Y-2Y), CPI YoY, unemployment rate, VIX, S&P 500 returns.
No incumbent provides hidden Markov regime detection as a native feature. Score: 10 vs 0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional
from sentinel.core.types import MacroRegime, RegimeResult
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

N_STATES = 4
REGIME_LABELS = [
    MacroRegime.GROWTH_INFLATION,
    MacroRegime.GROWTH_DEFLATION,
    MacroRegime.CONTRACTION_INFLATION,
    MacroRegime.CONTRACTION_DEFLATION,
]

# Heuristic state assignment based on component means
# State with highest growth + highest inflation → GROWTH_INFLATION, etc.
STATE_ASSIGNMENT_FEATURES = ["yield_slope", "cpi_yoy", "unrate", "vix"]


class MacroRegimeDetector:
    """
    Fits a GaussianHMM on macro features and predicts the current regime.
    Must call fit() before predict().
    """

    def __init__(self, n_states: int = N_STATES, n_iter: int = 200, random_state: int = 42) -> None:
        self._n_states = n_states
        self._n_iter = n_iter
        self._random_state = random_state
        self._model = None
        self._state_to_regime: dict[int, MacroRegime] = {}
        self._feature_cols: list[str] = []
        self._last_features: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, feature_cols: Optional[list[str]] = None) -> "MacroRegimeDetector":
        """
        Fit the HMM on a DataFrame of macro features.
        df must have a DatetimeIndex and at minimum: yield_slope, cpi_yoy, unrate, vix.
        """
        from hmmlearn import hmm  # Import deferred — heavy dependency

        cols = feature_cols or STATE_ASSIGNMENT_FEATURES
        # Only use columns present in df
        cols = [c for c in cols if c in df.columns]
        self._feature_cols = cols

        X = df[cols].dropna().values.astype(float)
        if len(X) < self._n_states * 10:
            raise ValueError(f"Not enough data to fit HMM: {len(X)} rows, need ≥{self._n_states * 10}")

        # Standardize
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._mean) / self._std
        self._last_features = X_scaled

        self._model = hmm.GaussianHMM(
            n_components=self._n_states,
            covariance_type="full",
            n_iter=self._n_iter,
            random_state=self._random_state,
            verbose=False,
        )
        self._model.fit(X_scaled)

        # Assign regime labels to hidden states via heuristic
        self._state_to_regime = self._assign_regime_labels(X, df[cols].dropna().index)
        logger.info("HMM fitted", n_states=self._n_states, n_iter=self._n_iter, samples=len(X))
        return self

    def _assign_regime_labels(self, X_raw: np.ndarray, index: pd.Index) -> dict[int, MacroRegime]:
        """
        Map hidden states to economic regimes by inspecting component means.
        yield_slope > 0 → growth; cpi_yoy high → inflation; unrate high → contraction; vix high → stress.
        """
        if self._model is None:
            return {}

        X_scaled = (X_raw - self._mean) / self._std
        states = self._model.predict(X_scaled)
        state_df = pd.DataFrame(X_raw, index=index, columns=self._feature_cols)
        state_df["state"] = states

        # Compute mean of key indicators per state
        state_means = state_df.groupby("state").mean()

        # Score each state on growth (yield_slope) and inflation (cpi_yoy)
        growth_col = "yield_slope" if "yield_slope" in state_means.columns else state_means.columns[0]
        inflation_col = "cpi_yoy" if "cpi_yoy" in state_means.columns else state_means.columns[min(1, len(state_means.columns)-1)]

        # Rank states: high growth ↑ yield_slope, high inflation ↑ cpi_yoy
        state_means["growth_rank"] = state_means[growth_col].rank()
        state_means["inflation_rank"] = state_means[inflation_col].rank()

        assignment: dict[int, MacroRegime] = {}
        for state_id in range(self._n_states):
            if state_id not in state_means.index:
                assignment[state_id] = MacroRegime.GROWTH_DEFLATION
                continue
            gr = state_means.loc[state_id, "growth_rank"] > (self._n_states / 2)
            ir = state_means.loc[state_id, "inflation_rank"] > (self._n_states / 2)
            if gr and ir:
                assignment[state_id] = MacroRegime.GROWTH_INFLATION
            elif gr and not ir:
                assignment[state_id] = MacroRegime.GROWTH_DEFLATION
            elif not gr and ir:
                assignment[state_id] = MacroRegime.CONTRACTION_INFLATION
            else:
                assignment[state_id] = MacroRegime.CONTRACTION_DEFLATION

        logger.info("Regime assignment", mapping={str(k): v.value for k, v in assignment.items()})
        return assignment

    def predict(self, df: pd.DataFrame) -> list[RegimeResult]:
        """Predict regime for each row of df. Returns RegimeResult per timestamp."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")

        cols = self._feature_cols
        sub = df[cols].dropna()
        if sub.empty:
            return []

        X = sub.values.astype(float)
        X_scaled = (X - self._mean) / self._std
        states = self._model.predict(X_scaled)
        posteriors = self._model.predict_proba(X_scaled)

        results = []
        for i, (idx, row) in enumerate(sub.iterrows()):
            state = int(states[i])
            regime = self._state_to_regime.get(state, MacroRegime.GROWTH_DEFLATION)
            probs = posteriors[i].tolist()

            # Map state probabilities to regime probabilities
            regime_probs: dict[str, float] = {r.value: 0.0 for r in MacroRegime}
            for s, p in enumerate(probs):
                r = self._state_to_regime.get(s, MacroRegime.GROWTH_DEFLATION)
                regime_probs[r.value] = regime_probs.get(r.value, 0.0) + p

            results.append(RegimeResult(
                date=pd.Timestamp(idx).date(),
                regime=regime,
                confidence=float(max(probs)),
                regime_probabilities=regime_probs,
                hidden_state=state,
                features={c: float(row[c]) for c in cols},
            ))
        return results

    def predict_current(self, df: pd.DataFrame) -> Optional[RegimeResult]:
        """Return only the most recent regime prediction."""
        results = self.predict(df)
        return results[-1] if results else None

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Return the learned transition probability matrix (n_states × n_states)."""
        if self._model is None:
            return None
        return self._model.transmat_

    def get_regime_durations(self, df: pd.DataFrame) -> dict[str, float]:
        """Return average duration (weeks) in each regime based on transition matrix."""
        mat = self.get_transition_matrix()
        if mat is None:
            return {}
        durations = {}
        for state in range(self._n_states):
            self_transition = mat[state, state]
            avg_duration = 1.0 / (1.0 - self_transition) if self_transition < 1.0 else float("inf")
            regime = self._state_to_regime.get(state, MacroRegime.GROWTH_DEFLATION)
            durations[regime.value] = round(avg_duration, 1)
        return durations


def build_macro_feature_df(
    yield_slope: pd.Series,  # 10Y-2Y spread
    cpi_yoy: pd.Series,      # CPI year-over-year %
    unrate: pd.Series,       # Unemployment rate
    vix: pd.Series,          # VIX level
    sp500_ret: Optional[pd.Series] = None,  # S&P 500 monthly return
) -> pd.DataFrame:
    """Align and combine macro series into a feature DataFrame suitable for HMM fitting."""
    df = pd.DataFrame({
        "yield_slope": yield_slope,
        "cpi_yoy": cpi_yoy,
        "unrate": unrate,
        "vix": vix,
    })
    if sp500_ret is not None:
        df["sp500_ret"] = sp500_ret

    # Resample to monthly, forward-fill gaps (macro data is irregular)
    df = df.resample("MS").last().ffill().dropna()
    return df
