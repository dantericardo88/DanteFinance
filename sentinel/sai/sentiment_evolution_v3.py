"""
Sentiment evolution tracking (SAI layer) — re-exports from sentinel.sma and
provides the named classes expected by dim_134 capability test.

dim_134 — Sentiment evolution tracking (time-series NLP drift)

Exports
-------
SentimentEvolution          — high-level tracker class (dim_134 interface)
SentimentDriftDetector      — drift and inflection detector
compute_sentiment_momentum  — convenience function

Full API (re-exported from sentinel.sma.sentiment_evolution_v3)
--------------------------------------------------------------
TextSentimentAnalyzer, SentimentEvolutionTracker, SentimentTimeSeries,
SentimentPoint, score_text, build_sentiment_series, sentiment_momentum,
sentiment_regime, lead_lag_correlation
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sentinel.sma.sentiment_evolution_v3 import (
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    SentimentEvolutionTracker,
    SentimentPoint,
    SentimentTimeSeries,
    TextSentimentAnalyzer,
    build_sentiment_series,
    lead_lag_correlation,
    score_text,
    sentiment_momentum,
    sentiment_regime,
)


# ---------------------------------------------------------------------------
# SAI-layer wrapper classes (dim_134 interface)
# ---------------------------------------------------------------------------


class SentimentEvolution:
    """High-level sentiment evolution tracker.

    Wraps TextSentimentAnalyzer + SentimentEvolutionTracker into a single
    convenient interface for the SAI (Sentiment AI) layer.

    Usage
    -----
    se = SentimentEvolution(texts)
    trend   = se.trend()
    regime  = se.regime()
    momentum = se.momentum(k=3)
    inflections = se.inflection_points()
    """

    def __init__(
        self,
        texts: List[str],
        positive_words: Optional[set] = None,
        negative_words: Optional[set] = None,
    ) -> None:
        self._analyzer = TextSentimentAnalyzer(
            positive_words=positive_words,
            negative_words=negative_words,
        )
        self._ts = self._analyzer.score_series(texts)
        self._tracker = SentimentEvolutionTracker(self._ts)

    # --- Delegation to time series ---

    @property
    def series(self) -> SentimentTimeSeries:
        """Underlying SentimentTimeSeries."""
        return self._ts

    @property
    def scores(self) -> np.ndarray:
        """Sentiment score array."""
        return self._ts.scores

    def drift(self) -> np.ndarray:
        """First-difference of scores (velocity)."""
        return self._ts.drift()

    def momentum(self, k: int = 3) -> np.ndarray:
        """k-period sentiment change."""
        return self._ts.momentum(k)

    def volatility(self, window: int = 5) -> np.ndarray:
        """Rolling sentiment volatility."""
        return self._ts.volatility(window)

    def regime(self) -> np.ndarray:
        """Per-point regime: 1=bull, -1=bear, 0=neutral."""
        return self._ts.regime()

    # --- Delegation to tracker ---

    def trend(self) -> float:
        """OLS slope of scores over time."""
        return self._tracker.trend()

    def inflection_points(self) -> List[int]:
        """Indices of regime changes."""
        return self._tracker.inflection_points()

    def predict_next(self) -> float:
        """AR(1) extrapolation of next score."""
        return self._tracker.predict_next()

    def correlation_with_returns(
        self, returns: np.ndarray, lag: int = 1
    ) -> float:
        """Pearson correlation of scores with lagged returns."""
        return self._tracker.correlation_with_returns(returns, lag)

    def lead_lag_analysis(
        self, returns: np.ndarray, max_lag: int = 5
    ) -> dict:
        """Best lag + full lag profile."""
        return self._tracker.lead_lag_analysis(returns, max_lag)

    def regime_returns(self, returns: np.ndarray) -> dict:
        """Mean return by sentiment regime."""
        return self._tracker.regime_returns(returns)

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._ts.points)
        return f"SentimentEvolution(n_texts={n})"


class SentimentDriftDetector:
    """Detects drift and inflection points in a sentiment time series.

    Usage
    -----
    detector = SentimentDriftDetector(ts)
    # or build from texts:
    detector = SentimentDriftDetector.from_texts(texts)

    inflections = detector.inflection_points()
    zero_crossings = detector.zero_crossings()
    momentum_shifts = detector.momentum_shifts(k=5)
    """

    def __init__(self, ts: SentimentTimeSeries) -> None:
        self._ts = ts
        self._tracker = SentimentEvolutionTracker(ts)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        positive_words: Optional[set] = None,
        negative_words: Optional[set] = None,
    ) -> "SentimentDriftDetector":
        """Build from a list of text strings."""
        analyzer = TextSentimentAnalyzer(
            positive_words=positive_words,
            negative_words=negative_words,
        )
        ts = analyzer.score_series(texts)
        return cls(ts)

    def inflection_points(self) -> List[int]:
        """Indices where the regime changes."""
        return self._tracker.inflection_points()

    def zero_crossings(self) -> List[int]:
        """Indices where drift changes sign (potential reversal)."""
        d = self._ts.drift()
        if len(d) < 2:
            return []
        crossings = []
        for i in range(1, len(d)):
            if np.sign(d[i]) != np.sign(d[i - 1]) and d[i - 1] != 0:
                crossings.append(i + 1)  # +1 to map back to score index
        return crossings

    def momentum_shifts(self, k: int = 5) -> List[int]:
        """Indices where k-period momentum changes sign."""
        s = self._ts.scores
        n = len(s)
        if n <= k:
            return []
        mom = s[k:] - s[: n - k]
        shifts = []
        for i in range(1, len(mom)):
            if np.sign(mom[i]) != np.sign(mom[i - 1]) and mom[i - 1] != 0:
                shifts.append(i + k)  # map back to original index
        return shifts

    def drift_sma(self, window: int = 5) -> np.ndarray:
        """Rolling mean of drift (drift smoothed)."""
        d = self._ts.drift()
        n = len(d)
        if n < window:
            return d.copy()
        out = np.zeros(n - window + 1)
        for i in range(n - window + 1):
            out[i] = d[i : i + window].mean()
        return out

    def trend(self) -> float:
        """OLS slope of scores over time."""
        return self._tracker.trend()

    def predict_next(self) -> float:
        """AR(1) extrapolation of next score."""
        return self._tracker.predict_next()

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._ts.points)
        return f"SentimentDriftDetector(n_points={n})"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def compute_sentiment_momentum(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """k-period sentiment momentum: scores[k:] - scores[:-k].

    Parameters
    ----------
    scores : array of sentiment scores
    k      : look-back periods

    Returns
    -------
    np.ndarray of length n-k
    """
    return sentiment_momentum(scores, k)


__all__ = [
    # SAI-layer classes
    "SentimentEvolution",
    "SentimentDriftDetector",
    # Convenience function
    "compute_sentiment_momentum",
    # Re-exported core types
    "SentimentTimeSeries",
    "SentimentPoint",
    "TextSentimentAnalyzer",
    "SentimentEvolutionTracker",
    # Re-exported convenience functions
    "score_text",
    "build_sentiment_series",
    "sentiment_momentum",
    "sentiment_regime",
    "lead_lag_correlation",
    # Lexicons
    "POSITIVE_WORDS",
    "NEGATIVE_WORDS",
]
