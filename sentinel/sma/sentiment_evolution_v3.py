"""
Sentiment evolution tracking — time-series NLP drift using simple lexicon.
Pure numpy + re — zero network calls, zero external NLP libs.

dim_134 — Sentiment evolution tracking (target: 9)

Classes
-------
SentimentPoint
    Single time-step sentiment reading.

SentimentTimeSeries
    Full time series of SentimentPoints with properties:
    .scores        → np.ndarray
    .timestamps    → np.ndarray
    .drift()       → first-differences of scores
    .momentum(k)   → k-period score change
    .volatility(w) → rolling std of scores
    .regime()      → 1=bull, -1=bear, 0=neutral per point

TextSentimentAnalyzer
    .score()          → single-text score in [-1, +1]
    .score_series()   → SentimentTimeSeries from list of texts
    .extract_topics() → per-topic keyword coverage fractions
    .score_by_topic() → per-topic score arrays

SentimentEvolutionTracker
    .trend()                       → OLS slope of scores over time
    .inflection_points()           → indices of regime changes
    .predict_next()                → AR(1) extrapolation
    .correlation_with_returns()    → Pearson(score_t, return_{t+lag})
    .lead_lag_analysis()           → best lag + full lag profile
    .regime_returns()              → mean return by regime

Convenience functions
---------------------
score_text, build_sentiment_series, sentiment_momentum,
sentiment_regime, lead_lag_correlation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Default lexicons
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "growth", "strong", "beat", "exceeded", "record", "robust", "momentum",
    "confident", "positive", "improve", "outperform", "increase", "expand",
    "accelerate", "solid", "raise", "raised", "ahead", "better", "gain",
    "surge", "rally", "bullish", "recovery", "upturn", "strengthen",
    "upbeat", "optimistic", "outperformed", "boosted", "accelerated",
    "healthy", "resilient", "improved", "rising", "grows", "grows",
}

NEGATIVE_WORDS = {
    "decline", "headwind", "pressure", "challenging", "miss", "weak", "below",
    "reduce", "decrease", "concern", "risk", "uncertain", "volatile", "loss",
    "impair", "downgrade", "cut", "disappoint", "bearish", "downturn", "soften",
    "compression", "struggle", "difficult", "deteriorate", "slowdown", "cautious",
    "disappointing", "declined", "reduced", "slowing", "missed", "downgraded",
    "negative", "falling", "dropped", "dropped", "losses", "weaker", "sluggish",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SentimentPoint:
    """Single time-step sentiment reading."""

    timestamp: int      # period index
    score: float        # -1 to +1
    n_positive: int
    n_negative: int
    text_length: int
    confidence: float   # (n_pos + n_neg) / n_words (coverage)


@dataclass
class SentimentTimeSeries:
    """Full time series of SentimentPoints."""

    points: List[SentimentPoint]

    @property
    def scores(self) -> np.ndarray:
        """Array of sentiment scores."""
        return np.array([p.score for p in self.points], dtype=float)

    @property
    def timestamps(self) -> np.ndarray:
        """Array of timestamp indices."""
        return np.array([p.timestamp for p in self.points], dtype=float)

    def drift(self) -> np.ndarray:
        """First difference of scores (sentiment velocity).

        Returns array of length n-1.
        """
        s = self.scores
        if len(s) < 2:
            return np.array([], dtype=float)
        return np.diff(s)

    def momentum(self, k: int = 3) -> np.ndarray:
        """k-period sentiment change.

        momentum[t] = score[t] - score[t-k], for t >= k.
        Returns array of length n-k.
        """
        s = self.scores
        n = len(s)
        if n <= k:
            return np.array([], dtype=float)
        return s[k:] - s[: n - k]

    def volatility(self, window: int = 5) -> np.ndarray:
        """Rolling std of scores over `window` periods.

        Returns array of length n-window+1.
        """
        s = self.scores
        n = len(s)
        if n < window:
            return np.array([], dtype=float)
        out = np.zeros(n - window + 1)
        for i in range(n - window + 1):
            out[i] = s[i : i + window].std()
        return out

    def regime(self) -> np.ndarray:
        """Per-point regime: 1=bull, -1=bear, 0=neutral.

        Thresholds: mean ± 0.5*std over the full series.
        """
        s = self.scores
        if len(s) == 0:
            return np.array([], dtype=int)
        mu = s.mean()
        sigma = s.std()
        threshold = 0.5 * sigma
        out = np.zeros(len(s), dtype=int)
        out[s > mu + threshold] = 1
        out[s < mu - threshold] = -1
        return out


# ---------------------------------------------------------------------------
# TextSentimentAnalyzer
# ---------------------------------------------------------------------------


class TextSentimentAnalyzer:
    """Lexicon-based text sentiment analyzer (no transformers required)."""

    def __init__(
        self,
        positive_words: Optional[set] = None,
        negative_words: Optional[set] = None,
    ) -> None:
        self._pos = positive_words if positive_words is not None else POSITIVE_WORDS
        self._neg = negative_words if negative_words is not None else NEGATIVE_WORDS

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase alphabetic tokens."""
        return re.findall(r"[a-z]+", text.lower())

    def score(self, text: str) -> float:
        """Score a single text.

        score = (n_positive - n_negative) / (n_positive + n_negative + 1)

        Returns float in (-1, +1).
        """
        tokens = self._tokenize(text)
        n_pos = sum(1 for t in tokens if t in self._pos)
        n_neg = sum(1 for t in tokens if t in self._neg)
        return (n_pos - n_neg) / (n_pos + n_neg + 1)

    def _score_detail(self, text: str, timestamp: int) -> SentimentPoint:
        """Internal: score a text and return a SentimentPoint."""
        tokens = self._tokenize(text)
        n_words = len(tokens)
        n_pos = sum(1 for t in tokens if t in self._pos)
        n_neg = sum(1 for t in tokens if t in self._neg)
        raw_score = (n_pos - n_neg) / (n_pos + n_neg + 1)
        coverage = (n_pos + n_neg) / n_words if n_words > 0 else 0.0
        return SentimentPoint(
            timestamp=timestamp,
            score=raw_score,
            n_positive=n_pos,
            n_negative=n_neg,
            text_length=n_words,
            confidence=coverage,
        )

    def score_series(self, texts: List[str]) -> SentimentTimeSeries:
        """Score a list of texts into a SentimentTimeSeries.

        Parameters
        ----------
        texts : list of strings (one per time period)

        Returns
        -------
        SentimentTimeSeries
        """
        points = [self._score_detail(t, i) for i, t in enumerate(texts)]
        return SentimentTimeSeries(points=points)

    def extract_topics(
        self, text: str, topics: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Per-topic keyword coverage fraction.

        Parameters
        ----------
        text   : source text
        topics : dict mapping topic_name → list of keywords

        Returns
        -------
        dict mapping topic_name → fraction of text tokens matching topic keywords
        """
        tokens = self._tokenize(text)
        n = len(tokens)
        if n == 0:
            return {k: 0.0 for k in topics}
        token_set = set(tokens)
        result: Dict[str, float] = {}
        for topic, keywords in topics.items():
            hits = sum(1 for t in tokens if t in set(keywords))
            result[topic] = hits / n
        return result

    def score_by_topic(
        self, texts: List[str], topics: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """Per-topic sentiment score arrays.

        For each topic, score only the tokens belonging to that topic's keyword
        context (approximated as full-text score weighted by topic coverage).

        Parameters
        ----------
        texts  : list of texts
        topics : dict mapping topic_name → list of keywords

        Returns
        -------
        dict mapping topic_name → np.ndarray of scores (one per text)
        """
        result: Dict[str, List[float]] = {k: [] for k in topics}
        analyzer = TextSentimentAnalyzer(self._pos, self._neg)
        for text in texts:
            full_score = analyzer.score(text)
            coverages = analyzer.extract_topics(text, topics)
            for topic in topics:
                # Weight: topic coverage * full score (simple approximation)
                result[topic].append(coverages[topic] * full_score)
        return {k: np.array(v, dtype=float) for k, v in result.items()}


# ---------------------------------------------------------------------------
# SentimentEvolutionTracker
# ---------------------------------------------------------------------------


class SentimentEvolutionTracker:
    """Tracks evolution of a sentiment time series and relates it to returns."""

    def __init__(self, ts: SentimentTimeSeries) -> None:
        self._ts = ts

    def trend(self) -> float:
        """OLS slope of sentiment scores over time.

        Returns float (positive → improving sentiment trend).
        """
        s = self._ts.scores
        t = self._ts.timestamps
        n = len(s)
        if n < 2:
            return 0.0
        tm, sm = t.mean(), s.mean()
        denom = np.sum((t - tm) ** 2)
        if denom < 1e-15:
            return 0.0
        return float(np.sum((t - tm) * (s - sm)) / denom)

    def inflection_points(self) -> List[int]:
        """Indices where the sentiment regime changes.

        A regime change is detected when the regime label switches
        (bull→neutral, neutral→bear, bear→bull, etc.).
        """
        reg = self._ts.regime()
        if len(reg) < 2:
            return []
        inflections = []
        for i in range(1, len(reg)):
            if reg[i] != reg[i - 1]:
                inflections.append(i)
        return inflections

    def predict_next(self) -> float:
        """AR(1) extrapolation: s_{n+1} = mu + phi * (s_n - mu).

        Phi estimated by OLS on lagged scores. Falls back to last score
        when insufficient data.
        """
        s = self._ts.scores
        if len(s) < 2:
            return float(s[-1]) if len(s) > 0 else 0.0
        # OLS: s[1:] = phi * s[:-1] + c
        x = s[:-1]
        y = s[1:]
        xm = x.mean()
        denom = np.sum((x - xm) ** 2)
        if denom < 1e-15:
            return float(s[-1])
        phi = float(np.sum((x - xm) * y) / denom)
        mu = float(y.mean() - phi * xm)
        return mu + phi * float(s[-1])

    def correlation_with_returns(
        self, returns: np.ndarray, lag: int = 1
    ) -> float:
        """Pearson correlation of score_t with forward return_{t+lag}.

        Parameters
        ----------
        returns : return series aligned with sentiment (same length)
        lag     : number of periods ahead

        Returns
        -------
        float in [-1, +1], or 0.0 if insufficient data
        """
        s = self._ts.scores
        r = np.asarray(returns, dtype=float)
        n = min(len(s), len(r))
        if n <= lag + 1:
            return 0.0
        s_lead = s[: n - lag]
        r_lag = r[lag:n]
        if len(s_lead) < 2:
            return 0.0
        return float(np.corrcoef(s_lead, r_lag)[0, 1])

    def lead_lag_analysis(
        self, returns: np.ndarray, max_lag: int = 5
    ) -> dict:
        """Find the lag at which sentiment best predicts returns.

        Parameters
        ----------
        returns : return series
        max_lag : maximum lag to consider

        Returns
        -------
        dict with keys:
            'best_lag'         : int
            'best_correlation' : float
            'lag_profile'      : dict {lag: correlation}
        """
        lag_profile: Dict[int, float] = {}
        for lag in range(0, max_lag + 1):
            lag_profile[lag] = self.correlation_with_returns(returns, lag=lag)

        best_lag = max(lag_profile, key=lambda k: abs(lag_profile[k]))
        return {
            "best_lag": best_lag,
            "best_correlation": lag_profile[best_lag],
            "lag_profile": lag_profile,
        }

    def regime_returns(self, returns: np.ndarray) -> dict:
        """Mean return by sentiment regime.

        Parameters
        ----------
        returns : return series aligned with sentiment

        Returns
        -------
        dict with keys 'bull', 'bear', 'neutral' → mean return
        """
        reg = self._ts.regime()
        r = np.asarray(returns, dtype=float)
        n = min(len(reg), len(r))
        reg = reg[:n]
        r = r[:n]

        def _mean(mask: np.ndarray) -> float:
            vals = r[mask]
            return float(vals.mean()) if len(vals) > 0 else 0.0

        return {
            "bull": _mean(reg == 1),
            "bear": _mean(reg == -1),
            "neutral": _mean(reg == 0),
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def score_text(text: str) -> float:
    """Score a single text using the default lexicon.

    Returns float in (-1, +1).
    """
    return TextSentimentAnalyzer().score(text)


def build_sentiment_series(texts: List[str]) -> SentimentTimeSeries:
    """Build a SentimentTimeSeries from a list of texts."""
    return TextSentimentAnalyzer().score_series(texts)


def sentiment_momentum(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """k-period sentiment momentum: scores[k:] - scores[:-k]."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n <= k:
        return np.array([], dtype=float)
    return s[k:] - s[: n - k]


def sentiment_regime(scores: np.ndarray) -> np.ndarray:
    """Classify each score as bull (1), bear (-1), or neutral (0).

    Thresholds: mean ± 0.5 * std of the series.
    """
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return np.array([], dtype=int)
    mu = s.mean()
    sigma = s.std()
    out = np.zeros(len(s), dtype=int)
    out[s > mu + 0.5 * sigma] = 1
    out[s < mu - 0.5 * sigma] = -1
    return out


def lead_lag_correlation(
    sentiment: np.ndarray, returns: np.ndarray, max_lag: int = 5
) -> dict:
    """Lead-lag correlation between sentiment and returns.

    Parameters
    ----------
    sentiment : sentiment score array
    returns   : return array (same length)
    max_lag   : maximum lag to consider

    Returns
    -------
    dict with keys: 'best_lag', 'best_correlation', 'lag_profile'
    """
    s = np.asarray(sentiment, dtype=float)
    r = np.asarray(returns, dtype=float)
    n = min(len(s), len(r))
    lag_profile: Dict[int, float] = {}
    for lag in range(0, max_lag + 1):
        if n <= lag + 1:
            lag_profile[lag] = 0.0
            continue
        s_lead = s[: n - lag]
        r_lag = r[lag:n]
        if len(s_lead) < 2:
            lag_profile[lag] = 0.0
        else:
            lag_profile[lag] = float(np.corrcoef(s_lead, r_lag)[0, 1])
    best_lag = max(lag_profile, key=lambda k: abs(lag_profile[k]))
    return {
        "best_lag": best_lag,
        "best_correlation": lag_profile[best_lag],
        "lag_profile": lag_profile,
    }
