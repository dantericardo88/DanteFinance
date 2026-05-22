"""
sentinel/sai/web_signals_v3.py
Web signal analytics framework — processes pre-collected text, no actual scraping.
Pure math: numpy + re only.
"""
import numpy as np
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional


POSITIVE_WORDS = {
    'growth', 'strong', 'beat', 'record', 'bullish', 'upgrade',
    'buy', 'positive', 'gain', 'surge', 'rally', 'exceed', 'robust'
}
NEGATIVE_WORDS = {
    'decline', 'weak', 'miss', 'bearish', 'downgrade', 'sell',
    'loss', 'risk', 'concern', 'drop', 'fall', 'disappoint', 'headwind'
}

SOURCE_WEIGHTS = {
    'bloomberg': 1.0,
    'reuters': 0.9,
    'wsj': 0.85,
    'seeking_alpha': 0.7,
    'reddit': 0.4,
    'twitter': 0.3,
}

_DEFAULT_SOURCE_WEIGHT = 0.5  # fallback for unknown sources


@dataclass
class WebArticle:
    url: str
    title: str
    body: str
    source: str       # 'reuters', 'seeking_alpha', 'reddit', 'twitter', 'bloomberg'
    timestamp: float  # unix timestamp
    ticker: str = ''

    def word_count(self) -> int:
        return len(self.body.split())


@dataclass
class SignalPoint:
    timestamp: float
    sentiment: float          # -1 to +1
    confidence: float         # 0 to 1
    n_articles: int
    source_breakdown: Dict[str, float]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def score_text(text: str) -> float:
    """
    Lexicon score: (n_pos - n_neg) / (n_pos + n_neg + 1).
    Range (-1, 1).
    """
    words = re.findall(r'[a-z]+', text.lower())
    n_pos = sum(1 for w in words if w in POSITIVE_WORDS)
    n_neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return (n_pos - n_neg) / (n_pos + n_neg + 1)


def recency_weight(timestamp: float, current_time: float,
                   halflife_hours: float = 24.0) -> float:
    """
    Exponential decay weight.  weight = 2^(-(age_hours / halflife_hours)).
    At age=0 -> 1.0, at age=halflife -> 0.5.
    """
    age_seconds = max(0.0, current_time - timestamp)
    age_hours = age_seconds / 3600.0
    return math.pow(2.0, -(age_hours / halflife_hours))


def aggregate_sentiment(articles: List[WebArticle]) -> float:
    """Simple unweighted mean of per-article lexicon scores."""
    if not articles:
        return 0.0
    scores = [score_text(a.body) for a in articles]
    return float(np.mean(scores))


def web_signal(articles: List[WebArticle]) -> float:
    """
    Source-credibility-weighted mean sentiment across all articles.
    Falls back to aggregate_sentiment if no weights are available.
    """
    if not articles:
        return 0.0
    agg = SentimentAggregator()
    sp = agg.aggregate(articles)
    return float(sp.sentiment)


# ---------------------------------------------------------------------------
# SentimentAggregator
# ---------------------------------------------------------------------------

class SentimentAggregator:
    def score_article(self, article: WebArticle) -> float:
        """Lexicon score on title + body: (n_pos - n_neg) / (n_pos + n_neg + 1)."""
        combined = article.title + ' ' + article.body
        return score_text(combined)

    def aggregate(self, articles: List[WebArticle],
                  decay_halflife_hours: float = 24.0) -> SignalPoint:
        """
        Weighted sentiment: weight = source_credibility * recency_decay.
        Returns SignalPoint with weighted average sentiment.
        """
        if not articles:
            return SignalPoint(
                timestamp=0.0, sentiment=0.0, confidence=0.0,
                n_articles=0, source_breakdown={}
            )

        current_time = max(a.timestamp for a in articles)
        total_weight = 0.0
        weighted_sum = 0.0
        source_sents: Dict[str, List[float]] = {}

        for a in articles:
            src_w = SOURCE_WEIGHTS.get(a.source, _DEFAULT_SOURCE_WEIGHT)
            rec_w = recency_weight(a.timestamp, current_time,
                                   halflife_hours=decay_halflife_hours)
            w = src_w * rec_w
            s = self.score_article(a)
            weighted_sum += w * s
            total_weight += w
            source_sents.setdefault(a.source, []).append(s)

        sentiment = weighted_sum / total_weight if total_weight > 1e-12 else 0.0
        # confidence: based on number of articles and total weight normalisation
        confidence = float(np.clip(total_weight / (len(articles) + 1e-8), 0.0, 1.0))

        source_breakdown = {
            src: float(np.mean(vals))
            for src, vals in source_sents.items()
        }

        return SignalPoint(
            timestamp=current_time,
            sentiment=float(np.clip(sentiment, -1.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            n_articles=len(articles),
            source_breakdown=source_breakdown,
        )

    def by_source(self, articles: List[WebArticle]) -> Dict[str, float]:
        """Average sentiment per source."""
        buckets: Dict[str, List[float]] = {}
        for a in articles:
            buckets.setdefault(a.source, []).append(self.score_article(a))
        return {src: float(np.mean(vals)) for src, vals in buckets.items()}


# ---------------------------------------------------------------------------
# SignalTimeSeries
# ---------------------------------------------------------------------------

class SignalTimeSeries:
    def __init__(self, points: List[SignalPoint]):
        self._points = points

    def sentiment_array(self) -> np.ndarray:
        return np.array([p.sentiment for p in self._points])

    def confidence_weighted_signal(self) -> np.ndarray:
        return np.array([p.sentiment * p.confidence for p in self._points])

    def momentum(self, window: int = 3) -> np.ndarray:
        """Rolling mean differences over the window."""
        arr = self.sentiment_array()
        if len(arr) < window + 1:
            return np.diff(arr)
        result = []
        for i in range(window, len(arr)):
            result.append(np.mean(arr[i - window:i]))
        return np.array(result)

    def acceleration(self) -> np.ndarray:
        """Second differences of sentiment array."""
        arr = self.sentiment_array()
        return np.diff(np.diff(arr))

    def trend(self) -> float:
        """OLS slope of sentiment over time (index as time axis)."""
        arr = self.sentiment_array()
        n = len(arr)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = arr.mean()
        num = float(np.sum((x - x_mean) * (arr - y_mean)))
        den = float(np.sum((x - x_mean) ** 2))
        if abs(den) < 1e-12:
            return 0.0
        return num / den

    def regime(self) -> str:
        """Classify: 'bullish' if trend > 0.01, 'bearish' if < -0.01, else 'neutral'."""
        slope = self.trend()
        arr = self.sentiment_array()
        mean_sent = float(arr.mean()) if len(arr) else 0.0
        if slope > 0.01 or mean_sent > 0.2:
            return 'bullish'
        elif slope < -0.01 or mean_sent < -0.2:
            return 'bearish'
        return 'neutral'


# ---------------------------------------------------------------------------
# WebSignalGenerator
# ---------------------------------------------------------------------------

class WebSignalGenerator:
    def contrarian_signal(self, sentiment: np.ndarray) -> float:
        """
        Extreme sentiment -> reversal signal.
        Returns -sign(mean) when |mean| > 0.5, else 0.
        """
        if len(sentiment) == 0:
            return 0.0
        mean_val = float(np.mean(sentiment))
        if abs(mean_val) > 0.5:
            return -float(np.sign(mean_val))
        return 0.0

    def momentum_signal(self, sentiment: np.ndarray, window: int = 5) -> float:
        """Mean of last window elements."""
        if len(sentiment) == 0:
            return 0.0
        return float(np.mean(sentiment[-window:]))

    def volume_signal(self, n_articles: np.ndarray) -> float:
        """Z-score of latest article count vs history."""
        if len(n_articles) < 2:
            return 0.0
        hist = n_articles[:-1].astype(float)
        latest = float(n_articles[-1])
        mean = float(np.mean(hist))
        std = float(np.std(hist))
        if std < 1e-12:
            return 0.0
        return (latest - mean) / std

    def combined_signal(self, points: List[SignalPoint]) -> float:
        """Weighted combination: 0.6 * momentum + 0.4 * contrarian."""
        if not points:
            return 0.0
        arr = np.array([p.sentiment for p in points])
        mom = self.momentum_signal(arr)
        con = self.contrarian_signal(arr)
        return 0.6 * mom + 0.4 * con


# ---------------------------------------------------------------------------
# Backward-compatible aliases for the old dim_131.sh stub
# (which imported from sentinel.sma.web_signals_v3)
# ---------------------------------------------------------------------------

class WebSignalScraper:
    """Thin compatibility shim for legacy tests."""
    pass


class SeekingAlphaSignal:
    """Thin compatibility shim for legacy tests."""
    pass


def compute_web_sentiment(articles: Optional[List[WebArticle]] = None) -> float:
    """Stub for legacy compatibility."""
    if articles is None:
        return 0.0
    return aggregate_sentiment(articles)
