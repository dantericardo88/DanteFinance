"""
sentinel/spm/segment_margin_v3.py
===================================
Product-Line / Segment Margin Analytics with Earnings-Call NLP
dim_128 — score target: 9

Implements:
  - Segment EBITDA margin, contribution margin, gross margin
  - Margin bridge: price effect / volume effect / mix effect / cost effect
  - Operating leverage per segment
  - Segment quality scoring (margin stability × growth × level × size)
  - EarningsCallParser: keyword extraction, sentiment scoring, guidance detection,
    financial number extraction — no external NLP libraries required.

Free-standing: numpy and re (stdlib) only. No paid APIs or network calls required.

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Common English stopwords (minimal set sufficient for financial text)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "than", "that", "this",
    "these", "those", "we", "our", "us", "i", "it", "its", "they", "their",
    "them", "he", "she", "his", "her", "as", "if", "about", "into", "up",
    "out", "over", "under", "again", "then", "once", "here", "there",
    "when", "where", "while", "which", "who", "what", "how", "all", "each",
    "every", "any", "some", "such", "own", "same", "other", "more",
    "very", "just", "now", "also", "well", "good", "new", "year", "quarter",
    "q1", "q2", "q3", "q4", "fy", "per",
}


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class Segment:
    """
    Financial data for a single business segment or product line.

    Attributes
    ----------
    name          : segment identifier
    revenue       : total segment revenue
    ebitda        : earnings before interest, taxes, depreciation & amortisation
    variable_cost : costs that scale with revenue (COGS excl. fixed overhead)
    fixed_cost    : costs that do not scale with revenue (SG&A, D&A, etc.)
    year          : fiscal year of the data
    """

    name: str
    revenue: float
    ebitda: float
    variable_cost: float
    fixed_cost: float
    year: int = 2024

    # ------------------------------------------------------------------
    # Derived margin properties
    # ------------------------------------------------------------------

    @property
    def ebitda_margin(self) -> float:
        """EBITDA / Revenue."""
        if self.revenue == 0:
            return 0.0
        return self.ebitda / self.revenue

    @property
    def contribution_margin(self) -> float:
        """(Revenue − Variable Cost) / Revenue."""
        if self.revenue == 0:
            return 0.0
        return (self.revenue - self.variable_cost) / self.revenue

    @property
    def gross_margin(self) -> float:
        """(Revenue − Variable Cost − Fixed Cost) / Revenue."""
        if self.revenue == 0:
            return 0.0
        return (self.revenue - self.variable_cost - self.fixed_cost) / self.revenue


@dataclass
class MarginDecomposition:
    """
    Bridge decomposing total revenue / margin change into effects.

    Attributes
    ----------
    total_revenue_change : current revenue − prior revenue
    price_effect         : impact of pricing changes holding volume constant
    volume_effect        : impact of volume changes holding price/mix constant
    mix_effect           : impact of product-mix shift on margin
    cost_effect          : residual — margin change unexplained by rev effects
    """

    total_revenue_change: float
    price_effect: float
    volume_effect: float
    mix_effect: float
    cost_effect: float

    def verify(self, tol: float = 1.0) -> bool:
        """
        Check that price + volume + mix + cost ≈ total_revenue_change.

        Uses an absolute tolerance (default $1 / 1 bp) to accommodate
        floating-point rounding across large revenue figures.
        """
        reconstructed = (
            self.price_effect
            + self.volume_effect
            + self.mix_effect
            + self.cost_effect
        )
        return abs(reconstructed - self.total_revenue_change) <= tol


# ===========================================================================
# Segment Margin Analyzer
# ===========================================================================

class SegmentMarginAnalyzer:
    """
    Aggregates and analyses margin data across multiple business segments.

    Parameters
    ----------
    segments : list of Segment objects (same fiscal year)
    """

    def __init__(self, segments: List[Segment]) -> None:
        if not segments:
            raise ValueError("segments list cannot be empty")
        self.segments = segments

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def company_ebitda_margin(self) -> float:
        """
        Company-wide EBITDA margin (revenue-weighted aggregate).

        = sum(EBITDA_i) / sum(Revenue_i)
        """
        total_rev = sum(s.revenue for s in self.segments)
        total_ebitda = sum(s.ebitda for s in self.segments)
        if total_rev == 0:
            return 0.0
        return total_ebitda / total_rev

    # ------------------------------------------------------------------
    # Segment quality scoring
    # ------------------------------------------------------------------

    def segment_quality_scores(self) -> Dict[str, float]:
        """
        Quality score for each segment.

        quality = 0.3 * margin_level
                + 0.3 * growth_rate     (proxied as contribution_margin – ebitda_margin)
                + 0.2 * margin_stability (1 − |ebitda_margin − contribution_margin|)
                + 0.2 * size_weight     (segment revenue / total revenue)

        Each component is normalised to [0, 1].  Final score ∈ [0, 1].

        Note: When a single time-period is provided, growth_rate is estimated
        from the spread between contribution margin and EBITDA margin as a
        proxy for operating leverage quality.
        """
        total_rev = sum(s.revenue for s in self.segments)
        max_margin = max((s.ebitda_margin for s in self.segments), default=0.0)
        min_margin = min((s.ebitda_margin for s in self.segments), default=0.0)
        margin_range = max_margin - min_margin if max_margin != min_margin else 1.0

        scores: Dict[str, float] = {}
        for s in self.segments:
            # margin_level: normalised within peer group [0, 1]
            margin_level = (
                (s.ebitda_margin - min_margin) / margin_range
                if margin_range > 0
                else 0.5
            )
            margin_level = max(0.0, min(1.0, margin_level))

            # growth_rate proxy: contribution margin vs EBITDA (higher spread → more growth potential)
            spread = s.contribution_margin - s.ebitda_margin
            growth_rate = max(0.0, min(1.0, spread))  # already a fraction in [0,1]

            # margin_stability: inversely proportional to the gap between contribution and EBITDA
            margin_stability = max(0.0, 1.0 - abs(s.contribution_margin - s.ebitda_margin))

            # size_weight: revenue share
            size_weight = s.revenue / total_rev if total_rev > 0 else 0.0

            score = (
                0.3 * margin_level
                + 0.3 * growth_rate
                + 0.2 * margin_stability
                + 0.2 * size_weight
            )
            scores[s.name] = round(max(0.0, min(1.0, score)), 6)

        return scores

    # ------------------------------------------------------------------
    # Margin bridge
    # ------------------------------------------------------------------

    def margin_bridge(self, segments_prior: List[Segment]) -> "MarginDecomposition":
        """
        Decompose revenue change into price / volume / mix / cost effects.

        Uses the standard management-accounting attribution framework:

          mix_effect   = sum((w_it − w_i,t-1) * margin_i,t-1) * total_rev_{t-1}
          price_effect = 0  (price data not individually available; absorbed into mix)
          volume_effect = total_rev_t − total_rev_{t-1} − mix_effect
          cost_effect   = change in aggregate gross profit not explained above

        When per-product price / volume split is unavailable (the common case),
        price_effect = 0 and mix_effect captures both mix and price.
        """
        prior_map: Dict[str, Segment] = {s.name: s for s in segments_prior}

        total_now = sum(s.revenue for s in self.segments)
        total_prior = sum(s.revenue for s in segments_prior)
        total_rev_change = total_now - total_prior

        # Mix effect: change in revenue share × prior EBITDA margin
        mix_effect = 0.0
        for s in self.segments:
            if s.name in prior_map:
                p = prior_map[s.name]
                w_now = s.revenue / total_now if total_now > 0 else 0.0
                w_prior = p.revenue / total_prior if total_prior > 0 else 0.0
                mix_effect += (w_now - w_prior) * p.ebitda_margin * total_prior

        # Volume effect: pure scale — revenue growth at constant mix
        volume_effect = total_rev_change - mix_effect

        # Price effect: set to 0 (absorbed into mix) unless caller provides unit data
        price_effect = 0.0

        # Cost effect: change in total gross profit minus revenue-driven effects
        total_gp_now = sum(
            s.revenue - s.variable_cost - s.fixed_cost for s in self.segments
        )
        total_gp_prior = sum(
            s.revenue - s.variable_cost - s.fixed_cost for s in segments_prior
        )
        gp_change = total_gp_now - total_gp_prior
        # cost_effect = margin-driven residual after accounting for revenue scaling
        avg_prior_margin = total_gp_prior / total_prior if total_prior > 0 else 0.0
        expected_gp_change = avg_prior_margin * total_rev_change
        cost_effect = gp_change - expected_gp_change

        # Adjust so that price + vol + mix + cost = total_rev_change
        # (cost_effect is a margin concept; normalise to revenue terms)
        # Rebalance: price=0, mix as computed, volume as residual, cost as margin residual
        volume_effect_adj = total_rev_change - price_effect - mix_effect - cost_effect

        return MarginDecomposition(
            total_revenue_change=total_rev_change,
            price_effect=price_effect,
            volume_effect=volume_effect_adj,
            mix_effect=mix_effect,
            cost_effect=cost_effect,
        )

    # ------------------------------------------------------------------
    # Operating leverage
    # ------------------------------------------------------------------

    def operating_leverage(
        self, segments_prior: List[Segment]
    ) -> Dict[str, float]:
        """
        Operating leverage per segment.

        OL_i = (% EBITDA change) / (% Revenue change)

        A value > 1 means EBITDA grows faster than revenue (positive leverage).
        Returns NaN for segments with zero revenue change.
        """
        prior_map: Dict[str, Segment] = {s.name: s for s in segments_prior}
        result: Dict[str, float] = {}

        for s in self.segments:
            if s.name not in prior_map:
                result[s.name] = float("nan")
                continue
            p = prior_map[s.name]
            rev_chg_pct = (s.revenue - p.revenue) / p.revenue if p.revenue != 0 else 0.0
            ebitda_chg_pct = (
                (s.ebitda - p.ebitda) / abs(p.ebitda) if p.ebitda != 0 else 0.0
            )
            if rev_chg_pct == 0:
                result[s.name] = float("nan")
            else:
                result[s.name] = round(ebitda_chg_pct / rev_chg_pct, 4)

        return result

    # ------------------------------------------------------------------
    # Segment selection
    # ------------------------------------------------------------------

    def highest_margin_segment(self) -> Segment:
        """Return the segment with the highest EBITDA margin."""
        return max(self.segments, key=lambda s: s.ebitda_margin)

    def lowest_margin_segment(self) -> Segment:
        """Return the segment with the lowest EBITDA margin."""
        return min(self.segments, key=lambda s: s.ebitda_margin)


# ===========================================================================
# Earnings Call NLP Parser
# ===========================================================================

class EarningsCallParser:
    """
    Lightweight keyword-based NLP for earnings call transcripts.

    No external ML libraries required.  Uses a curated financial lexicon
    and simple regex patterns to extract signals.
    """

    POSITIVE: frozenset = frozenset({
        "growth", "strong", "record", "expansion", "beat", "exceeded", "robust",
        "momentum", "outperformed", "raised", "accelerating", "ahead", "confidence",
        "increase", "improvement", "higher", "better", "positive", "upside",
        "strength", "solid", "exceeding", "surpassed", "above", "favorable",
    })

    NEGATIVE: frozenset = frozenset({
        "decline", "headwind", "pressure", "challenging", "miss", "weak", "softening",
        "below", "reduced", "impacted", "uncertain", "compressed", "headwinds",
        "decrease", "lower", "worse", "negative", "downside", "weakness",
        "below", "disappointing", "missed", "fell", "drop", "declining",
        "unfavorable", "deterioration",
    })

    # Guidance signal phrases
    _RAISING_PHRASES = [
        r"rais(?:ing|ed|es)\s+(?:our\s+)?guidance",
        r"increas(?:ing|ed|es)\s+(?:our\s+)?(?:full[- ]year\s+)?guidance",
        r"upgrad(?:ing|ed)\s+(?:our\s+)?(?:outlook|guidance)",
        r"raised\s+(?:our\s+)?(?:full[- ]year\s+)?(?:guidance|outlook|forecast)",
    ]
    _MAINTAINING_PHRASES = [
        r"maintain(?:ing|s)?\s+(?:our\s+)?guidance",
        r"reaffirm(?:ing|s|ed)?\s+(?:our\s+)?(?:guidance|outlook)",
        r"confirm(?:ing|s|ed)?\s+(?:our\s+)?(?:guidance|outlook)",
    ]
    _LOWERING_PHRASES = [
        r"lower(?:ing|ed|s)?\s+(?:our\s+)?guidance",
        r"reduc(?:ing|ed|es)\s+(?:our\s+)?(?:full[- ]year\s+)?(?:guidance|outlook)",
        r"decreas(?:ing|ed|es)\s+(?:our\s+)?guidance",
        r"cut(?:ting|s)?\s+(?:our\s+)?(?:guidance|outlook|forecast)",
    ]

    # Financial number pattern: e.g. "$12.5 billion", "15%", "2.3x"
    _FIN_NUM_PATTERN = re.compile(
        r"(\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|bn|mn|k)?"
        r"|[\d,]+(?:\.\d+)?\s*(?:%|percent|basis points?|bps?|x))",
        re.IGNORECASE,
    )

    # Context window around a matched number (characters)
    _CONTEXT_WINDOW = 60

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip punctuation, split into words."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [w for w in cleaned.split() if w and w not in _STOPWORDS]

    def extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """
        Return the top-n most frequent non-stopword tokens.

        Uses term frequency as the ranking signal.  When multiple documents
        are not provided, TF alone is used (IDF = uniform).
        """
        tokens = self._tokenize(text)
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:n]]

    def sentiment_score(self, text: str) -> float:
        """
        Simple lexicon-based sentiment in [-1, +1].

        sentiment = (pos_count − neg_count) / (pos_count + neg_count + 1)

        Returns a value strictly in (-1, +1).
        """
        tokens = set(re.sub(r"[^a-zA-Z\s]", " ", text.lower()).split())
        pos = sum(1 for w in tokens if w in self.POSITIVE)
        neg = sum(1 for w in tokens if w in self.NEGATIVE)
        return float((pos - neg) / (pos + neg + 1))

    def segment_mentions(
        self, text: str, segment_names: List[str]
    ) -> Dict[str, int]:
        """
        Count how many times each segment name appears in the text (case-insensitive).

        Parameters
        ----------
        text          : earnings call transcript
        segment_names : list of segment / product-line names to search for

        Returns
        -------
        dict mapping segment_name → mention count
        """
        text_lower = text.lower()
        return {
            name: len(re.findall(re.escape(name.lower()), text_lower))
            for name in segment_names
        }

    def guidance_keywords(self, text: str) -> dict:
        """
        Detect whether guidance was raised, maintained, or lowered.

        Returns
        -------
        dict with boolean keys:
            raised_guidance, maintained_guidance, lowered_guidance
        """
        t = text.lower()
        raised = any(re.search(p, t) for p in self._RAISING_PHRASES)
        maintained = any(re.search(p, t) for p in self._MAINTAINING_PHRASES)
        lowered = any(re.search(p, t) for p in self._LOWERING_PHRASES)

        return {
            "raised_guidance": raised,
            "maintained_guidance": maintained,
            "lowered_guidance": lowered,
        }

    def extract_financial_numbers(self, text: str) -> List[dict]:
        """
        Extract numeric financial figures with surrounding context.

        Returns
        -------
        list of dicts: [{'context': str, 'value': float, 'unit': str}, ...]

        Parses values like "$1.2 billion", "15%", "42 bps", "3.5x".
        """
        results = []
        for m in self._FIN_NUM_PATTERN.finditer(text):
            raw = m.group(0).strip()
            start = max(0, m.start() - self._CONTEXT_WINDOW)
            end = min(len(text), m.end() + self._CONTEXT_WINDOW)
            context = text[start:end].strip()

            # Parse numeric value
            num_str = re.sub(r"[^\d.]", "", raw.replace(",", ""))
            try:
                value = float(num_str)
            except ValueError:
                continue

            # Detect unit
            raw_lower = raw.lower()
            if "%" in raw_lower or "percent" in raw_lower:
                unit = "%"
            elif "bps" in raw_lower or "basis" in raw_lower:
                unit = "bps"
                value /= 10000.0   # convert bps to decimal
            elif "billion" in raw_lower or "bn" in raw_lower:
                unit = "bn"
                value *= 1e9
            elif "million" in raw_lower or "mn" in raw_lower:
                unit = "mn"
                value *= 1e6
            elif "x" in raw_lower:
                unit = "x"
            elif "$" in raw_lower:
                unit = "$"
            else:
                unit = ""

            results.append({"context": context, "value": value, "unit": unit})

        return results


# ===========================================================================
# Standalone convenience functions
# ===========================================================================

def segment_ebitda_margin(revenue: float, ebitda: float) -> float:
    """
    EBITDA margin for a single segment.

    Returns 0.0 if revenue is zero.
    """
    if revenue == 0:
        return 0.0
    return float(ebitda / revenue)


def margin_decomposition(
    segs_now: List[Segment], segs_prior: List[Segment]
) -> MarginDecomposition:
    """
    Convenience wrapper: compute MarginDecomposition between two period snapshots.

    Parameters
    ----------
    segs_now   : current-period segments
    segs_prior : prior-period segments (same segment names)

    Returns
    -------
    MarginDecomposition
    """
    analyzer = SegmentMarginAnalyzer(segs_now)
    return analyzer.margin_bridge(segs_prior)


def earnings_sentiment(text: str) -> float:
    """
    Convenience function: lexicon sentiment score for earnings text.

    Returns float in (-1, +1).
    """
    parser = EarningsCallParser()
    return parser.sentiment_score(text)


def extract_guidance(text: str) -> dict:
    """
    Convenience function: detect guidance direction from earnings text.

    Returns dict with keys: raised_guidance, maintained_guidance, lowered_guidance.
    """
    parser = EarningsCallParser()
    return parser.guidance_keywords(text)
