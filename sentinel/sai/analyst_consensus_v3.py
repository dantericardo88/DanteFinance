"""
Analyst consensus estimates aggregation.

Dimension: dim_018 — Analyst consensus aggregation (target score: 9)

Covers: price target aggregation (mean/median/high/low/dispersion),
recommendation distribution, EPS revision tracking, consensus scoring
(1-5 scale), sentiment composites, analyst accuracy ranking.

No network calls — numpy only.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ---------------------------------------------------------------------------
# Rating utilities
# ---------------------------------------------------------------------------

_REC_SCORE: Dict[str, int] = {
    "strong buy": 5,
    "buy": 4,
    "hold": 3,
    "neutral": 3,
    "market perform": 3,
    "underperform": 2,
    "sell": 1,
    "strong sell": 1,
}


def _rec_to_score(rec: str) -> int:
    return _REC_SCORE.get(rec.lower().strip(), 3)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AnalystEstimate:
    """One analyst's full set of estimates for a ticker."""

    analyst_id: str
    firm: str
    recommendation: str     # 'Strong Buy', 'Buy', 'Hold', 'Underperform', 'Sell'
    price_target: float
    eps_current_year: float
    eps_next_year: float
    date: str = "2025-01-01"

    @property
    def rec_score(self) -> int:
        """Numeric recommendation: Strong Buy=5 … Sell=1."""
        return _rec_to_score(self.recommendation)


@dataclass
class ConsensusSnapshot:
    """Aggregated analyst consensus for one ticker at a point in time."""

    ticker: str
    current_price: float
    estimates: List[AnalystEstimate] = field(default_factory=list)

    # ---- computed properties -----------------------------------------------

    @property
    def n_analysts(self) -> int:
        return len(self.estimates)

    @property
    def mean_target(self) -> float:
        if not self.estimates:
            return float("nan")
        return float(np.mean([e.price_target for e in self.estimates]))

    @property
    def median_target(self) -> float:
        if not self.estimates:
            return float("nan")
        return float(np.median([e.price_target for e in self.estimates]))

    @property
    def high_target(self) -> float:
        if not self.estimates:
            return float("nan")
        return float(max(e.price_target for e in self.estimates))

    @property
    def low_target(self) -> float:
        if not self.estimates:
            return float("nan")
        return float(min(e.price_target for e in self.estimates))

    @property
    def upside_pct(self) -> float:
        """Percentage upside from current price to mean target."""
        if not self.estimates or self.current_price <= 0:
            return float("nan")
        return (self.mean_target / self.current_price - 1.0) * 100.0

    @property
    def consensus_score(self) -> float:
        """Mean numeric recommendation score (1=Sell … 5=Strong Buy)."""
        if not self.estimates:
            return float("nan")
        return float(np.mean([e.rec_score for e in self.estimates]))

    @property
    def buy_pct(self) -> float:
        """Percentage of analysts with Buy or Strong Buy."""
        if not self.estimates:
            return float("nan")
        buy_count = sum(1 for e in self.estimates
                        if e.rec_score >= 4)
        return buy_count / len(self.estimates) * 100.0

    @property
    def eps_consensus_cy(self) -> float:
        """Consensus (mean) current-year EPS."""
        if not self.estimates:
            return float("nan")
        return float(np.mean([e.eps_current_year for e in self.estimates]))

    @property
    def eps_consensus_ny(self) -> float:
        """Consensus (mean) next-year EPS."""
        if not self.estimates:
            return float("nan")
        return float(np.mean([e.eps_next_year for e in self.estimates]))


# ---------------------------------------------------------------------------
# Consensus Analytics
# ---------------------------------------------------------------------------

class ConsensusAnalytics:
    """Rich analytics derived from a ConsensusSnapshot."""

    def __init__(self, snapshot: ConsensusSnapshot) -> None:
        self.snap = snapshot

    # ---- distributions -------------------------------------------------------

    def recommendation_distribution(self) -> Dict[str, int]:
        """Count of each canonical recommendation label."""
        dist: Dict[str, int] = {}
        canonical_map = {
            5: "Strong Buy",
            4: "Buy",
            3: "Hold",
            2: "Underperform",
            1: "Sell",
        }
        for e in self.snap.estimates:
            label = canonical_map.get(e.rec_score, e.recommendation)
            dist[label] = dist.get(label, 0) + 1
        return dist

    def target_distribution(self) -> dict:
        """Summary stats for price targets."""
        targets = [e.price_target for e in self.snap.estimates]
        if not targets:
            return {}
        arr = np.array(targets)
        mean_val = float(arr.mean())
        return {
            "mean": mean_val,
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "high": float(arr.max()),
            "low": float(arr.min()),
            "dispersion_cv": (
                float(arr.std(ddof=1) / abs(mean_val))
                if len(arr) > 1 and abs(mean_val) > 1e-9
                else 0.0
            ),
        }

    def eps_distribution(self) -> dict:
        """Summary stats for current-year EPS estimates."""
        cy = [e.eps_current_year for e in self.snap.estimates]
        ny = [e.eps_next_year for e in self.snap.estimates]
        if not cy:
            return {}

        def _stats(lst: list) -> dict:
            arr = np.array(lst)
            mean_val = float(arr.mean())
            return {
                "mean": mean_val,
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "high": float(arr.max()),
                "low": float(arr.min()),
                "dispersion": (
                    float(arr.std(ddof=1) / abs(mean_val))
                    if len(arr) > 1 and abs(mean_val) > 1e-9
                    else 0.0
                ),
            }

        return {
            "current_year": _stats(cy),
            "next_year": _stats(ny),
        }

    def dispersion_signal(self) -> float:
        """Coefficient of variation of price targets.

        Low CV → high conviction; high CV → high uncertainty.
        Returns a value in [0, inf]; typically < 0.30 is low dispersion.
        """
        td = self.target_distribution()
        return td.get("dispersion_cv", 0.0)

    def revision_trend(self, old_snapshot: ConsensusSnapshot) -> dict:
        """Compare current snapshot to an older one.

        Matches analysts by analyst_id and counts upgrades/downgrades
        and EPS/target revision percentages.
        """
        old_by_id = {e.analyst_id: e for e in old_snapshot.estimates}
        new_by_id = {e.analyst_id: e for e in self.snap.estimates}
        common = set(old_by_id) & set(new_by_id)

        n_upgrades = 0
        n_downgrades = 0
        eps_revisions: List[float] = []
        target_revisions: List[float] = []

        for aid in common:
            old_e = old_by_id[aid]
            new_e = new_by_id[aid]
            if new_e.rec_score > old_e.rec_score:
                n_upgrades += 1
            elif new_e.rec_score < old_e.rec_score:
                n_downgrades += 1
            if abs(old_e.eps_current_year) > 1e-9:
                eps_revisions.append(
                    (new_e.eps_current_year - old_e.eps_current_year)
                    / abs(old_e.eps_current_year)
                    * 100
                )
            if abs(old_e.price_target) > 1e-9:
                target_revisions.append(
                    (new_e.price_target - old_e.price_target)
                    / abs(old_e.price_target)
                    * 100
                )

        return {
            "n_upgrades": n_upgrades,
            "n_downgrades": n_downgrades,
            "eps_revision_pct": float(np.mean(eps_revisions)) if eps_revisions else 0.0,
            "target_revision_pct": (
                float(np.mean(target_revisions)) if target_revisions else 0.0
            ),
            "analysts_compared": len(common),
        }

    def sentiment_score(self) -> float:
        """Composite sentiment score 0-1 (1 = very bullish).

        Combines recommendation score (normalised) with upside to target.
        """
        if not self.snap.estimates:
            return 0.5
        rec_component = (self.snap.consensus_score - 1.0) / 4.0  # [0,1]
        upside = self.snap.upside_pct
        if np.isnan(upside):
            upside = 0.0
        upside_component = np.clip((upside + 20) / 60.0, 0.0, 1.0)  # 0% → 0.33, 40% → 1
        return float(0.6 * rec_component + 0.4 * upside_component)


# ---------------------------------------------------------------------------
# EPS Revision Tracker
# ---------------------------------------------------------------------------

class EPSRevisionTracker:
    """Track and score EPS estimate revisions."""

    def revision_ratio(self, old_eps: List[float], new_eps: List[float]) -> float:
        """Fraction of estimates revised upward."""
        if len(old_eps) != len(new_eps) or not old_eps:
            return 0.5
        n_up = sum(1 for o, n in zip(old_eps, new_eps) if n > o)
        n_dn = sum(1 for o, n in zip(old_eps, new_eps) if n < o)
        total = n_up + n_dn
        if total == 0:
            return 0.5
        return n_up / total

    def estimate_dispersion(self, estimates: List[float]) -> float:
        """Coefficient of variation = std / |mean|.

        Returns 0.0 for a single estimate.
        """
        if len(estimates) < 2:
            return 0.0
        arr = np.array(estimates)
        mean_val = float(arr.mean())
        if abs(mean_val) < 1e-9:
            return 0.0
        return float(arr.std(ddof=1) / abs(mean_val))

    def surprise_probability(self, consensus: float, std: float,
                              actual: float) -> float:
        """P(actual > consensus) assuming normal distribution of estimates."""
        if std < 1e-12:
            return 1.0 if actual > consensus else 0.0
        from scipy.stats import norm
        return float(norm.sf(consensus, loc=actual, scale=std))

    def momentum(self, estimates_over_time: List[float]) -> float:
        """OLS slope of EPS estimates over time, normalised by |mean|.

        Returns positive if trend is upward, negative if downward.
        """
        n = len(estimates_over_time)
        if n < 2:
            return 0.0
        arr = np.array(estimates_over_time, dtype=float)
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])
        mean_val = float(np.abs(arr).mean())
        if mean_val < 1e-9:
            return 0.0
        return slope / mean_val


# ---------------------------------------------------------------------------
# Analyst Accuracy Tracker
# ---------------------------------------------------------------------------

class AnalystTracker:
    """Track analyst forecast accuracy over time."""

    def accuracy(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """1 - MAE / mean(|actuals|).

        Clamped to [0, 1].
        """
        if len(forecasts) == 0 or len(actuals) == 0:
            return 0.0
        mae = float(np.abs(forecasts - actuals).mean())
        base = float(np.abs(actuals).mean())
        if base < 1e-9:
            return 1.0 if mae < 1e-9 else 0.0
        return float(np.clip(1.0 - mae / base, 0.0, 1.0))

    def directional_accuracy(self, forecast_changes: np.ndarray,
                              actual_changes: np.ndarray) -> float:
        """Fraction of times forecast direction matches actual direction."""
        if len(forecast_changes) == 0:
            return 0.0
        correct = np.sign(forecast_changes) == np.sign(actual_changes)
        return float(correct.mean())

    def rank_analysts(self, analyst_forecasts: Dict[str, np.ndarray],
                      actuals: np.ndarray) -> List[tuple]:
        """Rank analysts by accuracy, best first.

        Returns list of (analyst_id, accuracy_score).
        """
        results = []
        for analyst_id, forecasts in analyst_forecasts.items():
            acc = self.accuracy(np.array(forecasts), actuals)
            results.append((analyst_id, acc))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Standalone convenience functions
# ---------------------------------------------------------------------------

def consensus_target(estimates: List[AnalystEstimate]) -> float:
    """Mean price target across all analysts."""
    if not estimates:
        return float("nan")
    return float(np.mean([e.price_target for e in estimates]))


def buy_sell_ratio(estimates: List[AnalystEstimate]) -> float:
    """Ratio of buy-rated to sell-rated analysts.

    Returns inf if there are buys but no sells; nan if empty.
    """
    if not estimates:
        return float("nan")
    n_buy = sum(1 for e in estimates if e.rec_score >= 4)
    n_sell = sum(1 for e in estimates if e.rec_score <= 2)
    if n_sell == 0:
        return float("inf") if n_buy > 0 else 1.0
    return n_buy / n_sell


def eps_revision_signal(old_eps: float, new_eps: float) -> float:
    """Normalised EPS revision: (new - old) / |old|.

    Positive = upward revision, negative = downward.
    Returns 0.0 if old_eps is ~zero.
    """
    if abs(old_eps) < 1e-9:
        return 0.0
    return (new_eps - old_eps) / abs(old_eps)


def upside_to_target(current_price: float, mean_target: float) -> float:
    """Percentage upside from current_price to mean_target."""
    if current_price <= 0:
        return float("nan")
    return (mean_target / current_price - 1.0) * 100.0
