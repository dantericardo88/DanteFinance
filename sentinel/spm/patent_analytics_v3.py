"""
sentinel/spm/patent_analytics_v3.py
dim_125: Quantitative Patent Analytics and IP Strategy (score target: 9)

Implements h-index, technology concentration (IPC HHI), patent family breadth,
R&D efficiency metrics, and an IP moat composite score.  Works entirely with
supplied patent data — no live patent-database connection required.

Key mathematics
---------------
h-index (modified):
    Largest h such that h patents each have >= h forward citations.

Citation score (recency-adjusted):
    citation_score = sum(weighted_citations_i / (years_since_grant_i + 1))

Technology concentration (IPC HHI):
    tech_hhi = sum(s_k^2)  where s_k = share of portfolio in IPC class k
    tech_diversity = 1 - tech_hhi

Patent family breadth:
    family_score_i = n_jurisdictions_i / max_jurisdictions_globally
    avg_family_breadth = mean(family_score_i)

R&D efficiency:
    patent_efficiency    = total_patents   / R&D_spend_millions
    citation_efficiency  = total_citations / R&D_spend_millions

IP moat score (composite [0,1]):
    moat = 0.30 * citation_quality
         + 0.20 * tech_diversity
         + 0.20 * family_breadth
         + 0.30 * forward_citation_rate

Forward citation rate:
    forward_citation_rate = total_forward_citations / (total_patents * industry_avg_citations)
    clipped to [0, 1].
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Max jurisdictions globally (used to normalise family breadth).
# In practice the PCT covers ~155 contracting states; we use 150 as a
# conservative normalisation ceiling.
# ---------------------------------------------------------------------------
MAX_JURISDICTIONS_GLOBALLY: int = 150


# ===========================================================================
# Data structures
# ===========================================================================


@dataclass
class Patent:
    """Represents a single patent in a portfolio."""

    patent_id: str
    title: str
    filing_year: int
    grant_year: int
    ipc_class: str           # e.g. 'G06F', 'H04L', 'A61K'
    citations_received: int  # forward citations (others citing this patent)
    citations_made: int      # backward citations (prior art references)
    n_jurisdictions: int     # number of countries where protection is in force
    claims: int              # number of independent + dependent claims


@dataclass
class PatentPortfolio:
    """A company's patent portfolio with R&D spend context."""

    patents: List[Patent]
    company: str
    rd_spend_millions: float = 100.0

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def total_patents(self) -> int:
        return len(self.patents)

    @property
    def total_citations(self) -> int:
        return sum(p.citations_received for p in self.patents)

    # ------------------------------------------------------------------
    # h-index
    # ------------------------------------------------------------------

    def h_index(self) -> int:
        """Largest h such that h patents each have >= h forward citations."""
        return h_index([p.citations_received for p in self.patents])

    # ------------------------------------------------------------------
    # Citation score (recency-adjusted)
    # ------------------------------------------------------------------

    def citation_score(self, current_year: int = 2025) -> float:
        """
        Sum of per-patent recency-adjusted citation scores.

        score_i = citations_received_i / (years_since_grant_i + 1)

        Patents granted in the current year get denominator = 1.
        """
        total = 0.0
        for p in self.patents:
            years = max(0, current_year - p.grant_year)
            total += p.citations_received / (years + 1)
        return total

    # ------------------------------------------------------------------
    # Technology concentration
    # ------------------------------------------------------------------

    def tech_hhi(self) -> float:
        """Herfindahl–Hirschman Index over IPC classes (0=diverse, 1=monopoly)."""
        return tech_hhi([p.ipc_class for p in self.patents])

    def tech_diversity(self) -> float:
        """1 - tech_hhi; higher is more diverse."""
        return 1.0 - self.tech_hhi()

    # ------------------------------------------------------------------
    # R&D efficiency
    # ------------------------------------------------------------------

    def rd_efficiency(self) -> dict:
        """Returns patents/M$ and citations/M$ of R&D spend."""
        return rd_efficiency(
            n_patents=self.total_patents,
            n_citations=self.total_citations,
            rd_spend_millions=self.rd_spend_millions,
        )

    # ------------------------------------------------------------------
    # Age distribution
    # ------------------------------------------------------------------

    def age_distribution(self, current_year: int = 2025) -> dict:
        """Mean, median, newest, and oldest patent ages (years since grant)."""
        if not self.patents:
            return {"mean_age": 0.0, "median_age": 0.0, "newest": 0, "oldest": 0}
        ages = [max(0, current_year - p.grant_year) for p in self.patents]
        return {
            "mean_age": float(np.mean(ages)),
            "median_age": float(np.median(ages)),
            "newest": int(min(ages)),
            "oldest": int(max(ages)),
        }

    # ------------------------------------------------------------------
    # IPC breakdown
    # ------------------------------------------------------------------

    def ipc_breakdown(self) -> Dict[str, int]:
        """Count of patents per IPC class, sorted descending."""
        counts = Counter(p.ipc_class for p in self.patents)
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ===========================================================================
# IP Moat Scorer
# ===========================================================================


class IPMoatScorer:
    """
    Computes a composite IP moat score [0, 1] for a patent portfolio.

    moat = 0.30 * citation_quality
         + 0.20 * tech_diversity
         + 0.20 * family_breadth
         + 0.30 * forward_citation_rate
    """

    def __init__(self, portfolio: PatentPortfolio, current_year: int = 2025) -> None:
        self.portfolio = portfolio
        self.current_year = current_year

    def citation_quality(self) -> float:
        """
        Normalised citation quality.

        We normalise the recency-adjusted citation score by the portfolio size
        so larger portfolios do not automatically dominate, then apply a
        logistic-like saturation at a reference level of 10 citations/patent.

        Returns value in [0, 1].
        """
        if self.portfolio.total_patents == 0:
            return 0.0
        raw = self.portfolio.citation_score(self.current_year) / self.portfolio.total_patents
        # Saturate: score of 10 citations/patent -> ~0.91, 5 -> ~0.73
        return float(1.0 - math.exp(-raw / 5.0))

    def family_breadth(self) -> float:
        """
        Average family breadth across the portfolio, normalised by
        MAX_JURISDICTIONS_GLOBALLY.  Returns value in [0, 1].
        """
        if not self.portfolio.patents:
            return 0.0
        scores = [
            min(p.n_jurisdictions / MAX_JURISDICTIONS_GLOBALLY, 1.0)
            for p in self.portfolio.patents
        ]
        return float(np.mean(scores))

    def forward_citation_rate(self, industry_avg_citations: float = 5.0) -> float:
        """
        forward_citation_rate = total_forward_citations
                                / (total_patents * industry_avg_citations)

        Clipped to [0, 1].  A ratio of 1 means the portfolio is at the
        industry average citation rate; above 1 is clipped to 1.
        """
        if self.portfolio.total_patents == 0 or industry_avg_citations <= 0:
            return 0.0
        rate = self.portfolio.total_citations / (
            self.portfolio.total_patents * industry_avg_citations
        )
        return float(min(rate, 1.0))

    def moat_score(self, industry_avg_citations: float = 5.0) -> float:
        """Composite IP moat score in [0, 1]."""
        cq = self.citation_quality()
        td = self.portfolio.tech_diversity()
        fb = self.family_breadth()
        fcr = self.forward_citation_rate(industry_avg_citations)
        return 0.30 * cq + 0.20 * td + 0.20 * fb + 0.30 * fcr

    def moat_breakdown(self, industry_avg_citations: float = 5.0) -> dict:
        """Return the four component scores and the composite."""
        cq = self.citation_quality()
        td = self.portfolio.tech_diversity()
        fb = self.family_breadth()
        fcr = self.forward_citation_rate(industry_avg_citations)
        composite = 0.30 * cq + 0.20 * td + 0.20 * fb + 0.30 * fcr
        return {
            "citation_quality": cq,
            "tech_diversity": td,
            "family_breadth": fb,
            "forward_citation_rate": fcr,
            "moat_score": composite,
            # weights for reference
            "weights": {
                "citation_quality": 0.30,
                "tech_diversity": 0.20,
                "family_breadth": 0.20,
                "forward_citation_rate": 0.30,
            },
        }


# ===========================================================================
# Competitive IP Analysis
# ===========================================================================


class CompetitiveIPAnalysis:
    """Cross-portfolio comparison utilities."""

    def compare(
        self,
        portfolios: List[PatentPortfolio],
        current_year: int = 2025,
        industry_avg_citations: float = 5.0,
    ) -> dict:
        """
        Summarise key metrics for each company in the list.

        Returns a dict keyed by company name with sub-dicts containing
        h_index, total_patents, moat_score, tech_focus (top IPC class).
        """
        result: dict = {}
        for pf in portfolios:
            scorer = IPMoatScorer(pf, current_year)
            breakdown = pf.ipc_breakdown()
            top_class = next(iter(breakdown), "N/A") if breakdown else "N/A"
            result[pf.company] = {
                "h_index": pf.h_index(),
                "total_patents": pf.total_patents,
                "total_citations": pf.total_citations,
                "moat_score": scorer.moat_score(industry_avg_citations),
                "tech_diversity": pf.tech_diversity(),
                "patent_efficiency": pf.rd_efficiency()["patent_efficiency"],
                "tech_focus": top_class,
                "ipc_breakdown": breakdown,
            }
        return result

    def technology_overlap(
        self, p1: PatentPortfolio, p2: PatentPortfolio
    ) -> float:
        """
        Jaccard similarity of the IPC class sets between two portfolios.

        overlap = |IPC(p1) ∩ IPC(p2)| / |IPC(p1) ∪ IPC(p2)|

        Returns 0.0 if both portfolios are empty.
        """
        set1 = {p.ipc_class for p in p1.patents}
        set2 = {p.ipc_class for p in p2.patents}
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def citation_network_centrality(
        self, portfolios: List[PatentPortfolio]
    ) -> dict:
        """
        Simple degree centrality based on cross-citation counts.

        For each portfolio we compute a proxy centrality score as:
            centrality = total_citations / max_citations_in_group

        This approximates which company's patents are most influential
        within the peer group.  Returns scores in [0, 1].
        """
        if not portfolios:
            return {}
        citation_counts = {pf.company: pf.total_citations for pf in portfolios}
        max_cit = max(citation_counts.values()) if citation_counts else 1
        if max_cit == 0:
            max_cit = 1
        return {
            company: count / max_cit
            for company, count in citation_counts.items()
        }


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def h_index(citation_counts: List[int]) -> int:
    """
    Compute the h-index from a list of citation counts.

    Returns the largest h such that h patents each have >= h citations.
    """
    if not citation_counts:
        return 0
    sorted_counts = sorted(citation_counts, reverse=True)
    h = 0
    for i, c in enumerate(sorted_counts):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


def tech_hhi(ipc_classes: List[str]) -> float:
    """
    Herfindahl–Hirschman Index over IPC classes.

    tech_hhi = sum(s_k^2) where s_k = share of patents in class k.
    Returns 0.0 for an empty list, 1.0 for a single class.
    """
    if not ipc_classes:
        return 0.0
    n = len(ipc_classes)
    counts = Counter(ipc_classes)
    return float(sum((c / n) ** 2 for c in counts.values()))


def ip_moat_score(
    portfolio: PatentPortfolio,
    current_year: int = 2025,
    industry_avg_citations: float = 5.0,
) -> float:
    """Convenience wrapper: returns composite IP moat score for a portfolio."""
    return IPMoatScorer(portfolio, current_year).moat_score(industry_avg_citations)


def rd_efficiency(
    n_patents: int, n_citations: int, rd_spend_millions: float
) -> dict:
    """
    Compute R&D efficiency ratios.

    Returns:
        patent_efficiency   — patents per million $ of R&D
        citation_efficiency — citations per million $ of R&D
    """
    if rd_spend_millions <= 0:
        return {"patent_efficiency": float("inf"), "citation_efficiency": float("inf")}
    return {
        "patent_efficiency": n_patents / rd_spend_millions,
        "citation_efficiency": n_citations / rd_spend_millions,
    }
