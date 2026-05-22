"""
sentinel/spm/customer_concentration_v3.py
==========================================
Customer Concentration & Contract-Win Intelligence Analytics
dim_127 — score target: 9

Implements:
  - Herfindahl-Hirschman Index (HHI) for revenue concentration
  - Top-N customer concentration (CR5, CR10, etc.)
  - Revenue at risk from top-customer churn
  - Customer Lifetime Value (CLV) using retention-rate discounting
  - Renewal risk calendar based on contract end dates
  - Pipeline analytics: expected value, stage-weighted value, win rate
  - NPS-adjusted revenue growth proxy

Free-standing: numpy only. No paid APIs or network calls required.

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stage-based win probability weights used in weighted pipeline calculation
STAGE_WEIGHTS: Dict[str, float] = {
    "prospect": 0.10,
    "qualified": 0.25,
    "proposal": 0.50,
    "negotiation": 0.80,
}

# HHI thresholds (DOJ / FTC standard for market concentration)
HHI_LOW_THRESHOLD = 0.15        # < 0.15 → low concentration
HHI_MODERATE_THRESHOLD = 0.25   # 0.15–0.25 → moderate
HHI_HIGH_THRESHOLD = 0.40       # 0.25–0.40 → high
# >= 0.40 → critical


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class Customer:
    """Represents a single customer with revenue and relationship attributes."""
    name: str
    annual_revenue: float
    years_as_customer: float
    retention_probability: float    # 0–1: probability of renewing next year
    gross_margin_pct: float         # 0–1: gross margin on this customer's revenue
    contract_end_year: int = 2026
    industry: str = "unknown"


@dataclass
class ContractOpportunity:
    """A sales pipeline opportunity."""
    name: str
    value: float                    # total contract value
    win_probability: float          # 0–1 analyst estimate
    stage: str                      # 'prospect' | 'qualified' | 'proposal' | 'negotiation'
    expected_close_days: int = 90


@dataclass
class ConcentrationRisk:
    """Summary output from CustomerConcentrationAnalyzer.assess()."""
    hhi: float
    top_customer_pct: float         # revenue share of single largest customer (%)
    cr5_pct: float                  # combined share of top-5 customers (%)
    revenue_at_risk: float          # expected loss from top-customer churn ($)
    concentration_category: str     # 'low' | 'moderate' | 'high' | 'critical'
    clv_weighted_avg: float         # revenue-weighted average CLV across all customers


# ===========================================================================
# Core analyzer
# ===========================================================================

class CustomerConcentrationAnalyzer:
    """
    Analyses customer revenue concentration and associated risks.

    Parameters
    ----------
    customers     : list of Customer objects
    total_revenue : company total revenue (may exceed sum of customers if
                    there are anonymous / small customers grouped elsewhere)
    """

    def __init__(self, customers: List[Customer], total_revenue: float) -> None:
        if not customers:
            raise ValueError("customers list cannot be empty")
        if total_revenue <= 0:
            raise ValueError("total_revenue must be positive")
        self.customers = customers
        self.total_revenue = total_revenue

        # Precompute revenue shares
        self._shares = np.array(
            [c.annual_revenue / total_revenue for c in customers], dtype=float
        )

    # ------------------------------------------------------------------
    # Concentration metrics
    # ------------------------------------------------------------------

    def hhi(self) -> float:
        """
        Herfindahl-Hirschman Index.

        HHI = sum(s_i^2)  where s_i = revenue_i / total_revenue.

        Range: [1/n, 1].  A perfectly diversified portfolio → 1/n.
        """
        return float(np.sum(self._shares ** 2))

    def top_n_concentration(self, n: int = 5) -> float:
        """
        CR_n: combined revenue share of the top-n customers (as a percentage).

        Returns a value in [0, 100].
        """
        sorted_shares = np.sort(self._shares)[::-1]
        top = sorted_shares[:n]
        return float(np.sum(top) * 100.0)

    def revenue_at_risk(self, discount_rate: float = 0.10) -> float:
        """
        Expected revenue loss if the top customer churns.

        revenue_at_risk = top_customer_revenue * (1 - retention_probability)
        where (1 - retention) is interpreted as the churn probability.

        The discount_rate parameter is accepted for API compatibility but the
        primary driver is the customer's own retention_probability.
        """
        top_customer = max(self.customers, key=lambda c: c.annual_revenue)
        churn_prob = 1.0 - top_customer.retention_probability
        return float(top_customer.annual_revenue * churn_prob)

    def clv(self, customer: Customer, discount_rate: float = 0.10) -> float:
        """
        Customer Lifetime Value using the perpetuity / Gordon-growth model.

        CLV = gross_margin_$ * (retention_rate / (1 + discount_rate - retention_rate))

        where gross_margin_$ = annual_revenue * gross_margin_pct.
        """
        gross_margin_dollars = customer.annual_revenue * customer.gross_margin_pct
        r = customer.retention_probability
        d = discount_rate
        denominator = 1.0 + d - r
        if denominator <= 0:
            # Edge case: retention ≥ 1 + discount → infinite CLV; cap at 100× margin
            return gross_margin_dollars * 100.0
        return float(gross_margin_dollars * (r / denominator))

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(self, discount_rate: float = 0.10) -> ConcentrationRisk:
        """
        Produce a full ConcentrationRisk summary.
        """
        hhi_val = self.hhi()
        top_share = float(np.max(self._shares))
        cr5 = self.top_n_concentration(n=5)
        rar = self.revenue_at_risk(discount_rate=discount_rate)

        # Concentration category based on HHI
        if hhi_val < HHI_LOW_THRESHOLD:
            category = "low"
        elif hhi_val < HHI_MODERATE_THRESHOLD:
            category = "moderate"
        elif hhi_val < HHI_HIGH_THRESHOLD:
            category = "high"
        else:
            category = "critical"

        # Revenue-weighted average CLV
        total_rev = sum(c.annual_revenue for c in self.customers)
        clv_wavg = 0.0
        for c in self.customers:
            weight = c.annual_revenue / total_rev if total_rev > 0 else 0.0
            clv_wavg += weight * self.clv(c, discount_rate=discount_rate)

        return ConcentrationRisk(
            hhi=hhi_val,
            top_customer_pct=top_share * 100.0,
            cr5_pct=cr5,
            revenue_at_risk=rar,
            concentration_category=category,
            clv_weighted_avg=clv_wavg,
        )

    # ------------------------------------------------------------------
    # Breakdown helpers
    # ------------------------------------------------------------------

    def industry_breakdown(self) -> Dict[str, float]:
        """
        Revenue share by industry (as a fraction of total_revenue).

        Returns dict mapping industry → revenue fraction.
        """
        breakdown: Dict[str, float] = {}
        for c in self.customers:
            breakdown[c.industry] = breakdown.get(c.industry, 0.0) + c.annual_revenue
        return {k: v / self.total_revenue for k, v in breakdown.items()}

    def renewal_risk_calendar(self, current_year: int = 2025) -> List[dict]:
        """
        List upcoming contract renewals with associated risk scores.

        Each entry: {customer, contract_end_year, revenue, risk_score}

        risk_score = revenue_share * (1 - retention_probability)
        Higher → higher urgency to retain.

        Sorted by (contract_end_year ASC, risk_score DESC).
        """
        calendar = []
        for c in self.customers:
            years_to_renewal = c.contract_end_year - current_year
            churn_prob = 1.0 - c.retention_probability
            revenue_share = c.annual_revenue / self.total_revenue
            risk_score = revenue_share * churn_prob

            calendar.append({
                "customer": c.name,
                "contract_end_year": c.contract_end_year,
                "revenue": c.annual_revenue,
                "years_to_renewal": years_to_renewal,
                "retention_probability": c.retention_probability,
                "risk_score": round(risk_score, 6),
            })

        calendar.sort(key=lambda x: (x["contract_end_year"], -x["risk_score"]))
        return calendar


# ===========================================================================
# Pipeline analytics
# ===========================================================================

class PipelineAnalytics:
    """
    Analyses a sales pipeline of ContractOpportunity objects.

    Parameters
    ----------
    opportunities : list of ContractOpportunity
    """

    def __init__(self, opportunities: List[ContractOpportunity]) -> None:
        self.opportunities = opportunities

    def pipeline_value(self) -> float:
        """
        Raw probability-weighted pipeline value.

        pipeline_value = sum(deal_value * win_probability)
        """
        return float(sum(o.value * o.win_probability for o in self.opportunities))

    def weighted_pipeline(self) -> float:
        """
        Stage-adjusted pipeline value using canonical stage weights.

        Uses STAGE_WEIGHTS dict; falls back to raw win_probability if stage unknown.
        """
        total = 0.0
        for o in self.opportunities:
            stage_wt = STAGE_WEIGHTS.get(o.stage.lower(), o.win_probability)
            total += o.value * stage_wt
        return float(total)

    def win_rate(self) -> float:
        """
        Estimated aggregate win rate from stage distribution.

        Approximated as the revenue-weighted average win probability across all
        opportunities (a common pipeline heuristic when historical data is absent).
        """
        total_value = sum(o.value for o in self.opportunities)
        if total_value == 0:
            return 0.0
        return float(
            sum(o.value * o.win_probability for o in self.opportunities) / total_value
        )

    def expected_revenue_next_quarter(self) -> float:
        """
        Subset of pipeline expected to close within 90 days.

        Returns probability-weighted value of deals with expected_close_days <= 90.
        """
        return float(
            sum(
                o.value * o.win_probability
                for o in self.opportunities
                if o.expected_close_days <= 90
            )
        )

    def stage_breakdown(self) -> Dict[str, dict]:
        """
        Summary statistics grouped by stage.

        Returns dict mapping stage → {count, total_value, expected_value, avg_prob}.
        """
        stages: Dict[str, dict] = {}
        for o in self.opportunities:
            s = o.stage.lower()
            if s not in stages:
                stages[s] = {
                    "count": 0,
                    "total_value": 0.0,
                    "expected_value": 0.0,
                    "avg_prob": 0.0,
                    "_prob_sum": 0.0,
                }
            stages[s]["count"] += 1
            stages[s]["total_value"] += o.value
            stages[s]["expected_value"] += o.value * o.win_probability
            stages[s]["_prob_sum"] += o.win_probability

        # Finalise avg_prob
        for s, d in stages.items():
            d["avg_prob"] = d["_prob_sum"] / d["count"] if d["count"] > 0 else 0.0
            del d["_prob_sum"]

        return stages


# ===========================================================================
# Standalone convenience functions
# ===========================================================================

def hhi(revenue_list: List[float]) -> float:
    """
    Compute Herfindahl-Hirschman Index from a list of revenue figures.

    HHI = sum(s_i^2)  where s_i = revenue_i / sum(revenue_list).

    Parameters
    ----------
    revenue_list : list of non-negative floats

    Returns
    -------
    float in (0, 1]
    """
    arr = np.array(revenue_list, dtype=float)
    total = arr.sum()
    if total <= 0:
        raise ValueError("revenue_list must sum to a positive number")
    shares = arr / total
    return float(np.sum(shares ** 2))


def customer_lifetime_value(
    gross_margin: float,
    retention_rate: float,
    discount_rate: float,
) -> float:
    """
    Customer Lifetime Value (perpetuity model).

    CLV = gross_margin * (retention_rate / (1 + discount_rate - retention_rate))

    Parameters
    ----------
    gross_margin   : annual gross margin in dollars (or fraction × revenue)
    retention_rate : probability of retention per period (0–1)
    discount_rate  : periodic discount / hurdle rate (e.g. 0.10 = 10%)

    Returns
    -------
    float — CLV in same units as gross_margin
    """
    denominator = 1.0 + discount_rate - retention_rate
    if denominator <= 0:
        return gross_margin * 100.0   # infinite-CLV cap
    return float(gross_margin * (retention_rate / denominator))


def revenue_at_risk(top_customer_revenue: float, churn_probability: float) -> float:
    """
    Expected revenue loss from potential top-customer churn.

    revenue_at_risk = top_customer_revenue * churn_probability

    Parameters
    ----------
    top_customer_revenue : annual revenue from the largest customer
    churn_probability    : probability (0–1) that the customer churns

    Returns
    -------
    float — expected revenue loss
    """
    return float(top_customer_revenue * churn_probability)


def pipeline_expected_value(opportunities: List[ContractOpportunity]) -> float:
    """
    Probability-weighted expected value of a list of pipeline opportunities.

    pipeline_value = sum(deal_value * win_probability)

    Parameters
    ----------
    opportunities : list of ContractOpportunity

    Returns
    -------
    float — expected pipeline value
    """
    return float(sum(o.value * o.win_probability for o in opportunities))


# ---------------------------------------------------------------------------
# NPS-adjusted revenue growth (proxy for NLP intelligence signal)
# ---------------------------------------------------------------------------

def nps_adjusted_revenue_growth(
    revenue_growth: float,
    nps_score: float,
) -> float:
    """
    Adjust reported revenue growth by an NPS quality multiplier.

    revenue_growth_adj = revenue_growth * (1 + 0.05 * nps_score / 100)

    Parameters
    ----------
    revenue_growth : raw revenue growth rate (e.g. 0.12 = 12%)
    nps_score      : Net Promoter Score in range [-100, 100]

    Returns
    -------
    float — NPS-adjusted growth rate
    """
    return float(revenue_growth * (1.0 + 0.05 * nps_score / 100.0))
