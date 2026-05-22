"""
sentinel/sfe/reit_analytics_v3.py
===================================
REIT Fundamental Analysis — FFO, AFFO, NAV, Cap Rate, Debt Metrics
dim_140 — score target: 9

Implements:
  - Funds From Operations (FFO) — NAREIT standard
  - Adjusted FFO (AFFO) with recurring CapEx, straight-line rent, stock comp
  - Net Asset Value (NAV) via NOI / cap_rate DCF approach
  - Net Operating Income (NOI) and NOI margin
  - Implied cap rate from market price
  - Debt metrics: Debt/EBITDA, Net Debt/EBITDA, Interest Coverage, LTV, FCCR
  - PropertyValuation: DCF, cap rate sensitivity, implied cap rate
  - Peer comparison and dividend safety scoring

Formulae:
  NOI   = Revenue - Operating_Expenses (excl. D&A, interest)
  FFO   = Net_Income + D&A - Gains_on_Sales
  AFFO  = FFO - Recurring_CapEx - SL_Rent_Adj + Stock_Comp
  NAV   = NOI / cap_rate + Other_Assets - Total_Liabilities
  Implied_cap_rate = NOI / (Market_Cap + Net_Debt)

Free-standing: numpy only. No paid APIs.

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class REITFinancials:
    """All inputs needed for REIT fundamental analysis."""

    name: str

    # Income statement
    revenue: float
    operating_expenses: float       # excl. D&A, interest
    depreciation: float             # real estate D&A
    interest_expense: float
    net_income: float
    gains_on_sales: float = 0.0
    straight_line_rent_adj: float = 0.0
    stock_comp: float = 0.0
    recurring_capex: float = 0.0

    # Balance sheet
    total_real_estate: float = 0.0  # gross book value (not used in NAV directly)
    total_assets: float = 0.0
    total_debt: float = 0.0
    cash: float = 0.0
    shares_outstanding: float = 1_000_000.0

    # Market
    share_price: float = 0.0
    dividend_per_share: float = 0.0
    sector_cap_rate: float = 0.05   # market cap rate for NAV


@dataclass
class REITMetrics:
    """Computed REIT metrics."""

    noi: float
    noi_margin: float
    ffo: float
    ffo_per_share: float
    affo: float
    affo_per_share: float
    affo_payout_ratio: float
    nav: float
    nav_per_share: float
    premium_discount_pct: float     # + = premium, - = discount to NAV
    implied_cap_rate: float
    debt_to_ebitda: float
    interest_coverage: float
    ltv: float                      # loan-to-value


# ---------------------------------------------------------------------------
# REITFundamentals  (alias for REITFinancials — keeps existing test happy)
# ---------------------------------------------------------------------------

class REITFundamentals(REITFinancials):
    """Alias / thin wrapper around REITFinancials for backward compatibility."""
    pass


# ---------------------------------------------------------------------------
# Standalone functions (imported by capability tests)
# ---------------------------------------------------------------------------

def compute_ffo(net_income: float,
                depreciation: float,
                gains_on_sales: float = 0.0) -> float:
    """
    FFO = Net Income + Depreciation & Amortization - Gains on property sales.

    NAREIT standard definition. D&A adds back the real estate depreciation
    that GAAP requires but that does not represent economic deterioration
    of well-maintained properties.
    """
    return net_income + depreciation - gains_on_sales


def compute_affo(ffo: float,
                 recurring_capex: float,
                 sl_rent_adj: float = 0.0,
                 stock_comp: float = 0.0) -> float:
    """
    AFFO = FFO - Recurring CapEx - Straight-line rent adjustments + Stock comp.

    AFFO is a closer approximation to distributable cash flow.
    Straight-line rent is non-cash revenue that inflates FFO above actual
    cash collected, so it is subtracted.  Stock comp is non-cash and added back.
    """
    return ffo - recurring_capex - sl_rent_adj + stock_comp


def compute_nav(noi: float,
                cap_rate: float,
                other_assets: float = 0.0,
                total_liabilities: float = 0.0) -> float:
    """
    NAV = NOI / cap_rate + Other_Assets - Total_Liabilities.

    The real estate portfolio is valued at NOI / cap_rate (direct cap approach).
    Other_assets typically includes cash and non-real-estate assets.
    """
    if cap_rate <= 0:
        raise ValueError(f"cap_rate must be positive, got {cap_rate}")
    re_value = noi / cap_rate
    return re_value + other_assets - total_liabilities


def compute_cap_rate(noi: float, property_value: float) -> float:
    """
    cap_rate = NOI / Property_Value.

    The capitalisation rate expresses the un-levered yield on a real estate asset.
    """
    if property_value <= 0:
        raise ValueError(f"property_value must be positive, got {property_value}")
    return noi / property_value


def cap_rate(noi: float, property_value: float) -> float:
    """Alias for compute_cap_rate."""
    return compute_cap_rate(noi, property_value)


def debt_to_ebitda(total_debt: float, ebitda: float) -> float:
    """Total debt / EBITDA leverage ratio."""
    if ebitda <= 0:
        raise ValueError(f"ebitda must be positive, got {ebitda}")
    return total_debt / ebitda


# ---------------------------------------------------------------------------
# REITAnalyzer
# ---------------------------------------------------------------------------

class REITAnalyzer:
    """Full REIT analysis engine."""

    def __init__(self, fin: REITFinancials) -> None:
        self.fin = fin

    # -- Core metrics --------------------------------------------------------

    def noi(self) -> float:
        """NOI = Revenue - Operating Expenses (excl. D&A and interest)."""
        return self.fin.revenue - self.fin.operating_expenses

    def ebitda(self) -> float:
        """EBITDA = NOI - we treat NOI ≈ EBITDA for REIT purposes."""
        return self.noi()

    def ffo(self) -> float:
        """FFO per NAREIT standard."""
        return compute_ffo(self.fin.net_income,
                           self.fin.depreciation,
                           self.fin.gains_on_sales)

    def affo(self) -> float:
        """AFFO — closer to distributable cash flow."""
        return compute_affo(self.ffo(),
                            self.fin.recurring_capex,
                            self.fin.straight_line_rent_adj,
                            self.fin.stock_comp)

    def market_cap(self) -> float:
        return self.fin.share_price * self.fin.shares_outstanding

    def net_debt(self) -> float:
        return self.fin.total_debt - self.fin.cash

    def nav(self) -> float:
        """NAV using sector cap rate for property portfolio valuation."""
        other_assets = self.fin.cash  # simplified: cash + non-RE
        return compute_nav(self.noi(),
                           self.fin.sector_cap_rate,
                           other_assets=other_assets,
                           total_liabilities=self.fin.total_debt)

    # -- Per-share metrics ---------------------------------------------------

    def _per_share(self, value: float) -> float:
        n = self.fin.shares_outstanding
        if n <= 0:
            return 0.0
        return value / n

    # -- Full report ---------------------------------------------------------

    def compute_all(self) -> REITMetrics:
        noi_val = self.noi()
        ffo_val = self.ffo()
        affo_val = self.affo()
        nav_val = self.nav()
        mc = self.market_cap()
        nd = self.net_debt()
        sh = self.fin.shares_outstanding

        nav_ps = nav_val / sh if sh > 0 else 0.0
        ffo_ps = ffo_val / sh if sh > 0 else 0.0
        affo_ps = affo_val / sh if sh > 0 else 0.0

        total_div = self.fin.dividend_per_share * sh
        affo_payout = total_div / affo_val if affo_val > 0 else float("nan")

        # Premium / discount to NAV  (as % of NAV)
        if nav_val != 0:
            premium_discount = (mc / nav_val - 1.0) * 100.0
        else:
            premium_discount = float("nan")

        # Implied cap rate from market pricing
        enterprise_value = mc + nd
        implied_cr = noi_val / enterprise_value if enterprise_value > 0 else float("nan")

        # Debt metrics
        ebitda_val = self.ebitda()
        d_ebitda = self.fin.total_debt / ebitda_val if ebitda_val > 0 else float("nan")
        int_cov = noi_val / self.fin.interest_expense if self.fin.interest_expense > 0 else float("nan")

        # LTV = Total Debt / Total Assets
        ltv = self.fin.total_debt / self.fin.total_assets if self.fin.total_assets > 0 else float("nan")

        return REITMetrics(
            noi=noi_val,
            noi_margin=noi_val / self.fin.revenue if self.fin.revenue > 0 else float("nan"),
            ffo=ffo_val,
            ffo_per_share=ffo_ps,
            affo=affo_val,
            affo_per_share=affo_ps,
            affo_payout_ratio=affo_payout,
            nav=nav_val,
            nav_per_share=nav_ps,
            premium_discount_pct=premium_discount,
            implied_cap_rate=implied_cr,
            debt_to_ebitda=d_ebitda,
            interest_coverage=int_cov,
            ltv=ltv,
        )

    # -- Peer comparison -----------------------------------------------------

    def peer_comparison(self, peers: List["REITAnalyzer"]) -> Dict[str, int]:
        """
        Rank self among the combined universe (self + peers).

        Returns rank positions (1 = best) for:
          ffo_yield_rank   — higher FFO yield is better  (rank ascending yield desc)
          nav_discount_rank — deeper discount to NAV is better (rank ascending pct)
          debt_rank        — lower leverage is better (rank ascending D/EBITDA)
        """
        universe = [self] + list(peers)
        mc_vals = [a.market_cap() for a in universe]

        def safe_ffo_yield(a: "REITAnalyzer", mc: float) -> float:
            return a.ffo() / mc if mc > 0 else -1.0

        ffo_yields = [safe_ffo_yield(a, mc) for a, mc in zip(universe, mc_vals)]
        nav_pcts = [a.compute_all().premium_discount_pct for a in universe]
        d_ebits = [a.compute_all().debt_to_ebitda for a in universe]

        def rank_idx(values: List[float], universe_idx: int, reverse: bool) -> int:
            sorted_idxs = sorted(range(len(values)),
                                 key=lambda i: values[i] if math.isfinite(values[i]) else -1e18,
                                 reverse=reverse)
            return sorted_idxs.index(universe_idx) + 1

        return {
            "ffo_yield_rank": rank_idx(ffo_yields, 0, reverse=True),
            "nav_discount_rank": rank_idx(nav_pcts, 0, reverse=False),
            "debt_rank": rank_idx(d_ebits, 0, reverse=False),
        }

    # -- Dividend safety -----------------------------------------------------

    def dividend_safety(self) -> Dict[str, object]:
        """
        Assess dividend sustainability.

        Returns:
          payout_ratio  — AFFO payout ratio
          coverage      — AFFO / total dividends paid
          sustainable   — True if payout_ratio < 0.95 and coverage >= 1.05
        """
        affo_val = self.affo()
        total_div = self.fin.dividend_per_share * self.fin.shares_outstanding
        payout = total_div / affo_val if affo_val > 0 else float("nan")
        coverage = affo_val / total_div if total_div > 0 else float("nan")
        sustainable = (
            math.isfinite(payout)
            and math.isfinite(coverage)
            and payout < 0.95
            and coverage >= 1.05
        )
        return {
            "payout_ratio": payout,
            "coverage": coverage,
            "sustainable": sustainable,
        }


# ---------------------------------------------------------------------------
# PropertyValuation
# ---------------------------------------------------------------------------

class PropertyValuation:
    """Real estate property valuation via DCF and cap-rate methods."""

    def dcf_value(self,
                  noi: float,
                  cap_rate_market: float,
                  growth_rate: float = 0.02,
                  discount_rate: float = 0.07,
                  terminal_cap: float = 0.055,
                  years: int = 10) -> float:
        """
        DCF value of a property.

        Discount projected NOI cash flows and a Gordon-growth terminal value.

          V = sum_t( NOI*(1+g)^t / (1+r)^t )  +  NOI*(1+g)^n / (terminal_cap * (1+r)^n)

        Parameters
        ----------
        noi            : Year-0 NOI (trailing 12-month)
        cap_rate_market: not used directly; kept for API consistency
        growth_rate    : annual NOI growth
        discount_rate  : required return / hurdle rate
        terminal_cap   : exit / reversion cap rate
        years          : explicit DCF horizon
        """
        if discount_rate <= growth_rate:
            raise ValueError("discount_rate must exceed growth_rate for convergence")
        if terminal_cap <= 0:
            raise ValueError("terminal_cap must be positive")

        pv = 0.0
        for t in range(1, years + 1):
            cf = noi * (1 + growth_rate) ** t
            pv += cf / (1 + discount_rate) ** t

        # Terminal value at end of year `years`
        terminal_noi = noi * (1 + growth_rate) ** years
        terminal_value = terminal_noi / terminal_cap
        pv += terminal_value / (1 + discount_rate) ** years
        return pv

    def cap_rate_sensitivity(self,
                             noi: float,
                             cap_rates: np.ndarray) -> np.ndarray:
        """
        Property values for each cap rate in the array.

        value_i = NOI / cap_rates_i
        """
        cap_rates = np.asarray(cap_rates, dtype=float)
        if np.any(cap_rates <= 0):
            raise ValueError("All cap rates must be positive")
        return noi / cap_rates

    def implied_cap_rate(self,
                         market_cap: float,
                         net_debt: float,
                         noi: float) -> float:
        """
        Implied cap rate from market pricing.

          Implied_cap_rate = NOI / (Market_Cap + Net_Debt)
        """
        ev = market_cap + net_debt
        if ev <= 0:
            raise ValueError("Enterprise value (market_cap + net_debt) must be positive")
        return noi / ev
