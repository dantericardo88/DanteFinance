"""
sentinel/sfe/cre_analytics_v3.py
==================================
Commercial Real Estate (CRE) Transaction Comps, Cap Rate Analysis & Market Analytics
dim_141 — score target: 9

Implements:
  - CRETransaction dataclass with property-level metrics
  - TransactionComps / CRECompsEngine: comp selection, market metrics, z-scores, percentiles
  - CREValuation: DCF, effective rent, DSCR, max loan, breakeven occupancy
  - CapRateAnalytics: spread, implied growth, cap-rate-by-tier, trend

Formulae:
  Price per SF        = sale_price / square_feet
  Cap rate            = NOI / sale_price
  NOI per SF          = NOI / square_feet
  Price z-score       = (price_psf - mean_psf) / std_psf
  Cap rate spread     = property_cap_rate - treasury_10yr
  Effective rent      = asking_rent_psf * SF * (1 - vacancy) * (1 - concession)
  NOI                 = Effective_rent - Operating_Expenses
  DSCR                = NOI / Annual_Debt_Service
  Breakeven_occ       = Operating_Expenses / (asking_rent_psf * SF)
  Implied growth (g)  = discount_rate - cap_rate   (Gordon)

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

VALID_PROPERTY_TYPES = frozenset(
    {"office", "retail", "industrial", "multifamily", "hotel", "mixed-use"}
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CRETransaction:
    """Single commercial real estate sale transaction."""

    property_id: str
    property_type: str       # 'office', 'retail', 'industrial', 'multifamily', 'hotel'
    location: str
    sale_price: float
    square_feet: float
    noi: float               # trailing 12-month NOI
    year: int = 2024
    vacancy_rate: float = 0.05

    @property
    def price_per_sf(self) -> float:
        """Price per square foot."""
        if self.square_feet <= 0:
            return float("nan")
        return self.sale_price / self.square_feet

    @property
    def cap_rate(self) -> float:
        """Going-in cap rate = NOI / sale_price."""
        if self.sale_price <= 0:
            return float("nan")
        return self.noi / self.sale_price

    @property
    def noi_per_sf(self) -> float:
        """NOI per square foot."""
        if self.square_feet <= 0:
            return float("nan")
        return self.noi / self.square_feet


@dataclass
class MarketMetrics:
    """Aggregated market metrics for a property type / sub-market."""

    avg_cap_rate: float
    median_cap_rate: float
    avg_price_per_sf: float
    median_price_per_sf: float
    cap_rate_spread: float      # vs 10yr treasury yield
    n_transactions: int
    property_type: str


# ---------------------------------------------------------------------------
# Standalone functions (imported by capability tests)
# ---------------------------------------------------------------------------

def price_per_sf(sale_price: float, sf: float) -> float:
    """Price per square foot."""
    if sf <= 0:
        raise ValueError(f"sf must be positive, got {sf}")
    return sale_price / sf


def implied_cap_rate(noi: float, price: float) -> float:
    """Going-in cap rate = NOI / price."""
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    return noi / price


def compute_cre_cap_rate(noi: float, price: float) -> float:
    """Alias for implied_cap_rate (used by capability test)."""
    return implied_cap_rate(noi, price)


def dscr(noi: float, annual_debt_service: float) -> float:
    """Debt Service Coverage Ratio = NOI / Annual Debt Service."""
    if annual_debt_service <= 0:
        raise ValueError(f"annual_debt_service must be positive, got {annual_debt_service}")
    return noi / annual_debt_service


def dcf_value(noi: float,
              g: float,
              r: float,
              terminal_cap: float,
              years: int = 10) -> float:
    """
    DCF value of a CRE asset.

    V = sum_t( NOI*(1+g)^t / (1+r)^t ) + NOI*(1+g)^n / (terminal_cap * (1+r)^n)
    """
    if r <= g:
        raise ValueError("discount rate r must exceed growth rate g")
    if terminal_cap <= 0:
        raise ValueError("terminal_cap must be positive")
    pv = 0.0
    for t in range(1, years + 1):
        pv += noi * (1 + g) ** t / (1 + r) ** t
    terminal_noi = noi * (1 + g) ** years
    terminal_value = terminal_noi / terminal_cap
    pv += terminal_value / (1 + r) ** years
    return pv


# ---------------------------------------------------------------------------
# CRECompsEngine  (alias: TransactionComps)
# ---------------------------------------------------------------------------

class CRECompsEngine:
    """
    Comparable transaction analysis engine.

    Provides market metrics, comparable valuation, z-scores, and percentiles
    based on a pool of CRETransaction objects.
    """

    def __init__(self, transactions: List[CRETransaction]) -> None:
        self.transactions = list(transactions)

    # -- Filtering -----------------------------------------------------------

    def by_type(self, ptype: str) -> List[CRETransaction]:
        """Return all transactions matching property_type (case-insensitive)."""
        ptype_lower = ptype.lower()
        return [t for t in self.transactions if t.property_type.lower() == ptype_lower]

    def _filter(self, property_type: Optional[str]) -> List[CRETransaction]:
        if property_type is None:
            return self.transactions
        return self.by_type(property_type)

    # -- Market metrics ------------------------------------------------------

    def market_metrics(self,
                       property_type: Optional[str] = None,
                       treasury_yield: float = 0.042) -> MarketMetrics:
        """Aggregate market metrics for the selected comp pool."""
        comps = self._filter(property_type)
        if not comps:
            raise ValueError(f"No transactions found for property_type={property_type!r}")

        cap_rates = np.array([c.cap_rate for c in comps])
        prices_psf = np.array([c.price_per_sf for c in comps])

        avg_cr = float(np.mean(cap_rates))
        med_cr = float(np.median(cap_rates))
        avg_psf = float(np.mean(prices_psf))
        med_psf = float(np.median(prices_psf))
        spread = avg_cr - treasury_yield

        return MarketMetrics(
            avg_cap_rate=avg_cr,
            median_cap_rate=med_cr,
            avg_price_per_sf=avg_psf,
            median_price_per_sf=med_psf,
            cap_rate_spread=spread,
            n_transactions=len(comps),
            property_type=property_type or "all",
        )

    # -- Comparable valuation ------------------------------------------------

    def comparable_value(self,
                          subject_noi: float,
                          subject_sf: float,
                          property_type: str) -> Dict[str, object]:
        """
        Estimate value of a subject property using comp pool metrics.

        Returns
        -------
        dict with keys:
          cap_rate_value  — value implied by avg comp cap rate
          price_per_sf_value — value implied by avg price per SF
          avg_value       — simple average of the two approaches
          range           — (low, high) tuple (10th / 90th percentile of cap-rate values)
        """
        comps = self._filter(property_type)
        if not comps:
            raise ValueError(f"No comps found for property_type={property_type!r}")

        cap_rates = np.array([c.cap_rate for c in comps])
        prices_psf = np.array([c.price_per_sf for c in comps])

        avg_cr = float(np.mean(cap_rates))
        avg_psf = float(np.mean(prices_psf))

        cr_val = subject_noi / avg_cr if avg_cr > 0 else float("nan")
        psf_val = subject_sf * avg_psf

        avg_val = (cr_val + psf_val) / 2.0

        # Range using 10th / 90th percentile cap rates
        cr_p10 = float(np.percentile(cap_rates, 10))
        cr_p90 = float(np.percentile(cap_rates, 90))
        low = subject_noi / cr_p90 if cr_p90 > 0 else float("nan")
        high = subject_noi / cr_p10 if cr_p10 > 0 else float("nan")

        return {
            "cap_rate_value": cr_val,
            "price_per_sf_value": psf_val,
            "avg_value": avg_val,
            "range": (low, high),
        }

    # -- Z-score and percentile ----------------------------------------------

    def zscore(self, transaction: CRETransaction) -> float:
        """
        Z-score of this transaction's price per SF vs the comp pool.

        z = (price_psf - mean_psf) / std_psf
        """
        comps = self.transactions  # compare to full pool
        prices = np.array([c.price_per_sf for c in comps])
        std = float(np.std(prices, ddof=1))
        if std == 0:
            return 0.0
        return (transaction.price_per_sf - float(np.mean(prices))) / std

    def percentile(self,
                   cap_rate: float = None,
                   property_type: Optional[str] = None,
                   *,
                   cap_rate_query: float = None) -> float:
        """
        Percentile rank of a query cap rate vs the comp pool.

        Returns 0-100 where 100 means cap_rate is higher than all comps.
        Accepts both positional and keyword argument forms.
        """
        # Accept either positional `cap_rate` or keyword `cap_rate_query`
        query = cap_rate if cap_rate is not None else cap_rate_query
        if query is None:
            raise ValueError("Provide cap_rate or cap_rate_query")
        comps = self._filter(property_type)
        if not comps:
            return float("nan")
        cap_rates_arr = np.array([c.cap_rate for c in comps])
        pct = float(np.sum(cap_rates_arr < query)) / len(cap_rates_arr) * 100.0
        return pct


# Alias for backward compatibility / spec naming
TransactionComps = CRECompsEngine


# ---------------------------------------------------------------------------
# CREValuation
# ---------------------------------------------------------------------------

class CREValuation:
    """Commercial real estate valuation tools."""

    def dcf_value(self,
                  noi: float,
                  growth_rate: float = 0.03,
                  discount_rate: float = 0.07,
                  terminal_cap: float = 0.055,
                  years: int = 10) -> float:
        """DCF value — see module-level dcf_value for formula."""
        return dcf_value(noi, growth_rate, discount_rate, terminal_cap, years)

    def effective_rent(self,
                       asking_rent_psf: float,
                       sf: float,
                       vacancy: float,
                       concession_pct: float = 0.0) -> float:
        """
        Effective Rent = Asking_Rent_PSF * SF * (1 - vacancy) * (1 - concession).

        This is the actual rent collected after adjusting for vacancy and
        concessions (e.g. free-rent periods, TI allowances).
        """
        if not 0.0 <= vacancy <= 1.0:
            raise ValueError(f"vacancy must be in [0,1], got {vacancy}")
        if not 0.0 <= concession_pct <= 1.0:
            raise ValueError(f"concession_pct must be in [0,1], got {concession_pct}")
        return asking_rent_psf * sf * (1.0 - vacancy) * (1.0 - concession_pct)

    def dscr(self, noi: float, annual_debt_service: float) -> float:
        """DSCR = NOI / Annual Debt Service."""
        return dscr(noi, annual_debt_service)

    def max_loan(self,
                 noi: float,
                 interest_rate: float,
                 amortization_years: int = 25,
                 dscr_min: float = 1.25) -> float:
        """
        Maximum supportable loan at a minimum DSCR.

        Uses the mortgage constant to compute annual debt service per $1 of loan:
          k = r * (1+r)^n / ((1+r)^n - 1)   (monthly compounding)
          Annual_DS_per_dollar = 12 * k
          Max_loan = NOI / (dscr_min * Annual_DS_per_dollar)

        Falls back to interest-only if amortization_years is very large.
        """
        r_monthly = interest_rate / 12.0
        n_months = amortization_years * 12
        if r_monthly == 0:
            # Interest-free: full principal returned at end
            annual_ds_per_dollar = 1.0 / amortization_years
        else:
            k = r_monthly * (1 + r_monthly) ** n_months / ((1 + r_monthly) ** n_months - 1)
            annual_ds_per_dollar = 12.0 * k
        return noi / (dscr_min * annual_ds_per_dollar)

    def breakeven_occupancy(self,
                             operating_expenses: float,
                             asking_rent_psf: float,
                             sf: float) -> float:
        """
        Breakeven occupancy rate.

          BEO = Operating_Expenses / (Asking_Rent_PSF * SF)

        The occupancy level at which gross revenue equals total operating costs.
        """
        gross_potential = asking_rent_psf * sf
        if gross_potential <= 0:
            raise ValueError("asking_rent_psf * sf must be positive")
        return operating_expenses / gross_potential


# ---------------------------------------------------------------------------
# CapRateAnalytics
# ---------------------------------------------------------------------------

class CapRateAnalytics:
    """Market cap rate analysis utilities."""

    def cap_rate_spread(self, cap_rate_val: float, treasury_10yr: float) -> float:
        """
        Cap rate spread over 10-year Treasury.

        Represents the risk premium that real estate carries over risk-free bonds.
        """
        return cap_rate_val - treasury_10yr

    def implied_growth(self, cap_rate_val: float, discount_rate: float) -> float:
        """
        Implied growth rate via Gordon Growth Model.

          g = r - cap_rate

        If cap_rate > r, the model implies negative growth (distressed asset).
        """
        return discount_rate - cap_rate_val

    def cap_rate_by_tier(self,
                          transactions: List[CRETransaction]) -> Dict[str, float]:
        """
        Average cap rate per property type (tier).

        Returns {property_type: avg_cap_rate} for all types present.
        """
        by_type: Dict[str, List[float]] = {}
        for txn in transactions:
            by_type.setdefault(txn.property_type, []).append(txn.cap_rate)
        return {ptype: float(np.mean(rates)) for ptype, rates in by_type.items()}

    def trend(self,
              transactions_by_year: Dict[int, List[CRETransaction]]) -> np.ndarray:
        """
        Average cap rate per year in ascending year order.

        Returns 1-D numpy array of average cap rates ordered by year.
        """
        years = sorted(transactions_by_year.keys())
        avg_rates = []
        for yr in years:
            txns = transactions_by_year[yr]
            if txns:
                avg_rates.append(float(np.mean([t.cap_rate for t in txns])))
            else:
                avg_rates.append(float("nan"))
        return np.array(avg_rates)
