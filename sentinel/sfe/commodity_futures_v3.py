"""
commodity_futures_v3.py — Commodity futures roll analysis engine.

dim_142: Contango/backwardation, roll yield, carry, term structure

Architecture:
  FuturesContract           — Single futures contract dataclass
  TermStructure             — Term structure across maturities with shape analytics
  RollAnalytics             — Roll yield, calendar spread, optimal roll timing
  CostOfCarryModel          — Cost-of-carry pricing, implied convenience yield
  CommoditySignals          — Momentum, carry and term-structure signals
  CommodityRollAnalyzer     — Orchestrator (legacy alias for crusade compatibility)
  RollYieldCalculator       — Standalone roll yield calculator (legacy alias)

No network calls — numpy/scipy only.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FuturesContract:
    """A single futures contract."""
    commodity: str
    expiry_months_out: float   # months to expiry (e.g. 1.0, 3.0, 6.0, 12.0)
    price: float
    open_interest: int = 0
    volume: int = 0


@dataclass
class TermStructure:
    """Term structure of futures prices across maturities.

    Wraps a list of FuturesContract objects sorted by expiry and provides
    shape analytics (contango/backwardation, slope, curvature).
    """
    contracts: List[FuturesContract]
    spot_price: float

    def __post_init__(self) -> None:
        # Sort ascending by maturity
        self.contracts = sorted(self.contracts, key=lambda c: c.expiry_months_out)

    @property
    def maturities(self) -> np.ndarray:
        """Array of months-to-expiry for each contract."""
        return np.array([c.expiry_months_out for c in self.contracts])

    @property
    def prices(self) -> np.ndarray:
        """Array of futures prices aligned with maturities."""
        return np.array([c.price for c in self.contracts])

    @property
    def is_contango(self) -> bool:
        """True when prices generally rise with maturity (far > near)."""
        p = self.prices
        if len(p) < 2:
            return False
        return bool(p[-1] > p[0])

    @property
    def is_backwardation(self) -> bool:
        """True when prices generally fall with maturity (near > far)."""
        p = self.prices
        if len(p) < 2:
            return False
        return bool(p[0] > p[-1])

    def slope(self) -> float:
        """OLS slope of price on maturity ($/month).

        Positive slope ↔ contango, negative slope ↔ backwardation.
        """
        m = self.maturities
        p = self.prices
        if len(m) < 2:
            return 0.0
        result = linregress(m, p)
        return float(result.slope)

    def curvature(self) -> float:
        """Second-order polynomial coefficient (quadratic term).

        Positive curvature → curve bows upward (humped structure).
        """
        m = self.maturities
        p = self.prices
        if len(m) < 3:
            return 0.0
        coeffs = np.polyfit(m, p, 2)
        return float(coeffs[0])


# ---------------------------------------------------------------------------
# Roll Analytics
# ---------------------------------------------------------------------------

class RollAnalytics:
    """Compute roll yields, calendar spreads and optimal roll timing."""

    # ------------------------------------------------------------------
    def roll_yield(self, near: FuturesContract, far: FuturesContract) -> float:
        """Annualized roll yield for a long position rolling near→far.

        Positive → backwardation (profitable carry for long).
        Negative → contango (drag for long).

        Formula:
            roll_yield = (F_near / F_far - 1) * (365 / days_between)
        """
        days_between = (far.expiry_months_out - near.expiry_months_out) * 30.4375
        return self.annualized_roll(near.price, far.price, days_between)

    # ------------------------------------------------------------------
    def annualized_roll(
        self, near_price: float, far_price: float, days_between: float
    ) -> float:
        """Annualized roll yield given raw prices and days between contracts.

        Args:
            near_price:    Price of the front/near contract.
            far_price:     Price of the back/far contract.
            days_between:  Calendar days between the two expiries.

        Returns:
            Annualized roll yield as a decimal (e.g. -0.04 = -4 % p.a.)
        """
        if days_between <= 0 or far_price <= 0:
            return 0.0
        ratio = near_price / far_price - 1.0
        return ratio * (365.0 / days_between)

    # ------------------------------------------------------------------
    def roll_cost_total_return(
        self,
        spot_return: float,
        roll_yield: float,
        collateral_rate: float,
        T: float,
    ) -> float:
        """Total annualized return from a fully-collateralized futures position.

        total_return = spot_return + roll_yield + collateral_rate * T

        Args:
            spot_return:     (S_T/S_0 - 1) over the period.
            roll_yield:      Annualized roll yield (decimal).
            collateral_rate: T-bill yield on posted collateral (decimal).
            T:               Holding period in years.

        Returns:
            Total return as a decimal.
        """
        return spot_return + roll_yield + collateral_rate * T

    # ------------------------------------------------------------------
    def optimal_roll_timing(self, term_structure: TermStructure) -> int:
        """Index of the contract with the best (most positive) roll yield.

        Iterates adjacent pairs and returns the index of the near leg of the
        pair with the highest annualized roll yield.  In backwardation the
        front contract always wins; in contango you want to roll as late as
        possible.
        """
        contracts = term_structure.contracts
        if len(contracts) < 2:
            return 0
        best_idx = 0
        best_ry = -float("inf")
        for i in range(len(contracts) - 1):
            ry = self.roll_yield(contracts[i], contracts[i + 1])
            if ry > best_ry:
                best_ry = ry
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    def calendar_spread(self, near_price: float, far_price: float) -> dict:
        """Return calendar spread metrics.

        Returns:
            dict with keys:
              'absolute'       — F_near - F_far
              'pct'            — (F_near/F_far - 1) * 100
              'regime'         — 'backwardation' or 'contango'
        """
        absolute = near_price - far_price
        pct = (near_price / far_price - 1.0) * 100.0 if far_price != 0 else 0.0
        regime = "backwardation" if near_price > far_price else "contango"
        return {"absolute": absolute, "pct": pct, "regime": regime}


# ---------------------------------------------------------------------------
# Cost-of-Carry Model
# ---------------------------------------------------------------------------

class CostOfCarryModel:
    """Cost-of-carry (full carrying-charge) pricing model.

    F(T) = S * exp((r - q + u) * T)

    where:
        r  = risk-free rate (decimal per year)
        q  = convenience yield (decimal per year)
        u  = storage cost (decimal per year, percentage of spot)
        T  = time to expiry in years
    """

    def theoretical_price(
        self,
        spot: float,
        r: float,
        convenience_yield: float,
        storage_cost: float,
        T: float,
    ) -> float:
        """Compute the theoretical futures price.

        Args:
            spot:              Current spot price.
            r:                 Risk-free rate (e.g. 0.05 for 5 %).
            convenience_yield: Annualized convenience yield (e.g. 0.02).
            storage_cost:      Annualized storage cost as fraction of spot.
            T:                 Time to expiry in years.

        Returns:
            Theoretical futures price.
        """
        return spot * math.exp((r - convenience_yield + storage_cost) * T)

    def implied_convenience_yield(
        self,
        spot: float,
        futures: float,
        r: float,
        storage_cost: float,
        T: float,
    ) -> float:
        """Implied convenience yield from observed futures price.

        q = r + u - (1/T) * ln(F/S)

        Args:
            spot:         Current spot price.
            futures:      Observed futures price.
            r:            Risk-free rate.
            storage_cost: Storage cost (fraction of spot per year).
            T:            Time to expiry in years.

        Returns:
            Implied convenience yield (decimal per year).
        """
        if T <= 0 or spot <= 0 or futures <= 0:
            return 0.0
        return r + storage_cost - (1.0 / T) * math.log(futures / spot)

    def carry_return(
        self, spot: float, futures: float, T: float, r: float
    ) -> float:
        """Annualized carry return above risk-free rate.

        carry = (F/S - 1) / T - r

        Args:
            spot:    Current spot price.
            futures: Futures price.
            T:       Time to expiry in years.
            r:       Risk-free rate.

        Returns:
            Annualized carry return above risk-free (decimal).
        """
        if T <= 0 or spot <= 0:
            return 0.0
        return (futures / spot - 1.0) / T - r


# ---------------------------------------------------------------------------
# Commodity Signals
# ---------------------------------------------------------------------------

class CommoditySignals:
    """Composite trading signals: momentum, carry and term-structure slope."""

    def momentum(self, prices: np.ndarray, window: int = 12) -> float:
        """12-month (or ``window``-period) price momentum.

        Returns:
            (P_t / P_{t-window} - 1) as a decimal.  Returns 0.0 if there are
            fewer than ``window+1`` prices.
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) <= window:
            return 0.0
        return float(prices[-1] / prices[-(window + 1)] - 1.0)

    def term_structure_signal(self, term_structure: TermStructure) -> float:
        """Normalized term-structure slope signal.

        Returns the OLS slope divided by the front-month price so the signal
        is scale-independent.  Negative → contango (sell signal), positive →
        backwardation (buy signal).
        """
        slp = term_structure.slope()
        front = term_structure.prices[0] if len(term_structure.prices) > 0 else 1.0
        return float(slp / front) if front != 0 else 0.0

    def carry_signal(
        self, near: float, far: float, days: float
    ) -> float:
        """Annualized carry signal from near/far prices.

        Positive → backwardation → positive carry for long.
        Negative → contango → negative carry for long.
        """
        if days <= 0 or far <= 0:
            return 0.0
        return float((near / far - 1.0) * (365.0 / days))

    def combined_signal(
        self, momentum: float, carry: float, ts_signal: float
    ) -> float:
        """Equal-weighted composite of momentum, carry and term-structure signals."""
        return float((momentum + carry + ts_signal) / 3.0)


# ---------------------------------------------------------------------------
# Standalone convenience functions
# ---------------------------------------------------------------------------

def roll_yield(near_price: float, far_price: float, days_between: float) -> float:
    """Annualized roll yield.

    Positive → backwardation, negative → contango.
    """
    if days_between <= 0 or far_price <= 0:
        return 0.0
    return (near_price / far_price - 1.0) * (365.0 / days_between)


def term_structure_slope(maturities: np.ndarray, prices: np.ndarray) -> float:
    """OLS slope of prices on maturities."""
    maturities = np.asarray(maturities, dtype=float)
    prices = np.asarray(prices, dtype=float)
    if len(maturities) < 2:
        return 0.0
    result = linregress(maturities, prices)
    return float(result.slope)


def implied_convenience_yield(
    spot: float, futures: float, r: float, storage_cost: float, T: float
) -> float:
    """Implied convenience yield from cost-of-carry model.

    q = r + u - (1/T) * ln(F/S)
    """
    if T <= 0 or spot <= 0 or futures <= 0:
        return 0.0
    return r + storage_cost - (1.0 / T) * math.log(futures / spot)


def total_return_decomposition(
    spot_return: float,
    roll_yield_val: float,
    collateral_rate: float,
    T: float,
) -> dict:
    """Decompose total return into spot, roll and collateral components.

    Args:
        spot_return:      (S_T / S_0 - 1) over the period.
        roll_yield_val:   Annualized roll yield (decimal).
        collateral_rate:  T-bill rate on posted collateral (decimal).
        T:                Holding period in years.

    Returns:
        dict with keys: 'spot', 'roll', 'collateral', 'total'.
    """
    collateral = collateral_rate * T
    total = spot_return + roll_yield_val + collateral
    return {
        "spot": spot_return,
        "roll": roll_yield_val,
        "collateral": collateral,
        "total": total,
    }


def is_contango(term_structure: TermStructure) -> bool:
    """Return True if the term structure is in contango."""
    return term_structure.is_contango


def calendar_spread(near: float, far: float) -> float:
    """Absolute calendar spread (near - far).

    Positive → backwardation, negative → contango.
    """
    return near - far


# ---------------------------------------------------------------------------
# Legacy aliases — required by the existing dim_142.sh capability test
# ---------------------------------------------------------------------------

class CommodityRollAnalyzer:
    """Orchestrator combining RollAnalytics and CostOfCarryModel.

    Provided as a legacy alias for backward compatibility with the
    dim_142 capability test.
    """

    def __init__(self) -> None:
        self._roll = RollAnalytics()
        self._carry = CostOfCarryModel()

    # Delegate core methods
    def roll_yield(self, near: FuturesContract, far: FuturesContract) -> float:
        return self._roll.roll_yield(near, far)

    def annualized_roll(
        self, near_price: float, far_price: float, days_between: float
    ) -> float:
        return self._roll.annualized_roll(near_price, far_price, days_between)

    def calendar_spread(self, near_price: float, far_price: float) -> dict:
        return self._roll.calendar_spread(near_price, far_price)

    def theoretical_price(
        self, spot: float, r: float, q: float, u: float, T: float
    ) -> float:
        return self._carry.theoretical_price(spot, r, q, u, T)

    def implied_convenience_yield(
        self, spot: float, futures: float, r: float, storage_cost: float, T: float
    ) -> float:
        return self._carry.implied_convenience_yield(spot, futures, r, storage_cost, T)


class RollYieldCalculator:
    """Standalone roll yield calculator.

    Provided as a legacy alias for backward compatibility.
    """

    def annualized(
        self, near_price: float, far_price: float, days_between: float
    ) -> float:
        return roll_yield(near_price, far_price, days_between)

    def __call__(
        self, near_price: float, far_price: float, days_between: float = 30.0
    ) -> float:
        return self.annualized(near_price, far_price, days_between)


def detect_contango(term_structure: TermStructure) -> bool:
    """Return True when the term structure is in contango."""
    return term_structure.is_contango


def detect_backwardation(term_structure: TermStructure) -> bool:
    """Return True when the term structure is in backwardation."""
    return term_structure.is_backwardation
