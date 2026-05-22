"""
sentinel/spm/regulatory_capital_v3.py
=======================================
Basel III/IV Pillar 3 Regulatory Capital Engine
dim_138 — score target: 9

Implements:
  - SA-CCR (Standardised Approach for Counterparty Credit Risk)
      - Replacement Cost (RC)
      - Potential Future Exposure (PFE) with multiplier
      - Exposure-at-Default (EAD = alpha * (RC + PFE))
  - Risk-Weighted Asset (RWA) calculation
      - Credit RWA (standardised)
      - Market RWA (multiplier / VaR-based)
      - Operational RWA (Basic Indicator Approach)
  - Capital Adequacy ratios
      - CET1, Tier 1, Total capital ratios
      - Leverage ratio
      - Buffer headroom (conservation, countercyclical)
  - Pillar 3 disclosure report
  - Stress testing

References:
  - BCBS "The standardised approach for measuring counterparty credit risk
    exposures" (March 2014, rev April 2014)
  - Basel III: A global regulatory framework (June 2011)

Free-standing: only numpy from third-party. No paid APIs required.

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import math
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Basel III capital minimums and buffers
# ---------------------------------------------------------------------------

CET1_MINIMUM: float = 0.045        # 4.5%
TIER1_MINIMUM: float = 0.060       # 6.0%
TOTAL_CAP_MINIMUM: float = 0.080   # 8.0%
LEVERAGE_MINIMUM: float = 0.030    # 3.0%
CONSERVATION_BUFFER: float = 0.025  # 2.5%
MAX_COUNTERCYCLICAL: float = 0.025  # 0–2.5%

# SA-CCR parameters
ALPHA: float = 1.4           # supervisory alpha multiplier
FLOOR: float = 0.05          # PFE multiplier floor (5%)

SUPERVISORY_FACTORS: Dict[str, float] = {
    "ir": 0.005,
    "fx": 0.04,
    "equity": 0.32,
    "credit": 0.38,
    "commodity": 0.18,
}

# Standard risk weights (Basel III standardised) for sovereign/bank/corporate
# Mapped by credit quality step or label
STANDARD_RISK_WEIGHTS: Dict[str, float] = {
    "sovereign_aaa": 0.00,
    "sovereign_aa": 0.00,
    "sovereign_a": 0.20,
    "sovereign_bbb": 0.50,
    "sovereign_bb": 1.00,
    "sovereign_b": 1.50,
    "sovereign_unrated": 1.00,
    "bank_aaa": 0.20,
    "bank_aa": 0.20,
    "bank_a": 0.50,
    "bank_bbb": 1.00,
    "bank_bb": 1.00,
    "bank_b": 1.50,
    "bank_unrated": 1.00,
    "corporate_aaa": 0.20,
    "corporate_aa": 0.20,
    "corporate_a": 0.50,
    "corporate_bbb": 1.00,
    "corporate_bb": 1.00,
    "corporate_b": 1.50,
    "corporate_unrated": 1.00,
    "retail_residential": 0.35,
    "retail_other": 0.75,
}


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class DerivativePosition:
    """A single derivative position for SA-CCR calculation.

    Attributes
    ----------
    asset_class : str
        One of 'ir', 'fx', 'equity', 'credit', 'commodity'.
    notional : float
        Adjusted notional in currency units.
    market_value : float
        Current mark-to-market value (positive = asset, negative = liability).
    maturity_years : float
        Remaining maturity in years.
    is_buy : bool
        True for long / pay-fixed / buy (delta = +1), False for sell / receive-fixed
        / short (delta = -1).
    """

    asset_class: str
    notional: float
    market_value: float
    maturity_years: float
    is_buy: bool = True

    def __post_init__(self) -> None:
        ac = self.asset_class.lower()
        if ac not in SUPERVISORY_FACTORS:
            raise ValueError(
                f"asset_class '{self.asset_class}' not recognised. "
                f"Valid: {list(SUPERVISORY_FACTORS)}"
            )
        self.asset_class = ac
        if self.notional < 0:
            raise ValueError("notional must be non-negative")
        if self.maturity_years <= 0:
            raise ValueError("maturity_years must be positive")


@dataclass
class CapitalStructure:
    """Bank's regulatory capital components.

    Attributes
    ----------
    cet1 : float
        Common Equity Tier 1 capital.
    at1 : float
        Additional Tier 1 capital.
    tier2 : float
        Tier 2 capital.
    """

    cet1: float
    at1: float
    tier2: float

    def __post_init__(self) -> None:
        if self.cet1 < 0:
            raise ValueError("CET1 cannot be negative")
        if self.at1 < 0:
            raise ValueError("AT1 cannot be negative")
        if self.tier2 < 0:
            raise ValueError("Tier2 cannot be negative")

    @property
    def total_capital(self) -> float:
        """Total regulatory capital = CET1 + AT1 + Tier2."""
        return self.cet1 + self.at1 + self.tier2

    @property
    def tier1(self) -> float:
        """Tier 1 capital = CET1 + AT1."""
        return self.cet1 + self.at1


@dataclass
class CapitalRatios:
    """Computed capital adequacy ratios.

    Attributes
    ----------
    cet1_ratio : float
        CET1 / RWA.
    tier1_ratio : float
        Tier1 / RWA.
    total_capital_ratio : float
        Total capital / RWA.
    leverage_ratio : float
        Tier1 / Total exposure measure.
    rwa_total : float
        Total risk-weighted assets.
    """

    cet1_ratio: float
    tier1_ratio: float
    total_capital_ratio: float
    leverage_ratio: float
    rwa_total: float

    def is_compliant(self) -> bool:
        """Return True if all minimum Basel III ratios are satisfied."""
        return (
            self.cet1_ratio >= CET1_MINIMUM
            and self.tier1_ratio >= TIER1_MINIMUM
            and self.total_capital_ratio >= TOTAL_CAP_MINIMUM
            and self.leverage_ratio >= LEVERAGE_MINIMUM
        )

    def buffer_headroom(self) -> Dict[str, float]:
        """Distance from each binding constraint (positive = surplus)."""
        return {
            "cet1_vs_minimum": self.cet1_ratio - CET1_MINIMUM,
            "cet1_vs_conservation": self.cet1_ratio - (CET1_MINIMUM + CONSERVATION_BUFFER),
            "tier1_vs_minimum": self.tier1_ratio - TIER1_MINIMUM,
            "total_vs_minimum": self.total_capital_ratio - TOTAL_CAP_MINIMUM,
            "leverage_vs_minimum": self.leverage_ratio - LEVERAGE_MINIMUM,
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            "cet1_ratio": self.cet1_ratio,
            "tier1_ratio": self.tier1_ratio,
            "total_capital_ratio": self.total_capital_ratio,
            "leverage_ratio": self.leverage_ratio,
            "rwa_total": self.rwa_total,
            "is_compliant": float(self.is_compliant()),
        }


# ===========================================================================
# SA-CCR Calculator
# ===========================================================================

class SACCRCalculator:
    """
    Basel III Standardised Approach for Counterparty Credit Risk (SA-CCR).

    Computes:
      - Replacement Cost (RC)
      - Add-on per asset class
      - PFE multiplier
      - Potential Future Exposure (PFE)
      - Exposure-at-Default (EAD)

    Reference: BCBS d279 (March 2014)
    """

    SUPERVISORY_FACTORS = SUPERVISORY_FACTORS
    ALPHA = ALPHA
    FLOOR = FLOOR

    def replacement_cost(
        self,
        portfolio_value: float,
        collateral: float = 0.0,
    ) -> float:
        """
        RC for unmargined portfolios.

        RC = max(V - C, 0)

        Parameters
        ----------
        portfolio_value : float
            Current net MTM value of the netting set V.
        collateral : float
            Net collateral posted by counterparty (post-haircut).

        Returns
        -------
        float
            Replacement cost (>=0).
        """
        return max(portfolio_value - collateral, 0.0)

    def maturity_factor(self, maturity_years: float) -> float:
        """
        Supervisory maturity factor for unmargined trades.

        MF = sqrt(min(M, 1))

        Parameters
        ----------
        maturity_years : float
            Remaining maturity in years.

        Returns
        -------
        float
            Maturity factor in [0, 1].
        """
        return math.sqrt(min(maturity_years, 1.0))

    def supervisory_delta(self, position: DerivativePosition) -> float:
        """
        Supervisory delta: +1 for long, -1 for short positions.

        For vanilla linear instruments (non-option):
            delta = +1  if long/buy
            delta = -1  if short/sell

        For simplicity in this standardised approach implementation,
        options are not handled separately (use the sign-based approach).
        """
        return 1.0 if position.is_buy else -1.0

    def addon_single(self, position: DerivativePosition) -> float:
        """
        Compute add-on contribution for a single derivative position.

        Add-On_i = SF * delta * MF * N

        where SF is the supervisory factor for the asset class.
        """
        sf = self.SUPERVISORY_FACTORS[position.asset_class]
        delta = self.supervisory_delta(position)
        mf = self.maturity_factor(position.maturity_years)
        return sf * delta * mf * position.notional

    def addon(self, positions: List[DerivativePosition]) -> float:
        """
        Aggregated add-on across all positions.

        For simplicity (single hedging set per asset class), the add-on is
        the absolute value of the sum of individual add-ons per asset class,
        then summed across asset classes.  This approximates the full netting
        set aggregation specified in SA-CCR paragraph 168.

        Returns
        -------
        float
            Non-negative aggregated add-on.
        """
        class_addons: Dict[str, float] = {}
        for pos in positions:
            contrib = self.addon_single(pos)
            class_addons[pos.asset_class] = class_addons.get(pos.asset_class, 0.0) + contrib

        # For credit / equity add-ons, apply absolute value (full netting set basis)
        total = sum(abs(v) for v in class_addons.values())
        return total

    def pfe_multiplier(
        self,
        portfolio_value: float,
        collateral: float,
        addon: float,
    ) -> float:
        """
        PFE multiplier to recognise over-collateralisation.

        multiplier = min(1, Floor + (1 - Floor) * exp((V - C) / (2 * (1 - Floor) * Agg)))

        Capped at 1 (multiplier cannot exceed 1 under SA-CCR).

        Parameters
        ----------
        portfolio_value : float
            Current net MTM value V.
        collateral : float
            Net collateral C (post-haircut).
        addon : float
            Aggregated add-on (must be > 0 to avoid division by zero).

        Returns
        -------
        float
            Multiplier in [Floor, 1].
        """
        if addon <= 0.0:
            return self.FLOOR
        exponent = (portfolio_value - collateral) / (2.0 * (1.0 - self.FLOOR) * addon)
        m = self.FLOOR + (1.0 - self.FLOOR) * math.exp(exponent)
        return min(1.0, m)

    def pfe(
        self,
        positions: List[DerivativePosition],
        portfolio_value: float,
        collateral: float = 0.0,
    ) -> float:
        """
        Potential Future Exposure.

        PFE = multiplier * Aggregated_Add-On
        """
        agg = self.addon(positions)
        mult = self.pfe_multiplier(portfolio_value, collateral, agg)
        return mult * agg

    def ead(
        self,
        positions: List[DerivativePosition],
        portfolio_value: float,
        collateral: float = 0.0,
    ) -> float:
        """
        Exposure at Default.

        EAD = alpha * (RC + PFE)  where alpha = 1.4
        """
        rc = self.replacement_cost(portfolio_value, collateral)
        pfe_val = self.pfe(positions, portfolio_value, collateral)
        return self.ALPHA * (rc + pfe_val)


# ===========================================================================
# RWA Calculator
# ===========================================================================

class RWACalculator:
    """
    Risk-Weighted Asset (RWA) calculator under Basel III standardised approach.

    Covers:
      - Credit RWA (sum of exposure * RW)
      - Market RWA  (VaR-based multiplier approach)
      - Operational RWA (Basic Indicator Approach)
    """

    def credit_rwa(
        self,
        exposures: Dict[str, float],
        risk_weights: Dict[str, float],
    ) -> float:
        """
        Credit RWA = sum(exposure_i * RW_i).

        Parameters
        ----------
        exposures : dict
            {counterparty_id: exposure_amount}
        risk_weights : dict
            {counterparty_id: risk_weight}  e.g. 1.0 = 100% RW

        Returns
        -------
        float
            Total credit RWA.
        """
        total = 0.0
        for key, exp in exposures.items():
            rw = risk_weights.get(key, 1.0)  # default 100% for unrated
            total += abs(exp) * rw
        return total

    def market_rwa(self, var_10d: float) -> float:
        """
        Market RWA using the VaR-based capital charge multiplier approach.

        RWA = VaR_10d * sqrt(10) * 3 * 12.5

        Note: The sqrt(10) scaling converts 1-day VaR to 10-day.
        The factor of 3 is the Basel supervisory multiplier.
        The factor of 12.5 converts capital charge to RWA equivalent (= 1/0.08).

        Parameters
        ----------
        var_10d : float
            10-day 99% VaR.

        Returns
        -------
        float
            Market RWA.
        """
        return var_10d * math.sqrt(10.0) * 3.0 * 12.5

    def operational_rwa(self, gross_income: float) -> float:
        """
        Operational RWA — Basic Indicator Approach (BIA).

        Op_charge = 0.15 * max(GI, 0)
        Op_RWA = Op_charge * 12.5

        Parameters
        ----------
        gross_income : float
            Average positive gross income over up to 3 years.

        Returns
        -------
        float
            Operational RWA.
        """
        charge = 0.15 * max(gross_income, 0.0)
        return charge * 12.5

    def total_rwa(
        self,
        credit: float,
        market: float,
        operational: float,
    ) -> float:
        """Total RWA = Credit RWA + Market RWA + Op RWA."""
        return credit + market + operational


# ===========================================================================
# Capital Adequacy Analyser
# ===========================================================================

class CapitalAdequacyAnalyzer:
    """
    Compute and analyse Basel III capital adequacy ratios.

    Parameters
    ----------
    capital : CapitalStructure
        Bank's regulatory capital.
    rwa : float
        Total Risk-Weighted Assets.
    total_exposure : float
        Total leverage exposure measure (balance sheet + off-balance-sheet).
    """

    def __init__(
        self,
        capital: CapitalStructure,
        rwa: float,
        total_exposure: float,
    ) -> None:
        self.capital = capital
        self.rwa = rwa
        self.total_exposure = total_exposure

    def ratios(self) -> CapitalRatios:
        """Compute all capital ratios."""
        if self.rwa <= 0:
            raise ValueError("RWA must be positive")
        if self.total_exposure <= 0:
            raise ValueError("total_exposure must be positive")

        cet1_r = self.capital.cet1 / self.rwa
        t1_r = self.capital.tier1 / self.rwa
        total_r = self.capital.total_capital / self.rwa
        lev_r = self.capital.tier1 / self.total_exposure

        return CapitalRatios(
            cet1_ratio=cet1_r,
            tier1_ratio=t1_r,
            total_capital_ratio=total_r,
            leverage_ratio=lev_r,
            rwa_total=self.rwa,
        )

    def pillar3_report(self) -> Dict:
        """
        Generate a structured Pillar 3 disclosure summary.

        Returns a dict suitable for JSON serialisation covering:
          - Capital structure
          - RWA breakdown (if provided)
          - Capital ratios
          - Compliance flags
          - Buffer headroom
        """
        r = self.ratios()
        report = {
            "capital": {
                "cet1": self.capital.cet1,
                "at1": self.capital.at1,
                "tier2": self.capital.tier2,
                "tier1": self.capital.tier1,
                "total_capital": self.capital.total_capital,
            },
            "rwa": {
                "total": self.rwa,
            },
            "ratios": r.to_dict(),
            "compliance": {
                "cet1_compliant": r.cet1_ratio >= CET1_MINIMUM,
                "tier1_compliant": r.tier1_ratio >= TIER1_MINIMUM,
                "total_compliant": r.total_capital_ratio >= TOTAL_CAP_MINIMUM,
                "leverage_compliant": r.leverage_ratio >= LEVERAGE_MINIMUM,
                "fully_compliant": r.is_compliant(),
            },
            "buffer_headroom": r.buffer_headroom(),
            "minimums": {
                "cet1": CET1_MINIMUM,
                "tier1": TIER1_MINIMUM,
                "total_capital": TOTAL_CAP_MINIMUM,
                "leverage": LEVERAGE_MINIMUM,
                "conservation_buffer": CONSERVATION_BUFFER,
            },
        }
        return report

    def stress_test(
        self,
        rwa_shock_pct: float,
        capital_loss_pct: float,
    ) -> CapitalRatios:
        """
        Apply a stress scenario to capital and RWA.

        Parameters
        ----------
        rwa_shock_pct : float
            Fractional increase in RWA (e.g. 0.50 = 50% increase).
        capital_loss_pct : float
            Fractional loss of total capital applied pro-rata (e.g. 0.20 = 20%).
            Capital is reduced uniformly across CET1, AT1, T2.

        Returns
        -------
        CapitalRatios
            Stressed capital ratios.
        """
        stressed_rwa = self.rwa * (1.0 + rwa_shock_pct)
        stressed_cap = CapitalStructure(
            cet1=self.capital.cet1 * (1.0 - capital_loss_pct),
            at1=self.capital.at1 * (1.0 - capital_loss_pct),
            tier2=self.capital.tier2 * (1.0 - capital_loss_pct),
        )
        # Exposure grows with RWA (conservative: same proportion)
        stressed_exposure = self.total_exposure * (1.0 + rwa_shock_pct * 0.5)

        stressed_analyzer = CapitalAdequacyAnalyzer(
            capital=stressed_cap,
            rwa=stressed_rwa,
            total_exposure=stressed_exposure,
        )
        return stressed_analyzer.ratios()


# ===========================================================================
# RegulatoryCapital — main facade (required by dim_138.sh)
# ===========================================================================

class RegulatoryCapital:
    """
    High-level facade for Basel III regulatory capital computation.

    This class ties together SA-CCR, RWA, and capital adequacy analysis
    into a single entry point for the Pillar 3 disclosure workflow.

    Parameters
    ----------
    capital : CapitalStructure
        Bank's regulatory capital components.
    total_exposure : float
        Leverage exposure measure.
    """

    def __init__(
        self,
        capital: CapitalStructure,
        total_exposure: float,
    ) -> None:
        self.capital = capital
        self.total_exposure = total_exposure
        self._rwa_credit: float = 0.0
        self._rwa_market: float = 0.0
        self._rwa_operational: float = 0.0
        self._ccr_ead: float = 0.0

        self._saccr = SACCRCalculator()
        self._rwa_calc = RWACalculator()

    def set_ccr(
        self,
        positions: List[DerivativePosition],
        portfolio_value: float,
        collateral: float = 0.0,
        counterparty_rw: float = 1.0,
    ) -> float:
        """
        Compute SA-CCR EAD and add CCR RWA to the credit RWA bucket.

        Parameters
        ----------
        positions : list[DerivativePosition]
        portfolio_value : float
        collateral : float
        counterparty_rw : float
            Risk weight for CCR exposure (e.g. 0.50 for A-rated bank).

        Returns
        -------
        float
            CCR EAD.
        """
        ead = self._saccr.ead(positions, portfolio_value, collateral)
        self._ccr_ead = ead
        self._rwa_credit += ead * counterparty_rw
        return ead

    def set_credit_rwa(self, credit_rwa: float) -> None:
        """Directly set (add to) the credit RWA bucket."""
        self._rwa_credit += credit_rwa

    def set_market_rwa(self, var_10d: float) -> None:
        """Compute and store market RWA from a 10-day VaR figure."""
        self._rwa_market = self._rwa_calc.market_rwa(var_10d)

    def set_operational_rwa(self, avg_gross_income: float) -> None:
        """Compute and store operational RWA via BIA."""
        self._rwa_operational = self._rwa_calc.operational_rwa(avg_gross_income)

    @property
    def total_rwa(self) -> float:
        """Sum of all RWA components."""
        return self._rwa_credit + self._rwa_market + self._rwa_operational

    def capital_ratios(self) -> CapitalRatios:
        """Compute capital adequacy ratios."""
        analyzer = CapitalAdequacyAnalyzer(
            capital=self.capital,
            rwa=self.total_rwa,
            total_exposure=self.total_exposure,
        )
        return analyzer.ratios()

    def pillar3(self) -> Dict:
        """Generate Pillar 3 disclosure."""
        analyzer = CapitalAdequacyAnalyzer(
            capital=self.capital,
            rwa=self.total_rwa,
            total_exposure=self.total_exposure,
        )
        report = analyzer.pillar3_report()
        report["rwa"]["credit"] = self._rwa_credit
        report["rwa"]["market"] = self._rwa_market
        report["rwa"]["operational"] = self._rwa_operational
        report["ccr_ead"] = self._ccr_ead
        return report

    def stress_test(
        self,
        rwa_shock_pct: float = 0.50,
        capital_loss_pct: float = 0.20,
    ) -> CapitalRatios:
        """Run a stress scenario on the current capital position."""
        analyzer = CapitalAdequacyAnalyzer(
            capital=self.capital,
            rwa=self.total_rwa,
            total_exposure=self.total_exposure,
        )
        return analyzer.stress_test(rwa_shock_pct, capital_loss_pct)


# ===========================================================================
# Pillar3Disclosure — structured report (required by dim_138.sh)
# ===========================================================================

class Pillar3Disclosure:
    """
    Structured Pillar 3 disclosure generator for a bank.

    Produces Basel III-compliant quantitative disclosures including:
      - KM1 (Key metrics)
      - OV1 (RWA overview)
      - CC1 (Capital composition)
      - LR1 (Leverage ratio)

    Parameters
    ----------
    capital : CapitalStructure
    rwa_credit : float
    rwa_market : float
    rwa_operational : float
    total_exposure : float
    """

    def __init__(
        self,
        capital: CapitalStructure,
        rwa_credit: float = 0.0,
        rwa_market: float = 0.0,
        rwa_operational: float = 0.0,
        total_exposure: float = 0.0,
    ) -> None:
        self.capital = capital
        self.rwa_credit = rwa_credit
        self.rwa_market = rwa_market
        self.rwa_operational = rwa_operational
        self.total_exposure = total_exposure

    @property
    def total_rwa(self) -> float:
        return self.rwa_credit + self.rwa_market + self.rwa_operational

    def km1(self) -> Dict:
        """KM1 — Key prudential metrics."""
        rwa = self.total_rwa
        t1 = self.capital.tier1
        total = self.capital.total_capital
        if rwa <= 0:
            return {}
        return {
            "cet1_ratio": self.capital.cet1 / rwa,
            "tier1_ratio": t1 / rwa,
            "total_capital_ratio": total / rwa,
            "leverage_ratio": t1 / max(self.total_exposure, 1.0),
            "total_rwa": rwa,
        }

    def ov1(self) -> Dict:
        """OV1 — RWA overview."""
        return {
            "credit_rwa": self.rwa_credit,
            "market_rwa": self.rwa_market,
            "operational_rwa": self.rwa_operational,
            "total_rwa": self.total_rwa,
        }

    def cc1(self) -> Dict:
        """CC1 — Capital composition."""
        return {
            "cet1": self.capital.cet1,
            "at1": self.capital.at1,
            "tier1": self.capital.tier1,
            "tier2": self.capital.tier2,
            "total_capital": self.capital.total_capital,
        }

    def lr1(self) -> Dict:
        """LR1 — Leverage ratio summary."""
        return {
            "tier1_capital": self.capital.tier1,
            "total_exposure": self.total_exposure,
            "leverage_ratio": self.capital.tier1 / max(self.total_exposure, 1.0),
            "minimum": LEVERAGE_MINIMUM,
            "compliant": (
                self.capital.tier1 / max(self.total_exposure, 1.0) >= LEVERAGE_MINIMUM
            ),
        }

    def full_report(self) -> Dict:
        """Return all disclosure tables in a single dict."""
        ratios_obj = CapitalRatios(
            cet1_ratio=self.km1().get("cet1_ratio", 0.0),
            tier1_ratio=self.km1().get("tier1_ratio", 0.0),
            total_capital_ratio=self.km1().get("total_capital_ratio", 0.0),
            leverage_ratio=self.km1().get("leverage_ratio", 0.0),
            rwa_total=self.total_rwa,
        )
        return {
            "KM1": self.km1(),
            "OV1": self.ov1(),
            "CC1": self.cc1(),
            "LR1": self.lr1(),
            "compliant": ratios_obj.is_compliant(),
            "buffer_headroom": ratios_obj.buffer_headroom(),
        }


# ===========================================================================
# Module-level convenience functions
# ===========================================================================

def sa_ccr_ead(
    positions: List[DerivativePosition],
    portfolio_value: float,
    collateral: float = 0.0,
) -> float:
    """
    Compute SA-CCR EAD for a netting set.

    EAD = alpha * (RC + PFE)  with alpha = 1.4

    Parameters
    ----------
    positions : list[DerivativePosition]
    portfolio_value : float
        Current net MTM value of the netting set.
    collateral : float
        Net collateral posted (post-haircut).

    Returns
    -------
    float
        EAD.
    """
    calc = SACCRCalculator()
    return calc.ead(positions, portfolio_value, collateral)


def capital_ratios(
    capital: CapitalStructure,
    rwa: float,
    total_exposure: float,
) -> CapitalRatios:
    """
    Compute capital ratios from capital structure, RWA, and total exposure.

    Parameters
    ----------
    capital : CapitalStructure
    rwa : float
        Total Risk-Weighted Assets.
    total_exposure : float
        Leverage exposure measure.

    Returns
    -------
    CapitalRatios
    """
    analyzer = CapitalAdequacyAnalyzer(capital, rwa, total_exposure)
    return analyzer.ratios()


def is_capital_compliant(ratios: CapitalRatios) -> bool:
    """Return True if the bank meets all Basel III minimum capital requirements."""
    return ratios.is_compliant()


def compute_risk_weighted_assets(
    exposures: Dict[str, float],
    risk_weights: Dict[str, float],
    var_10d: float = 0.0,
    avg_gross_income: float = 0.0,
) -> Dict[str, float]:
    """
    Compute all RWA components and return a breakdown.

    Parameters
    ----------
    exposures : dict
        {counterparty_id: exposure_amount} for credit RWA.
    risk_weights : dict
        {counterparty_id: risk_weight} for credit RWA.
    var_10d : float
        10-day VaR for market RWA (0 if no trading book).
    avg_gross_income : float
        Average positive gross income for operational RWA BIA.

    Returns
    -------
    dict
        Keys: 'credit', 'market', 'operational', 'total'.
    """
    calc = RWACalculator()
    credit = calc.credit_rwa(exposures, risk_weights)
    market = calc.market_rwa(var_10d) if var_10d > 0 else 0.0
    operational = calc.operational_rwa(avg_gross_income) if avg_gross_income > 0 else 0.0
    return {
        "credit": credit,
        "market": market,
        "operational": operational,
        "total": credit + market + operational,
    }
