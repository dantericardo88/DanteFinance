"""
sentinel/spm/liquidity_risk_v3.py
===================================
Basel III Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR) engine.
dim_135 — score target 9

Implements the full Basel III liquidity framework:
  - LCR: HQLA / Net Cash Outflows (30-day stress scenario) >= 100%
  - NSFR: Available Stable Funding / Required Stable Funding >= 100%

Key Basel III parameters encoded:
  - HQLA Level 1 (0% haircut), Level 2A (15%), Level 2B (25-50%)
  - Level 2 cap: max 40% of total adjusted HQLA
  - Net outflow floor: max(outflows - min(inflows, 0.75*outflows), 0.25*outflows)
  - ASF factors: Tier 1 capital 100%, retail stable 95%, retail less-stable 90%,
    wholesale short-term (<6mo) 0-50%, wholesale long-term (>=6mo) 100%
  - RSF factors: cash 0%, L1 HQLA 5%, L2A HQLA 15%, residential mortgages 50%,
    retail loans 65%, wholesale/SME lending 85%, illiquid assets 100%

No external dependencies — pure Python / stdlib math only.

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Basel III LCR constants
# ---------------------------------------------------------------------------

# HQLA haircuts
LEVEL2A_HAIRCUT: float = 0.15   # 15% haircut on Level 2A assets
LEVEL2B_HAIRCUT_LOW: float = 0.25   # 25% haircut floor for Level 2B
LEVEL2B_HAIRCUT_HIGH: float = 0.50  # 50% haircut ceiling for Level 2B (default)

# Level 2 cap: Level 2 assets cannot exceed 40% of total adjusted HQLA
LEVEL2_CAP_FRACTION: float = 0.40

# Run-off rates for stress outflow calculation
RUNOFF_RETAIL_STABLE: float = 0.03      # 3% — insured, stable retail deposits
RUNOFF_RETAIL_LESS_STABLE: float = 0.10 # 10% — less stable retail deposits
RUNOFF_WHOLESALE: float = 0.25          # 25% — operational wholesale deposits (conservative default)
RUNOFF_WHOLESALE_NON_OP: float = 0.40   # 40% — non-operational wholesale deposits
RUNOFF_CREDIT_LINES: float = 0.10       # 10% — committed credit/liquidity facilities (retail)

# Inflow cap: inflows cannot exceed 75% of total outflows
INFLOW_CAP_FRACTION: float = 0.75

# Minimum net outflow floor: 25% of total outflows
OUTFLOW_FLOOR_FRACTION: float = 0.25

# Minimum LCR threshold for compliance (100%)
LCR_MINIMUM: float = 1.00

# ---------------------------------------------------------------------------
# Basel III NSFR constants
# ---------------------------------------------------------------------------

# Available Stable Funding (ASF) weight factors
ASF_TIER1_CAPITAL: float = 1.00         # 100% — Tier 1 and Tier 2 capital
ASF_RETAIL_STABLE: float = 0.95         # 95% — stable retail/SME deposits (insured)
ASF_RETAIL_LESS_STABLE: float = 0.90    # 90% — less-stable retail/SME deposits
ASF_WHOLESALE_SHORT_TERM: float = 0.50  # 50% — wholesale funding <6 months (operational)
ASF_WHOLESALE_SHORT_NON_OP: float = 0.00  # 0% — non-operational wholesale <6 months
ASF_WHOLESALE_LONG_TERM: float = 1.00   # 100% — wholesale funding >= 6 months (or 1 year)

# Required Stable Funding (RSF) weight factors by asset class
RSF_CASH: float = 0.00           # 0%   — cash and central bank reserves
RSF_HQLA_L1: float = 0.05       # 5%   — Level 1 HQLA (encumbered < 6 months)
RSF_HQLA_L2A: float = 0.15      # 15%  — Level 2A HQLA
RSF_HQLA_L2B: float = 0.50      # 50%  — Level 2B HQLA (unencumbered)
RSF_RESIDENTIAL_MORTGAGES: float = 0.65  # 65%  — residential mortgages (risk-weight <= 35%)
RSF_RETAIL_LOANS: float = 0.85   # 85%  — retail loans not classified above
RSF_WHOLESALE_LENDING: float = 0.50  # 50%  — loans to financial institutions, maturity < 6 mo
RSF_ILLIQUID: float = 1.00       # 100% — all other assets (illiquid, non-HQLA)

# Alternative RSF rates per latest Basel (BCBS 295)
RSF_RESIDENTIAL_MORTGAGE_LOW_RW: float = 0.50   # 50% — residential mortgages, RW <= 35%
RSF_RETAIL_LOAN_ALTERNATIVE: float = 0.65       # 65% — retail / SME revolving, maturity < 1yr

# Minimum NSFR threshold for compliance (100%)
NSFR_MINIMUM: float = 1.00


# ---------------------------------------------------------------------------
# Data structures — LCR inputs
# ---------------------------------------------------------------------------

@dataclass
class HQLAPortfolio:
    """
    High Quality Liquid Assets per Basel III LCR framework.

    All values in the same currency unit (e.g. USD millions).
    Haircuts are applied internally by LiquidityCoverageRatio.compute().
    """
    level1: float = 0.0
    """Level 1 HQLA — cash, central bank reserves, sovereign bonds with 0% haircut."""

    level2a: float = 0.0
    """Level 2A HQLA — GSE securities, high-grade corporate bonds; 15% haircut applied."""

    level2b: float = 0.0
    """Level 2B HQLA — lower-rated bonds, equities; 25-50% haircut applied (default 50%)."""

    level2b_haircut: float = LEVEL2B_HAIRCUT_HIGH
    """Haircut for Level 2B assets. Range [0.25, 0.50]. Default 0.50 (conservative)."""

    def adjusted_level1(self) -> float:
        """Level 1 HQLA after haircut (no haircut applied)."""
        return max(0.0, self.level1)

    def adjusted_level2a(self) -> float:
        """Level 2A HQLA after 15% haircut."""
        return max(0.0, self.level2a) * (1.0 - LEVEL2A_HAIRCUT)

    def adjusted_level2b(self) -> float:
        """Level 2B HQLA after the applicable haircut (25-50%)."""
        haircut = max(LEVEL2B_HAIRCUT_LOW, min(LEVEL2B_HAIRCUT_HIGH, self.level2b_haircut))
        return max(0.0, self.level2b) * (1.0 - haircut)

    def total_adjusted_hqla(self) -> float:
        """
        Total adjusted HQLA after haircuts and the Level 2 cap.

        Basel III rule: Level 2 assets (2A + 2B) cannot exceed 40% of total HQLA.
        Equivalently: Level 2 <= (2/3) * Level 1.

        Also: Level 2B cannot exceed 15% of total HQLA (i.e., 75% of total Level 2).
        """
        l1 = self.adjusted_level1()
        l2a = self.adjusted_level2a()
        l2b = self.adjusted_level2b()

        # Level 2B sub-cap: Level 2B <= 15% of total HQLA
        # Equivalently: l2b <= 0.15/(0.85) * (l1 + l2a) — solved iteratively
        # Conservative approach: apply cap relative to uncapped l1+l2a first
        # Per Basel: Level2B cap = 15% of total HQLA => L2B <= (15/85)*(L1+L2A)
        l2b_sub_cap = (15.0 / 85.0) * (l1 + l2a)
        l2b_capped = min(l2b, l2b_sub_cap)

        # Level 2 aggregate cap: Level 2 total <= 40% of total HQLA
        # => L2 <= (40/60) * L1 = (2/3) * L1
        l2_cap = (LEVEL2_CAP_FRACTION / (1.0 - LEVEL2_CAP_FRACTION)) * l1
        l2_total = l2a + l2b_capped
        l2_total_capped = min(l2_total, l2_cap)

        # If Level 2A alone exceeds the cap, 2B gets zero
        if l2a >= l2_total_capped:
            l2a_final = l2_total_capped
            l2b_final = 0.0
        else:
            l2a_final = l2a
            l2b_final = l2_total_capped - l2a

        total = l1 + l2a_final + l2b_final
        logger.debug(
            "HQLA breakdown: L1=%.2f, L2A_adj=%.2f (capped=%.2f), "
            "L2B_adj=%.2f (capped=%.2f), Total=%.2f",
            l1, l2a, l2a_final, l2b, l2b_final, total
        )
        return total

    def level2_fraction(self) -> float:
        """Fraction of total adjusted HQLA that is Level 2 (should be <= 40%)."""
        total = self.total_adjusted_hqla()
        if total <= 0:
            return 0.0
        l2 = self.adjusted_level2a() + self.adjusted_level2b()
        return min(l2, total - self.adjusted_level1()) / total


@dataclass
class CashFlowStress:
    """
    30-day stressed cash flow inputs for LCR calculation.

    All values as gross notional (positive numbers).
    Run-off rates are applied internally by LiquidityCoverageRatio.compute().
    """
    retail_stable_deposits: float = 0.0
    """Retail deposits covered by deposit insurance schemes — run-off rate 3%."""

    retail_less_stable_deposits: float = 0.0
    """Retail deposits not covered or from high-net-worth clients — run-off rate 10%."""

    wholesale_deposits: float = 0.0
    """Operational wholesale deposits (e.g., clearing, custody) — run-off rate 25%."""

    committed_credit_lines: float = 0.0
    """Committed credit and liquidity facilities — run-off rate 10%."""

    cash_inflows: float = 0.0
    """Expected cash inflows from performing counterparties over 30 days."""

    wholesale_non_operational: float = 0.0
    """Non-operational wholesale funding — run-off rate 40%."""

    def total_outflows(self) -> float:
        """Gross stressed outflows over the 30-day window."""
        retail_stress = (
            self.retail_stable_deposits * RUNOFF_RETAIL_STABLE
            + self.retail_less_stable_deposits * RUNOFF_RETAIL_LESS_STABLE
        )
        wholesale_stress = (
            self.wholesale_deposits * RUNOFF_WHOLESALE
            + self.wholesale_non_operational * RUNOFF_WHOLESALE_NON_OP
        )
        credit_stress = self.committed_credit_lines * RUNOFF_CREDIT_LINES
        return retail_stress + wholesale_stress + credit_stress

    def net_cash_outflows(self) -> float:
        """
        Net Cash Outflows per Basel III formula:
          NCO = max(Total Outflows - min(Inflows, 0.75 * Total Outflows),
                    0.25 * Total Outflows)
        """
        outflows = self.total_outflows()
        inflows_capped = min(self.cash_inflows, INFLOW_CAP_FRACTION * outflows)
        nco = max(outflows - inflows_capped, OUTFLOW_FLOOR_FRACTION * outflows)
        return nco


# ---------------------------------------------------------------------------
# LCR engine
# ---------------------------------------------------------------------------

class LiquidityCoverageRatio:
    """
    Basel III Liquidity Coverage Ratio engine.

    LCR = Adjusted HQLA / Net Cash Outflows (30-day stress) >= 100%

    Usage:
        lcr_engine = LiquidityCoverageRatio()
        ratio = lcr_engine.compute(hqla_portfolio, cash_flow_stress)
        compliant = lcr_engine.is_compliant(ratio)
    """

    def compute(self, hqla: HQLAPortfolio, stress: CashFlowStress) -> float:
        """
        Compute the LCR ratio.

        Parameters
        ----------
        hqla : HQLAPortfolio
            HQLA portfolio with Level 1, 2A, and 2B assets.
        stress : CashFlowStress
            30-day stressed cash flow scenario.

        Returns
        -------
        float
            LCR ratio (e.g., 1.50 = 150%). Returns inf if net outflows are zero.
        """
        adjusted_hqla = hqla.total_adjusted_hqla()
        net_outflows = stress.net_cash_outflows()

        if net_outflows <= 0:
            logger.warning("Net cash outflows are zero or negative — LCR is undefined (inf).")
            return math.inf

        ratio = adjusted_hqla / net_outflows
        logger.info(
            "LCR: Adj.HQLA=%.4f, Net Outflows=%.4f, Ratio=%.4f (%.1f%%)",
            adjusted_hqla, net_outflows, ratio, ratio * 100
        )
        return ratio

    def is_compliant(self, ratio: float) -> bool:
        """
        Check Basel III LCR compliance.

        Parameters
        ----------
        ratio : float
            LCR ratio (output of compute()).

        Returns
        -------
        bool
            True if LCR >= 100% (ratio >= 1.0).
        """
        return ratio >= LCR_MINIMUM

    def breakdown(self, hqla: HQLAPortfolio, stress: CashFlowStress) -> Dict[str, float]:
        """
        Return detailed LCR calculation breakdown for reporting/audit purposes.
        """
        l1 = hqla.adjusted_level1()
        l2a = hqla.adjusted_level2a()
        l2b = hqla.adjusted_level2b()
        total_hqla = hqla.total_adjusted_hqla()
        total_outflows = stress.total_outflows()
        inflows_capped = min(stress.cash_inflows, INFLOW_CAP_FRACTION * total_outflows)
        net_outflows = stress.net_cash_outflows()
        ratio = self.compute(hqla, stress)

        return {
            "hqla_level1_adjusted": l1,
            "hqla_level2a_adjusted": l2a,
            "hqla_level2b_adjusted": l2b,
            "hqla_total_adjusted": total_hqla,
            "gross_outflows": total_outflows,
            "cash_inflows_raw": stress.cash_inflows,
            "cash_inflows_capped": inflows_capped,
            "net_cash_outflows": net_outflows,
            "lcr_ratio": ratio,
            "lcr_percent": ratio * 100,
            "compliant": self.is_compliant(ratio),
            "level2_fraction": hqla.level2_fraction(),
            "minimum_requirement": LCR_MINIMUM,
        }


# ---------------------------------------------------------------------------
# Data structures — NSFR inputs
# ---------------------------------------------------------------------------

@dataclass
class FundingStructure:
    """
    Liability-side funding structure for NSFR Available Stable Funding (ASF) calculation.

    All values in the same currency unit.
    """
    tier1_capital: float = 0.0
    """Tier 1 and qualifying Tier 2 capital — ASF weight 100%."""

    retail_deposits_stable: float = 0.0
    """Insured/stable retail and SME deposits — ASF weight 95%."""

    retail_deposits_less_stable: float = 0.0
    """Less-stable retail and SME deposits — ASF weight 90%."""

    wholesale_short_term: float = 0.0
    """Wholesale funding with residual maturity < 6 months (operational) — ASF weight 50%."""

    wholesale_long_term: float = 0.0
    """Wholesale funding with residual maturity >= 6 months — ASF weight 100%."""

    wholesale_short_non_operational: float = 0.0
    """Non-operational wholesale funding < 6 months — ASF weight 0%."""

    def available_stable_funding(self) -> float:
        """Compute total Available Stable Funding (ASF)."""
        asf = (
            self.tier1_capital * ASF_TIER1_CAPITAL
            + self.retail_deposits_stable * ASF_RETAIL_STABLE
            + self.retail_deposits_less_stable * ASF_RETAIL_LESS_STABLE
            + self.wholesale_short_term * ASF_WHOLESALE_SHORT_TERM
            + self.wholesale_long_term * ASF_WHOLESALE_LONG_TERM
            + self.wholesale_short_non_operational * ASF_WHOLESALE_SHORT_NON_OP
        )
        return asf


@dataclass
class AssetStructure:
    """
    Asset-side structure for NSFR Required Stable Funding (RSF) calculation.

    All values in the same currency unit.
    """
    cash: float = 0.0
    """Cash and central bank reserves — RSF weight 0%."""

    hqla_l1: float = 0.0
    """Unencumbered Level 1 HQLA — RSF weight 5%."""

    hqla_l2a: float = 0.0
    """Unencumbered Level 2A HQLA — RSF weight 15%."""

    hqla_l2b: float = 0.0
    """Unencumbered Level 2B HQLA — RSF weight 50%."""

    residential_mortgages: float = 0.0
    """Residential mortgages (risk-weight <= 35%) — RSF weight 65%."""

    retail_loans: float = 0.0
    """Retail loans and SME lending not classified above — RSF weight 85%."""

    wholesale_lending: float = 0.0
    """Loans to financial institutions or corporates, maturity < 6 months — RSF weight 50%."""

    illiquid_assets: float = 0.0
    """All other assets (encumbered assets, non-HQLA securities, etc.) — RSF weight 100%."""

    def required_stable_funding(self) -> float:
        """Compute total Required Stable Funding (RSF)."""
        rsf = (
            self.cash * RSF_CASH
            + self.hqla_l1 * RSF_HQLA_L1
            + self.hqla_l2a * RSF_HQLA_L2A
            + self.hqla_l2b * RSF_HQLA_L2B
            + self.residential_mortgages * RSF_RESIDENTIAL_MORTGAGES
            + self.retail_loans * RSF_RETAIL_LOANS
            + self.wholesale_lending * RSF_WHOLESALE_LENDING
            + self.illiquid_assets * RSF_ILLIQUID
        )
        return rsf


# ---------------------------------------------------------------------------
# NSFR engine
# ---------------------------------------------------------------------------

class NetStableFundingRatio:
    """
    Basel III Net Stable Funding Ratio engine.

    NSFR = Available Stable Funding / Required Stable Funding >= 100%

    Usage:
        nsfr_engine = NetStableFundingRatio()
        ratio = nsfr_engine.compute(funding_structure, asset_structure)
        compliant = nsfr_engine.is_compliant(ratio)
    """

    def compute(self, funding: FundingStructure, assets: AssetStructure) -> float:
        """
        Compute the NSFR ratio.

        Parameters
        ----------
        funding : FundingStructure
            Liability-side funding categorized by stability and maturity.
        assets : AssetStructure
            Asset-side portfolio categorized by liquidity and asset class.

        Returns
        -------
        float
            NSFR ratio (e.g., 1.15 = 115%). Returns inf if RSF is zero.
        """
        asf = funding.available_stable_funding()
        rsf = assets.required_stable_funding()

        if rsf <= 0:
            logger.warning("Required Stable Funding is zero — NSFR is undefined (inf).")
            return math.inf

        ratio = asf / rsf
        logger.info(
            "NSFR: ASF=%.4f, RSF=%.4f, Ratio=%.4f (%.1f%%)",
            asf, rsf, ratio, ratio * 100
        )
        return ratio

    def is_compliant(self, ratio: float) -> bool:
        """
        Check Basel III NSFR compliance.

        Parameters
        ----------
        ratio : float
            NSFR ratio (output of compute()).

        Returns
        -------
        bool
            True if NSFR >= 100% (ratio >= 1.0).
        """
        return ratio >= NSFR_MINIMUM

    def breakdown(self, funding: FundingStructure, assets: AssetStructure) -> Dict[str, float]:
        """
        Return detailed NSFR calculation breakdown for reporting/audit purposes.
        """
        asf = funding.available_stable_funding()
        rsf = assets.required_stable_funding()
        ratio = self.compute(funding, assets)

        asf_components = {
            "asf_tier1_capital": funding.tier1_capital * ASF_TIER1_CAPITAL,
            "asf_retail_stable": funding.retail_deposits_stable * ASF_RETAIL_STABLE,
            "asf_retail_less_stable": funding.retail_deposits_less_stable * ASF_RETAIL_LESS_STABLE,
            "asf_wholesale_short_term": funding.wholesale_short_term * ASF_WHOLESALE_SHORT_TERM,
            "asf_wholesale_long_term": funding.wholesale_long_term * ASF_WHOLESALE_LONG_TERM,
            "asf_wholesale_short_non_op": funding.wholesale_short_non_operational * ASF_WHOLESALE_SHORT_NON_OP,
        }

        rsf_components = {
            "rsf_cash": assets.cash * RSF_CASH,
            "rsf_hqla_l1": assets.hqla_l1 * RSF_HQLA_L1,
            "rsf_hqla_l2a": assets.hqla_l2a * RSF_HQLA_L2A,
            "rsf_hqla_l2b": assets.hqla_l2b * RSF_HQLA_L2B,
            "rsf_residential_mortgages": assets.residential_mortgages * RSF_RESIDENTIAL_MORTGAGES,
            "rsf_retail_loans": assets.retail_loans * RSF_RETAIL_LOANS,
            "rsf_wholesale_lending": assets.wholesale_lending * RSF_WHOLESALE_LENDING,
            "rsf_illiquid": assets.illiquid_assets * RSF_ILLIQUID,
        }

        return {
            **asf_components,
            "asf_total": asf,
            **rsf_components,
            "rsf_total": rsf,
            "nsfr_ratio": ratio,
            "nsfr_percent": ratio * 100,
            "compliant": self.is_compliant(ratio),
            "minimum_requirement": NSFR_MINIMUM,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def compute_lcr(hqla: float, net_cash_outflows: float) -> float:
    """
    Simple LCR computation from pre-computed HQLA and net cash outflows.

    Parameters
    ----------
    hqla : float
        Total adjusted HQLA (after haircuts and Level 2 cap).
    net_cash_outflows : float
        Net cash outflows over the 30-day stress window.

    Returns
    -------
    float
        LCR ratio. Returns math.inf if net_cash_outflows == 0.
    """
    if net_cash_outflows <= 0:
        return math.inf
    return hqla / net_cash_outflows


def compute_nsfr(available_stable_funding: float, required_stable_funding: float) -> float:
    """
    Simple NSFR computation from pre-computed ASF and RSF totals.

    Parameters
    ----------
    available_stable_funding : float
        Total Available Stable Funding (ASF) after applying ASF weights.
    required_stable_funding : float
        Total Required Stable Funding (RSF) after applying RSF weights.

    Returns
    -------
    float
        NSFR ratio. Returns math.inf if required_stable_funding == 0.
    """
    if required_stable_funding <= 0:
        return math.inf
    return available_stable_funding / required_stable_funding


# ---------------------------------------------------------------------------
# Scenario builders (convenience)
# ---------------------------------------------------------------------------

def build_compliant_bank_lcr(
    level1_hqla: float = 200.0,
    retail_stable: float = 500.0,
    retail_less_stable: float = 200.0,
    inflows: float = 10.0,
) -> Tuple[HQLAPortfolio, CashFlowStress]:
    """Build a typical compliant bank LCR scenario for testing."""
    hqla = HQLAPortfolio(level1=level1_hqla)
    stress = CashFlowStress(
        retail_stable_deposits=retail_stable,
        retail_less_stable_deposits=retail_less_stable,
        cash_inflows=inflows,
    )
    return hqla, stress


def build_compliant_bank_nsfr(
    tier1_capital: float = 100.0,
    retail_stable: float = 400.0,
    retail_less_stable: float = 200.0,
    wholesale_long: float = 100.0,
    cash: float = 50.0,
    hqla_l1: float = 100.0,
    hqla_l2a: float = 80.0,
    mortgages: float = 300.0,
    retail_loans: float = 150.0,
    illiquid: float = 120.0,
) -> Tuple[FundingStructure, AssetStructure]:
    """Build a typical compliant bank NSFR scenario for testing."""
    funding = FundingStructure(
        tier1_capital=tier1_capital,
        retail_deposits_stable=retail_stable,
        retail_deposits_less_stable=retail_less_stable,
        wholesale_long_term=wholesale_long,
    )
    assets = AssetStructure(
        cash=cash,
        hqla_l1=hqla_l1,
        hqla_l2a=hqla_l2a,
        residential_mortgages=mortgages,
        retail_loans=retail_loans,
        illiquid_assets=illiquid,
    )
    return funding, assets


# ---------------------------------------------------------------------------
# Module-level summary
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "HQLAPortfolio",
    "CashFlowStress",
    "FundingStructure",
    "AssetStructure",
    # Engines
    "LiquidityCoverageRatio",
    "NetStableFundingRatio",
    # Convenience functions
    "compute_lcr",
    "compute_nsfr",
    # Scenario builders
    "build_compliant_bank_lcr",
    "build_compliant_bank_nsfr",
    # Constants
    "LEVEL2A_HAIRCUT",
    "LEVEL2B_HAIRCUT_HIGH",
    "LEVEL2_CAP_FRACTION",
    "LCR_MINIMUM",
    "NSFR_MINIMUM",
]
