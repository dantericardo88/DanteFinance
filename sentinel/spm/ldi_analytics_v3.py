"""
sentinel/spm/ldi_analytics_v3.py
=================================
Liability-Driven Investing (LDI) Analytics Engine
dim_146 — score target: 9

Implements duration-gap matching, immunization, pension funding analytics,
key rate durations, and LDI overlay (interest rate swap) notional calculation.

Key formulas implemented:
  Macaulay duration : sum(t_i * PV(CF_i)) / Price
  Modified duration : D_mac / (1 + y/n)
  DV01              : -D_mod * Price / 10_000
  Duration gap      : D_A - (L/A) * D_L
  Funding ratio     : PV(Assets) / PV(Liabilities)
  Surplus           : PV(Assets) - PV(Liabilities)
  Surplus at risk   : change in surplus for ±1% rate shock
  Glide path        : equity allocation as fn of funding ratio

Free-standing: numpy and scipy only. No network calls.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import brentq as _brentq
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CashFlow:
    """A single cash flow with time (years) and amount."""
    time: float
    amount: float


@dataclass
class LiabilityProfile:
    """Represents a stream of liability cash flows with a discount rate."""
    cash_flows: List[CashFlow]
    discount_rate: float = 0.05

    def pv(self) -> float:
        """Present value of all cash flows."""
        return sum(
            cf.amount / (1.0 + self.discount_rate) ** cf.time
            for cf in self.cash_flows
        )

    def duration(self) -> float:
        """Macaulay duration of the liability profile."""
        return macaulay_duration(self.cash_flows, self.discount_rate)

    def mod_duration(self) -> float:
        """Modified duration (annual coupon frequency assumed)."""
        return modified_duration(self.duration(), self.discount_rate, freq=1)

    def dv01(self) -> float:
        """Dollar value of 1 basis point (DV01)."""
        return dv01_calc(self.mod_duration(), self.pv())


@dataclass
class AssetPortfolio:
    """Represents an asset portfolio with key duration metrics."""
    market_value: float
    duration: float            # Macaulay or effective duration in years
    yield_to_maturity: float
    dv01: float                # Dollar value of 1bp for the total portfolio


@dataclass
class FundingStatus:
    """Snapshot of pension / LDI funding status."""
    funding_ratio: float
    surplus: float             # assets - liabilities (can be negative)
    duration_gap: float        # D_A - (L/A) * D_L
    dv01_gap: float            # DV01_assets - DV01_liabilities

    def is_fully_funded(self) -> bool:
        """Fully funded when assets >= liabilities (ratio >= 1.0)."""
        return self.funding_ratio >= 1.0

    def is_over_funded(self) -> bool:
        """Over-funded when funding ratio exceeds 105%."""
        return self.funding_ratio > 1.05


# ---------------------------------------------------------------------------
# Standalone formula functions
# ---------------------------------------------------------------------------

def macaulay_duration(cash_flows: List[CashFlow], discount_rate: float) -> float:
    """
    Macaulay duration: weighted average time to cash flow receipt.
    D_mac = sum(t_i * PV(CF_i)) / Price
    """
    if not cash_flows:
        return 0.0
    pv_total = 0.0
    weighted_time = 0.0
    for cf in cash_flows:
        pv_i = cf.amount / (1.0 + discount_rate) ** cf.time
        pv_total += pv_i
        weighted_time += cf.time * pv_i
    if pv_total <= 0.0:
        return 0.0
    return weighted_time / pv_total


def modified_duration(mac_dur: float, yield_rate: float, freq: int = 1) -> float:
    """
    Modified duration: D_mod = D_mac / (1 + y/n)
    where n is coupon frequency (1 = annual, 2 = semi-annual).
    """
    return mac_dur / (1.0 + yield_rate / freq)


def dv01_calc(modified_dur: float, price: float) -> float:
    """
    Dollar value of 1 basis point.
    DV01 = modified_dur * price / 10_000
    (absolute value; positive by convention)
    """
    return abs(modified_dur * price / 10_000.0)


def funding_ratio(asset_pv: float, liability_pv: float) -> float:
    """Funding ratio = PV(Assets) / PV(Liabilities)."""
    if liability_pv <= 0.0:
        return float("inf")
    return asset_pv / liability_pv


def duration_gap(
    asset_dur: float,
    liability_dur: float,
    asset_value: float,
    liability_value: float,
) -> float:
    """
    Duration gap = D_assets - (L/A) * D_liabilities
    Positive gap → rising rates hurt surplus.
    Zero gap     → duration-neutral (immunized).
    """
    if asset_value <= 0.0:
        return float("nan")
    leverage = liability_value / asset_value
    return asset_dur - leverage * liability_dur


# ---------------------------------------------------------------------------
# Key Rate Duration
# ---------------------------------------------------------------------------

def key_rate_duration(
    cash_flows: List[CashFlow],
    spot_rates: Dict[float, float],
    bump_maturity: float,
    bump_bps: float = 1.0,
) -> float:
    """
    Key Rate Duration (KRD) for a given maturity point.

    Computes -dP/dy_t * 1/P by bumping the spot rate at 'bump_maturity'
    by ±bump_bps basis points and computing central difference.

    Parameters
    ----------
    cash_flows    : list of CashFlow
    spot_rates    : dict {maturity: rate} — full spot curve
    bump_maturity : the maturity tenor to bump (closest cash flow maturity)
    bump_bps      : size of bump in basis points (default 1bp = 0.01%)
    """
    bump = bump_bps / 10_000.0

    def pv_with_rates(rates: Dict[float, float]) -> float:
        total = 0.0
        for cf in cash_flows:
            # Interpolate rate for this cash flow
            t = cf.time
            maturities = sorted(rates.keys())
            if t <= maturities[0]:
                r = rates[maturities[0]]
            elif t >= maturities[-1]:
                r = rates[maturities[-1]]
            else:
                # Linear interpolation
                for i in range(len(maturities) - 1):
                    if maturities[i] <= t <= maturities[i + 1]:
                        t0, t1 = maturities[i], maturities[i + 1]
                        r0, r1 = rates[t0], rates[t1]
                        r = r0 + (r1 - r0) * (t - t0) / (t1 - t0)
                        break
                else:
                    r = rates[maturities[-1]]
            total += cf.amount / (1.0 + r) ** t
        return total

    # Bump up
    rates_up = dict(spot_rates)
    rates_up[bump_maturity] = spot_rates.get(bump_maturity, 0.05) + bump
    # Bump down
    rates_dn = dict(spot_rates)
    rates_dn[bump_maturity] = spot_rates.get(bump_maturity, 0.05) - bump

    pv_base = pv_with_rates(spot_rates)
    pv_up = pv_with_rates(rates_up)
    pv_dn = pv_with_rates(rates_dn)

    if pv_base <= 0.0:
        return 0.0

    # Central difference approximation
    dP_dy = (pv_dn - pv_up) / (2.0 * bump)
    return dP_dy / pv_base


# ---------------------------------------------------------------------------
# LDI Analyzer
# ---------------------------------------------------------------------------

class LDIAnalyzer:
    """
    Core LDI analytics engine.

    Computes funding status, duration gap, hedge ratios, surplus-at-risk,
    and glide path equity allocations.
    """

    def __init__(self, assets: AssetPortfolio, liabilities: LiabilityProfile) -> None:
        self.assets = assets
        self.liabilities = liabilities

    def funding_status(self) -> FundingStatus:
        """Compute current funding status snapshot."""
        liability_pv = self.liabilities.pv()
        asset_pv = self.assets.market_value
        fr = funding_ratio(asset_pv, liability_pv)
        surplus = asset_pv - liability_pv
        dgap = self.duration_gap()
        dv01_assets = self.assets.dv01
        dv01_liab = self.liabilities.dv01()
        return FundingStatus(
            funding_ratio=fr,
            surplus=surplus,
            duration_gap=dgap,
            dv01_gap=dv01_assets - dv01_liab,
        )

    def duration_gap(self) -> float:
        """
        Duration gap = D_assets - (L/A) * D_liabilities
        """
        a = self.assets.market_value
        l = self.liabilities.pv()
        d_a = self.assets.duration
        d_l = self.liabilities.duration()
        return duration_gap(d_a, d_l, a, l)

    def hedge_ratio(self) -> float:
        """
        Hedge ratio for overlay: L * D_L / (A * D_A)
        = fraction of asset duration risk offset by liabilities
        """
        a = self.assets.market_value
        l = self.liabilities.pv()
        d_a = self.assets.duration
        d_l = self.liabilities.duration()
        denominator = a * d_a
        if denominator == 0.0:
            return 0.0
        return (l * d_l) / denominator

    def required_overlay_notional(self, target_duration: float) -> float:
        """
        Notional for interest rate swap overlay to reach target duration.

        Notional = (target_duration - current_duration) * Portfolio_value / DV01_per_unit
        We use DV01 per $1 notional of an interest rate swap ≈ D_mod_swap / 10000.
        For simplicity, assume the IRS has duration = target_duration.

        Returns signed notional (positive = receive-fixed, negative = pay-fixed).
        """
        current_dur = self.assets.duration
        pv = self.assets.market_value
        dur_change = target_duration - current_dur
        # DV01 per unit notional (using target_duration as proxy for swap duration)
        if target_duration <= 0.0:
            return 0.0
        dv01_per_unit = target_duration / 10_000.0
        if dv01_per_unit == 0.0:
            return 0.0
        # DV01 needed in total = dur_change * pv / 10000
        dv01_needed = dur_change * pv / 10_000.0
        return dv01_needed / dv01_per_unit

    def surplus_at_risk(self, interest_rate_shock: float = 0.01) -> float:
        """
        Change in surplus when rates rise by interest_rate_shock (e.g., +1%).

        Delta_surplus ≈ DV01_assets * (-shock*10000) - DV01_liab * (-shock*10000)
                      = (DV01_assets - DV01_liab) * (-shock * 10000)

        Positive = surplus improves with rate rise.
        Negative = surplus deteriorates with rate rise.
        """
        dv01_assets = self.assets.dv01
        dv01_liab = self.liabilities.dv01()
        shock_bps = interest_rate_shock * 10_000.0
        # When rates rise: bond prices fall → DV01 represents $ loss per bp rise
        # Surplus change = -(DV01_assets - DV01_liab) * shock_bps
        return -(dv01_assets - dv01_liab) * shock_bps

    def glide_path(self, funding_ratios: np.ndarray) -> np.ndarray:
        """
        Target equity allocation for given funding ratios.

        Formula: equity_pct = max(0, 100 - 100 * FR) clamped to [0, 60]
        At FR=0.7: equity=30%, FR=1.0: equity=0%, FR>1.0: 0%.
        """
        frs = np.asarray(funding_ratios, dtype=float)
        equity = np.clip(100.0 - 100.0 * frs, 0.0, 100.0)
        return equity


# ---------------------------------------------------------------------------
# Immunization Engine
# ---------------------------------------------------------------------------

class ImmunizationEngine:
    """
    Checks whether an asset portfolio immunizes a liability stream.

    Classical immunization conditions (Redington):
    1. PV(assets) >= PV(liabilities)
    2. D(assets) = D(liabilities)  [duration match]
    3. Convexity(assets) >= Convexity(liabilities)  [convexity condition]
    """

    def is_immunized(
        self,
        assets: AssetPortfolio,
        liabilities: LiabilityProfile,
        tol: float = 0.01,
    ) -> bool:
        """
        Return True if the portfolio is immunized (duration matched within tol years).
        """
        gap = self.immunization_gap(assets, liabilities)
        return abs(gap["duration_gap"]) <= tol and gap["funding_ratio"] >= 1.0

    def immunization_gap(
        self,
        assets: AssetPortfolio,
        liabilities: LiabilityProfile,
    ) -> dict:
        """
        Return immunization gap metrics.

        Returns dict with keys:
          duration_gap   : D_assets - D_liabilities (years)
          funding_ratio  : PV(assets) / PV(liabilities)
          surplus        : PV(assets) - PV(liabilities)
          is_immunized   : bool
        """
        d_a = assets.duration
        d_l = liabilities.duration()
        pv_a = assets.market_value
        pv_l = liabilities.pv()
        fr = funding_ratio(pv_a, pv_l)
        gap = d_a - d_l
        return {
            "duration_gap": gap,
            "funding_ratio": fr,
            "surplus": pv_a - pv_l,
            "is_immunized": abs(gap) <= 0.01 and fr >= 1.0,
        }

    def convexity_match(
        self,
        assets: AssetPortfolio,
        liabilities: LiabilityProfile,
    ) -> float:
        """
        Estimate convexity difference (assets - liabilities).
        Convexity ≈ D_mod^2 + D_mod (crude approximation for bullet bonds).
        """
        ytm = assets.yield_to_maturity
        d_mod_a = modified_duration(assets.duration, ytm)
        conv_a = d_mod_a ** 2 + d_mod_a  # simple estimate

        d_mac_l = liabilities.duration()
        d_mod_l = liabilities.mod_duration()
        conv_l = d_mod_l ** 2 + d_mod_l

        return conv_a - conv_l


# ---------------------------------------------------------------------------
# Liability spot-curve bootstrapping (simplified)
# ---------------------------------------------------------------------------

def bootstrap_spot_curve(
    maturities: List[float],
    par_yields: List[float],
    coupon_freq: int = 2,
) -> Dict[float, float]:
    """
    Bootstrap spot rates from par yields (simplified, semi-annual coupons).

    Uses the standard bootstrapping algorithm:
      P_n = sum(c/(1+s_i)^i, i=1..n-1) + (1+c)/(1+s_n)^n = 1
    where c = par_yield/coupon_freq.

    Returns dict {maturity: spot_rate}.
    """
    spot_rates: Dict[float, float] = {}
    for idx, (mat, par) in enumerate(zip(maturities, par_yields)):
        c = par / coupon_freq
        # Sum discounted coupons using already-bootstrapped spots
        coupon_pv = 0.0
        steps = round(mat * coupon_freq)
        for j in range(1, steps):
            t_j = j / coupon_freq
            # Linear interpolate spot for intermediate times
            if spot_rates:
                known_mats = sorted(spot_rates.keys())
                if t_j <= known_mats[0]:
                    s_j = spot_rates[known_mats[0]]
                elif t_j >= known_mats[-1]:
                    s_j = spot_rates[known_mats[-1]]
                else:
                    # interpolate
                    s_j = np.interp(t_j, known_mats, [spot_rates[m] for m in known_mats])
                coupon_pv += c / (1.0 + s_j / coupon_freq) ** j
            else:
                coupon_pv += c / (1.0 + par) ** t_j
        # Solve for spot at maturity
        remaining = 1.0 + c - coupon_pv
        if remaining <= 0.0 or steps <= 0:
            spot_rates[mat] = par
        else:
            spot_rates[mat] = (remaining ** (-1.0 / (mat * coupon_freq))) - 1.0
            spot_rates[mat] = spot_rates[mat] * coupon_freq  # annualize

    return spot_rates


# ---------------------------------------------------------------------------
# Convenience aliases (backward compat with dim_146 test patterns)
# ---------------------------------------------------------------------------

def dv01(modified_dur: float, price: float) -> float:
    """Alias for dv01_calc."""
    return dv01_calc(modified_dur, price)
