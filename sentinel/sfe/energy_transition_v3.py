"""
energy_transition_v3.py — Energy transition analytics engine.

dim_143: Stranded asset risk, EV adoption curves, renewable capacity models

Architecture:
  FossilAsset               — Fossil fuel asset dataclass (coal/oil/gas/refinery)
  StrandedAssetRisk         — Result dataclass for stranded asset assessment
  StrandedAssetAnalyzer     — NPV-based stranded value computation under carbon cost
  EVAdoptionModel           — Bass diffusion S-curve for EV penetration
  RenewableLearningCurve    — Wright's Law cost trajectory for solar/wind
  EnergyTransitionMetrics   — Carbon-adjusted NPV, temperature scenarios, intensity
  EnergyTransitionModel     — Orchestrator (legacy alias for crusade compatibility)
  EVDemandCurve             — Legacy alias for EVAdoptionModel
  compute_transition_risk   — Standalone risk aggregator (legacy alias)

No network calls — numpy/scipy only.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import odeint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FossilAsset:
    """A fossil-fuel-based asset subject to transition risk."""

    name: str
    asset_type: str             # 'coal', 'oil', 'gas', 'refinery'
    book_value: float           # current book value ($)
    remaining_life_years: float # years of useful life remaining
    total_life_years: float     # total designed useful life
    annual_cf: float            # free cash flow per year ($)
    emissions_mtpa: float       # CO₂ equivalent, million tonnes per year
    carbon_price_per_ton: float = 50.0    # $/tonne CO₂
    policy_transition_year: int = 2035    # year hard policy cap kicks in


@dataclass
class StrandedAssetRisk:
    """Output of a stranded asset assessment."""

    stranded_value: float        # $ of book value at risk
    stranded_pct: float          # fraction of book value (0–1)
    npv_base: float              # NPV without carbon cost
    npv_with_carbon: float       # NPV after subtracting carbon costs
    npv_haircut: float           # npv_base - npv_with_carbon (always ≥ 0)
    break_even_carbon_price: float  # carbon price that zeroes out NPV


# ---------------------------------------------------------------------------
# Stranded Asset Analyzer
# ---------------------------------------------------------------------------

class StrandedAssetAnalyzer:
    """Compute stranded value and NPV adjustments for fossil assets."""

    # ------------------------------------------------------------------
    def assess(
        self, asset: FossilAsset, discount_rate: float = 0.10
    ) -> StrandedAssetRisk:
        """Full stranded asset assessment.

        Stranded value formula:
            stranded_pct = max(0, 1 - (remaining_life / total_life))
            stranded_value = book_value * stranded_pct

        Base NPV:
            NPV_base = sum_{t=1}^{T} CF / (1+r)^t

        Carbon-adjusted NPV (carbon cost paid each year):
            carbon_cost_pa = emissions_mtpa * 1e6 * carbon_price_per_ton
            NPV_carbon = sum_{t=1}^{T} (CF - carbon_cost_pa) / (1+r)^t
        """
        T = int(math.ceil(asset.remaining_life_years))
        if T <= 0:
            return StrandedAssetRisk(
                stranded_value=asset.book_value,
                stranded_pct=1.0,
                npv_base=0.0,
                npv_with_carbon=0.0,
                npv_haircut=0.0,
                break_even_carbon_price=0.0,
            )

        # Stranded value
        stranded_pct = max(
            0.0, 1.0 - asset.remaining_life_years / asset.total_life_years
        )
        stranded_value = asset.book_value * stranded_pct

        # Base NPV (annuity)
        times = np.arange(1, T + 1, dtype=float)
        discount = 1.0 / (1.0 + discount_rate) ** times
        npv_base = float(np.sum(asset.annual_cf * discount))

        # Carbon annual cost (in dollars)
        carbon_cost_pa = asset.emissions_mtpa * 1e6 * asset.carbon_price_per_ton
        npv_with_carbon = float(np.sum((asset.annual_cf - carbon_cost_pa) * discount))

        npv_haircut = max(0.0, npv_base - npv_with_carbon)

        # Break-even carbon price: solve npv_base - sum(emission_cost/denom) = 0
        # sum(disc) * emit * 1e6 * p_c = npv_base
        sum_disc = float(np.sum(discount))
        if sum_disc > 0 and asset.emissions_mtpa > 0:
            break_even = npv_base / (sum_disc * asset.emissions_mtpa * 1e6)
        else:
            break_even = float("inf")

        return StrandedAssetRisk(
            stranded_value=stranded_value,
            stranded_pct=stranded_pct,
            npv_base=npv_base,
            npv_with_carbon=npv_with_carbon,
            npv_haircut=npv_haircut,
            break_even_carbon_price=break_even,
        )

    # ------------------------------------------------------------------
    def portfolio_risk(
        self, assets: List[FossilAsset], discount_rate: float = 0.10
    ) -> dict:
        """Aggregate stranded asset risk across a portfolio.

        Returns:
            dict with keys:
              'total_stranded_value'  — sum of individual stranded values
              'avg_stranded_pct'      — equal-weight average stranded pct
              'worst_asset'           — name of asset with highest stranded pct
        """
        results = [self.assess(a, discount_rate) for a in assets]
        total_sv = sum(r.stranded_value for r in results)
        avg_pct = np.mean([r.stranded_pct for r in results]) if results else 0.0
        worst_idx = int(np.argmax([r.stranded_pct for r in results])) if results else 0
        worst = assets[worst_idx].name if assets else ""
        return {
            "total_stranded_value": total_sv,
            "avg_stranded_pct": float(avg_pct),
            "worst_asset": worst,
        }

    # ------------------------------------------------------------------
    def transition_scenario(
        self,
        asset: FossilAsset,
        carbon_prices: List[float],
        years: List[int],
    ) -> np.ndarray:
        """NPV under each (carbon_price, year) scenario.

        Args:
            asset:         Asset to value.
            carbon_prices: List of carbon prices ($/tonne).
            years:         Corresponding list of transition years (unused in
                           NPV but stored for labelling).

        Returns:
            1-D array of NPVs, one per scenario.
        """
        npvs = []
        for cp in carbon_prices:
            a_copy = FossilAsset(
                name=asset.name,
                asset_type=asset.asset_type,
                book_value=asset.book_value,
                remaining_life_years=asset.remaining_life_years,
                total_life_years=asset.total_life_years,
                annual_cf=asset.annual_cf,
                emissions_mtpa=asset.emissions_mtpa,
                carbon_price_per_ton=cp,
            )
            risk = self.assess(a_copy)
            npvs.append(risk.npv_with_carbon)
        return np.array(npvs)


# ---------------------------------------------------------------------------
# EV Adoption Model (Bass Diffusion)
# ---------------------------------------------------------------------------

class EVAdoptionModel:
    """Bass diffusion model for electric vehicle adoption.

    dN/dt = (p + q * N/M) * (M - N)

    N(t) = M * (1 - exp(-(p+q)*t)) / (1 + (q/p)*exp(-(p+q)*t))

    Args:
        p:           Innovation coefficient (default 0.003).
        q:           Imitation coefficient (default 0.38).
        market_size: Total addressable market in vehicles (default 1.4e9).
    """

    def __init__(
        self,
        p: float = 0.003,
        q: float = 0.38,
        market_size: float = 1.4e9,
    ) -> None:
        self.p = p
        self.q = q
        self.M = market_size

    # ------------------------------------------------------------------
    def bass_diffusion(self, t_years: int) -> np.ndarray:
        """Cumulative EV adoptions at each integer year [1 .. t_years].

        Returns:
            1-D array of length t_years with cumulative units adopted.
        """
        p, q, M = self.p, self.q, self.M
        times = np.arange(1, t_years + 1, dtype=float)
        pq = p + q
        numerator = 1.0 - np.exp(-pq * times)
        denominator = 1.0 + (q / p) * np.exp(-pq * times)
        return M * numerator / denominator

    # ------------------------------------------------------------------
    def penetration_at(self, year: int, base_year: int = 2024) -> float:
        """EV penetration rate at ``year`` relative to ``base_year``.

        Returns:
            Fraction of total market (0–1).
        """
        t = max(0, year - base_year)
        if t == 0:
            return 0.0
        arr = self.bass_diffusion(t)
        return float(arr[-1] / self.M)

    # ------------------------------------------------------------------
    def peak_adoption_year(self) -> int:
        """Year of peak *new* adoptions (inflection point of cumulative curve).

        At the Bass inflection:
            t_peak = ln(q/p) / (p + q)
        """
        p, q = self.p, self.q
        t_peak = math.log(q / p) / (p + q)
        # Convert to calendar year (assume t=0 is 2024)
        return int(2024 + math.ceil(t_peak))

    # ------------------------------------------------------------------
    def oil_demand_displacement(
        self,
        baseline_demand_mbpd: float,
        liters_per_ev_saved: float = 2200.0,
    ) -> np.ndarray:
        """Oil demand displaced by EV fleet over time.

        Each EV is assumed to save ``liters_per_ev_saved`` litres of petrol
        per year.  1 barrel ≈ 159 litres.

        Args:
            baseline_demand_mbpd: Baseline oil demand in million barrels/day.
            liters_per_ev_saved:  Litres of petrol saved per EV per year.

        Returns:
            1-D array of displaced demand in million barrels/day for each year.
        """
        t_years = 20  # 20-year horizon
        ev_counts = self.bass_diffusion(t_years)                    # units
        liters_saved = ev_counts * liters_per_ev_saved              # litres/year
        barrels_saved_pa = liters_saved / 159.0                     # barrels/year
        mbpd_saved = barrels_saved_pa / 365.0 / 1e6                 # million bbl/day
        return mbpd_saved


# ---------------------------------------------------------------------------
# Renewable Learning Curve (Wright's Law)
# ---------------------------------------------------------------------------

class RenewableLearningCurve:
    """Wright's Law cost trajectory for renewable energy technologies.

    Cost(X) = Cost(1) * X^b    where b = log2(1 - LR)

    Args:
        technology:    'solar' or 'wind' (affects defaults).
        initial_cost:  Cost at 1 GW cumulative capacity ($/MWh).
        learning_rate: Fraction by which cost falls for each doubling of
                       cumulative capacity (e.g. 0.20 = 20 %).
    """

    def __init__(
        self,
        technology: str = "solar",
        initial_cost: float = 350.0,
        learning_rate: float = 0.20,
    ) -> None:
        self.technology = technology
        self.initial_cost = initial_cost
        self.learning_rate = learning_rate
        # Wright's Law exponent: b = log2(1 - LR)
        self._b = math.log2(1.0 - learning_rate)

    # ------------------------------------------------------------------
    def cost_at_capacity(self, cumulative_gw: float) -> float:
        """Cost at a given level of cumulative installed capacity.

        Args:
            cumulative_gw: Cumulative installed capacity in GW.

        Returns:
            Levelised cost of energy in $/MWh.
        """
        if cumulative_gw <= 0:
            return self.initial_cost
        return self.initial_cost * (cumulative_gw ** self._b)

    # ------------------------------------------------------------------
    def cost_trajectory(
        self,
        initial_gw: float,
        final_gw: float,
        n_points: int = 20,
    ) -> np.ndarray:
        """Cost at each of ``n_points`` capacity levels from initial to final.

        Returns:
            1-D array of costs ($/MWh), decreasing as capacity grows.
        """
        capacities = np.linspace(initial_gw, final_gw, n_points)
        capacities = np.where(capacities <= 0, 1e-9, capacities)
        return self.initial_cost * (capacities ** self._b)

    # ------------------------------------------------------------------
    def breakeven_year(
        self,
        fossil_cost: float,
        capacity_growth_pct: float = 0.25,
    ) -> int:
        """Year when renewable cost falls below ``fossil_cost``.

        Assumes cumulative capacity starts at 1 GW and grows by
        ``capacity_growth_pct`` per year.

        Args:
            fossil_cost:          Competing fossil fuel cost ($/MWh).
            capacity_growth_pct:  Annual growth rate of cumulative capacity.

        Returns:
            Year (integer) of grid parity.  Returns 9999 if parity is not
            reached within 100 years.
        """
        gw = 1.0
        for yr in range(100):
            if self.cost_at_capacity(gw) <= fossil_cost:
                return 2024 + yr
            gw *= (1.0 + capacity_growth_pct)
        return 9999


# ---------------------------------------------------------------------------
# Energy Transition Metrics
# ---------------------------------------------------------------------------

class EnergyTransitionMetrics:
    """Portfolio-level energy transition KPIs."""

    _SCENARIO_HAIRCUTS: Dict[str, float] = {
        "1.5C": 0.8,
        "2C":   0.5,
        "BAU":  0.0,
    }

    # ------------------------------------------------------------------
    def carbon_adjusted_npv(
        self,
        base_npv: float,
        emissions_mtpa: float,
        carbon_price: float,
        discount_rate: float,
        years: int,
    ) -> float:
        """NPV after subtracting PV of future carbon costs.

        PV(carbon) = sum_{t=1}^{years} (emissions * 1e6 * carbon_price) / (1+r)^t

        Returns:
            Adjusted NPV (will be < base_npv when carbon_price > 0).
        """
        times = np.arange(1, years + 1, dtype=float)
        discount = 1.0 / (1.0 + discount_rate) ** times
        pv_carbon = float(
            np.sum(emissions_mtpa * 1e6 * carbon_price * discount)
        )
        return base_npv - pv_carbon

    # ------------------------------------------------------------------
    def temperature_scenario_impact(
        self, asset: "FossilAsset", scenario: str = "1.5C"
    ) -> float:
        """Haircut multiplier for a given temperature scenario.

        Args:
            asset:    Fossil asset (unused in current version; kept for API
                      compatibility with future implementation).
            scenario: One of '1.5C', '2C', 'BAU'.

        Returns:
            Haircut multiplier: 1.5C→0.8, 2C→0.5, BAU→0.0.
        """
        return self._SCENARIO_HAIRCUTS.get(scenario, 0.0)

    # ------------------------------------------------------------------
    def portfolio_carbon_intensity(
        self, assets: List["FossilAsset"]
    ) -> float:
        """Weighted average carbon intensity across the portfolio.

        Intensity = total_emissions_mtpa / total_annual_cf

        Returns:
            Emission intensity in Mt CO₂ / $ revenue.  Returns 0 if the
            portfolio is empty or has zero aggregate cash flow.
        """
        if not assets:
            return 0.0
        total_cf = sum(a.annual_cf for a in assets)
        total_em = sum(a.emissions_mtpa for a in assets)
        if total_cf <= 0:
            return 0.0
        return total_em / total_cf


# ---------------------------------------------------------------------------
# Standalone convenience functions
# ---------------------------------------------------------------------------

def bass_ev_adoption(
    t_years: int,
    p: float = 0.003,
    q: float = 0.38,
    M: float = 1.4e9,
) -> np.ndarray:
    """Cumulative EV adoptions over ``t_years`` via Bass diffusion.

    Returns:
        1-D array of length t_years.
    """
    model = EVAdoptionModel(p=p, q=q, market_size=M)
    return model.bass_diffusion(t_years)


def renewable_cost(
    cumulative_gw: float,
    initial_cost: float = 350.0,
    LR: float = 0.20,
) -> float:
    """Cost of renewable energy at a given cumulative installed capacity.

    Uses Wright's Law: Cost(X) = Cost(1) * X^b, b = log2(1 - LR).

    Args:
        cumulative_gw: Cumulative capacity in GW.
        initial_cost:  Cost at 1 GW ($/MWh).
        LR:            Learning rate (e.g. 0.20).

    Returns:
        Cost in $/MWh.
    """
    curve = RenewableLearningCurve(initial_cost=initial_cost, learning_rate=LR)
    return curve.cost_at_capacity(cumulative_gw)


def stranded_asset_npv(
    annual_cf: float,
    remaining_years: int,
    carbon_cost_annual: float,
    discount_rate: float = 0.10,
) -> float:
    """NPV of a stranded asset after subtracting annual carbon costs.

    Args:
        annual_cf:          Free cash flow per year ($).
        remaining_years:    Remaining useful life (integer years).
        carbon_cost_annual: Annual carbon cost ($).
        discount_rate:      Discount rate.

    Returns:
        Net present value ($).
    """
    if remaining_years <= 0:
        return 0.0
    times = np.arange(1, remaining_years + 1, dtype=float)
    discount = 1.0 / (1.0 + discount_rate) ** times
    net_cf = annual_cf - carbon_cost_annual
    return float(np.sum(net_cf * discount))


def oil_demand_peak(
    baseline_mbpd: float,
    ev_penetration: float,
    efficiency_factor: float = 0.25,
) -> float:
    """Oil demand after adjusting for EV penetration.

    peak_demand = baseline * (1 - ev_penetration * efficiency_factor)

    Args:
        baseline_mbpd:    Baseline demand in million barrels/day.
        ev_penetration:   EV fraction of total vehicle fleet (0–1).
        efficiency_factor: Fraction of baseline demand displaced per unit of
                           EV penetration.

    Returns:
        Adjusted demand in million barrels/day.
    """
    return baseline_mbpd * (1.0 - ev_penetration * efficiency_factor)


# ---------------------------------------------------------------------------
# Legacy aliases — required by the existing dim_143.sh capability test
# ---------------------------------------------------------------------------

class EnergyTransitionModel:
    """Orchestrator combining all transition analytics.

    Provided as a legacy alias for backward compatibility with the
    dim_143 capability test.
    """

    def __init__(self) -> None:
        self._stranded = StrandedAssetAnalyzer()
        self._ev = EVAdoptionModel()
        self._renewable = RenewableLearningCurve()
        self._metrics = EnergyTransitionMetrics()

    def assess_stranded(
        self, asset: FossilAsset, discount_rate: float = 0.10
    ) -> StrandedAssetRisk:
        return self._stranded.assess(asset, discount_rate)

    def ev_penetration(self, year: int, base_year: int = 2024) -> float:
        return self._ev.penetration_at(year, base_year)

    def renewable_cost(self, cumulative_gw: float) -> float:
        return self._renewable.cost_at_capacity(cumulative_gw)

    def carbon_adjusted_npv(
        self,
        base_npv: float,
        emissions_mtpa: float,
        carbon_price: float,
        discount_rate: float,
        years: int,
    ) -> float:
        return self._metrics.carbon_adjusted_npv(
            base_npv, emissions_mtpa, carbon_price, discount_rate, years
        )


class EVDemandCurve:
    """Legacy alias for EVAdoptionModel.

    Compatible with dim_143.sh import: ``from sentinel.sfe.energy_transition_v3 import EVDemandCurve``
    """

    def __init__(
        self,
        p: float = 0.003,
        q: float = 0.38,
        market_size: float = 1.4e9,
    ) -> None:
        self._model = EVAdoptionModel(p=p, q=q, market_size=market_size)

    def penetration(self, year: int, base_year: int = 2024) -> float:
        return self._model.penetration_at(year, base_year)

    def bass_diffusion(self, t_years: int) -> np.ndarray:
        return self._model.bass_diffusion(t_years)

    def oil_demand_displacement(
        self, baseline_mbpd: float, liters_per_ev: float = 2200.0
    ) -> np.ndarray:
        return self._model.oil_demand_displacement(baseline_mbpd, liters_per_ev)


def compute_transition_risk(
    assets: List[FossilAsset],
    discount_rate: float = 0.10,
) -> dict:
    """Aggregate energy transition risk across a list of fossil assets.

    Returns:
        dict with 'total_stranded_value', 'avg_stranded_pct', 'worst_asset'.
    """
    analyzer = StrandedAssetAnalyzer()
    return analyzer.portfolio_risk(assets, discount_rate)
