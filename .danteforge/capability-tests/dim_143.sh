#!/usr/bin/env bash
# dim_143: Energy transition analytics (stranded assets / EV demand curves)
# Comprehensive capability verification for sentinel.sfe.energy_transition_v3
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---------------------------------------------------------------------------
# 1. Imports — full public API
# ---------------------------------------------------------------------------
from sentinel.sfe.energy_transition_v3 import (
    FossilAsset,
    StrandedAssetRisk,
    StrandedAssetAnalyzer,
    EVAdoptionModel,
    RenewableLearningCurve,
    EnergyTransitionMetrics,
    bass_ev_adoption,
    renewable_cost,
    stranded_asset_npv,
    oil_demand_peak,
    # Legacy aliases required by original dim_143.sh
    EnergyTransitionModel,
    EVDemandCurve,
    compute_transition_risk,
)
print("[OK] All classes and functions imported from energy_transition_v3")

# ---------------------------------------------------------------------------
# 2. FossilAsset + StrandedAssetAnalyzer
#    Coal plant: book=500M, remaining=20yr, total=40yr,
#    CF=50M/yr, emissions=5MT/yr, carbon_price=50
# ---------------------------------------------------------------------------
coal = FossilAsset(
    name="Coal Plant Alpha",
    asset_type="coal",
    book_value=500e6,
    remaining_life_years=20.0,
    total_life_years=40.0,
    annual_cf=50e6,
    emissions_mtpa=5.0,
    carbon_price_per_ton=50.0,
    policy_transition_year=2035,
)

analyzer = StrandedAssetAnalyzer()
risk = analyzer.assess(coal, discount_rate=0.10)

assert isinstance(risk, StrandedAssetRisk), "assess() must return StrandedAssetRisk"
assert risk.stranded_value > 0,  f"stranded_value must be > 0, got {risk.stranded_value}"
assert 0 < risk.stranded_pct < 1, f"stranded_pct must be in (0,1), got {risk.stranded_pct}"
assert risk.npv_with_carbon < risk.npv_base, \
    f"npv_with_carbon ({risk.npv_with_carbon:.0f}) must be < npv_base ({risk.npv_base:.0f})"
assert risk.npv_haircut >= 0, f"npv_haircut must be >= 0, got {risk.npv_haircut}"
assert risk.break_even_carbon_price > 0, f"break_even_carbon_price must be > 0"

print(f"[OK] stranded_value = ${risk.stranded_value/1e6:.1f}M")
print(f"[OK] stranded_pct   = {risk.stranded_pct:.4f}  (in (0,1))")
print(f"[OK] npv_base       = ${risk.npv_base/1e6:.1f}M")
print(f"[OK] npv_with_carbon= ${risk.npv_with_carbon/1e6:.1f}M  (< npv_base)")
print(f"[OK] npv_haircut    = ${risk.npv_haircut/1e6:.1f}M")
print(f"[OK] break_even_carbon_price = ${risk.break_even_carbon_price:.2f}/tonne")

# ---------------------------------------------------------------------------
# 3. Portfolio risk
# ---------------------------------------------------------------------------
oil_asset = FossilAsset(
    name="Oil Field Beta",
    asset_type="oil",
    book_value=200e6,
    remaining_life_years=15.0,
    total_life_years=30.0,
    annual_cf=30e6,
    emissions_mtpa=2.0,
    carbon_price_per_ton=50.0,
)
portfolio_result = analyzer.portfolio_risk([coal, oil_asset])
assert "total_stranded_value" in portfolio_result
assert "avg_stranded_pct"     in portfolio_result
assert "worst_asset"          in portfolio_result
assert portfolio_result["total_stranded_value"] > 0
print(f"[OK] portfolio_risk total_stranded_value = ${portfolio_result['total_stranded_value']/1e6:.1f}M")
print(f"[OK] portfolio_risk worst_asset = '{portfolio_result['worst_asset']}'")

# ---------------------------------------------------------------------------
# 4. transition_scenario
# ---------------------------------------------------------------------------
npvs = analyzer.transition_scenario(coal, carbon_prices=[0, 50, 100, 200], years=[2035]*4)
assert len(npvs) == 4, f"Expected 4 NPVs, got {len(npvs)}"
assert npvs[0] > npvs[-1], "Higher carbon prices → lower NPV"
print(f"[OK] transition_scenario: NPVs at $0/$50/$100/$200 = {npvs/1e6}")

# ---------------------------------------------------------------------------
# 5. EVAdoptionModel (Bass Diffusion)
# ---------------------------------------------------------------------------
ev = EVAdoptionModel(p=0.003, q=0.38, market_size=1.4e9)

bass = ev.bass_diffusion(20)
assert len(bass) == 20, f"bass_diffusion(20) must return array of length 20, got {len(bass)}"
assert np.all(np.diff(bass) >= 0), "bass_diffusion must be monotone non-decreasing"
assert bass[-1] > bass[0], "cumulative adoptions grow over 20 years"
print(f"[OK] bass_diffusion(20): length={len(bass)}, monotone non-decreasing")
print(f"     year 1 = {bass[0]/1e6:.1f}M EVs, year 20 = {bass[-1]/1e6:.1f}M EVs")

pen_2025 = ev.penetration_at(2025)
pen_2035 = ev.penetration_at(2035)
assert pen_2035 > pen_2025, f"penetration grows: 2025={pen_2025:.4f}, 2035={pen_2035:.4f}"
print(f"[OK] penetration_at(2025) = {pen_2025:.4f}, penetration_at(2035) = {pen_2035:.4f}")

peak_yr = ev.peak_adoption_year()
assert 2030 <= peak_yr <= 2060, f"peak_adoption_year should be 2030-2060, got {peak_yr}"
print(f"[OK] peak_adoption_year = {peak_yr}")

disp = ev.oil_demand_displacement(baseline_demand_mbpd=100.0)
assert len(disp) == 20, f"oil_demand_displacement must have 20 entries, got {len(disp)}"
assert np.all(np.diff(disp) >= 0), "displacement grows with EV adoption"
print(f"[OK] oil_demand_displacement: length=20, monotone growing")
print(f"     year 20 displacement = {disp[-1]:.4f} million bbl/day")

# ---------------------------------------------------------------------------
# 6. RenewableLearningCurve (Wright's Law)
# ---------------------------------------------------------------------------
solar = RenewableLearningCurve(technology="solar", initial_cost=350.0, learning_rate=0.20)

cost_100gw = solar.cost_at_capacity(100.0)
assert cost_100gw < 350.0, \
    f"solar at 100 GW must be < initial 350 $/MWh, got {cost_100gw:.2f}"
print(f"[OK] renewable_cost at 100 GW = ${cost_100gw:.2f}/MWh  (< $350 initial)")

trajectory = solar.cost_trajectory(initial_gw=1.0, final_gw=500.0, n_points=20)
assert len(trajectory) == 20, f"cost_trajectory must have 20 points, got {len(trajectory)}"
assert np.all(np.diff(trajectory) <= 0), "cost_trajectory must be monotone decreasing"
print(f"[OK] cost_trajectory: 20 points, monotone decreasing")
print(f"     start=${trajectory[0]:.2f}/MWh, end=${trajectory[-1]:.2f}/MWh")

be_yr = solar.breakeven_year(fossil_cost=80.0, capacity_growth_pct=0.25)
assert isinstance(be_yr, int), "breakeven_year must return int"
print(f"[OK] breakeven_year vs $80/MWh fossil = {be_yr}")

# ---------------------------------------------------------------------------
# 7. EnergyTransitionMetrics
# ---------------------------------------------------------------------------
etm = EnergyTransitionMetrics()

base_npv = 300e6
adj_npv = etm.carbon_adjusted_npv(
    base_npv=base_npv, emissions_mtpa=5.0, carbon_price=50.0,
    discount_rate=0.10, years=20
)
assert adj_npv < base_npv, \
    f"carbon_adjusted_npv ({adj_npv/1e6:.1f}M) must be < base_npv ({base_npv/1e6:.1f}M)"
print(f"[OK] carbon_adjusted_npv = ${adj_npv/1e6:.1f}M  (< base ${base_npv/1e6:.0f}M)")

haircut_15 = etm.temperature_scenario_impact(coal, scenario="1.5C")
assert abs(haircut_15 - 0.8) < 1e-9, f"1.5C haircut must be 0.8, got {haircut_15}"
haircut_2  = etm.temperature_scenario_impact(coal, scenario="2C")
assert abs(haircut_2 - 0.5) < 1e-9,  f"2C haircut must be 0.5, got {haircut_2}"
haircut_bau = etm.temperature_scenario_impact(coal, scenario="BAU")
assert abs(haircut_bau - 0.0) < 1e-9, f"BAU haircut must be 0.0, got {haircut_bau}"
print(f"[OK] temperature_scenario_impact: 1.5C={haircut_15}, 2C={haircut_2}, BAU={haircut_bau}")

intensity = etm.portfolio_carbon_intensity([coal, oil_asset])
assert intensity > 0, "portfolio_carbon_intensity must be > 0"
print(f"[OK] portfolio_carbon_intensity = {intensity:.6f} MT CO2 / $")

# ---------------------------------------------------------------------------
# 8. Standalone functions
# ---------------------------------------------------------------------------
bass_arr = bass_ev_adoption(20)
assert len(bass_arr) == 20, f"bass_ev_adoption(20) length wrong: {len(bass_arr)}"
print(f"[OK] standalone bass_ev_adoption(20) length = {len(bass_arr)}")

rc = renewable_cost(100.0, initial_cost=350.0, LR=0.20)
assert rc < 350.0, f"standalone renewable_cost at 100 GW should be < 350, got {rc:.2f}"
print(f"[OK] standalone renewable_cost(100 GW) = ${rc:.2f}/MWh")

npv_sa = stranded_asset_npv(50e6, 20, 5e6 * 50, 0.10)
assert np.isfinite(npv_sa), f"stranded_asset_npv must be finite, got {npv_sa}"
print(f"[OK] standalone stranded_asset_npv = ${npv_sa/1e6:.1f}M")

peak_demand = oil_demand_peak(100.0, ev_penetration=0.30, efficiency_factor=0.25)
assert 0 < peak_demand < 100.0, f"oil_demand_peak must be in (0, 100), got {peak_demand}"
print(f"[OK] standalone oil_demand_peak = {peak_demand:.2f} million bbl/day")

# ---------------------------------------------------------------------------
# 9. Legacy aliases (original dim_143.sh requirements)
# ---------------------------------------------------------------------------
etm_model = EnergyTransitionModel()
assert etm_model is not None, "EnergyTransitionModel must be instantiable"

evc = EVDemandCurve()
assert evc is not None, "EVDemandCurve must be instantiable"
evc_bass = evc.bass_diffusion(10)
assert len(evc_bass) == 10, "EVDemandCurve.bass_diffusion(10) must have length 10"

ctr = compute_transition_risk([coal, oil_asset])
assert "total_stranded_value" in ctr, "compute_transition_risk missing 'total_stranded_value'"
assert ctr["total_stranded_value"] > 0

print("[OK] Legacy aliases: EnergyTransitionModel, EVDemandCurve, compute_transition_risk")

print("\n[PASS] dim_143: Energy transition analytics -- all checks passed")
PYEOF
