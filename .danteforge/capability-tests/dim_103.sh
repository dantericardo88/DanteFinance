#!/usr/bin/env bash
# dim_103: TCFD climate — pillar structure, physical risk, transition risk, scenarios, Climate VaR
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.tcfd_climate_v3 import (
    TCFD_DISCLOSURES,
    TCFDPillarScorer,
    GHGEmissions,
    ClimateTarget,
    TCFDAssessment,
    # Physical risk
    get_state_physical_risk,
    get_company_physical_risk,
    # Transition risk
    get_sic_transition_risk,
    compute_carbon_intensity,
    # Scenario analysis
    climate_scenario_analysis,
    # Climate VaR
    compute_climate_var,
)

# ── TCFD_DISCLOSURES structure ───────────────────────────────────────────────
assert len(TCFD_DISCLOSURES) == 11
pillars_found = {d[0] for d in TCFD_DISCLOSURES}
for pillar in ["governance", "strategy", "risk_mgmt", "metrics"]:
    assert pillar in pillars_found, f"Missing pillar: {pillar}"
gov_d   = [d for d in TCFD_DISCLOSURES if d[0] == "governance"]
strat_d = [d for d in TCFD_DISCLOSURES if d[0] == "strategy"]
risk_d  = [d for d in TCFD_DISCLOSURES if d[0] == "risk_mgmt"]
met_d   = [d for d in TCFD_DISCLOSURES if d[0] == "metrics"]
assert len(gov_d) == 2 and len(strat_d) == 3 and len(risk_d) == 3 and len(met_d) == 3
disc_keys = {d[1] for d in TCFD_DISCLOSURES}
assert "board_oversight" in disc_keys
assert "scenario_analysis" in disc_keys
assert "ghg_scope1_2" in disc_keys
assert "targets" in disc_keys
print(f"[OK] TCFD_DISCLOSURES: 11 disclosures across 4 pillars")

# ── TCFDPillarScorer ─────────────────────────────────────────────────────────
scorer = TCFDPillarScorer()

gov_text = """
The Board of Directors committee oversees and monitors climate risk.
The Chief Sustainability Officer provides management's role in assessing climate risks
and the sustainability committee of management reviews progress quarterly.
"""
gov_result = scorer.score_governance(gov_text)
assert gov_result.pillar == "governance"
assert 0 <= gov_result.score <= 100
print(f"[OK] score_governance: score={gov_result.score:.1f}")

strat_text = """
Climate scenario analysis including 1.5°C scenarios and 2°C Paris pathways.
Climate risk identification considers short, medium and long-term horizons.
Climate risk has a material impact on business strategy and financial planning.
"""
strat_result = scorer.score_strategy(strat_text)
assert strat_result.score > 0
poor_strat_result = scorer.score_strategy("Revenue grew 15% driven by product innovation.")
assert poor_strat_result.score <= strat_result.score
print(f"[OK] score_strategy: good={strat_result.score:.1f} poor={poor_strat_result.score:.1f}")

risk_text = """
Climate risk identification and assessment process is integrated into enterprise risk management (ERM).
The company manages and mitigates climate risk through a formal framework.
Physical and transition risk are identified in our climate risk register.
"""
risk_result = scorer.score_risk_mgmt(risk_text)
assert risk_result.score > 0
print(f"[OK] score_risk_mgmt: score={risk_result.score:.1f}")

# ── Pydantic models ──────────────────────────────────────────────────────────
ghg = GHGEmissions(ticker="XOM", filing_year=2023, scope1_mt=120_000_000.0,
                   scope2_mt_location=15_000_000.0, scope1_source="xbrl",
                   ghg_verified=True, data_quality="high")
assert ghg.ticker == "XOM" and ghg.scope1_mt == 120_000_000.0
print(f"[OK] GHGEmissions: scope1={ghg.scope1_mt:,.0f} tCO2e verified={ghg.ghg_verified}")

target = ClimateTarget(ticker="MSFT", target_type="net_zero", target_year=2030,
                       sbti_approved=True, is_net_zero=True)
assert target.sbti_approved and target.target_year == 2030
print(f"[OK] ClimateTarget: {target.ticker} net_zero by {target.target_year}")

assessment = TCFDAssessment(ticker="AAPL", filing_year=2023, alignment_score=72.5,
                             alignment_level="advancing", disclosures_found_count=8)
assert 0.0 <= assessment.alignment_score <= 100.0
print(f"[OK] TCFDAssessment: score={assessment.alignment_score} level={assessment.alignment_level}")

# ── Physical risk: state-level flood/water risk ──────────────────────────────
fl_risk = get_state_physical_risk("FL")
co_risk = get_state_physical_risk("CO")
la_risk = get_state_physical_risk("LA")
ny_risk = get_state_physical_risk("NY")

# FL and LA should be the highest-risk states
assert fl_risk >= 9.0, f"FL should score >= 9.0: {fl_risk}"
assert la_risk >= 9.0, f"LA should score >= 9.0: {la_risk}"
# FL must score higher than CO (inland, arid)
assert fl_risk > co_risk, f"FL ({fl_risk}) should score higher than CO ({co_risk})"
assert co_risk <= 3.5, f"CO should be low risk (<= 3.5): {co_risk}"
print(f"[OK] State physical risk: FL={fl_risk} LA={la_risk} NY={ny_risk} CO={co_risk}")

# Company-level aggregation
company_risk = get_company_physical_risk("FL", operations_states=["TX", "LA", "MS"])
assert company_risk["aggregate_risk_score"] >= 7.0, \
    f"FL/TX/LA/MS heavy company should be high risk: {company_risk['aggregate_risk_score']}"
assert company_risk["risk_tier"] in ("high", "very_high"), \
    f"Coastal company should be high/very_high tier: {company_risk['risk_tier']}"
print(f"[OK] Company physical risk: aggregate={company_risk['aggregate_risk_score']} tier={company_risk['risk_tier']}")

low_risk_company = get_company_physical_risk("CO", operations_states=["WY", "MT", "ID"])
assert low_risk_company["aggregate_risk_score"] < fl_risk, \
    f"CO/WY/MT company should be less risky than FL HQ: {low_risk_company['aggregate_risk_score']} vs {fl_risk}"
print(f"[OK] Low-risk company (CO/WY/MT): aggregate={low_risk_company['aggregate_risk_score']}")

# ── Transition risk: SIC-based carbon tier ───────────────────────────────────
# SIC 1311 = Crude Petroleum & Natural Gas → must be very_high
oil_risk = get_sic_transition_risk("1311")
assert oil_risk["transition_risk_tier"] == "very_high", \
    f"Oil company (SIC 1311) should be very_high tier: {oil_risk['transition_risk_tier']}"
assert oil_risk["carbon_intensity_proxy"] >= 0.85, \
    f"Oil SIC 1311 should have high carbon intensity: {oil_risk['carbon_intensity_proxy']}"
print(f"[OK] SIC 1311 (oil): tier={oil_risk['transition_risk_tier']} intensity={oil_risk['carbon_intensity_proxy']}")

# SIC 7372 = Prepackaged Software → should be low
sw_risk = get_sic_transition_risk("7372")
assert sw_risk["transition_risk_tier"] in ("low", "very_low"), \
    f"Software (SIC 7372) should be low/very_low: {sw_risk['transition_risk_tier']}"
print(f"[OK] SIC 7372 (software): tier={sw_risk['transition_risk_tier']}")

# SIC 6022 = National Commercial Banks → very_low
bank_risk = get_sic_transition_risk("6022")
assert bank_risk["transition_risk_tier"] == "very_low", \
    f"Banks (SIC 6022) should be very_low: {bank_risk['transition_risk_tier']}"
print(f"[OK] SIC 6022 (banks): tier={bank_risk['transition_risk_tier']}")

# Oil must have higher tier than software
assert oil_risk["transition_risk_tier"] != sw_risk["transition_risk_tier"]
print(f"[OK] Oil tier ({oil_risk['transition_risk_tier']}) != software tier ({sw_risk['transition_risk_tier']})")

# Carbon intensity computation
ci = compute_carbon_intensity(scope1_mt=120_000_000, revenue_m=400_000)
assert ci is not None and ci > 0, f"Carbon intensity should be positive: {ci}"
print(f"[OK] compute_carbon_intensity: {ci} tCO2e/M USD")

# Edge cases
assert compute_carbon_intensity(0, 1000) == 0.0
assert compute_carbon_intensity(1000, 0) is None
print(f"[OK] compute_carbon_intensity edge cases: zero_scope1=0, zero_revenue=None")

# ── Climate scenario analysis ────────────────────────────────────────────────
# Fossil fuel company (energy sector) with $10B assets
scenarios = climate_scenario_analysis(
    ticker="XOM", sector="energy", total_assets_m=10_000.0
)
assert "scenarios" in scenarios
assert "1.5C" in scenarios["scenarios"]
assert "2C"   in scenarios["scenarios"]
assert "4C"   in scenarios["scenarios"]
print(f"[OK] climate_scenario_analysis: 3 scenarios returned for energy sector")

sc_15 = scenarios["scenarios"]["1.5C"]
sc_2  = scenarios["scenarios"]["2C"]
sc_4  = scenarios["scenarios"]["4C"]

# 1.5°C scenario should strand more fossil assets than 4°C (more aggressive transition)
assert sc_15["stranding_risk_pct"] > sc_4["stranding_risk_pct"], \
    f"1.5C stranding ({sc_15['stranding_risk_pct']}) must > 4C ({sc_4['stranding_risk_pct']})"
assert sc_15["stranding_risk_pct"] > sc_2["stranding_risk_pct"], \
    f"1.5C stranding ({sc_15['stranding_risk_pct']}) must > 2C ({sc_2['stranding_risk_pct']})"
print(f"[OK] Stranding risk: 1.5C={sc_15['stranding_risk_pct']:.0%} > 2C={sc_2['stranding_risk_pct']:.0%} > 4C={sc_4['stranding_risk_pct']:.0%}")

# 1.5°C carbon price should be highest
assert sc_15["carbon_price_2030"] > sc_2["carbon_price_2030"] > sc_4["carbon_price_2030"], \
    f"Carbon prices should decrease: {sc_15['carbon_price_2030']} > {sc_2['carbon_price_2030']} > {sc_4['carbon_price_2030']}"
print(f"[OK] Carbon prices: 1.5C=${sc_15['carbon_price_2030']}/t > 2C=${sc_2['carbon_price_2030']}/t > 4C=${sc_4['carbon_price_2030']}/t")

# Stranded assets in USD should be non-trivial for fossil company
assert sc_15["stranded_assets_m"] > 1000, \
    f"Fossil company should have >$1B stranded assets in 1.5C: {sc_15['stranded_assets_m']}"
print(f"[OK] Stranded assets (1.5C, energy): ${sc_15['stranded_assets_m']:,.0f}M")

# Tech/low-carbon company should have lower stranding risk
tech_scenarios = climate_scenario_analysis(
    ticker="MSFT", sector="information_technology", total_assets_m=10_000.0
)
tech_15 = tech_scenarios["scenarios"]["1.5C"]
assert tech_15["stranding_risk_pct"] < sc_15["stranding_risk_pct"], \
    f"Tech stranding ({tech_15['stranding_risk_pct']:.2%}) must < energy ({sc_15['stranding_risk_pct']:.2%})"
print(f"[OK] Tech vs energy stranding: tech={tech_15['stranding_risk_pct']:.2%} < energy={sc_15['stranding_risk_pct']:.2%}")

# ── Climate VaR ─────────────────────────────────────────────────────────────
# Fossil-heavy portfolio
fossil_portfolio = [
    {"ticker": "XOM",  "weight": 0.40, "sic_code": "1311"},  # oil → very_high beta
    {"ticker": "CVX",  "weight": 0.30, "sic_code": "2911"},  # refining → very_high
    {"ticker": "NEE",  "weight": 0.20, "sic_code": "4911"},  # electric utility → high
    {"ticker": "AAPL", "weight": 0.10, "sic_code": "7372"},  # software → low
]
var_result = compute_climate_var(fossil_portfolio, temperature_delta=2.0)
assert "climate_var" in var_result
assert var_result["climate_var"] > 0, \
    f"Climate VaR should be positive for fossil-heavy portfolio: {var_result['climate_var']}"
assert abs(var_result["total_weight"] - 1.0) < 1e-6, \
    f"Weights should sum to 1.0: {var_result['total_weight']}"
assert len(var_result["breakdown"]) == 4
print(f"[OK] Climate VaR (fossil-heavy, 2°C): {var_result['climate_var']:.4f} ({var_result['climate_var']*100:.2f}% of portfolio)")

# Low-carbon portfolio should have lower VaR
clean_portfolio = [
    {"ticker": "MSFT", "weight": 0.50, "sic_code": "7372"},
    {"ticker": "GOOGL","weight": 0.30, "sic_code": "7371"},
    {"ticker": "JNJ",  "weight": 0.20, "sic_code": "2836"},
]
var_clean = compute_climate_var(clean_portfolio, temperature_delta=2.0)
assert var_clean["climate_var"] < var_result["climate_var"], \
    f"Clean portfolio VaR ({var_clean['climate_var']:.4f}) must < fossil ({var_result['climate_var']:.4f})"
print(f"[OK] Climate VaR (clean-tech, 2°C): {var_clean['climate_var']:.4f} < fossil {var_result['climate_var']:.4f}")

# Higher temperature delta → higher VaR (linear relationship)
var_4c = compute_climate_var(fossil_portfolio, temperature_delta=4.0)
assert var_4c["climate_var"] > var_result["climate_var"], \
    f"4°C VaR ({var_4c['climate_var']:.4f}) should > 2°C VaR ({var_result['climate_var']:.4f})"
print(f"[OK] Climate VaR scales with temperature: 2°C={var_result['climate_var']:.4f} 4°C={var_4c['climate_var']:.4f}")

# Override climate_beta directly
custom_portfolio = [
    {"ticker": "TEST", "weight": 1.0, "climate_beta": 0.30},
]
var_custom = compute_climate_var(custom_portfolio, temperature_delta=2.0)
expected = 0.30 * 2.0  # weight=1, beta=0.30, delta=2
assert abs(var_custom["climate_var"] - expected) < 1e-9, \
    f"Custom beta should give exact result: {var_custom['climate_var']} != {expected}"
print(f"[OK] Climate VaR with explicit beta: {var_custom['climate_var']:.4f} (expected {expected:.4f})")

print("\n[PASS] dim_103: TCFD climate disclosure")
PYEOF
