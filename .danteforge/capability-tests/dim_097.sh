#!/usr/bin/env bash
# dim_097: Private company profiles — SIC sector mapping, constants, pure logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import re

from sentinel.sfe.private_company_profiles import (
    _SIC_SECTORS,
    _EXEMPTION_LABELS,
    _HOT_SICS,
    _REVENUE_MULTIPLES,
    _TOP_VC_STATES,
    _REVENUE_MIDPOINTS,
    _clean_company_name,
    _strip_ns,
    _float_or_none,
    PrivateCompanyProfile,
    FinancingRound,
    ExecutiveProfile,
    SectorTrend,
)

# Test _SIC_SECTORS mapping
assert len(_SIC_SECTORS) >= 15, f"Expected >= 15 SIC sectors: {len(_SIC_SECTORS)}"
assert "7372" in _SIC_SECTORS, "SIC 7372 (Software) should be mapped"
assert _SIC_SECTORS["7372"] == "Software", f"SIC 7372 should be Software: {_SIC_SECTORS['7372']}"
assert "3674" in _SIC_SECTORS, "SIC 3674 (Semiconductors) should be mapped"
assert "2836" in _SIC_SECTORS, "SIC 2836 (BioTech) should be mapped"
print(f"[OK] _SIC_SECTORS: {len(_SIC_SECTORS)} sectors, 7372=Software 3674=Semiconductors")

# Test _EXEMPTION_LABELS
assert len(_EXEMPTION_LABELS) >= 8, f"Expected >= 8 exemptions: {len(_EXEMPTION_LABELS)}"
assert "06b" in _EXEMPTION_LABELS, "Rule 506(b) should be mapped"
assert "06c" in _EXEMPTION_LABELS, "Rule 506(c) should be mapped"
assert "3C1" in _EXEMPTION_LABELS, "3(c)(1) should be mapped"
assert "CF" in _EXEMPTION_LABELS, "Regulation Crowdfunding should be mapped"
print(f"[OK] _EXEMPTION_LABELS: {len(_EXEMPTION_LABELS)} exemption types")

# Test _HOT_SICS
assert len(_HOT_SICS) >= 8, f"Expected >= 8 hot SICs: {len(_HOT_SICS)}"
assert "7372" in _HOT_SICS, "Software SIC should be hot"
assert "3674" in _HOT_SICS, "Semiconductor SIC should be hot"
# All hot SICs should be in _SIC_SECTORS
for sic in _HOT_SICS:
    assert sic in _SIC_SECTORS, f"Hot SIC {sic} should be in _SIC_SECTORS"
print(f"[OK] _HOT_SICS: {len(_HOT_SICS)} sectors, all present in _SIC_SECTORS")

# Test _REVENUE_MULTIPLES
assert "Software" in _REVENUE_MULTIPLES, "Software should have revenue multiples"
assert "BioTech" in _REVENUE_MULTIPLES, "BioTech should have revenue multiples"
for sector, mults in _REVENUE_MULTIPLES.items():
    assert "low" in mults and "mid" in mults and "high" in mults, \
        f"Sector {sector} should have low/mid/high multiples"
    assert mults["low"] <= mults["mid"] <= mults["high"], \
        f"Multiples should be ordered low <= mid <= high for {sector}"
sw = _REVENUE_MULTIPLES["Software"]
assert sw["low"] >= 3.0, f"Software low multiple should be substantial: {sw['low']}"
assert sw["high"] > sw["low"], f"Software high > low: {sw}"
print(f"[OK] _REVENUE_MULTIPLES: {len(_REVENUE_MULTIPLES)} sectors, Software low={sw['low']}x mid={sw['mid']}x high={sw['high']}x")

# Test _TOP_VC_STATES
assert len(_TOP_VC_STATES) >= 8, f"Expected >= 8 VC states: {len(_TOP_VC_STATES)}"
assert "CA" in _TOP_VC_STATES, "California should be a top VC state"
assert "NY" in _TOP_VC_STATES, "New York should be a top VC state"
assert "MA" in _TOP_VC_STATES, "Massachusetts should be a top VC state"
print(f"[OK] _TOP_VC_STATES: {len(_TOP_VC_STATES)} states including CA, NY, MA")

# Test _REVENUE_MIDPOINTS
assert "No Revenues" in _REVENUE_MIDPOINTS, "Should have 'No Revenues' category"
assert _REVENUE_MIDPOINTS["No Revenues"] == 0, "No revenues → $0"
# Verify midpoints are in increasing order
sorted_keys = sorted(_REVENUE_MIDPOINTS.items(), key=lambda x: x[1])
# The last entry should be the largest
assert sorted_keys[-1][1] >= 1_000_000_000, f"Top category should be >= $1B: {sorted_keys[-1]}"
print(f"[OK] _REVENUE_MIDPOINTS: {len(_REVENUE_MIDPOINTS)} bands, max={sorted_keys[-1][1]:,.0f}")

# Test _clean_company_name
assert _clean_company_name("Apple Inc.") in ("Apple", "Apple Inc"), \
    f"Should clean 'Inc.': {_clean_company_name('Apple Inc.')}"
assert _clean_company_name("TechVentures LLC") in ("TechVentures", "Tech"), \
    f"Should clean 'LLC': {_clean_company_name('TechVentures LLC')}"
# Corp should also be cleaned
clean = _clean_company_name("Acme Corp")
assert "Corp" not in clean, f"Should clean Corp: '{clean}'"
print(f"[OK] _clean_company_name: 'Apple Inc.' -> '{_clean_company_name('Apple Inc.')}', 'Acme Corp' -> '{_clean_company_name('Acme Corp')}'")

# Test _strip_ns
ns_tag = "{http://www.sec.gov/XMLSchema/document}issuerName"
stripped = _strip_ns(ns_tag)
assert stripped == "issuerName", f"Should strip namespace: '{stripped}'"
no_ns_tag = "issuerName"
assert _strip_ns(no_ns_tag) == "issuerName", "Non-namespaced tag should pass through"
print(f"[OK] _strip_ns: '{ns_tag}' -> '{stripped}'")

# Test _float_or_none
assert _float_or_none("1000000") == 1000000.0
assert _float_or_none("1,000,000") == 1000000.0
assert _float_or_none("$1,500,000") == 1500000.0
assert _float_or_none("invalid") is None
assert _float_or_none("") is None
assert _float_or_none(None) is None
print(f"[OK] _float_or_none: '$1,500,000'={_float_or_none('$1,500,000'):,.0f} 'invalid'=None")

# Test PrivateCompanyProfile Pydantic model
profile = PrivateCompanyProfile(
    company_name="TechStartup Inc.",
    clean_name="TechStartup",
    cik="0001234567",
    issuer_state="CA",
    issuer_sic="7372",
    sector="Software",
    revenue_range="1-1000000",
    revenue_estimate=500_000,
    total_raised=5_000_000,
    round_count=2,
    is_hot_sector=True,
    is_unicorn_candidate=False,
)
assert profile.company_name == "TechStartup Inc."
assert profile.cik == "0001234567"
assert profile.sector == "Software"
assert profile.is_hot_sector is True
assert profile.total_raised == 5_000_000
print(f"[OK] PrivateCompanyProfile: {profile.company_name} sector={profile.sector} raised=${profile.total_raised:,.0f}")

# Test FinancingRound
round1 = FinancingRound(
    accession_number="0001234567-24-000001",
    filed_date="2024-03-15",
    total_offering_amount=5_000_000.0,
    amount_sold=4_200_000.0,
    federal_exemptions=["06b"],
    investor_count=12,
    round_label="Series A",
)
assert round1.round_label == "Series A"
assert round1.amount_sold < round1.total_offering_amount
assert "06b" in round1.federal_exemptions
print(f"[OK] FinancingRound: round_label={round1.round_label} amount=${round1.amount_sold:,.0f}")

# Test ExecutiveProfile
exec1 = ExecutiveProfile(
    name="Jane Smith",
    roles=["CEO", "Director"],
    companies=["TechStartup Inc.", "PrevVenture LLC"],
    filing_count=5,
    is_serial_entrepreneur=True,
    reputation_score=7.5,
)
assert exec1.is_serial_entrepreneur is True
assert 0 <= exec1.reputation_score <= 10
assert "CEO" in exec1.roles
print(f"[OK] ExecutiveProfile: {exec1.name} serial_entrepreneur={exec1.is_serial_entrepreneur} score={exec1.reputation_score}")

# Test SIC → sector lookup logic
def get_sector(sic_code: str) -> str:
    return _SIC_SECTORS.get(sic_code, "Other")

assert get_sector("7372") == "Software"
assert get_sector("3674") == "Semiconductors"
assert get_sector("9999") == "Other"  # unknown SIC
print(f"[OK] SIC lookup: 7372=Software 3674=Semiconductors 9999=Other")

# Test valuation estimation logic (mirrors PrivateMarketComps)
def estimate_valuation(revenue: float, sector: str) -> dict:
    mults = _REVENUE_MULTIPLES.get(sector, _REVENUE_MULTIPLES["Other"])
    return {
        "low": revenue * mults["low"],
        "mid": revenue * mults["mid"],
        "high": revenue * mults["high"],
    }

sw_val = estimate_valuation(10_000_000, "Software")
assert sw_val["low"] < sw_val["mid"] < sw_val["high"]
assert sw_val["low"] >= 40_000_000, f"Software $10M ARR low valuation: ${sw_val['low']:,.0f}"
print(f"[OK] Valuation estimate (Software $10M ARR): low=${sw_val['low']:,.0f} mid=${sw_val['mid']:,.0f} high=${sw_val['high']:,.0f}")

# ------------------------------------------------------------------
# Wave-9 additions: compute_implied_valuation, estimate_growth_stage,
# compute_acquisition_likelihood_score
# ------------------------------------------------------------------
from sentinel.sfe.private_company_profiles import (
    compute_implied_valuation,
    estimate_growth_stage,
    compute_acquisition_likelihood_score,
)

# Test compute_implied_valuation — SaaS: 3× multiple
saas_val = compute_implied_valuation(10_000_000, "Software")
assert saas_val["revenue_multiple"] == 3.0, \
    f"SaaS multiple should be 3.0: {saas_val['revenue_multiple']}"
assert saas_val["implied_ev"] == 30_000_000.0, \
    f"SaaS implied EV should be 30M: {saas_val['implied_ev']}"
print(f"[OK] compute_implied_valuation SaaS: {saas_val['revenue_multiple']}x -> EV=${saas_val['implied_ev']:,.0f}")

# Industrial: 2× multiple
ind_val = compute_implied_valuation(5_000_000, "Industrial")
assert ind_val["revenue_multiple"] == 2.0, \
    f"Industrial multiple should be 2.0: {ind_val['revenue_multiple']}"
assert ind_val["implied_ev"] == 10_000_000.0, \
    f"Industrial implied EV should be 10M: {ind_val['implied_ev']}"
print(f"[OK] compute_implied_valuation Industrial: {ind_val['revenue_multiple']}x -> EV=${ind_val['implied_ev']:,.0f}")

# Retail: 1.5× multiple
retail_val = compute_implied_valuation(8_000_000, "Retail")
assert retail_val["revenue_multiple"] == 1.5, \
    f"Retail multiple should be 1.5: {retail_val['revenue_multiple']}"
assert abs(retail_val["implied_ev"] - 12_000_000.0) < 1.0, \
    f"Retail implied EV should be 12M: {retail_val['implied_ev']}"
print(f"[OK] compute_implied_valuation Retail: {retail_val['revenue_multiple']}x -> EV=${retail_val['implied_ev']:,.0f}")

# Test estimate_growth_stage
assert estimate_growth_stage(0) == "seed",  f"$0 revenue -> seed"
assert estimate_growth_stage(100_000) == "early", f"$100K revenue -> early"
assert estimate_growth_stage(4_999_999) == "early", f"<$5M -> early"
assert estimate_growth_stage(5_000_000) == "growth", f"$5M -> growth"
assert estimate_growth_stage(49_999_999) == "growth", f"<$50M -> growth"
assert estimate_growth_stage(50_000_000) == "late", f"$50M -> late"
assert estimate_growth_stage(200_000_000) == "late", f"$200M -> late"
print("[OK] estimate_growth_stage: 0->seed, <5M->early, <50M->growth, >=50M->late")

# Test compute_acquisition_likelihood_score
# Small + strong FCF + strategic sector = 30+30+40 = 100
high_score = compute_acquisition_likelihood_score("small", True, "Software")
assert high_score == 100.0, f"Max score should be 100: {high_score}"
print(f"[OK] compute_acquisition_likelihood_score (small/FCF/tech): {high_score}")

# Small + no FCF + non-strategic = 30
small_no_fcf = compute_acquisition_likelihood_score("small", False, "Other")
assert small_no_fcf == 30.0, f"Small no-FCF non-strategic: {small_no_fcf}"
print(f"[OK] compute_acquisition_likelihood_score (small/no-FCF/other): {small_no_fcf}")

# Large + FCF + strategic sector = 10+30+40 = 80
large_score = compute_acquisition_likelihood_score("large", True, "Healthcare")
assert large_score == 80.0, f"Large FCF strategic: {large_score}"
print(f"[OK] compute_acquisition_likelihood_score (large/FCF/healthcare): {large_score}")

# ------------------------------------------------------------------
# Wave-35 Crusade additions: Berkus / Scorecard / VC Method / Composite
# valuations for pre-revenue + early-stage private companies.
# This block exists because the prior Wave 3 LIED about implementing
# these — the harsh audit caught it. These asserts force honesty.
# ------------------------------------------------------------------
from sentinel.sfe.private_company_profiles import (
    berkus_valuation,
    scorecard_valuation,
    vc_method_valuation,
    composite_valuation,
)

assert callable(berkus_valuation),     "berkus_valuation must exist"
assert callable(scorecard_valuation),  "scorecard_valuation must exist"
assert callable(vc_method_valuation),  "vc_method_valuation must exist"
assert callable(composite_valuation),  "composite_valuation must exist"

# Berkus — 5 factors x $500k = $2.5M ceiling
b_max = berkus_valuation(500_000, 500_000, 500_000, 500_000, 500_000)
assert b_max["method"] == "berkus"
assert b_max["mid"] == 2_500_000, f"Berkus ceiling should be $2.5M: {b_max['mid']}"
assert b_max["ceiling"] == 2_500_000.0
assert set(b_max["components"].keys()) == {
    "sound_idea", "prototype", "mgmt_quality",
    "strategic_relationships", "product_rollout",
}, "Berkus must expose all 5 components"
b_default = berkus_valuation()
assert b_default["mid"] == 500_000, "Default sound_idea = $500k baseline"
print(f"[OK] berkus_valuation: ceiling=${b_max['mid']:,.0f}, default=${b_default['mid']:,.0f}")

# Scorecard — US seed Software peer-median == $2.5M * 1.30 = $3.25M
sc = scorecard_valuation("AcmeAI", sector="Software", region="US", stage="seed")
assert sc["method"] == "scorecard"
assert sc["comparables"]["regional_baseline"] == 2_500_000.0, "US seed = $2.5M"
assert abs(sc["mid"] - 3_250_000.0) < 1.0, f"Software seed peer-median: {sc['mid']}"
assert "adjustments" in sc and len(sc["adjustments"]) == 7, "7 scorecard factors"
sc_a = scorecard_valuation("AcmeAI", sector="Software", region="US", stage="series_a")
assert sc_a["comparables"]["regional_baseline"] == 8_000_000.0, "US Series A = $8M"
print(f"[OK] scorecard_valuation: seed mid=${sc['mid']:,.0f} series_a mid=${sc_a['mid']:,.0f}")

# VC Method — Exit $500M, 30% IRR, 5 years → post-money ≈ $134.66M
vc = vc_method_valuation(
    projected_exit_revenue=100_000_000,
    projected_exit_multiple=5,
    years_to_exit=5,
    target_irr=0.30,
    dilution_to_exit=0.20,
    investment_amount=5_000_000,
)
assert vc["method"] == "vc_method"
assert vc["exit_value"] == 500_000_000, f"Exit value: {vc['exit_value']}"
expected_post = 500_000_000 / (1.30**5)
assert abs(vc["post_money"] - expected_post) < 1.0, f"Post-money: {vc['post_money']}"
assert vc["pre_money"] == round(vc["post_money"] - 5_000_000, 2)
assert 0 < vc["ownership_required"] < 1.0, "Ownership in (0,1)"
print(f"[OK] vc_method_valuation: exit=${vc['exit_value']:,.0f} post=${vc['post_money']:,.0f} ownership={vc['ownership_required']:.2%}")

# Composite — pre-revenue: Berkus 60% / Scorecard 40%
comp_pre = composite_valuation("PreRevCo", sector="Software", revenue=None, stage="seed")
assert comp_pre["stage_bucket"] == "pre_revenue"
assert comp_pre["weights"]["berkus"] == 0.60
assert comp_pre["weights"]["scorecard"] == 0.40
assert comp_pre["weights"]["vc"] == 0.0

# Composite — early stage: VC method active
comp_early = composite_valuation("EarlyCo", sector="Software", revenue=2_000_000, stage="seed")
assert comp_early["stage_bucket"] == "early"
assert comp_early["weights"]["vc"] > 0
assert comp_early["components"]["vc_method"] is not None

# Composite — growth: VC dominates
comp_growth = composite_valuation("GrowthCo", sector="Software", revenue=20_000_000, stage="series_a")
assert comp_growth["stage_bucket"] == "growth"
assert comp_growth["weights"]["vc"] >= 0.50

# Composite — late: comps significant
comp_late = composite_valuation("LateCo", sector="Software", revenue=100_000_000, stage="series_b")
assert comp_late["stage_bucket"] == "late"
assert comp_late["weights"]["comps"] >= 0.30

print(f"[OK] composite_valuation: pre=${comp_pre['valuation_mid']:,.0f} early=${comp_early['valuation_mid']:,.0f} growth=${comp_growth['valuation_mid']:,.0f} late=${comp_late['valuation_mid']:,.0f}")

# Honesty grep — make absolutely sure the 4 function defs exist in source.
import inspect, sentinel.sfe.private_company_profiles as mod
src = inspect.getsource(mod)
for fn_name in ("berkus_valuation", "scorecard_valuation",
                "vc_method_valuation", "composite_valuation"):
    assert f"def {fn_name}(" in src, f"def {fn_name}( missing from source"
print("[OK] honesty grep: all 4 function defs present in source")

print("\n[PASS] dim_097: Private company profiles")
PYEOF
