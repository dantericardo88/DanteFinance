#!/bin/bash
# dim_049: global_macro_v3 -- 180-country, EM stress, CB divergence, country health score
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.global_macro_v3 import (
    WB_BASE,
    IMF_BASE,
    COUNTRY_UNIVERSE,
    DEVELOPED,
    EMERGING,
    _WB3,
    _NAMES,
    _POLICY_RATES,
    CountryScorer,
    CountryMacroSnapshot,
    compute_country_health_score,
    compute_em_stress_index,
    compute_cb_divergence_score,
    classify_market_type,
    get_em_stress_index,
    get_g10_cb_divergence,
    _G10_RATE_PATHS,
    _EM_SOVEREIGN_SPREADS,
)

# --- base constants ---
assert "worldbank.org" in WB_BASE, f"WB_BASE missing worldbank.org: {WB_BASE}"
assert "imf.org" in IMF_BASE, f"IMF_BASE missing imf.org: {IMF_BASE}"
print(f"[OK] WB_BASE and IMF_BASE constants present")

# --- universe coverage ---
assert len(COUNTRY_UNIVERSE) >= 40, f"Universe too small: {len(COUNTRY_UNIVERSE)}"
assert "US" in COUNTRY_UNIVERSE
assert "CN" in COUNTRY_UNIVERSE
assert "IN" in COUNTRY_UNIVERSE
assert "BR" in COUNTRY_UNIVERSE
assert "MX" in COUNTRY_UNIVERSE
print(f"[OK] COUNTRY_UNIVERSE covers {len(COUNTRY_UNIVERSE)} countries including US, CN, IN, BR, MX")

# --- WB3 code mapping ---
assert _WB3.get("US") == "USA"
assert _WB3.get("CN") == "CHN"
assert _WB3.get("IN") == "IND"
assert _WB3.get("GB") == "GBR"
print("[OK] WB3 ISO2->ISO3 code mapping verified (US->USA, CN->CHN, IN->IND)")

# --- country health score: pure math ---
# USA: moderate growth + low inflation + small CA deficit + moderate debt
us_score = compute_country_health_score(
    gdp_growth=2.5,
    inflation=3.2,
    current_account_pct_gdp=-3.0,
    debt_pct_gdp=122.0,
    inflation_target=2.0,
)
assert 0 < us_score <= 100, f"Score out of range: {us_score}"
print(f"[OK] CountryScore USA: {us_score}/100 (from hardcoded data)")

# CHN: high growth + low inflation + CA surplus + moderate debt
cn_score = compute_country_health_score(
    gdp_growth=4.8,
    inflation=0.5,
    current_account_pct_gdp=2.5,
    debt_pct_gdp=83.0,
    inflation_target=3.0,
)
assert 0 < cn_score <= 100
print(f"[OK] CountryScore CHN: {cn_score}/100")

# IND: strong growth + moderate inflation + CA deficit + moderate debt
ind_score = compute_country_health_score(
    gdp_growth=6.5,
    inflation=4.8,
    current_account_pct_gdp=-1.5,
    debt_pct_gdp=85.0,
    inflation_target=4.0,
)
assert 0 < ind_score <= 100
print(f"[OK] CountryScore IND: {ind_score}/100")

# BRA: moderate growth + high inflation + CA deficit + high debt
bra_score = compute_country_health_score(
    gdp_growth=2.0,
    inflation=5.5,
    current_account_pct_gdp=-2.5,
    debt_pct_gdp=92.0,
    inflation_target=3.0,
)
assert 0 < bra_score <= 100
print(f"[OK] CountryScore BRA: {bra_score}/100")

# MEX: moderate growth + moderate inflation + small CA deficit + moderate debt
mex_score = compute_country_health_score(
    gdp_growth=2.2,
    inflation=4.0,
    current_account_pct_gdp=-1.0,
    debt_pct_gdp=55.0,
    inflation_target=3.0,
)
assert 0 < mex_score <= 100
print(f"[OK] CountryScore MEX: {mex_score}/100")

# Country with high growth and low inflation should score higher than high inflation
high_growth_score = compute_country_health_score(3.5, 1.5, 1.0, 40.0)
high_inflation_score = compute_country_health_score(1.0, 12.0, -4.0, 90.0)
assert high_growth_score > high_inflation_score, (
    f"Better macro should score higher: {high_growth_score} vs {high_inflation_score}"
)
print(f"[OK] Country health scoring: better macro ({high_growth_score}) > worse macro ({high_inflation_score})")

# --- EM stress index: pure math ---
# Given 5 EM spread values, index is weighted (equal) average
em_spreads = [180.0, 420.0, 950.0, 220.0, 115.0]
stress_idx = compute_em_stress_index(em_spreads)
expected = sum(em_spreads) / len(em_spreads)
assert abs(stress_idx - expected) < 0.01, f"EM stress index: expected {expected:.2f}, got {stress_idx}"
print(f"[OK] EM stress index: avg of {em_spreads} = {stress_idx:.2f} bps")

# With all spreads = 200, index = 200
homogeneous = compute_em_stress_index([200.0] * 10)
assert abs(homogeneous - 200.0) < 0.01
print(f"[OK] EM stress index: uniform 200bps -> index={homogeneous:.2f}bps")

# Full index has 20 EM countries
full_idx = get_em_stress_index()
assert full_idx["n_countries"] >= 15
assert full_idx["em_stress_index_bps"] > 0
print(f"[OK] Full EM stress index: {full_idx['em_stress_index_bps']:.0f}bps over {full_idx['n_countries']} countries")

# --- CB divergence score: pure math ---
# Fed=+50bps, ECB=-25bps -> divergence score
simple_paths = {"FED": 50.0, "ECB": -25.0}
div_score = compute_cb_divergence_score(simple_paths)
assert div_score > 0, f"Divergence score should be positive: {div_score}"
print(f"[OK] CB divergence: FED=+50bps, ECB=-25bps -> divergence={div_score:.2f}bps")

# More divergence = higher score
high_div = compute_cb_divergence_score({"A": 100.0, "B": -100.0, "C": 0.0})
low_div = compute_cb_divergence_score({"A": 5.0, "B": 5.0, "C": 5.0})
assert high_div > low_div, f"High divergence should score higher: {high_div} vs {low_div}"
print(f"[OK] High divergence ({high_div:.1f}) > low divergence ({low_div:.1f}) as expected")

# G10 full divergence
g10_result = get_g10_cb_divergence()
assert g10_result["divergence_score_bps"] > 0
assert "most_hawkish" in g10_result
print(f"[OK] G10 CB divergence score: {g10_result['divergence_score_bps']:.1f}bps, "
      f"most hawkish: {g10_result['most_hawkish']['iso2']}, "
      f"most dovish: {g10_result['most_dovish']['iso2']}")

# --- DM/EM classification ---
assert classify_market_type("US") == "DM", "US should be DM"
assert classify_market_type("IN") == "EM", "IN should be EM"
assert classify_market_type("DE") == "DM", "DE should be DM"
assert classify_market_type("BR") == "EM", "BR should be EM"
assert classify_market_type("SG") == "DM", "SG should be DM"
print("[OK] DM/EM classification: USA->DM, IND->EM, DE->DM, BR->EM, SG->DM")

# --- CountryScorer with snapshot (pure math, no network) ---
scorer = CountryScorer()
snap = CountryMacroSnapshot(
    iso2="US",
    name="United States",
    gdp_growth_imf=2.5,
    inflation_fred=3.2,
    unemployment=3.7,
    debt_pct_gdp=122.0,
    fiscal_balance_imf=-6.5,
    current_account_imf=-3.0,
    composite_pmi=52.3,
    policy_rate=5.25,
    inflation_target=2.0,
    gdp_5y_avg=2.3,
)
sc = scorer.score(snap)
assert 0 < sc.total_score <= 100
assert sc.growth_score >= 0
assert sc.inflation_score >= 0
print(f"[OK] CountryScorer (pure math): US total={sc.total_score}/100, "
      f"growth={sc.growth_score}, inflation={sc.inflation_score}")

print("\n[PASS] dim_049: global_macro_v3 -- country health score, EM stress, CB divergence, DM/EM all verified")
PYEOF
