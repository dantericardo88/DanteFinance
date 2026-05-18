#!/usr/bin/env bash
# dim_102: ESG ratings — SIC sector mapping, score-to-grade logic, sector weights
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.esg_ratings_engine import (
    _SIC_TO_SECTOR,
    SECTOR_WEIGHTS,
    _score_to_grade_10,
    _count_keyword_hits,
    _bool_hit,
    _WEAPONS_SIC,
    _TOBACCO_SIC,
    _GAMBLING_SIC,
    _FOSSIL_SIC,
    CLIMATE_TRANSITION_RISK,
    GovernanceScore,
    EnvironmentalScore,
    SocialScore,
)

# Test _SIC_TO_SECTOR mapping
assert len(_SIC_TO_SECTOR) >= 30, f"Expected >= 30 SIC codes: {len(_SIC_TO_SECTOR)}"
assert "7372" in _SIC_TO_SECTOR, "SIC 7372 (Software) should be mapped"
assert _SIC_TO_SECTOR["7372"] == "information_technology"
assert "4911" in _SIC_TO_SECTOR, "SIC 4911 (Electric Utilities) should be mapped"
assert _SIC_TO_SECTOR["4911"] == "utilities"
assert "2836" in _SIC_TO_SECTOR, "SIC 2836 (BioTech) should be mapped"
assert _SIC_TO_SECTOR["2836"] == "health_care"
print(f"[OK] _SIC_TO_SECTOR: {len(_SIC_TO_SECTOR)} mappings, 7372=information_technology")

# Test SECTOR_WEIGHTS
assert "default" in SECTOR_WEIGHTS, "Should have 'default' sector weights"
assert "energy" in SECTOR_WEIGHTS, "Should have 'energy' sector weights"
assert "information_technology" in SECTOR_WEIGHTS, "Should have IT sector weights"

for sector, weights in SECTOR_WEIGHTS.items():
    assert "E" in weights and "S" in weights and "G" in weights, \
        f"Sector '{sector}' missing E/S/G keys"
    total = weights["E"] + weights["S"] + weights["G"]
    assert abs(total - 1.0) < 1e-9, \
        f"Sector '{sector}' E+S+G should sum to 1.0: {total}"
print(f"[OK] SECTOR_WEIGHTS: {len(SECTOR_WEIGHTS)} sectors, all E+S+G=1.0")

# Energy should be E-heavy (pollution-intensive)
energy_w = SECTOR_WEIGHTS["energy"]
assert energy_w["E"] >= 0.45, f"Energy should be E-heavy: E={energy_w['E']}"
# Financials should be G-heavy (governance matters most)
fin_w = SECTOR_WEIGHTS["financials"]
assert fin_w["G"] >= 0.45, f"Financials should be G-heavy: G={fin_w['G']}"
# HealthCare should be S-heavy (patient/worker outcomes)
hc_w = SECTOR_WEIGHTS["health_care"]
assert hc_w["S"] >= 0.45, f"Healthcare should be S-heavy: S={hc_w['S']}"
print(f"[OK] Sector weight logic: energy E={energy_w['E']} financials G={fin_w['G']} health_care S={hc_w['S']}")

# Test _score_to_grade_10
assert _score_to_grade_10(9.5) == "A+"
assert _score_to_grade_10(9.0) == "A+"
assert _score_to_grade_10(8.5) == "A"
assert _score_to_grade_10(8.0) == "A"
assert _score_to_grade_10(7.5) == "A-"
assert _score_to_grade_10(7.0) == "A-"
assert _score_to_grade_10(6.5) == "BBB"
assert _score_to_grade_10(6.0) == "BBB"
assert _score_to_grade_10(5.5) == "BB"
assert _score_to_grade_10(5.0) == "BB"
assert _score_to_grade_10(4.5) == "B"
assert _score_to_grade_10(4.0) == "B"
assert _score_to_grade_10(3.5) == "B-"
assert _score_to_grade_10(3.0) == "B-"
assert _score_to_grade_10(2.0) == "CCC"
assert _score_to_grade_10(0.0) == "CCC"
print("[OK] _score_to_grade_10: 9.5->A+ 8.0->A 7.0->A- 6.0->BBB 5.0->BB 4.0->B 3.0->B- 2.0->CCC")

# Verify monotonic ordering
scores = [9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.0]
grades = [_score_to_grade_10(s) for s in scores]
grade_order = ["A+", "A", "A-", "BBB", "BB", "B", "B-", "CCC"]
assert grades == grade_order, f"Grade ordering should be monotonic: {grades}"
print(f"[OK] Score-to-grade monotonicity: {list(zip(scores, grades))}")

# Test _count_keyword_hits
text = "The company has reduced carbon emissions and committed to net zero targets. Carbon disclosure was submitted to CDP."
hits = _count_keyword_hits(text, ["carbon", "net zero", "CDP"])
assert hits >= 3, f"Should find at least 3 keyword hits: {hits}"
assert _count_keyword_hits(text, ["carbon"]) >= 2, "Should find 2+ occurrences of 'carbon'"
assert _count_keyword_hits("clean text", ["xyz_nonexistent"]) == 0, "Non-matching keyword -> 0"
print(f"[OK] _count_keyword_hits: {hits} hits for carbon/net zero/CDP in sample text")

# Test _bool_hit
esg_text = "The board includes independent directors and diverse representation."
assert _bool_hit(esg_text, ["independent director"]) is True, "Should find 'independent director'"
assert _bool_hit(esg_text, ["classified board"]) is False, "Should not find 'classified board'"
assert _bool_hit("", ["carbon disclosure"]) is False, "Empty text -> False"
print(f"[OK] _bool_hit: pattern matching works correctly")

# Test negative screens
assert "2911" in _FOSSIL_SIC, "Oil refining SIC 2911 should be in fossil fuels"
assert "1311" in _FOSSIL_SIC, "Crude oil SIC 1311 should be in fossil fuels"
assert "2100" in _TOBACCO_SIC, "Tobacco SIC 2100 should be in tobacco"
assert "3761" in _WEAPONS_SIC, "Defense SIC 3761 should be in weapons"
print(f"[OK] Negative screens: fossil={len(_FOSSIL_SIC)} tobacco={len(_TOBACCO_SIC)} weapons={len(_WEAPONS_SIC)}")

# Test CLIMATE_TRANSITION_RISK
assert "energy" in CLIMATE_TRANSITION_RISK, "Energy should have climate transition risk"
assert CLIMATE_TRANSITION_RISK["energy"] == "HIGH", f"Energy should be HIGH risk: {CLIMATE_TRANSITION_RISK['energy']}"
assert CLIMATE_TRANSITION_RISK["information_technology"] == "LOW", \
    f"IT should be LOW risk: {CLIMATE_TRANSITION_RISK['information_technology']}"
print(f"[OK] CLIMATE_TRANSITION_RISK: energy=HIGH IT=LOW")

# Test composite ESG score computation (mirrors ESGRatingsEngine.get_composite)
def compute_composite(e_score: float, s_score: float, g_score: float,
                       sector: str = "default") -> float:
    weights = SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS["default"])
    return weights["E"] * e_score + weights["S"] * s_score + weights["G"] * g_score

# Energy company: E matters most
energy_composite = compute_composite(8.0, 5.0, 6.0, "energy")
default_composite = compute_composite(8.0, 5.0, 6.0, "default")
# Energy weights E more heavily, so similar E-score -> energy_composite should reflect E-weight
assert 0 < energy_composite <= 10, f"Composite should be in (0,10]: {energy_composite}"
print(f"[OK] ESG composite (energy): E=8.0 S=5.0 G=6.0 -> {energy_composite:.2f}")
print(f"[OK] ESG composite (default): E=8.0 S=5.0 G=6.0 -> {default_composite:.2f}")

# Test GovernanceScore model
gov = GovernanceScore(
    ticker="AAPL",
    score=7.5,
    grade=_score_to_grade_10(7.5),
    board_independence_pct=0.80,
    board_diversity_pct=0.40,
    ceo_duality=False,
    say_on_pay_pct=0.92,
    has_poison_pill=False,
    has_classified_board=False,
    ceo_pay_ratio=254.0,
)
assert gov.grade == "A-", f"Score 7.5 should be grade A-: {gov.grade}"
assert gov.score == 7.5
assert gov.board_independence_pct == 0.80
print(f"[OK] GovernanceScore: ticker={gov.ticker} score={gov.score} grade={gov.grade}")

# Test EnvironmentalScore model
env = EnvironmentalScore(
    ticker="XOM",
    score=3.5,
    grade=_score_to_grade_10(3.5),
    has_carbon_disclosure=True,
    has_net_zero_target=False,
    has_science_based_target=False,
    has_environmental_litigation=True,
    sector="energy",
    keyword_hits={"carbon": 45, "emissions": 28, "climate": 15},
)
assert env.grade == "B-", f"Score 3.5 should be grade B-: {env.grade}"
assert env.has_environmental_litigation is True
print(f"[OK] EnvironmentalScore: ticker={env.ticker} score={env.score} grade={env.grade}")

# ------------------------------------------------------------------
# Wave-9 additions: compute_esg_momentum, detect_greenwashing_risk,
# compute_controversy_adjusted_esg
# ------------------------------------------------------------------
from sentinel.sfe.esg_ratings_engine import (
    compute_esg_momentum,
    detect_greenwashing_risk,
    compute_controversy_adjusted_esg,
)

# Test compute_esg_momentum
momentum = compute_esg_momentum(7.5, 6.0)
assert momentum["delta"] == 1.5, f"Delta should be 1.5: {momentum['delta']}"
assert momentum["momentum_signal"] == "positive", \
    f"Positive delta -> positive signal: {momentum['momentum_signal']}"
print(f"[OK] compute_esg_momentum: delta={momentum['delta']} signal={momentum['momentum_signal']}")

momentum_neg = compute_esg_momentum(5.0, 7.0)
assert momentum_neg["delta"] == -2.0, f"Negative delta: {momentum_neg['delta']}"
assert momentum_neg["momentum_signal"] == "negative"
print(f"[OK] compute_esg_momentum (negative): delta={momentum_neg['delta']}")

momentum_flat = compute_esg_momentum(6.0, 6.0)
assert momentum_flat["delta"] == 0.0
assert momentum_flat["momentum_signal"] == "neutral"
print(f"[OK] compute_esg_momentum (flat): signal={momentum_flat['momentum_signal']}")

# Test detect_greenwashing_risk
# gap > 15 pts → greenwashing flag
gw = detect_greenwashing_risk(75.0, 55.0)
assert gw["gap"] == 20.0, f"Gap should be 20: {gw['gap']}"
assert gw["greenwashing_risk"] is True, f"Gap>15 -> greenwashing risk: {gw['greenwashing_risk']}"
assert gw["risk_level"] == "moderate", f"20pt gap -> moderate: {gw['risk_level']}"
print(f"[OK] detect_greenwashing_risk: gap={gw['gap']} risk={gw['greenwashing_risk']} level={gw['risk_level']}")

# gap <= 15 pts → no greenwashing
no_gw = detect_greenwashing_risk(60.0, 55.0)
assert no_gw["greenwashing_risk"] is False, f"Gap<=15 -> no greenwashing: {no_gw}"
assert no_gw["risk_level"] in ("low", "none"), f"5pt gap -> low/none: {no_gw['risk_level']}"
print(f"[OK] detect_greenwashing_risk (safe): gap={no_gw['gap']} risk={no_gw['greenwashing_risk']}")

# Test compute_controversy_adjusted_esg
# base_score=80, num_controversies=3, penalty=3*0.05=0.15 -> 80*(1-0.15)=68.0
adjusted = compute_controversy_adjusted_esg(80.0, 3)
expected_penalty = 3 * 0.05
expected_score = 80.0 * (1.0 - expected_penalty)
assert abs(adjusted["controversy_penalty"] - expected_penalty) < 1e-9, \
    f"Penalty should be {expected_penalty}: {adjusted['controversy_penalty']}"
assert abs(adjusted["controversy_adjusted"] - expected_score) < 1e-9, \
    f"Adjusted score should be {expected_score}: {adjusted['controversy_adjusted']}"
print(f"[OK] compute_controversy_adjusted_esg: 3 controversies -> penalty={adjusted['controversy_penalty']:.2f} adjusted={adjusted['controversy_adjusted']:.2f}")

# Max penalty capped at 0.30: 10 controversies * 0.05 = 0.50 -> capped at 0.30
capped = compute_controversy_adjusted_esg(100.0, 10)
assert capped["controversy_penalty"] == 0.30, \
    f"Penalty capped at 0.30: {capped['controversy_penalty']}"
assert abs(capped["controversy_adjusted"] - 70.0) < 1e-9, \
    f"100*(1-0.30)=70: {capped['controversy_adjusted']}"
print(f"[OK] compute_controversy_adjusted_esg (cap): 10 controversies -> penalty={capped['controversy_penalty']} adjusted={capped['controversy_adjusted']}")

print("\n[PASS] dim_102: ESG ratings")
PYEOF
