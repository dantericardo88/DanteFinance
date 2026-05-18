#!/usr/bin/env bash
# dim_105: SDG impact scoring — SDGFramework, keyword scoring, 17-SDG metadata
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.sdg_impact_scoring import (
    SDGFramework,
)

# Test SDGFramework metadata
framework = SDGFramework()

# 17 SDGs
assert len(framework.SDG_METADATA) == 17, \
    f"Should have exactly 17 SDGs: {len(framework.SDG_METADATA)}"
for sdg_id in range(1, 18):
    assert sdg_id in framework.SDG_METADATA, f"SDG {sdg_id} should be in metadata"
    name, desc = framework.SDG_METADATA[sdg_id]
    assert name, f"SDG {sdg_id} should have a name"
    assert desc, f"SDG {sdg_id} should have a description"
print(f"[OK] SDGFramework: 17 SDGs, all IDs 1-17 present")

# Verify specific SDG names
assert "Climate Action" in framework.SDG_METADATA[13][0], \
    f"SDG 13 should be Climate Action: {framework.SDG_METADATA[13][0]}"
assert "No Poverty" in framework.SDG_METADATA[1][0], \
    f"SDG 1 should be No Poverty: {framework.SDG_METADATA[1][0]}"
assert "Gender Equality" in framework.SDG_METADATA[5][0], \
    f"SDG 5 should be Gender Equality: {framework.SDG_METADATA[5][0]}"
print(f"[OK] SDG names: SDG1=No Poverty SDG5=Gender Equality SDG13=Climate Action")

# Test SDG_POSITIVE_KEYWORDS
assert len(framework.SDG_POSITIVE_KEYWORDS) == 17, \
    f"Should have positive keywords for all 17 SDGs: {len(framework.SDG_POSITIVE_KEYWORDS)}"
total_pos = sum(len(v) for v in framework.SDG_POSITIVE_KEYWORDS.values())
assert total_pos >= 200, f"Expected >= 200 total positive keywords: {total_pos}"
print(f"[OK] SDG_POSITIVE_KEYWORDS: {total_pos} total keywords across 17 SDGs")

# SDG 7 (Clean Energy) should have renewable energy keywords
sdg7_pos = framework.SDG_POSITIVE_KEYWORDS.get(7, [])
assert "renewable energy" in sdg7_pos or "solar" in sdg7_pos, \
    f"SDG 7 should have clean energy keywords: {sdg7_pos[:5]}"
assert len(sdg7_pos) >= 10, f"SDG 7 should have >= 10 positive keywords: {len(sdg7_pos)}"
print(f"[OK] SDG 7 (Clean Energy): {len(sdg7_pos)} positive keywords")

# SDG 13 (Climate Action) should have climate keywords
sdg13_pos = framework.SDG_POSITIVE_KEYWORDS.get(13, [])
assert len(sdg13_pos) >= 8, f"SDG 13 should have >= 8 positive keywords: {len(sdg13_pos)}"
print(f"[OK] SDG 13 (Climate Action): {len(sdg13_pos)} positive keywords")

# Test SDG_NEGATIVE_KEYWORDS
assert len(framework.SDG_NEGATIVE_KEYWORDS) == 17, \
    f"Should have negative keywords for all 17 SDGs: {len(framework.SDG_NEGATIVE_KEYWORDS)}"
total_neg = sum(len(v) for v in framework.SDG_NEGATIVE_KEYWORDS.values())
assert total_neg >= 80, f"Expected >= 80 total negative keywords: {total_neg}"
print(f"[OK] SDG_NEGATIVE_KEYWORDS: {total_neg} total keywords across 17 SDGs")

# SDG 13 negative keywords should include climate-harm terms
sdg13_neg = framework.SDG_NEGATIVE_KEYWORDS.get(13, [])
climate_harm_terms = {"deforestation", "carbon fraud", "greenwashing climate",
                      "emission increase", "climate denial", "fossil fuel expansion",
                      "coal", "coal power"}
assert any(term in sdg13_neg for term in climate_harm_terms), \
    f"SDG 13 negative keywords should include climate-harm terms: {sdg13_neg[:8]}"
print(f"[OK] SDG 13 negative keywords include climate-harm terms: {sdg13_neg[:4]}")

# Test keyword scoring logic (pure computation)
def score_text_for_sdg(text: str, sdg_id: int) -> float:
    """Score text against a single SDG: positive hits increase, negative decrease."""
    text_lower = text.lower()
    pos_kws = framework.SDG_POSITIVE_KEYWORDS.get(sdg_id, [])
    neg_kws = framework.SDG_NEGATIVE_KEYWORDS.get(sdg_id, [])
    pos_hits = sum(1 for kw in pos_kws if kw in text_lower)
    neg_hits = sum(1 for kw in neg_kws if kw in text_lower)
    raw = pos_hits - (2 * neg_hits)   # negative hits penalized 2x
    # Normalize to [-10, +10] range
    return max(-10.0, min(10.0, raw))

# Strongly positive text for SDG 7 (clean energy company)
clean_energy_text = """
Our company operates renewable energy solar and wind energy projects globally.
We have committed to 100% clean energy by 2030 with a net zero energy transition strategy.
Our EV charging network is powered entirely by green power and photovoltaic arrays.
Energy efficiency is central to our decarbonization approach with smart grid technology.
"""
sdg7_score = score_text_for_sdg(clean_energy_text, 7)
assert sdg7_score > 0, f"Clean energy text should score positive for SDG 7: {sdg7_score}"
print(f"[OK] Clean energy text SDG 7 score: {sdg7_score:.1f} > 0")

# Negative text for SDG 13 (fossil fuel company)
# Use terms only in SDG 13 negatives: deforestation, fossil fuel expansion, coal power, climate denial
fossil_text = """
The company expanded coal power operations and engaged in fossil fuel expansion.
There is widespread concern about climate denial in the boardroom and deforestation
activities. Greenwashing climate claims have been disputed by regulators.
"""
sdg13_score = score_text_for_sdg(fossil_text, 13)
# Should be negative (neg hits 2x penalty > any pos hits)
assert sdg13_score < 0, f"Climate-harm text should score < 0 for SDG 13: {sdg13_score}"
print(f"[OK] Climate-harm text SDG 13 score: {sdg13_score:.1f} < 0")

# Irrelevant text should score near 0
irrelevant_text = "The company had strong revenue growth and expanded into new markets."
sdg7_irr = score_text_for_sdg(irrelevant_text, 7)
assert abs(sdg7_irr) < 5.0, f"Irrelevant text should score near 0 for SDG 7: {sdg7_irr}"
print(f"[OK] Irrelevant text SDG 7 score: {sdg7_irr:.1f} (near 0)")

# Test computing SDG alignment across all 17 for a healthcare company
healthcare_text = """
Our pharmaceutical company develops vaccines and medical devices for global health.
We invest in healthcare access for underserved communities and mental health programs.
Our telemedicine platform supports disease prevention and public health outcomes.
We focus on gender equality in our workforce with equal pay and inclusive hiring policies.
"""
all_sdg_scores = {sdg_id: score_text_for_sdg(healthcare_text, sdg_id)
                  for sdg_id in range(1, 18)}
# Healthcare company should score well on SDG 3 (Good Health)
assert all_sdg_scores[3] > 0, f"Healthcare company should score positive for SDG 3: {all_sdg_scores[3]}"
# Should score positive on SDG 5 (Gender Equality)
assert all_sdg_scores[5] > 0, f"Should score positive for SDG 5: {all_sdg_scores[5]}"
# All scores should be in [-10, +10]
for sdg_id, score in all_sdg_scores.items():
    assert -10 <= score <= 10, f"SDG {sdg_id} score out of bounds: {score}"
print(f"[OK] Healthcare company SDG scores: SDG3={all_sdg_scores[3]} SDG5={all_sdg_scores[5]}")
print(f"[OK] All 17 SDG scores in [-10, +10] range")

# Test portfolio-level SDG aggregation
# Weighted average across holdings
holdings_weights = {"healthcare_co": 0.40, "energy_co": 0.30, "tech_co": 0.30}
sdg3_scores = {"healthcare_co": 8.0, "energy_co": 2.0, "tech_co": 4.0}
portfolio_sdg3 = sum(holdings_weights[co] * sdg3_scores[co] for co in holdings_weights)
expected = 0.40 * 8.0 + 0.30 * 2.0 + 0.30 * 4.0
assert abs(portfolio_sdg3 - expected) < 1e-9, \
    f"Portfolio SDG3 should be {expected}: {portfolio_sdg3}"
print(f"[OK] Portfolio SDG3 score: {portfolio_sdg3:.2f} (weighted average)")

print("\n[PASS] dim_105: SDG impact scoring")
PYEOF
