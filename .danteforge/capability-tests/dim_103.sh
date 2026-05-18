#!/usr/bin/env bash
# dim_103: TCFD climate disclosure — pillar structure, patterns, models
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
)

# Test TCFD_DISCLOSURES structure — 11 recommended disclosures across 4 pillars
assert len(TCFD_DISCLOSURES) == 11, f"TCFD has 11 recommended disclosures: {len(TCFD_DISCLOSURES)}"
pillars_found = {d[0] for d in TCFD_DISCLOSURES}
for pillar in ["governance", "strategy", "risk_mgmt", "metrics"]:
    assert pillar in pillars_found, f"TCFD should have '{pillar}' pillar"
gov_disclosures = [d for d in TCFD_DISCLOSURES if d[0] == "governance"]
strat_disclosures = [d for d in TCFD_DISCLOSURES if d[0] == "strategy"]
risk_disclosures = [d for d in TCFD_DISCLOSURES if d[0] == "risk_mgmt"]
metrics_disclosures = [d for d in TCFD_DISCLOSURES if d[0] == "metrics"]
assert len(gov_disclosures) == 2, f"Governance should have 2 disclosures: {len(gov_disclosures)}"
assert len(strat_disclosures) == 3, f"Strategy should have 3 disclosures: {len(strat_disclosures)}"
assert len(risk_disclosures) == 3, f"Risk mgmt should have 3 disclosures: {len(risk_disclosures)}"
assert len(metrics_disclosures) == 3, f"Metrics should have 3 disclosures: {len(metrics_disclosures)}"
print(f"[OK] TCFD_DISCLOSURES: 11 disclosures across 4 pillars (gov={len(gov_disclosures)} strat={len(strat_disclosures)} risk={len(risk_disclosures)} metrics={len(metrics_disclosures)})")

# Verify disclosure keys include key recommendations
disc_keys = {d[1] for d in TCFD_DISCLOSURES}
assert "board_oversight" in disc_keys, "Should have board_oversight disclosure"
assert "scenario_analysis" in disc_keys, "Should have scenario_analysis disclosure"
assert "ghg_scope1_2" in disc_keys, "Should have ghg_scope1_2 disclosure"
assert "targets" in disc_keys, "Should have targets disclosure"
print(f"[OK] Key disclosure keys present: board_oversight, scenario_analysis, ghg_scope1_2, targets")

# Test TCFDPillarScorer — regex-based classification
scorer = TCFDPillarScorer()

# Good governance text
gov_text = """
The Board of Directors committee oversees and monitors climate risk.
The Chief Sustainability Officer provides management's role in assessing climate risks
and the sustainability committee of management reviews progress quarterly.
"""
gov_result = scorer.score_governance(gov_text)
assert gov_result.pillar == "governance", f"Pillar should be governance: {gov_result.pillar}"
assert 0 <= gov_result.score <= 100, f"Pillar score should be in [0,100]: {gov_result.score}"
print(f"[OK] score_governance: score={gov_result.score:.1f} disclosures_found={gov_result.disclosures_found}")

# Good strategy text
strat_text = """
The company conducts climate scenario analysis including 1.5 degree C scenarios and 2 degree
Paris Agreement pathways. Climate risk and opportunity identification considers short, medium
and long-term horizons. Climate risk has a material impact on business strategy and financial planning.
"""
strat_result = scorer.score_strategy(strat_text)
assert strat_result.pillar == "strategy", f"Pillar should be strategy: {strat_result.pillar}"
assert strat_result.score > 0, f"Good strategy text should score > 0: {strat_result.score}"
print(f"[OK] score_strategy: score={strat_result.score:.1f} disclosures_found={strat_result.disclosures_found}")

# Poor strategy text should score low
poor_strat = "Revenue grew 15% this year driven by product innovation and market expansion."
poor_strat_result = scorer.score_strategy(poor_strat)
assert poor_strat_result.score <= strat_result.score, \
    f"Poor strategy should score <= good: {poor_strat_result.score} vs {strat_result.score}"
print(f"[OK] Poor strategy scores lower ({poor_strat_result.score:.3f}) than good ({strat_result.score:.3f})")

# Good risk management text
risk_text = """
Climate risk identification and assessment process is integrated into enterprise risk management (ERM).
The company manages and mitigates climate risk through a formal climate risk management framework.
Physical and transition risk are identified in our climate risk register.
"""
risk_result = scorer.score_risk_mgmt(risk_text)
assert risk_result.pillar == "risk_mgmt", f"Pillar should be risk_mgmt: {risk_result.pillar}"
assert risk_result.score > 0, f"Good risk text should score > 0: {risk_result.score}"
print(f"[OK] score_risk_mgmt: score={risk_result.score:.1f}")

# Test GHGEmissions Pydantic model
ghg = GHGEmissions(
    ticker="XOM",
    filing_year=2023,
    scope1_mt=120_000_000.0,
    scope2_mt_location=15_000_000.0,
    scope3_mt=600_000_000.0,
    scope1_source="xbrl",
    ghg_verified=True,
    verification_body="Bureau Veritas",
    data_quality="high",
)
assert ghg.ticker == "XOM"
assert ghg.scope1_mt == 120_000_000.0
assert ghg.ghg_verified is True
assert ghg.data_quality == "high"
print(f"[OK] GHGEmissions: ticker={ghg.ticker} scope1={ghg.scope1_mt:,.0f} tCO2e verified={ghg.ghg_verified}")

# Test ClimateTarget Pydantic model
target = ClimateTarget(
    ticker="MSFT",
    target_type="net_zero",
    target_year=2030,
    base_year=2020,
    scope_coverage=["scope1", "scope2", "scope3"],
    reduction_pct=100.0,
    verification_status="sbti_validated",
    sbti_committed=True,
    sbti_approved=True,
    is_net_zero=True,
    description="Carbon negative by 2030",
)
assert target.ticker == "MSFT"
assert target.is_net_zero is True
assert target.sbti_approved is True
assert target.target_year == 2030
print(f"[OK] ClimateTarget: ticker={target.ticker} type={target.target_type} year={target.target_year} sbti_approved={target.sbti_approved}")

# Test TCFDAssessment Pydantic model
assessment = TCFDAssessment(
    ticker="AAPL",
    company_name="Apple Inc.",
    filing_year=2023,
    alignment_score=72.5,
    alignment_level="advancing",
    disclosures_found_count=8,
    disclosures_total=11,
    has_scenario_analysis=True,
    has_net_zero_target=True,
)
assert assessment.ticker == "AAPL"
assert 0.0 <= assessment.alignment_score <= 100.0
assert assessment.disclosures_found_count <= assessment.disclosures_total
coverage = assessment.disclosures_found_count / assessment.disclosures_total
assert coverage > 0.5, f"Coverage should be > 50%: {coverage:.2%}"
print(f"[OK] TCFDAssessment: ticker={assessment.ticker} score={assessment.alignment_score} level={assessment.alignment_level} coverage={coverage:.0%}")

print("\n[PASS] dim_103: TCFD climate disclosure")
PYEOF
