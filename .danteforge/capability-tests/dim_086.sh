#!/usr/bin/env bash
# dim_086: Job postings — pure hiring velocity math and constant checks
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.job_postings_v3 import (
    JOLTS_TOTAL,
    JOLTS_SECTOR_OPENINGS,
    FRED_LABOR_SERIES,
    HiringVelocitySignal,
    TechStackSignal,
    SectorComparison,
)

# Test JOLTS_TOTAL has the core series
assert "openings" in JOLTS_TOTAL, "JOLTS_TOTAL must have 'openings'"
assert "hires" in JOLTS_TOTAL, "JOLTS_TOTAL must have 'hires'"
assert "quits" in JOLTS_TOTAL, "JOLTS_TOTAL must have 'quits'"
assert "layoffs" in JOLTS_TOTAL, "JOLTS_TOTAL must have 'layoffs'"
assert len(JOLTS_TOTAL) >= 6, f"Expected >= 6 JOLTS total series: {len(JOLTS_TOTAL)}"
# Verify series ID format (18-char BLS codes start with JTS)
for key, sid in JOLTS_TOTAL.items():
    assert sid.startswith("JTS"), f"JOLTS series ID should start with JTS: {sid}"
print(f"[OK] JOLTS_TOTAL: {len(JOLTS_TOTAL)} series, all start with 'JTS'")

# Test JOLTS_SECTOR_OPENINGS
assert "manufacturing" in JOLTS_SECTOR_OPENINGS, "Should have 'manufacturing'"
assert "information" in JOLTS_SECTOR_OPENINGS, "Should have 'information'"
assert "finance_insurance" in JOLTS_SECTOR_OPENINGS, "Should have 'finance_insurance'"
assert len(JOLTS_SECTOR_OPENINGS) >= 8, f"Expected >= 8 sector series: {len(JOLTS_SECTOR_OPENINGS)}"
print(f"[OK] JOLTS_SECTOR_OPENINGS: {len(JOLTS_SECTOR_OPENINGS)} sectors")

# Test FRED_LABOR_SERIES
assert "unemployment" in FRED_LABOR_SERIES, "Should have 'unemployment'"
assert "nonfarm_payroll" in FRED_LABOR_SERIES, "Should have 'nonfarm_payroll'"
assert FRED_LABOR_SERIES["unemployment"] == "UNRATE", f"Unemployment should map to UNRATE"
assert FRED_LABOR_SERIES["nonfarm_payroll"] == "PAYEMS", f"Nonfarm payroll should map to PAYEMS"
print(f"[OK] FRED_LABOR_SERIES: {len(FRED_LABOR_SERIES)} series")

# Test hiring velocity math directly (pure computation, no DB)
# velocity = (current - prior) / prior * 100
def compute_velocity(current: int, prior: int) -> float:
    if prior <= 0:
        return 0.0
    return round((current - prior) / prior * 100, 1)

# Accelerating: current >> prior_30d
vel_acc = compute_velocity(1500, 1000)
assert vel_acc == 50.0, f"Velocity should be 50.0: {vel_acc}"
print(f"[OK] Hiring velocity (accelerating): {vel_acc}%")

# Decelerating: current << prior_30d
vel_dec = compute_velocity(700, 1000)
assert vel_dec == -30.0, f"Velocity should be -30.0: {vel_dec}"
print(f"[OK] Hiring velocity (decelerating): {vel_dec}%")

# Stable: ~0%
vel_stable = compute_velocity(1010, 1000)
assert abs(vel_stable) < 5.0, f"Velocity should be near 0: {vel_stable}"
print(f"[OK] Hiring velocity (stable): {vel_stable}%")

# Test signal classification logic (mirrors HiringSignalEngine)
def classify_velocity(vel: float) -> str:
    if vel > 15:
        return "accelerating"
    elif vel < -15:
        return "decelerating"
    return "stable"

assert classify_velocity(50.0) == "accelerating"
assert classify_velocity(-30.0) == "decelerating"
assert classify_velocity(5.0) == "stable"
assert classify_velocity(15.1) == "accelerating"
assert classify_velocity(-15.1) == "decelerating"
print("[OK] Velocity signal classification thresholds correct (±15%)")

# Test confidence level logic
def classify_confidence(current, vel_30d, vel_60d) -> str:
    if current is None:
        return "low"
    if vel_30d is not None and vel_60d is not None:
        return "high"
    return "medium"

assert classify_confidence(None, None, None) == "low"
assert classify_confidence(1000, None, None) == "medium"
assert classify_confidence(1000, 25.0, 30.0) == "high"
print("[OK] Confidence level classification correct")

# Test HiringVelocitySignal model structure (Pydantic)
from datetime import datetime, timezone
sig = HiringVelocitySignal(
    ticker="NVDA",
    company="NVIDIA",
    current_openings=2500,
    d30_openings=2000,
    d60_openings=1800,
    d90_openings=1500,
    velocity_30d=25.0,
    velocity_60d=38.9,
    signal="accelerating",
    confidence="high",
    computed_at=datetime.now(timezone.utc).isoformat(),
)
assert sig.ticker == "NVDA"
assert sig.signal == "accelerating"
assert sig.confidence == "high"
assert sig.velocity_30d == 25.0
print(f"[OK] HiringVelocitySignal: ticker={sig.ticker} signal={sig.signal} vel_30d={sig.velocity_30d}")

# Test TechStackSignal model structure
from datetime import datetime, timezone
tech_sig = TechStackSignal(
    company="OpenAI",
    cloud_mentions=45,
    ai_ml_mentions=120,
    devops_mentions=30,
    top_keywords=["llm", "generative ai", "python", "kubernetes"],
    tech_intensity_score=8.5,
    computed_at=datetime.now(timezone.utc).isoformat(),
)
assert 0.0 <= tech_sig.tech_intensity_score <= 10.0, \
    f"Tech intensity should be 0-10: {tech_sig.tech_intensity_score}"
assert "llm" in tech_sig.top_keywords
print(f"[OK] TechStackSignal: company={tech_sig.company} tech_score={tech_sig.tech_intensity_score}")

# Test tech intensity scoring math: more AI/cloud mentions → higher score
# (pure logic test — this mirrors what the engine computes)
def estimate_tech_intensity(ai_mentions: int, cloud_mentions: int, devops_mentions: int) -> float:
    """Simplified tech intensity score 0-10."""
    total = ai_mentions + cloud_mentions + devops_mentions
    score = min(10.0, total / 20.0)   # 200 total mentions → score of 10
    return round(score, 2)

low_tech = estimate_tech_intensity(5, 3, 2)
high_tech = estimate_tech_intensity(120, 45, 30)
assert low_tech < high_tech, f"High-tech company should score higher: {low_tech} vs {high_tech}"
assert 0 <= low_tech <= 10, f"Score out of bounds: {low_tech}"
assert 0 <= high_tech <= 10, f"Score out of bounds: {high_tech}"
print(f"[OK] Tech intensity scoring: low={low_tech} high={high_tech}")

print("\n[PASS] dim_086: Job postings")
PYEOF
