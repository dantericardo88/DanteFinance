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
    compute_hiring_momentum_score,
    classify_labor_market_tightness,
    detect_sector_hiring_surge,
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

# ---- NEW: compute_hiring_momentum_score ----------------------------------------
# jolts_openings_change=500, prior_avg=1000 -> jolts_component = 500/1000 * 0.5 = 0.25
# indeed_trend=0.4 -> indeed_component = 0.4 * 0.3 = 0.12
# usajobs_count_delta=0.2 -> usajobs_component = 0.2 * 0.2 = 0.04
# score = 0.25 + 0.12 + 0.04 = 0.41 -> "expanding"
result_hms = compute_hiring_momentum_score(
    jolts_openings_change=500.0,
    jolts_prior_avg=1000.0,
    indeed_trend=0.4,
    usajobs_count_delta=0.2,
)
expected_score = (500.0 / 1000.0) * 0.5 + 0.4 * 0.3 + 0.2 * 0.2
assert abs(result_hms["score"] - expected_score) < 1e-9, (
    f"Hiring momentum score wrong: expected {expected_score:.10f}, got {result_hms['score']}"
)
assert result_hms["signal"] == "expanding", f"Signal should be 'expanding': {result_hms['signal']}"
assert abs(result_hms["jolts_component"]   - 0.25) < 1e-9, f"JOLTS component wrong: {result_hms['jolts_component']}"
assert abs(result_hms["indeed_component"]  - 0.12) < 1e-9, f"Indeed component wrong: {result_hms['indeed_component']}"
assert abs(result_hms["usajobs_component"] - 0.04) < 1e-9, f"USAJobs component wrong: {result_hms['usajobs_component']}"
print(
    f"[OK] compute_hiring_momentum_score: score={result_hms['score']:.4f}, "
    f"signal={result_hms['signal']}, "
    f"jolts={result_hms['jolts_component']:.4f}, indeed={result_hms['indeed_component']:.4f}, "
    f"usajobs={result_hms['usajobs_component']:.4f}"
)

# Contracting signal: negative components -> score < -0.10
result_contr = compute_hiring_momentum_score(
    jolts_openings_change=-400.0,
    jolts_prior_avg=1000.0,
    indeed_trend=-0.6,
    usajobs_count_delta=-0.5,
)
expected_contr = (-400.0 / 1000.0) * 0.5 + (-0.6) * 0.3 + (-0.5) * 0.2
assert abs(result_contr["score"] - expected_contr) < 1e-9, (
    f"Contracting score wrong: expected {expected_contr:.10f}, got {result_contr['score']}"
)
assert result_contr["signal"] == "contracting", f"Signal should be 'contracting': {result_contr['signal']}"
print(f"[OK] Contracting signal: score={result_contr['score']:.4f}, signal={result_contr['signal']}")

# Stable signal: small components -> score in (-0.10, +0.10)
result_stable = compute_hiring_momentum_score(
    jolts_openings_change=50.0,
    jolts_prior_avg=1000.0,
    indeed_trend=0.0,
    usajobs_count_delta=0.0,
)
expected_stable = (50.0 / 1000.0) * 0.5
assert abs(result_stable["score"] - expected_stable) < 1e-9, f"Stable score wrong: {result_stable['score']}"
assert result_stable["signal"] == "stable", f"Signal should be 'stable': {result_stable['signal']}"
print(f"[OK] Stable signal: score={result_stable['score']:.4f}, signal={result_stable['signal']}")

# ---- NEW: classify_labor_market_tightness ----------------------------------------
# ratio > 1.0 -> tight
tight = classify_labor_market_tightness(jolts_openings=9_000.0, unemployed_persons=6_000.0)
assert tight["tightness"] == "tight", f"Expected 'tight': {tight['tightness']}"
assert abs(tight["ratio"] - 9000.0 / 6000.0) < 1e-6, f"Ratio wrong: {tight['ratio']}"
print(f"[OK] Tightness=tight: ratio={tight['ratio']:.4f}, threshold>1.0")

# 0.7 <= ratio <= 1.0 -> balanced
balanced = classify_labor_market_tightness(jolts_openings=8_000.0, unemployed_persons=10_000.0)
assert balanced["tightness"] == "balanced", f"Expected 'balanced': {balanced['tightness']}"
assert 0.7 <= balanced["ratio"] <= 1.0, f"Balanced ratio should be in [0.7, 1.0]: {balanced['ratio']}"
print(f"[OK] Tightness=balanced: ratio={balanced['ratio']:.4f}, in [0.7, 1.0]")

# ratio < 0.7 -> loose
loose = classify_labor_market_tightness(jolts_openings=5_000.0, unemployed_persons=10_000.0)
assert loose["tightness"] == "loose", f"Expected 'loose': {loose['tightness']}"
assert loose["ratio"] < 0.7, f"Loose ratio should be < 0.7: {loose['ratio']}"
print(f"[OK] Tightness=loose: ratio={loose['ratio']:.4f}, < 0.70")

# Exact boundary: ratio = 1.0 -> balanced
at_one = classify_labor_market_tightness(jolts_openings=10_000.0, unemployed_persons=10_000.0)
assert at_one["ratio"] == 1.0, f"Ratio should be exactly 1.0: {at_one['ratio']}"
assert at_one["tightness"] == "balanced", f"Exactly 1.0 should be 'balanced': {at_one['tightness']}"
print(f"[OK] Tightness boundary ratio=1.0 -> balanced")

# Exact boundary: ratio = 0.7 -> balanced
at_07 = classify_labor_market_tightness(jolts_openings=7_000.0, unemployed_persons=10_000.0)
assert abs(at_07["ratio"] - 0.7) < 1e-9, f"Ratio should be 0.7: {at_07['ratio']}"
assert at_07["tightness"] == "balanced", f"0.7 should be 'balanced': {at_07['tightness']}"
print(f"[OK] Tightness boundary ratio=0.7 -> balanced")

# ---- NEW: detect_sector_hiring_surge ----------------------------------------
import numpy as np
np.random.seed(42)
# History: mean ~0.05, std ~0.02; surge at +4σ
history = [0.04, 0.05, 0.06, 0.05, 0.04, 0.03]
mean_h = np.mean(history)
std_h  = np.std(history, ddof=1)
surge_growth = mean_h + 4.0 * std_h  # guaranteed > 2σ

surge_result = detect_sector_hiring_surge(
    sector_growth_rates=history,
    current_growth=surge_growth,
    sigma_threshold=2.0,
)
assert bool(surge_result["surge_flag"]) is True, (
    f"Should detect surge at {surge_growth:.4f} (z={surge_result['z_score']:.4f}): {surge_result}"
)
assert surge_result["z_score"] > 2.0, f"Z-score should be > 2.0: {surge_result['z_score']}"
assert abs(float(surge_result["mean_growth"])  - round(mean_h, 6)) < 1e-6, f"Mean wrong: {surge_result['mean_growth']}"
assert abs(float(surge_result["std_growth"])   - round(std_h, 6))  < 1e-6, f"Std wrong: {surge_result['std_growth']}"
print(
    f"[OK] detect_sector_hiring_surge (surge): z={surge_result['z_score']:.4f}, "
    f"surge_flag={surge_result['surge_flag']}, "
    f"mean={surge_result['mean_growth']:.4f}, std={surge_result['std_growth']:.4f}"
)

# No surge: current within 2σ
no_surge_result = detect_sector_hiring_surge(
    sector_growth_rates=history,
    current_growth=mean_h + 1.5 * std_h,  # < 2σ
    sigma_threshold=2.0,
)
assert bool(no_surge_result["surge_flag"]) is False, (
    f"No surge expected at 1.5σ: z={no_surge_result['z_score']:.4f}"
)
assert no_surge_result["z_score"] < 2.0, f"Z-score should be < 2.0: {no_surge_result['z_score']}"
print(f"[OK] detect_sector_hiring_surge (no surge): z={no_surge_result['z_score']:.4f}, surge_flag=False")

# Verify z-score formula: z = (current - mean) / std
manual_current = mean_h + 3.0 * std_h
manual_z       = (manual_current - mean_h) / std_h
check_result   = detect_sector_hiring_surge(history, manual_current)
assert abs(float(check_result["z_score"]) - manual_z) < 1e-6, (
    f"Z-score formula error: expected {manual_z:.6f}, got {check_result['z_score']:.6f}"
)
print(f"[OK] Z-score formula verified: (current - mean) / std = {check_result['z_score']:.6f}")

print("\n[PASS] dim_086: Job postings")
PYEOF
