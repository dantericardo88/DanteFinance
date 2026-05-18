#!/bin/bash
# dim_019: Earnings KPI tracker — SurpriseResult, accruals metrics (pure computation)
set -e
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np

from sentinel.sfe.earnings_kpi_tracker_v3 import (
    QuarterlyResult, SurpriseResult, GuidanceEntry,
    QualityMetrics, CalendarEntry, BatchSurpriseItem,
    compute_accruals_metrics,
    compute_earnings_quality_score,
    detect_guidance_walk_up,
    compute_beat_rate,
)

# Test 1: QuarterlyResult model
qr = QuarterlyResult(
    ticker="AAPL",
    period_label="Q1-2024",
    period_end="2024-03-31",
    actual_eps=2.18,
    revenue=119_575_000_000.0,
    net_income=23_636_000_000.0,
    cfo=26_700_000_000.0,
    diluted_shares=15_443_000_000.0
)
assert qr.ticker == "AAPL"
assert qr.actual_eps == 2.18
print(f"[OK] QuarterlyResult: {qr.ticker} {qr.period_label} EPS={qr.actual_eps}")

# Test 2: SurpriseResult — beat calculation
actual_eps = 2.18
estimate_eps = 2.10
surprise_pct = (actual_eps - estimate_eps) / abs(estimate_eps) * 100
surprise_cat = "BEAT" if surprise_pct > 1 else "MISS" if surprise_pct < -1 else "IN-LINE"
sr = SurpriseResult(
    ticker="AAPL",
    period_label="Q1-2024",
    actual_eps=actual_eps,
    estimate_eps=estimate_eps,
    surprise_pct=round(surprise_pct, 2),
    surprise_cat=surprise_cat,
    estimate_source="finviz",
    beat_streak=6
)
assert sr.surprise_cat == "BEAT"
assert abs(sr.surprise_pct - 3.81) < 0.01
print(f"[OK] SurpriseResult: {sr.ticker} {sr.surprise_cat} by {sr.surprise_pct:.2f}%")

# Test 3: QualityMetrics — accruals ratio interpretation
qm = QualityMetrics(
    ticker="AAPL",
    fiscal_year="FY2024",
    accruals_ratio=-0.03,  # negative = CFO > NI (good quality)
    sloan_ratio=-0.03,
    cash_eps=1.73,
    gaap_eps=1.53,
    cash_eps_gap=0.20,
    quality_flag="HIGH"
)
assert qm.quality_flag == "HIGH"
assert qm.accruals_ratio < 0  # negative accruals = high quality earnings
print(f"[OK] QualityMetrics: {qm.ticker} accruals={qm.accruals_ratio:.3f} (HIGH quality)")

# Test 4: compute_accruals_metrics pure computation with synthetic data
ni_df = pd.DataFrame({
    "end": pd.to_datetime(["2022-09-30", "2023-09-30", "2024-09-30"]),
    "val": [99_803_000_000.0, 96_995_000_000.0, 93_736_000_000.0],
})
cfo_df = pd.DataFrame({
    "end": pd.to_datetime(["2022-09-30", "2023-09-30", "2024-09-30"]),
    "val": [122_151_000_000.0, 113_996_000_000.0, 109_432_000_000.0],
})
assets_df = pd.DataFrame({
    "end": pd.to_datetime(["2022-09-30", "2023-09-30", "2024-09-30"]),
    "val": [352_755_000_000.0, 352_583_000_000.0, 364_980_000_000.0],
})
profile = {
    "net_income_annual": ni_df,
    "cfo_annual": cfo_df,
    "total_assets_annual": assets_df,
}
records = compute_accruals_metrics(profile)
assert len(records) >= 2, f"Expected 2+ records, got {len(records)}"
# Accruals = NI - CFO should be negative for AAPL (CFO > NI = quality)
first_rec = records[0]
assert "accruals_ratio" in first_rec
assert first_rec["accruals_ratio"] < 0, \
    f"AAPL accruals ratio should be negative: {first_rec['accruals_ratio']}"
print(f"[OK] compute_accruals_metrics: {len(records)} records, FY1 accruals_ratio={first_rec['accruals_ratio']:.4f}")

# Test 5: GuidanceEntry language detection logic
guidance_keywords = ["raised", "lowered", "maintained", "withdrawn", "initiated"]
for kw in guidance_keywords:
    ge = GuidanceEntry(
        ticker="NVDA", period_label="Q2-2025",
        guidance_action=kw,
        raw_excerpt=f"Company {kw} guidance for next quarter."
    )
    assert ge.guidance_action == kw
print(f"[OK] GuidanceEntry: all {len(guidance_keywords)} action types valid")

# --------------------------------------------------------------------------
# NEW: Test 6 — compute_earnings_quality_score (accruals ratio math)
# --------------------------------------------------------------------------
# Accruals ratio = (NI - CFO) / avg_assets
# Use AAPL-like numbers (millions):
# NI = 93_736, CFO = 109_432, avg_assets = (352_583 + 364_980) / 2 = 358_781.5
ni_m      = 93_736.0
cfo_m     = 109_432.0
avg_a_m   = (352_583.0 + 364_980.0) / 2.0   # 358_781.5

expected_ratio = (ni_m - cfo_m) / avg_a_m    # (93736 - 109432) / 358781.5 ≈ -0.04374
result = compute_earnings_quality_score(ni_m, cfo_m, avg_a_m)

assert result["quality_flag"] in ("HIGH", "MODERATE", "LOW", "VERY_LOW")
assert abs(result["accruals_ratio"] - expected_ratio) < 1e-6, \
    f"Expected {expected_ratio:.6f}, got {result['accruals_ratio']}"
assert result["accruals_ratio"] < 0, "CFO > NI → negative accruals → high quality"
print(f"[OK] compute_earnings_quality_score: accruals_ratio={result['accruals_ratio']:.6f}, flag={result['quality_flag']}")

# Zero avg_assets must raise
try:
    compute_earnings_quality_score(100.0, 80.0, 0.0)
    assert False, "Should have raised"
except ValueError:
    pass
print("[OK] compute_earnings_quality_score: zero avg_assets raises ValueError")

# --------------------------------------------------------------------------
# NEW: Test 7 — detect_guidance_walk_up
# --------------------------------------------------------------------------
history_walk_up = [
    {"period_label": "2024-Q1", "guidance_action": "lowers"},
    {"period_label": "2024-Q2", "guidance_action": "raises"},
    {"period_label": "2024-Q3", "guidance_action": "raises"},
]
wu = detect_guidance_walk_up(history_walk_up)
assert wu["walk_up_detected"] is True, "Two consecutive raises should be detected"
assert wu["consecutive_raises"] == 2
assert wu["periods"] == ["2024-Q2", "2024-Q3"]
print(f"[OK] detect_guidance_walk_up: walk_up={wu['walk_up_detected']}, raises={wu['consecutive_raises']}")

# Single raise — no walk-up
history_single = [
    {"period_label": "2024-Q1", "guidance_action": "reaffirms"},
    {"period_label": "2024-Q2", "guidance_action": "raises"},
]
wu2 = detect_guidance_walk_up(history_single)
assert wu2["walk_up_detected"] is False
assert wu2["consecutive_raises"] == 1
print(f"[OK] detect_guidance_walk_up: single raise correctly not flagged as walk-up")

# Empty history
wu3 = detect_guidance_walk_up([])
assert wu3["walk_up_detected"] is False
print(f"[OK] detect_guidance_walk_up: empty history handled")

# --------------------------------------------------------------------------
# NEW: Test 8 — compute_beat_rate
# --------------------------------------------------------------------------
surprise_8 = [
    {"surprise_cat": "BEAT"},
    {"surprise_cat": "STRONG_BEAT"},
    {"surprise_cat": "BEAT"},
    {"surprise_cat": "MISS"},
    {"surprise_cat": "BEAT"},
    {"surprise_cat": "IN_LINE"},
    {"surprise_cat": "BEAT"},
    {"surprise_cat": "STRONG_BEAT"},
]
br = compute_beat_rate(surprise_8)
assert br["quarters_used"] == 8
assert br["beats"] == 6, f"Expected 6 beats, got {br['beats']}"
assert abs(br["beat_rate"] - 0.75) < 1e-6, f"Expected 0.75, got {br['beat_rate']}"
assert br["assessment"] == "HIGH"
print(f"[OK] compute_beat_rate: beat_rate={br['beat_rate']:.2f}, assessment={br['assessment']}")

# Fewer than 8 quarters — uses all available
surprise_3 = [
    {"surprise_cat": "MISS"},
    {"surprise_cat": "MISS"},
    {"surprise_cat": "MISS"},
]
br2 = compute_beat_rate(surprise_3)
assert br2["quarters_used"] == 3
assert br2["beat_rate"] == 0.0
assert br2["assessment"] == "LOW"
print(f"[OK] compute_beat_rate: 0/3 beats → beat_rate=0.0, assessment=LOW")

# Window capped at 8 when more supplied
surprise_10 = [{"surprise_cat": "MISS"}] * 2 + [{"surprise_cat": "BEAT"}] * 8
br3 = compute_beat_rate(surprise_10)
assert br3["quarters_used"] == 8, f"Expected 8, got {br3['quarters_used']}"
print(f"[OK] compute_beat_rate: window correctly capped at 8 quarters")

print("[PASS]")
PYEOF
