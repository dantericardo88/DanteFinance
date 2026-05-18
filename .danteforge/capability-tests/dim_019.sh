#!/bin/bash
# dim_019: Earnings KPI tracker — SurpriseResult, accruals metrics (pure computation)
set -e
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

print("[PASS]")
PYEOF
