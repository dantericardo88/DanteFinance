#!/bin/bash
# dim_020: Historical PIT — data freshness, filing anomaly, revision impact
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import date

from sentinel.sfe.historical_pit_v3 import (
    PITSnapshot, FilingRecord, LagReport,
    RestatementRecord, VintageRecord,
    compute_data_freshness_score,
    detect_filing_anomaly,
    compute_revision_impact,
)

# Test 1: compute_data_freshness_score — formula: max(0, 100 - days * 0.5)
# 0 days → 100
s0 = compute_data_freshness_score(0)
assert s0 == 100.0, f"Expected 100.0, got {s0}"
print(f"[OK] freshness_score 0 days: {s0}")

# 40 days → 100 - 40*0.5 = 80
s40 = compute_data_freshness_score(40)
expected_40 = 100.0 - 40 * 0.5
assert abs(s40 - expected_40) < 1e-9, f"Expected {expected_40}, got {s40}"
print(f"[OK] freshness_score 40 days: {s40} (expected {expected_40})")

# 100 days → 100 - 50 = 50
s100 = compute_data_freshness_score(100)
expected_100 = 50.0
assert abs(s100 - expected_100) < 1e-9, f"Expected {expected_100}, got {s100}"
print(f"[OK] freshness_score 100 days: {s100}")

# 200 days → 100 - 100 = 0.0 (floor)
s200 = compute_data_freshness_score(200)
assert s200 == 0.0, f"Expected 0.0 (floor), got {s200}"
print(f"[OK] freshness_score 200 days (floor): {s200}")

# 500 days → still 0.0 (floor, never negative)
s500 = compute_data_freshness_score(500)
assert s500 == 0.0, f"Expected 0.0, got {s500}"
print(f"[OK] freshness_score 500 days (floor never negative): {s500}")

# Test 2: detect_filing_anomaly
# 10-K: period_end=2023-09-30, filed=2023-11-03 → lag=34 days < 90 → on time
r1 = detect_filing_anomaly("2023-09-30", "2023-11-03", "10-K")
assert r1["lag_days"] == 34
assert r1["is_late"] is False
assert r1["threshold_days"] == 90
assert r1["severity"] == "none"
print(f"[OK] detect_filing_anomaly 10-K on-time: lag={r1['lag_days']}d, late={r1['is_late']}")

# 10-K: lag=100 days → late (>90), excess=10 → mild
r2 = detect_filing_anomaly("2023-01-31", "2023-05-11", "10-K")
# 2023-05-11 - 2023-01-31 = 100 days
assert r2["lag_days"] == 100
assert r2["is_late"] is True
assert r2["severity"] == "mild"
print(f"[OK] detect_filing_anomaly 10-K late mild: lag={r2['lag_days']}d, severity={r2['severity']}")

# 10-K: lag=150 days → severe (excess=60 > 30)
r3 = detect_filing_anomaly("2022-12-31", "2023-05-30", "10-K")
# 2023-05-30 - 2022-12-31 = 150 days
assert r3["lag_days"] == 150
assert r3["is_late"] is True
assert r3["severity"] == "severe"
print(f"[OK] detect_filing_anomaly 10-K severe: lag={r3['lag_days']}d, severity={r3['severity']}")

# 10-Q: threshold = 45 days; lag=40 → on time
r4 = detect_filing_anomaly("2023-09-30", "2023-11-09", "10-Q")
# 2023-11-09 - 2023-09-30 = 40 days
assert r4["lag_days"] == 40
assert r4["is_late"] is False
assert r4["threshold_days"] == 45
print(f"[OK] detect_filing_anomaly 10-Q on-time: lag={r4['lag_days']}d, threshold={r4['threshold_days']}d")

# Bad date → graceful error
r5 = detect_filing_anomaly("not-a-date", "2023-11-03", "10-K")
assert r5["lag_days"] is None
assert r5["severity"] == "unknown"
print("[OK] detect_filing_anomaly bad-date guard")

# Test 3: compute_revision_impact
# 5% deviation threshold
# Original=100M, Restated=106M → magnitude=0.06 → material (>0.05)
rv1 = compute_revision_impact(100_000_000.0, 106_000_000.0, threshold_pct=0.05)
expected_mag = abs(106_000_000.0 - 100_000_000.0) / abs(100_000_000.0)  # = 0.06
assert abs(rv1["restatement_magnitude"] - round(expected_mag, 6)) < 1e-9, (
    f"magnitude: expected {expected_mag:.6f}, got {rv1['restatement_magnitude']}"
)
assert rv1["is_material"] is True
assert rv1["direction"] == "upward"
print(f"[OK] compute_revision_impact material upward: mag={rv1['restatement_magnitude']:.4f}, material={rv1['is_material']}")

# Original=100M, Restated=102M → magnitude=0.02 → immaterial (<0.05)
rv2 = compute_revision_impact(100_000_000.0, 102_000_000.0, threshold_pct=0.05)
assert rv2["restatement_magnitude"] == 0.02
assert rv2["is_material"] is False
assert rv2["direction"] == "upward"
print(f"[OK] compute_revision_impact immaterial: mag={rv2['restatement_magnitude']:.4f}, material={rv2['is_material']}")

# Downward restatement
rv3 = compute_revision_impact(100_000_000.0, 90_000_000.0, threshold_pct=0.05)
assert rv3["direction"] == "downward"
assert rv3["is_material"] is True
assert abs(rv3["restatement_magnitude"] - 0.10) < 1e-9
print(f"[OK] compute_revision_impact material downward: mag={rv3['restatement_magnitude']:.4f}")

# Zero original → graceful
rv4 = compute_revision_impact(0.0, 100_000_000.0)
assert rv4["restatement_magnitude"] is None
assert rv4["is_material"] is False
print(f"[OK] compute_revision_impact zero-original guard: mag={rv4['restatement_magnitude']}")

# Unchanged
rv5 = compute_revision_impact(100_000_000.0, 100_000_000.0)
assert rv5["restatement_magnitude"] == 0.0
assert rv5["direction"] == "unchanged"
print(f"[OK] compute_revision_impact unchanged: mag={rv5['restatement_magnitude']}")

# Test 4: PITSnapshot model (pre-existing)
snap = PITSnapshot(
    ticker="AAPL", cik="0000320193", company_name="Apple Inc.",
    as_of_date="2024-01-31", fiscal_period_end="2023-09-30",
    filed_date="2023-11-03",
    filing_lag_days=(date(2023, 11, 3) - date(2023, 9, 30)).days,
    revenue=383_285_000_000.0, net_income=96_995_000_000.0,
    total_assets=352_583_000_000.0, n_concepts_found=15, data_vintage="original"
)
assert snap.filing_lag_days == 34
assert snap.data_vintage == "original"
print(f"[OK] PITSnapshot: {snap.ticker} filed {snap.filing_lag_days} days after period end")

print("[PASS]")
PYEOF
