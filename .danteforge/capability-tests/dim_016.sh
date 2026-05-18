#!/bin/bash
# dim_016: Segment analytics — HHI concentration, SEGMENT_REFERENCE, attribution (pure computation)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.segment_analytics_v3 import (
    SegmentRow, GeoRow, ConcentrationMetrics,
    SEGMENT_REFERENCE, SegmentAnalyticsV3,
)

# Test 1: SEGMENT_REFERENCE static data
assert "AAPL" in SEGMENT_REFERENCE, "AAPL should be in static segment reference"
aapl_segs = SEGMENT_REFERENCE["AAPL"]
assert "segments" in aapl_segs
assert "iPhone" in aapl_segs["segments"]
assert "Services" in aapl_segs["segments"]
print(f"[OK] SEGMENT_REFERENCE has {len(SEGMENT_REFERENCE)} companies; AAPL segments: {aapl_segs['segments'][:3]}")

# Test 2: compute_hhi — Herfindahl-Hirschman Index
analytics = SegmentAnalyticsV3.__new__(SegmentAnalyticsV3)
analytics._fetcher = None  # skip network init

rows = [
    SegmentRow(period="FY2024", segment_name="iPhone",     revenue=200_000, pct_of_total=0.52),
    SegmentRow(period="FY2024", segment_name="Mac",        revenue=30_000,  pct_of_total=0.08),
    SegmentRow(period="FY2024", segment_name="iPad",       revenue=30_000,  pct_of_total=0.08),
    SegmentRow(period="FY2024", segment_name="Wearables",  revenue=40_000,  pct_of_total=0.09),
    SegmentRow(period="FY2024", segment_name="Services",   revenue=85_000,  pct_of_total=0.23),
]
total = sum(r.revenue for r in rows)
hhi = analytics.compute_hhi(rows, period="FY2024")
# Expected HHI: sum(share^2) * 10000
shares = [r.revenue / total for r in rows]
expected_hhi = sum(s**2 for s in shares) * 10_000
assert abs(hhi - round(expected_hhi, 1)) < 1e-3, f"HHI mismatch: {hhi} vs {expected_hhi}"
print(f"[OK] HHI = {hhi:.1f} (expected {expected_hhi:.1f})")

# Test 3: concentration_label
assert analytics.concentration_label(3000) == "high"
assert analytics.concentration_label(2000) == "moderate"
assert analytics.concentration_label(1000) == "low"
print("[OK] concentration_label: high>2500, moderate 1500-2500, low<1500")

# Test 4: segment_attribution
rows_prior = [
    SegmentRow(period="FY2023", segment_name="iPhone",     revenue=185_000, pct_of_total=0.53),
    SegmentRow(period="FY2023", segment_name="Mac",        revenue=27_000,  pct_of_total=0.08),
    SegmentRow(period="FY2023", segment_name="iPad",       revenue=28_000,  pct_of_total=0.08),
    SegmentRow(period="FY2023", segment_name="Wearables",  revenue=38_000,  pct_of_total=0.10),
    SegmentRow(period="FY2023", segment_name="Services",   revenue=72_000,  pct_of_total=0.21),
]
all_rows = rows + rows_prior
attr = analytics.segment_attribution(all_rows, "FY2024", "FY2023")
assert len(attr) == 5, f"Expected 5 segments in attribution: {len(attr)}"
# Total delta = 385K - 350K = 35K
total_delta = 385_000 - 350_000
pcts = {a["segment"]: a["pct_contribution_to_change"] for a in attr}
assert "iPhone" in pcts
# iPhone delta = 15K, total delta = 35K => iPhone attribution ~42.9%
expected_iphone_pct = (200_000 - 185_000) / total_delta * 100
assert abs(pcts["iPhone"] - round(expected_iphone_pct, 2)) < 1e-3, \
    f"iPhone attribution wrong: {pcts['iPhone']} vs {expected_iphone_pct}"
print(f"[OK] Segment attribution: iPhone {pcts['iPhone']:.1f}% of revenue delta")

# Test 5: ConcentrationMetrics model
cm = ConcentrationMetrics(
    ticker="AAPL", period="FY2024",
    hhi=hhi,
    concentration_label=analytics.concentration_label(hhi),
    n_segments=5,
    top_segment="iPhone",
    top_segment_pct=52.0,
    weighted_avg_margin=None
)
assert cm.ticker == "AAPL"
print(f"[OK] ConcentrationMetrics: HHI={cm.hhi}, label={cm.concentration_label}")

print("[PASS]")
PYEOF
