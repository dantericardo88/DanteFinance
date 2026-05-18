#!/bin/bash
# dim_016: Segment analytics — HHI, HTML parser, validation, regex fallback, deduplication
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.segment_analytics_v3 import (
    SegmentRow, GeoRow, ConcentrationMetrics,
    SEGMENT_REFERENCE, SegmentAnalyticsV3,
    _extract_segment_tables_from_html,
    _validate_table,
    _regex_fallback_parse,
    _fuzzy_name_match,
    _deduplicate_segments,
    _NUMBER_RE,
)

# ── 1. SEGMENT_REFERENCE static data (regression) ────────────────────────────
assert "AAPL" in SEGMENT_REFERENCE
aapl = SEGMENT_REFERENCE["AAPL"]
assert "iPhone" in aapl["segments"]
assert "Services" in aapl["segments"]
print(f"[OK] SEGMENT_REFERENCE: {len(SEGMENT_REFERENCE)} companies, AAPL segments verified")

# ── 2. HHI computation (regression) ──────────────────────────────────────────
analytics = SegmentAnalyticsV3.__new__(SegmentAnalyticsV3)
analytics._fetcher = None

rows = [
    SegmentRow(period="FY2024", segment_name="iPhone",    revenue=200_000, pct_of_total=0.52),
    SegmentRow(period="FY2024", segment_name="Mac",       revenue=30_000,  pct_of_total=0.08),
    SegmentRow(period="FY2024", segment_name="iPad",      revenue=30_000,  pct_of_total=0.08),
    SegmentRow(period="FY2024", segment_name="Wearables", revenue=40_000,  pct_of_total=0.09),
    SegmentRow(period="FY2024", segment_name="Services",  revenue=85_000,  pct_of_total=0.23),
]
total = sum(r.revenue for r in rows)
hhi = analytics.compute_hhi(rows, period="FY2024")
expected_hhi = sum((r.revenue / total) ** 2 for r in rows) * 10_000
assert abs(hhi - round(expected_hhi, 1)) < 1e-3
print(f"[OK] HHI = {hhi:.1f} (expected {expected_hhi:.1f})")

# ── 3. Concentration labels (regression) ─────────────────────────────────────
assert analytics.concentration_label(3000) == "high"
assert analytics.concentration_label(2000) == "moderate"
assert analytics.concentration_label(1000) == "low"
print("[OK] concentration_label: high>2500, moderate 1500-2500, low<1500")

# ── 4. HTML table parser: given sample HTML with segment table, extracts correct values ──
sample_html = """
<html><body>
<h3>Note 12 - Segment Information</h3>
<table>
  <tr><th>Segment</th><th>2024</th><th>2023</th><th>2022</th></tr>
  <tr><td>North America</td><td>1,234,567</td><td>1,100,000</td><td>980,000</td></tr>
  <tr><td>Europe</td><td>567,890</td><td>510,000</td><td>450,000</td></tr>
  <tr><td>Asia Pacific</td><td>345,678</td><td>310,000</td><td>280,000</td></tr>
  <tr><td>Total</td><td>2,148,135</td><td>1,920,000</td><td>1,710,000</td></tr>
</table>
</body></html>
"""

tables = _extract_segment_tables_from_html(
    sample_html,
    ["segment information", "business segments", "geographic areas"],
)
assert len(tables) >= 1, f"Expected at least 1 table, got {len(tables)}"

tbl = tables[0]
# Headers should contain "Segment", "2024", "2023", "2022"
headers = tbl["headers"]
assert any("2024" in h for h in headers), f"Header should contain '2024': {headers}"

# Rows: extract North America 2024
na_row = next((r for r in tbl["rows"] if "North America" in r[0]), None)
assert na_row is not None, "North America row not found"
# Value should be "1,234,567" in first data column
na_val_str = na_row[1].replace(",", "")
assert abs(float(na_val_str) - 1_234_567.0) < 1.0, f"North America 2024 value: {na_val_str}"
print(f"[OK] HTML table parser: found {len(tbl['rows'])} rows, North America 2024 = {float(na_val_str):,.0f}")

# ── 5. Validation rejects tables with < 2 numeric columns ────────────────────
# Good table: 3+ rows, 2+ numeric columns
good_rows = [
    ["North America", "1,234,567", "1,100,000"],
    ["Europe",        "567,890",   "510,000"],
    ["Asia Pacific",  "345,678",   "310,000"],
]
assert _validate_table(["Segment", "2024", "2023"], good_rows) is True, \
    "Good table should pass validation"

# Bad: only 1 numeric column (second col is text)
bad_rows_1num = [
    ["North America", "1,234,567", "N/A"],
    ["Europe",        "567,890",   "N/A"],
    ["Asia Pacific",  "345,678",   "N/A"],
]
assert _validate_table(["Segment", "2024", "Status"], bad_rows_1num) is False, \
    "Table with 1 numeric col and text 'N/A' should fail"

# Bad: fewer than 3 rows
sparse_rows = [
    ["North America", "1,234,567", "1,100,000"],
    ["Europe",        "567,890",   "510,000"],
]
assert _validate_table(["Segment", "2024", "2023"], sparse_rows) is False, \
    "Table with < 3 rows should fail validation"

print("[OK] Table validation: good passes, <2 numeric cols rejected, <3 rows rejected")

# ── 6. Regex fallback: parses "North America $1,234,567" correctly ────────────
plain_text = """
North America $1,234,567
Europe 567,890
Asia Pacific 345,678
"""
parsed = _regex_fallback_parse(plain_text)
names = [p["segment_name"] for p in parsed]
values = {p["segment_name"]: p["value"] for p in parsed}

assert "North America" in names, f"North America not found in {names}"
assert abs(values["North America"] - 1_234_567.0) < 1.0, \
    f"North America value: {values['North America']}"

assert "Europe" in names, f"Europe not found in {names}"
assert abs(values["Europe"] - 567_890.0) < 1.0, \
    f"Europe value: {values['Europe']}"

assert "Asia Pacific" in names, f"Asia Pacific not found"
assert abs(values["Asia Pacific"] - 345_678.0) < 1.0, \
    f"Asia Pacific value: {values['Asia Pacific']}"

print(f"[OK] Regex fallback: parsed {len(parsed)} segments — North America={values['North America']:,.0f}")

# ── 7. Deduplication: "North America" and "North American" merge correctly ────
xbrl_segs = [
    {"segment_name": "North America", "revenue": 1_234_567.0},
    {"segment_name": "Europe",        "revenue": 567_890.0},
]
html_segs = [
    {"segment_name": "North American", "revenue": 1_200_000.0},  # duplicate → should merge/drop
    {"segment_name": "Asia Pacific",   "revenue": 345_678.0},    # new → should be added
]

merged = _deduplicate_segments(xbrl_segs, html_segs)
merged_names = [s["segment_name"] for s in merged]

# "North American" should NOT appear (merged with "North America")
assert "North American" not in merged_names, \
    f"Duplicate 'North American' should have been dropped. Got: {merged_names}"

# "North America" (from XBRL) should still be present
assert "North America" in merged_names, f"'North America' should remain. Got: {merged_names}"

# "Asia Pacific" (new from HTML) should be added
assert "Asia Pacific" in merged_names, f"'Asia Pacific' should be added. Got: {merged_names}"

# "Europe" should remain
assert "Europe" in merged_names, f"'Europe' should remain. Got: {merged_names}"

# Total: 3 unique segments (North America, Europe, Asia Pacific)
assert len(merged) == 3, f"Expected 3 merged segments, got {len(merged)}: {merged_names}"

print(f"[OK] Deduplication: {len(merged)} segments — 'North American' merged into 'North America', 'Asia Pacific' added")

# Additional fuzzy match check
assert _fuzzy_name_match("North America", "North American") is True, \
    "North America vs North American should fuzzy-match"
assert _fuzzy_name_match("Europe", "Asia Pacific") is False, \
    "Europe vs Asia Pacific should NOT match"
assert _fuzzy_name_match("United States", "United States of America") is True, \
    "United States vs United States of America should match"
print("[OK] Fuzzy name match: North America/North American=True, Europe/Asia Pacific=False")

# ── 8. Segment attribution (regression) ──────────────────────────────────────
rows_prior = [
    SegmentRow(period="FY2023", segment_name="iPhone",    revenue=185_000, pct_of_total=0.53),
    SegmentRow(period="FY2023", segment_name="Mac",       revenue=27_000,  pct_of_total=0.08),
    SegmentRow(period="FY2023", segment_name="iPad",      revenue=28_000,  pct_of_total=0.08),
    SegmentRow(period="FY2023", segment_name="Wearables", revenue=38_000,  pct_of_total=0.10),
    SegmentRow(period="FY2023", segment_name="Services",  revenue=72_000,  pct_of_total=0.21),
]
all_rows = rows + rows_prior
attr = analytics.segment_attribution(all_rows, "FY2024", "FY2023")
assert len(attr) == 5
total_delta = 385_000 - 350_000
pcts = {a["segment"]: a["pct_contribution_to_change"] for a in attr}
expected_iphone_pct = (200_000 - 185_000) / total_delta * 100
assert abs(pcts["iPhone"] - round(expected_iphone_pct, 2)) < 1e-3
print(f"[OK] Segment attribution: iPhone {pcts['iPhone']:.1f}% of revenue delta")

print("\n[PASS] dim_016: Segment analytics — HTML parser, table validation, regex fallback, deduplication verified")
PYEOF
