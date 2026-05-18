#!/bin/bash
# dim_011: Extended hours — gap fill probability, overnight drift, gap classification
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sds.adapters.extended_hours_v3 import (
    _PRE_START_MIN, _PRE_END_MIN, _AH_START_MIN, _AH_END_MIN,
    _GAP_ALERT_THRESHOLD,
    ExtendedBar, GapStatistics, GapAlert, SessionVolume, ExtendedQuote,
    compute_gap_fill_probability,
    compute_overnight_drift,
    classify_gap_type,
)
from datetime import datetime, date, timezone

# Test 1: Session boundary constants
assert _PRE_START_MIN == 240
assert _PRE_END_MIN   == 570
assert _AH_START_MIN  == 960
assert _AH_END_MIN    == 1200
assert _GAP_ALERT_THRESHOLD == 0.02
print("[OK] Session boundary constants correct")

# Test 2: compute_gap_fill_probability — boundary checks
# <2% → 78%
r1 = compute_gap_fill_probability(1.5)
assert r1["fill_probability"] == 0.78, f"Expected 0.78, got {r1['fill_probability']}"
assert r1["category"] == "common"
print(f"[OK] gap_fill_prob 1.5%: fill={r1['fill_probability']}, cat={r1['category']}")

# >5% → 31%
r2 = compute_gap_fill_probability(6.0)
assert r2["fill_probability"] == 0.31, f"Expected 0.31, got {r2['fill_probability']}"
assert r2["category"] == "large"
print(f"[OK] gap_fill_prob 6.0%: fill={r2['fill_probability']}, cat={r2['category']}")

# 2% exactly → starts interpolation
r3 = compute_gap_fill_probability(2.0)
assert r3["fill_probability"] == 0.78, f"At 2% boundary expect 0.78, got {r3['fill_probability']}"
print(f"[OK] gap_fill_prob 2.0% boundary: fill={r3['fill_probability']}")

# 3.5% → linear interpolation midpoint between 78% and 31%
# fill = 0.78 + (0.31 - 0.78) * (3.5 - 2.0) / (5.0 - 2.0) = 0.78 - 0.235 = 0.545
r4 = compute_gap_fill_probability(3.5)
expected_r4 = 0.78 + (0.31 - 0.78) * (3.5 - 2.0) / (5.0 - 2.0)
assert abs(r4["fill_probability"] - round(expected_r4, 4)) < 1e-9, (
    f"3.5% interp: expected {expected_r4:.4f}, got {r4['fill_probability']}"
)
print(f"[OK] gap_fill_prob 3.5% interpolation: fill={r4['fill_probability']:.4f} (expected {expected_r4:.4f})")

# Negative gap treated same as positive (abs)
r5 = compute_gap_fill_probability(-1.5)
assert r5["fill_probability"] == 0.78
assert r5["gap_pct"] == -1.5
print(f"[OK] gap_fill_prob negative gap handled: fill={r5['fill_probability']}")

# Test 3: compute_overnight_drift — math verified
# The formula: overnight_return[i] = (opens[i] - closes[i-1]) / closes[i-1]
# so overnight_return[1] = (opens[1] - closes[0]) / closes[0]
#    overnight_return[2] = (opens[2] - closes[1]) / closes[1]
closes = [100.0, 101.0, 102.0]
opens  = [101.0, 102.0, 103.0]
# r1 = (opens[1] - closes[0]) / closes[0] = (102.0 - 100.0) / 100.0 = 0.02
# r2 = (opens[2] - closes[1]) / closes[1] = (103.0 - 101.0) / 101.0 ≈ 0.019802
expected_r1 = (opens[1] - closes[0]) / closes[0]
expected_r2 = (opens[2] - closes[1]) / closes[1]
expected_avg = (expected_r1 + expected_r2) / 2.0
drift_result = compute_overnight_drift(opens, closes)
assert drift_result["n_days"] == 2, f"Expected 2 days, got {drift_result['n_days']}"
assert abs(drift_result["overnight_returns"][0] - round(expected_r1, 6)) < 1e-9, (
    f"r1: expected {round(expected_r1,6)}, got {drift_result['overnight_returns'][0]}"
)
assert abs(drift_result["overnight_returns"][1] - round(expected_r2, 6)) < 1e-9, (
    f"r2: expected {round(expected_r2,6)}, got {drift_result['overnight_returns'][1]}"
)
assert abs(drift_result["avg_drift"] - round(expected_avg, 6)) < 1e-9, (
    f"avg drift: expected {round(expected_avg,6):.8f}, got {drift_result['avg_drift']}"
)
print(f"[OK] compute_overnight_drift: returns={drift_result['overnight_returns']}, avg={drift_result['avg_drift']:.6f}")

# Edge case: fewer than 2 days
short_result = compute_overnight_drift([100.0], [100.0])
assert short_result["avg_drift"] is None
print("[OK] compute_overnight_drift: single-day guard")

# Test 4: classify_gap_type
g1 = classify_gap_type(0.5)
assert g1["gap_type"] == "common"
assert g1["direction"] == "up"
print(f"[OK] classify_gap_type 0.5%: {g1['gap_type']}")

g2 = classify_gap_type(2.0)
assert g2["gap_type"] == "breakaway"
print(f"[OK] classify_gap_type 2.0%: {g2['gap_type']}")

g3 = classify_gap_type(4.0)
assert g3["gap_type"] == "runaway"
print(f"[OK] classify_gap_type 4.0%: {g3['gap_type']}")

g4 = classify_gap_type(-5.5)
assert g4["gap_type"] == "exhaustion"
assert g4["direction"] == "down"
print(f"[OK] classify_gap_type -5.5%: {g4['gap_type']}, direction={g4['direction']}")

g5 = classify_gap_type(0.0)
assert g5["direction"] == "flat"
print(f"[OK] classify_gap_type 0%: direction={g5['direction']}")

# Test 5: ExtendedBar pydantic model
UTC = timezone.utc
bar = ExtendedBar(
    ticker="AAPL",
    time=datetime(2024, 3, 15, 8, 0, 0, tzinfo=UTC),
    open=180.0, high=181.5, low=179.5, close=181.0,
    volume=50000, session="pre_market",
)
assert bar.ticker == "AAPL" and bar.session == "pre_market"
print(f"[OK] ExtendedBar model: {bar.ticker} session={bar.session}")

print("\n[PASS] dim_011: extended_hours_v3 -- all checks passed")
PYEOF
