#!/bin/bash
# dim_009: Short interest — squeeze signals, ladder attack, DTC signal
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())
from datetime import date

from sentinel.sds.adapters.short_interest_v3 import (
    ShortInterestRecord, DailyShortVolume, FTDRecord,
    ShortMetrics, SqueezeCandidateRecord, ShortChangeRecord,
    _compute_squeeze_score,
    compute_days_to_cover_signal,
    compute_short_squeeze_probability,
    detect_short_ladder_attack,
)

# Test 1: _compute_squeeze_score pure computation
score1 = _compute_squeeze_score(dtc=10.0, short_pct_float=0.50)
assert abs(score1 - 500.0) < 1e-6, f"Expected 500, got {score1}"
score2 = _compute_squeeze_score(dtc=1.0, short_pct_float=0.05)
assert abs(score2 - 5.0) < 1e-6, f"Expected 5, got {score2}"
assert _compute_squeeze_score(None, 0.5) is None
assert _compute_squeeze_score(5.0, None) is None
print(f"[OK] _compute_squeeze_score: GME-like={score1}, low={score2}, None handled")

# Test 2: compute_days_to_cover_signal
# DTC = 15M / 1M = 15 → extreme
sig = compute_days_to_cover_signal(15_000_000, 1_000_000)
assert sig["short_ratio"] == 15.0, f"Expected 15.0, got {sig['short_ratio']}"
assert sig["signal"] == "extreme_squeeze_risk", f"Expected extreme, got {sig['signal']}"
print(f"[OK] compute_days_to_cover_signal extreme: DTC={sig['short_ratio']}, signal={sig['signal']}")

# DTC = 3M / 1M = 3 → moderate
sig2 = compute_days_to_cover_signal(3_000_000, 1_000_000)
assert sig2["short_ratio"] == 3.0
assert sig2["signal"] == "moderate"
print(f"[OK] compute_days_to_cover_signal moderate: DTC={sig2['short_ratio']}")

# Zero volume → unavailable
sig3 = compute_days_to_cover_signal(1_000_000, 0)
assert sig3["signal"] == "unavailable"
print(f"[OK] compute_days_to_cover_signal zero-volume guard")

# Test 3: compute_short_squeeze_probability — verify formula to 1e-9
# P = 1 / (1 + exp(-(si_pct - 0.20) / 0.05)) * momentum_factor
import math as _math
for si_pct, mom, expected in [
    (0.20, 1.0, 1.0 / (1.0 + _math.exp(0.0))),            # = 0.5
    (0.30, 1.0, 1.0 / (1.0 + _math.exp(-2.0))),           # ≈ 0.8808
    (0.10, 1.0, 1.0 / (1.0 + _math.exp(2.0))),            # ≈ 0.1192
    (0.30, 1.5, min(1.0, 1.5 / (1.0 + _math.exp(-2.0)))), # clamped
]:
    got = compute_short_squeeze_probability(si_pct, mom)
    assert abs(got - expected) < 1e-9, (
        f"squeeze_prob({si_pct}, {mom}): expected {expected:.10f}, got {got:.10f}"
    )
print("[OK] compute_short_squeeze_probability: formula verified to 1e-9")

# Test 4: detect_short_ladder_attack
# 4 consecutive days with >40% short volume → pattern detected
short_vols = [4_500_000, 4_200_000, 4_800_000, 4_600_000, 2_000_000]
total_vols = [10_000_000, 10_000_000, 10_000_000, 10_000_000, 10_000_000]
result = detect_short_ladder_attack(short_vols, total_vols)
assert result["pattern_detected"] is True, "Pattern should be detected"
assert result["max_consecutive"] == 4, f"Expected 4 consecutive, got {result['max_consecutive']}"
assert result["flagged_days"] == 4
print(f"[OK] detect_short_ladder_attack: detected={result['pattern_detected']}, consec={result['max_consecutive']}")

# No pattern — only 2 consecutive flagged days
short_vols2 = [4_500_000, 4_200_000, 1_000_000, 1_000_000, 1_000_000]
result2 = detect_short_ladder_attack(short_vols2, total_vols)
assert result2["pattern_detected"] is False, "Should not detect with only 2 consecutive"
assert result2["max_consecutive"] == 2
print(f"[OK] detect_short_ladder_attack no-pattern: max_consec={result2['max_consecutive']}")

# Empty / mismatched → safe guard
result3 = detect_short_ladder_attack([], [])
assert result3["pattern_detected"] is False
print("[OK] detect_short_ladder_attack empty guard")

# Test 5: ShortInterestRecord model (field validation)
si = ShortInterestRecord(
    ticker="GME",
    settlement_date=date(2021, 1, 15),
    short_interest=71_200_000,
    avg_daily_volume=29_000_000,
    days_to_cover=round(71_200_000 / 29_000_000, 4),
    source="finra"
)
assert si.ticker == "GME"
assert si.days_to_cover is not None and si.days_to_cover > 2.0
print(f"[OK] ShortInterestRecord: GME DTC={si.days_to_cover:.2f}")

print("[PASS]")
PYEOF
