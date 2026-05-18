#!/bin/bash
# dim_017: Non-GAAP — EQI, cookie-jar reserves, street vs GAAP spread
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.non_gaap_v3 import (
    XBRLFact, AdjustmentLineItem, ReconciliationRow,
    NonGAAPQualityScore, MarginSpread, NonGAAPQualityEngine,
    _GAAP_BASE_CONCEPTS,
    compute_earnings_quality_index,
    detect_cookie_jar_reserves,
    compute_street_vs_gaap_spread,
)

# Test 1: _GAAP_BASE_CONCEPTS mapping
assert "net_income" in _GAAP_BASE_CONCEPTS
assert "NetIncomeLoss" in _GAAP_BASE_CONCEPTS["net_income"]
assert len(_GAAP_BASE_CONCEPTS) >= 8
print(f"[OK] _GAAP_BASE_CONCEPTS has {len(_GAAP_BASE_CONCEPTS)} entries")

# Test 2: compute_earnings_quality_index — formula verified
# EQI = (CFO/NI)*0.4 + (1-accruals_ratio)*0.3 + revenue_quality*0.3
# High quality: CFO=2*NI, accruals=0.02, rev_quality=0.90
#   = (2.0)*0.4 + (0.98)*0.3 + (0.90)*0.3
#   = 0.800 + 0.294 + 0.270 = 1.364
eqi_high = compute_earnings_quality_index(
    cfo=2_000_000_000.0,
    net_income=1_000_000_000.0,
    accruals_ratio=0.02,
    revenue_quality=0.90,
)
expected_high = (2.0) * 0.4 + (1.0 - 0.02) * 0.3 + 0.90 * 0.3
assert abs(eqi_high - round(expected_high, 6)) < 1e-9, (
    f"EQI high-quality: expected {expected_high:.6f}, got {eqi_high}"
)
print(f"[OK] compute_earnings_quality_index high-quality: EQI={eqi_high:.6f} (expected {expected_high:.6f})")

# Low quality: CFO=0.5*NI, accruals=0.15, rev_quality=0.40
#   = (0.5)*0.4 + (0.85)*0.3 + (0.40)*0.3
#   = 0.200 + 0.255 + 0.120 = 0.575
eqi_low = compute_earnings_quality_index(
    cfo=500_000_000.0,
    net_income=1_000_000_000.0,
    accruals_ratio=0.15,
    revenue_quality=0.40,
)
expected_low = (0.5) * 0.4 + (1.0 - 0.15) * 0.3 + 0.40 * 0.3
assert abs(eqi_low - round(expected_low, 6)) < 1e-9, (
    f"EQI low-quality: expected {expected_low:.6f}, got {eqi_low}"
)
print(f"[OK] compute_earnings_quality_index low-quality: EQI={eqi_low:.6f} (expected {expected_low:.6f})")

# Zero net income → 0.0
eqi_zero = compute_earnings_quality_index(1e9, 0.0, 0.05, 0.80)
assert eqi_zero == 0.0
print(f"[OK] compute_earnings_quality_index zero-NI guard: EQI={eqi_zero}")

# CFO exactly equals NI, accruals=0, rev_quality=1.0 → EQI = 0.4 + 0.3 + 0.3 = 1.0
eqi_perfect = compute_earnings_quality_index(1e9, 1e9, 0.0, 1.0)
assert abs(eqi_perfect - 1.0) < 1e-9, f"Expected 1.0, got {eqi_perfect}"
print(f"[OK] compute_earnings_quality_index perfect: EQI={eqi_perfect:.6f}")

# Test 3: detect_cookie_jar_reserves
# 3 years: years 0,1 build (excess provisions), year 2 release
provisions   = [120_000_000.0,  100_000_000.0, -50_000_000.0]
net_incomes  = [500_000_000.0, 600_000_000.0,  200_000_000.0]
# threshold=0.10 → build if prov > 0.10 * NI
# year 0: 120M > 0.10*500M=50M → build
# year 1: 100M > 0.10*600M=60M → build
# year 2: -50M < 0 → release
result = detect_cookie_jar_reserves(provisions, net_incomes, threshold_pct=0.10)
assert result["smoothing_detected"] is True, "Should detect smoothing"
assert result["build_years"] == 2, f"Expected 2 build years, got {result['build_years']}"
assert result["release_years"] == 1
assert result["signal_strength"] == "moderate"
print(f"[OK] detect_cookie_jar_reserves: detected={result['smoothing_detected']}, "
      f"build={result['build_years']}, release={result['release_years']}, strength={result['signal_strength']}")

# No smoothing — all normal provisions, no releases
provisions2  = [40_000_000.0, 45_000_000.0, 50_000_000.0]
net_incomes2 = [500_000_000.0, 600_000_000.0, 700_000_000.0]
result2 = detect_cookie_jar_reserves(provisions2, net_incomes2)
assert result2["smoothing_detected"] is False
print(f"[OK] detect_cookie_jar_reserves no-pattern: detected={result2['smoothing_detected']}")

# Insufficient data guard
result3 = detect_cookie_jar_reserves([1.0, 2.0], [1.0, 2.0])
assert result3["smoothing_detected"] is False
assert result3["signal_strength"] == "none"
print("[OK] detect_cookie_jar_reserves insufficient-data guard")

# Test 4: compute_street_vs_gaap_spread
# Spread > $0.50 → red flag
r1 = compute_street_vs_gaap_spread(non_gaap_eps=3.50, gaap_eps=2.80)
assert abs(r1["spread"] - 0.70) < 1e-9, f"Expected spread=0.70, got {r1['spread']}"
assert r1["is_red_flag"] is True, "Should be red flag"
assert r1["credibility"] == "red_flag"
print(f"[OK] compute_street_vs_gaap_spread red-flag: spread={r1['spread']:.2f}, flag={r1['is_red_flag']}")

# Spread = $0.30 → low credibility, not red flag
r2 = compute_street_vs_gaap_spread(non_gaap_eps=2.50, gaap_eps=2.20)
assert abs(r2["spread"] - 0.30) < 1e-9
assert r2["is_red_flag"] is False
assert r2["credibility"] == "low"
print(f"[OK] compute_street_vs_gaap_spread low: spread={r2['spread']:.2f}")

# Spread = $0.05 → high credibility
r3 = compute_street_vs_gaap_spread(non_gaap_eps=2.05, gaap_eps=2.00)
assert abs(r3["abs_spread"] - 0.05) < 1e-9
assert r3["credibility"] == "high"
print(f"[OK] compute_street_vs_gaap_spread high: spread={r3['spread']:.2f}")

# Negative spread (GAAP > non-GAAP) also detected
r4 = compute_street_vs_gaap_spread(non_gaap_eps=1.00, gaap_eps=2.00)
assert r4["spread"] == -1.00
assert r4["is_red_flag"] is True
print(f"[OK] compute_street_vs_gaap_spread negative: spread={r4['spread']:.2f}, flag={r4['is_red_flag']}")

# Test 5: NonGAAPQualityEngine margins (pre-existing)
engine = NonGAAPQualityEngine.__new__(NonGAAPQualityEngine)
class MockDB:
    def get_adjustment_history(self, *a, **kw): return []
    def get_reconciliations(self, *a, **kw): return []
    def track_adjustment(self, *a, **kw): pass
engine._db = MockDB()
adj_sbc = AdjustmentLineItem(name="SBC", canonical_name="stock_based_compensation",
                              value=500_000_000.0, source="xbrl", category="red_flag")
recon = ReconciliationRow(
    ticker="META", period_end="2024-09-30", filing_type="10-Q",
    gaap_value=15_000_000_000.0, adjustments=[adj_sbc],
    total_adjustments=500_000_000.0, nongaap_value=15_500_000_000.0,
    adjustment_pct_of_gaap=500_000_000.0 / 15_000_000_000.0 * 100,
)
gaap_m, ng_m = engine._compute_margins(recon, revenue=100_000_000_000.0)
assert abs(gaap_m - 15.0) < 1e-4
assert abs(ng_m - 15.5) < 1e-4
print(f"[OK] NonGAAPQualityEngine margins: GAAP={gaap_m:.1f}%, Non-GAAP={ng_m:.1f}%")

print("[PASS]")
PYEOF
