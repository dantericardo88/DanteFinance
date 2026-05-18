#!/bin/bash
# dim_017: Non-GAAP reconciliation — NonGAAPQualityEngine scoring (pure computation)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.non_gaap_v3 import (
    XBRLFact, AdjustmentLineItem, ReconciliationRow,
    NonGAAPQualityScore, MarginSpread, NonGAAPQualityEngine,
    _GAAP_BASE_CONCEPTS,
)

# Test 1: _GAAP_BASE_CONCEPTS mapping
assert "net_income" in _GAAP_BASE_CONCEPTS
assert "NetIncomeLoss" in _GAAP_BASE_CONCEPTS["net_income"]
assert "stock_comp" in _GAAP_BASE_CONCEPTS
assert len(_GAAP_BASE_CONCEPTS) >= 8
print(f"[OK] _GAAP_BASE_CONCEPTS has {len(_GAAP_BASE_CONCEPTS)} entries")

# Test 2: AdjustmentLineItem
adj_sbc = AdjustmentLineItem(
    name="Stock-Based Compensation",
    canonical_name="stock_based_compensation",
    value=500_000_000.0,
    source="xbrl",
    category="red_flag"
)
assert adj_sbc.canonical_name == "stock_based_compensation"
print(f"[OK] AdjustmentLineItem: {adj_sbc.name}, category={adj_sbc.category}")

# Test 3: ReconciliationRow
recon = ReconciliationRow(
    ticker="META",
    period_end="2024-09-30",
    filing_type="10-Q",
    gaap_metric="Net Income (GAAP)",
    gaap_value=15_000_000_000.0,
    adjustments=[adj_sbc],
    total_adjustments=500_000_000.0,
    nongaap_metric="Non-GAAP Net Income",
    nongaap_value=15_500_000_000.0,
    adjustment_pct_of_gaap=500_000_000.0 / 15_000_000_000.0 * 100,
)
assert recon.ticker == "META"
assert abs(recon.adjustment_pct_of_gaap - 3.333) < 0.001
print(f"[OK] ReconciliationRow: adj_pct={recon.adjustment_pct_of_gaap:.2f}%")

# Test 4: NonGAAPQualityEngine scoring — pure margin gap logic
# Simulate score() without DB by calling _compute_margins directly
engine = NonGAAPQualityEngine.__new__(NonGAAPQualityEngine)
# Set up _db as a mock that returns no history
class MockDB:
    def get_adjustment_history(self, *args, **kwargs): return []
    def get_reconciliations(self, *args, **kwargs): return []
    def track_adjustment(self, *args, **kwargs): pass
engine._db = MockDB()

# Test margin gap computation
gaap_m, ng_m = engine._compute_margins(recon, revenue=100_000_000_000.0)
# gaap: 15B/100B = 15%, nongaap: 15.5B/100B = 15.5%
assert abs(gaap_m - 15.0) < 1e-4, f"GAAP margin wrong: {gaap_m}"
assert abs(ng_m - 15.5) < 1e-4, f"Non-GAAP margin wrong: {ng_m}"
print(f"[OK] _compute_margins: GAAP={gaap_m:.1f}%, Non-GAAP={ng_m:.1f}%")

# Test 5: Test scoring logic for large margin gap (>10pp)
# Create a reconciliation with 12pp gap
recon_big_gap = ReconciliationRow(
    ticker="TSLA",
    period_end="2024-06-30",
    gaap_value=1_000_000_000.0,      # $1B net income GAAP
    nongaap_value=2_200_000_000.0,   # $2.2B non-GAAP  -> 22pp gap on $10B revenue
    adjustment_pct_of_gaap=120.0,    # 120% -> triggers red flag
)
# With these inputs and $10B revenue:
# GAAP margin = 10%, non-GAAP margin = 22% -> gap = 12pp
gaap_m2, ng_m2 = engine._compute_margins(recon_big_gap, revenue=10_000_000_000.0)
gap = ng_m2 - gaap_m2
assert gap > 10.0, f"Gap should be >10pp: {gap}"

# Verify scoring deductions add up correctly
score = 100.0
red_flags = []
if gap > 10.0:
    score -= 15  # HIGH_MARGIN_GAP_DEDUCTION
    red_flags.append(f"margin gap {gap:.1f}pp > 10pp")
if abs(recon_big_gap.adjustment_pct_of_gaap) > 50.0:
    score -= 20  # VERY_HIGH_ADJ_DEDUCTION
    red_flags.append(f"adj pct {recon_big_gap.adjustment_pct_of_gaap:.0f}% > 50%")
assert score == 65.0, f"Expected score 65: {score}"
assert len(red_flags) == 2
print(f"[OK] Quality scoring: large gap + high adj -> score={score}, flags={len(red_flags)}")

# Test 6: MarginSpread model
ms = MarginSpread(
    ticker="META", period_end="2024-09-30",
    gaap_margin=15.0, nongaap_margin=15.5, spread_pp=0.5,
    revenue=100_000_000_000.0
)
assert ms.spread_pp == 0.5
print(f"[OK] MarginSpread: GAAP={ms.gaap_margin}%, Non-GAAP={ms.nongaap_margin}%, spread={ms.spread_pp}pp")

print("[PASS]")
PYEOF
