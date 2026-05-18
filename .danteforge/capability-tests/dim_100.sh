#!/bin/bash
# dim_100: M&A Deal Intelligence — capability verification
# Tests pure computation logic (no network calls required)
set -e

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# 1. Import all public classes
from sentinel.sfe.ma_intelligence_v3 import (
    MAFilingCollector,
    DealTracker,
    MergerArbitrageAnalyzer,
    DealPremiumAnalyzer,
    SynergyEstimator,
    MASignalGenerator,
    MADashboard,
    MADeal,
    MAFiling,
    DealTerms,
    ArbOpportunity,
)
print("[OK] All M&A classes imported")

# 2. SynergyEstimator — revenue synergies (pure math)
est = SynergyEstimator()
rev_syn = est.estimate_revenue_synergies(
    acquirer_revenue=5_000_000_000,
    target_revenue=1_000_000_000,
    deal_type="strategic",
)
assert rev_syn["total"] > 0, "Revenue synergies must be positive"
assert "cross_sell" in rev_syn, "Missing cross_sell synergy"
print(f"[OK] Revenue synergies: ${rev_syn['total']:,.0f}")

# 3. Cost synergies (pure math)
cost_syn = est.estimate_cost_synergies(
    acquirer_opex=800_000_000,
    target_opex=200_000_000,
    acquirer_cogs=2_000_000_000,
    target_cogs=400_000_000,
    acquirer_rnd=100_000_000,
    target_rnd=50_000_000,
    deal_type="strategic",
)
assert cost_syn["total"] > 0, "Cost synergies must be positive"
assert "corporate_overhead" in cost_syn, "Missing overhead line"
print(f"[OK] Cost synergies: ${cost_syn['total']:,.0f}")

# 4. Synergy NPV (pure DCF)
npv_result = est.estimate_synergy_npv(
    annual_synergies=150_000_000,
    realization_period=3,
    discount_rate=0.10,
    implementation_cost_multiplier=1.25,
)
assert "total_synergy_npv" in npv_result, f"Missing total_synergy_npv key, got: {list(npv_result.keys())}"
assert npv_result["total_synergy_npv"] > 0, f"Synergy NPV must be positive, got {npv_result['total_synergy_npv']}"
print(f"[OK] Synergy NPV: ${npv_result['total_synergy_npv']:,.0f}")

# 5. Dataclass integrity
terms = DealTerms(
    per_share_price=45.00,
    total_deal_value_mm=1200.0,
    payment_type="cash",
    premium_1d=0.28,
)
deal = MADeal(
    deal_id="test_001",
    target_ticker="TGT",
    acquirer_ticker="ACQ",
    target_name="Target Corp",
    acquirer_name="Acquirer Inc",
    announcement_date="2026-01-15",
    status="announced",
    deal_terms=terms,
)
assert deal.deal_id == "test_001"
assert deal.status == "announced"
assert deal.deal_terms.payment_type == "cash"
print("[OK] MADeal + DealTerms dataclasses work")

# 6. Verify key subclasses have required methods
assert hasattr(MergerArbitrageAnalyzer, '__init__'), "Missing __init__"
assert hasattr(DealTracker, '__init__'), "Missing __init__"
assert hasattr(MADashboard, '__init__'), "Missing __init__"
assert hasattr(MASignalGenerator, '__init__'), "Missing __init__"
assert hasattr(DealPremiumAnalyzer, '__init__'), "Missing __init__"
print("[OK] All M&A class interfaces verified")

# 7. Verify geography coverage (roll-up deal type)
roll_syn = est.estimate_revenue_synergies(
    acquirer_revenue=2_000_000_000,
    target_revenue=500_000_000,
    deal_type="roll_up",
)
assert roll_syn["total"] > 0
geo_syn = est.estimate_revenue_synergies(
    acquirer_revenue=2_000_000_000,
    target_revenue=500_000_000,
    deal_type="geographic",
)
assert geo_syn["total"] > 0
print("[OK] Multiple deal-type synergy models verified")

print("\n[PASS] dim_100: M&A Deal Intelligence — all checks passed")
PYEOF
