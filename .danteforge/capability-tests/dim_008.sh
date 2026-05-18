#!/bin/bash
# dim_008: Corporate actions — Pydantic models, M&A deal quality, arb spread,
#          accretion/dilution, deal outcome tracker (pure math)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import date

from sentinel.sds.adapters.corporate_actions_v3 import (
    CorporateActionRecord, DividendRecord, SplitRecord,
    SpinoffRecord, MAActionRecord, AdjustmentFactor, UpcomingAction,
    score_deal_quality, compute_arb_spread,
    compute_eps_accretion_dilution, DealOutcomeTracker,
)

# ── 1. CorporateActionRecord (regression) ────────────────────────────────────
rec = CorporateActionRecord(
    ticker="AAPL", action_type="split",
    ex_date=date(2020, 8, 28), ratio="4:1", factor=0.25,
    source="sec_8937", notes="AAPL 4-for-1 split"
)
assert rec.ticker == "AAPL"
assert rec.factor == 0.25
print("[OK] CorporateActionRecord instantiates correctly")

# ── 2. DividendRecord (regression) ───────────────────────────────────────────
div = DividendRecord(
    ticker="T", ex_date=date(2022, 4, 8), amount=0.2775,
    div_type="cut", source="yfinance", prev_amount=0.52,
    pct_change=(0.2775 - 0.52) / 0.52 * 100, is_cut=True
)
assert div.is_cut is True
assert div.pct_change < 0
print(f"[OK] DividendRecord: cut={div.is_cut}, pct_change={div.pct_change:.1f}%")

# ── 3. SplitRecord (regression) ──────────────────────────────────────────────
split_fwd = SplitRecord(
    ticker="TSLA", ex_date=date(2022, 8, 25),
    ratio_new=3, ratio_old=1, factor=round(1/3, 6),
    split_type="forward", source="yfinance"
)
assert abs(split_fwd.factor - 1/3) < 1e-5
split_rev = SplitRecord(
    ticker="GME", ex_date=date(2023, 7, 20),
    ratio_new=1, ratio_old=20, factor=20.0,
    split_type="reverse", source="yfinance"
)
assert split_rev.factor == 20.0
print(f"[OK] SplitRecord: fwd factor={split_fwd.factor:.4f}, rev factor={split_rev.factor}")

# ── 4. AdjustmentFactor (regression) ─────────────────────────────────────────
cumulative = 0.25 * 0.5
assert abs(cumulative - 0.125) < 1e-10
af = AdjustmentFactor(
    ticker="AAPL", as_of_date=date(2020, 8, 28),
    factor=cumulative, source="sec_8937", action_type="split"
)
assert af.factor == 0.125
print(f"[OK] AdjustmentFactor: cumulative={af.factor}")

# ── 5. MAActionRecord (regression) ───────────────────────────────────────────
ma = MAActionRecord(
    ticker="ATVI", action_type="merger_completion",
    announcement_date=date(2022, 1, 18), expiry_date=date(2023, 10, 13),
    consideration="$95 per share cash", acquirer="Microsoft", source="edgar"
)
assert ma.acquirer == "Microsoft"
print(f"[OK] MAActionRecord: {ma.ticker} acquired by {ma.acquirer}")

# ── 6. Deal quality: all-cash at premium to 52w high → high certainty ────────
result = score_deal_quality(
    current_price=48.0,
    acquisition_price=55.0,
    week_52_high=50.0,         # offer is 10% above 52w high
    consideration_type="cash", # all-cash → certainty_score = 100
    is_cross_border=False,
    acquirer_market_share_pct=5.0,  # low market share → no antitrust discount
    target_revenue=200_000_000.0,
    deal_value=1_000_000_000.0,
)

# All-cash deal → certainty score must be 100
assert result["deal_certainty_score"] == 100.0, (
    f"All-cash deal certainty must be 100, got {result['deal_certainty_score']}"
)

# Offer at 55 vs 52w high 50 → premium = (55/50 - 1)*100 = 10%
assert abs(result["premium_to_52w_high_pct"] - 10.0) < 0.001, (
    f"premium_to_52w_high_pct: {result['premium_to_52w_high_pct']:.4f} (expected 10.0)"
)

# No cross-border, low market share → regulatory_risk_score = 100
assert result["regulatory_risk_score"] == 100.0, (
    f"Regulatory risk score: {result['regulatory_risk_score']} (expected 100)"
)

# Composite must be > 75 (high quality deal)
assert result["composite_score"] > 75.0, (
    f"Composite score too low: {result['composite_score']}"
)
print(f"[OK] Deal quality: composite={result['composite_score']:.1f}, certainty={result['deal_certainty_score']}, premium_52w={result['premium_to_52w_high_pct']:.1f}%")

# Stock deal with cross-border and high market share → low quality
result_bad = score_deal_quality(
    current_price=48.0, acquisition_price=50.0,
    week_52_high=60.0,            # below 52w high
    consideration_type="stock",   # stock deal → certainty = 40
    is_cross_border=True,         # -30 regulatory
    acquirer_market_share_pct=35.0, # -40 regulatory (>30%)
    target_revenue=100_000_000.0,
    deal_value=200_000_000.0,
)
assert result_bad["deal_certainty_score"] == 40.0, f"Stock certainty: {result_bad['deal_certainty_score']}"
assert result_bad["regulatory_risk_score"] == 30.0, f"Regulatory: {result_bad['regulatory_risk_score']}"  # 100-30-40=30
assert result_bad["composite_score"] < result["composite_score"], "Bad deal must score below good deal"
print(f"[OK] Deal quality: bad deal composite={result_bad['composite_score']:.1f} < good deal {result['composite_score']:.1f}")

# ── 7. Arb spread math: acquire at $50 when stock trades at $48 → 4.17% ─────
arb = compute_arb_spread(current_price=48.0, acquisition_price=50.0)
expected_spread = (50.0 / 48.0 - 1.0) * 100.0  # = 4.166...%
assert abs(arb["arb_spread_pct"] - expected_spread) < 0.0001, (
    f"Arb spread: {arb['arb_spread_pct']:.6f}% (expected {expected_spread:.6f}%)"
)
assert abs(arb["arb_spread_pct"] - 4.1667) < 0.001, f"Expected ~4.17%, got {arb['arb_spread_pct']:.4f}%"
print(f"[OK] Arb spread: {arb['arb_spread_pct']:.4f}% (expected ~4.17%)")

# Annualised arb spread (30 days to close)
arb_ann = compute_arb_spread(current_price=48.0, acquisition_price=50.0, annualise=True, days_to_close=30)
assert "annualised_arb_spread_pct" in arb_ann
# Annualised must be much larger than raw spread (365/30 ≈ 12.2x amplification)
assert arb_ann["annualised_arb_spread_pct"] > arb_ann["arb_spread_pct"] * 5
print(f"[OK] Annualised arb spread (30d): {arb_ann['annualised_arb_spread_pct']:.2f}%")

# ── 8. Accretion/dilution: target EPS $2 / 0.5 exchange ratio = $4 ──────────
# spec: target_earnings / share_exchange_ratio = $4 per acquirer share equivalent
result_eac = compute_eps_accretion_dilution(
    acquirer_eps=5.0,
    target_earnings=2.0,        # $2 per unit (will be divided by ratio)
    share_exchange_ratio=0.5,   # 0.5 acquirer shares per target share
    new_shares_issued=0.0,      # pure stock math, no new shares in this test
    acquirer_shares_outstanding=1.0,
    financing_cost_after_tax=0.0,
)
# eps_contribution_per_acquirer_share = target_earnings / share_exchange_ratio = 2 / 0.5 = 4
assert abs(result_eac["eps_contribution_per_acquirer_share"] - 4.0) < 1e-9, (
    f"EPS contribution: {result_eac['eps_contribution_per_acquirer_share']} (expected 4.0)"
)
print(f"[OK] Accretion: target_earnings/ratio = {result_eac['eps_contribution_per_acquirer_share']:.1f} (expected 4.0)")

# Real accretion/dilution: acquirer EPS $10, 1M shares; target earns $2M, issue 100k shares
result_full = compute_eps_accretion_dilution(
    acquirer_eps=10.0,
    target_earnings=2_000_000.0,
    share_exchange_ratio=1.0,
    new_shares_issued=100_000.0,
    acquirer_shares_outstanding=1_000_000.0,
    financing_cost_after_tax=0.0,
)
# Acquirer total = 10M, + 2M target - 0 cost = 12M earnings
# Pro-forma shares = 1M + 100K = 1.1M
# Pro-forma EPS = 12M / 1.1M = 10.909...
expected_pf_eps = 12_000_000.0 / 1_100_000.0
assert abs(result_full["pro_forma_eps"] - expected_pf_eps) < 0.001, (
    f"Pro-forma EPS: {result_full['pro_forma_eps']:.6f} (expected {expected_pf_eps:.6f})"
)
assert result_full["is_accretive"] is True, "Deal should be accretive"
print(f"[OK] Accretion full model: pro_forma_eps={result_full['pro_forma_eps']:.4f} (expected {expected_pf_eps:.4f}), accretive={result_full['is_accretive']}")

# Dilutive deal: issue too many shares
result_dil = compute_eps_accretion_dilution(
    acquirer_eps=10.0,
    target_earnings=100_000.0,    # tiny target earnings
    share_exchange_ratio=1.0,
    new_shares_issued=5_000_000.0, # massive dilution
    acquirer_shares_outstanding=1_000_000.0,
    financing_cost_after_tax=0.0,
)
assert result_dil["is_accretive"] is False, "Should be dilutive"
print(f"[OK] Dilution check: pro_forma_eps={result_dil['pro_forma_eps']:.4f} < acquirer_eps={result_dil['acquirer_eps']:.1f}")

# ── 9. Deal outcome tracker: 8 closed + 2 failed → 80% completion rate ───────
tracker = DealOutcomeTracker()
for i in range(8):
    tracker.record_deal(
        ticker=f"TGT{i}", deal_type="tender_offer",
        consideration_type="cash", outcome="closed",
        announced_date=date(2022, 1, 1), closed_date=date(2022, 6, 1),
    )
for i in range(2):
    tracker.record_deal(
        ticker=f"TGT{8+i}", deal_type="tender_offer",
        consideration_type="cash", outcome="failed",
        announced_date=date(2022, 1, 1),
    )
# Add 3 pending (should NOT count in denominator)
for i in range(3):
    tracker.record_deal(
        ticker=f"PEND{i}", deal_type="merger_completion",
        consideration_type="stock", outcome="pending",
    )

stats = tracker.get_completion_rate()
assert stats["closed"] == 8,  f"closed: {stats['closed']}"
assert stats["failed"] == 2,  f"failed: {stats['failed']}"
assert stats["pending"] == 3, f"pending: {stats['pending']}"
assert stats["total_resolved"] == 10, f"total_resolved: {stats['total_resolved']}"
assert abs(stats["completion_rate"] - 0.80) < 1e-9, (
    f"Completion rate: {stats['completion_rate']} (expected 0.80)"
)
assert abs(stats["completion_rate_pct"] - 80.0) < 1e-6, (
    f"Completion rate pct: {stats['completion_rate_pct']} (expected 80.0)"
)
print(f"[OK] Completion rate: {stats['closed']} closed / {stats['total_resolved']} resolved = {stats['completion_rate_pct']:.1f}%")

# Filter by deal type
cash_stats = tracker.get_completion_rate(deal_type="tender_offer")
assert cash_stats["total_resolved"] == 10  # all cash deals are tender_offer
pending_only_stats = tracker.get_completion_rate(deal_type="merger_completion")
assert pending_only_stats["total_resolved"] == 0  # all pending, none resolved
assert pending_only_stats["completion_rate"] is None
print(f"[OK] Filtered completion: tender_offer={cash_stats['completion_rate_pct']:.1f}%, merger_completion (all pending) rate=None")

print("\n[PASS] dim_008: Corporate actions — M&A deal quality, arb spread, accretion/dilution, outcome tracker verified")
PYEOF
