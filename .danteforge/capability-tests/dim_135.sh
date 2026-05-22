#!/usr/bin/env bash
# dim_135: Liquidity risk framework (Basel III LCR / NSFR)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.liquidity_risk_v3 import (
        LiquidityCoverageRatio, NetStableFundingRatio,
        compute_lcr, compute_nsfr,
        HQLAPortfolio, CashFlowStress,
        FundingStructure, AssetStructure,
        LEVEL2A_HAIRCUT, LEVEL2_CAP_FRACTION,
    )
except ImportError as e:
    print(f"[NOT-BUILT] dim_135: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.liquidity_risk_v3")
    sys.exit(0)

errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  [OK] {label}" + (f": {detail}" if detail else ""))
    else:
        print(f"  [FAIL] {label}" + (f": {detail}" if detail else ""))
        errors.append(label)

print("=" * 60)
print("dim_135: Basel III LCR / NSFR Engine")
print("=" * 60)

# ------------------------------------------------------------------
# TEST 1: LCR compliant bank
# Bank with $200M Level 1 HQLA, $100M net cash outflows => LCR = 200%
# ------------------------------------------------------------------
print("\n[TEST 1] LCR compliant bank: $200M HQLA / $100M outflows")

lcr_engine = LiquidityCoverageRatio()

# Build a scenario where net outflows = exactly $100M
# Using compute_lcr() convenience function directly
ratio_t1 = compute_lcr(hqla=200.0, net_cash_outflows=100.0)
check("compute_lcr(200, 100) == 2.0", abs(ratio_t1 - 2.0) < 1e-9,
      f"got {ratio_t1:.6f}")
check("LCR = 200% (compliant)", lcr_engine.is_compliant(ratio_t1),
      f"ratio={ratio_t1:.2f}")

# Also verify using the full class with HQLAPortfolio (pure L1, no outflow stress)
# Build stress such that NCO = 100M exactly:
#   NCO = max(outflows - min(inflows, 0.75*outflows), 0.25*outflows)
#   With inflows=0: NCO = max(outflows - 0, 0.25*outflows) = outflows
#   With outflows = retail_stable * 0.03 = X, set X = 100 => retail_stable = 3333.33
hqla_t1 = HQLAPortfolio(level1=200.0)
stress_t1 = CashFlowStress(
    retail_stable_deposits=3333.333333333,  # * 3% = 100.0
    cash_inflows=0.0,
)
nco_t1 = stress_t1.net_cash_outflows()
ratio_full_t1 = lcr_engine.compute(hqla_t1, stress_t1)
check("Net cash outflows = 100.0", abs(nco_t1 - 100.0) < 0.01,
      f"NCO={nco_t1:.4f}")
check("Full LCR class: ratio ~= 2.0", abs(ratio_full_t1 - 2.0) < 0.01,
      f"ratio={ratio_full_t1:.4f}")
check("Full LCR class: is_compliant()", lcr_engine.is_compliant(ratio_full_t1))

# Breakdown dict contains expected keys
bd_t1 = lcr_engine.breakdown(hqla_t1, stress_t1)
check("Breakdown has 'hqla_total_adjusted'", "hqla_total_adjusted" in bd_t1)
check("Breakdown has 'lcr_ratio'", "lcr_ratio" in bd_t1)
check("Breakdown compliant=True", bd_t1["compliant"] is True)

# ------------------------------------------------------------------
# TEST 2: LCR non-compliant bank ($80M HQLA, $100M outflows -> LCR=80%)
# ------------------------------------------------------------------
print("\n[TEST 2] LCR non-compliant bank: $80M HQLA / $100M outflows")

ratio_t2 = compute_lcr(hqla=80.0, net_cash_outflows=100.0)
check("compute_lcr(80, 100) == 0.80", abs(ratio_t2 - 0.80) < 1e-9,
      f"got {ratio_t2:.6f}")
check("LCR = 80% (non-compliant)", not lcr_engine.is_compliant(ratio_t2),
      f"ratio={ratio_t2:.2f}")

# Via full class
hqla_t2 = HQLAPortfolio(level1=80.0)
stress_t2 = CashFlowStress(
    retail_stable_deposits=3333.333333333,
    cash_inflows=0.0,
)
ratio_full_t2 = lcr_engine.compute(hqla_t2, stress_t2)
check("Full class LCR ~= 0.80", abs(ratio_full_t2 - 0.80) < 0.01,
      f"ratio={ratio_full_t2:.4f}")
check("Full class non-compliant", not lcr_engine.is_compliant(ratio_full_t2))

# ------------------------------------------------------------------
# TEST 3: NSFR compliant bank
# ------------------------------------------------------------------
print("\n[TEST 3] NSFR compliant bank")

nsfr_engine = NetStableFundingRatio()

# Build a clearly compliant bank
funding_t3 = FundingStructure(
    tier1_capital=100.0,          # ASF: 100*1.00 = 100
    retail_deposits_stable=400.0,  # ASF: 400*0.95 = 380
    retail_deposits_less_stable=200.0,  # ASF: 200*0.90 = 180
    wholesale_long_term=100.0,     # ASF: 100*1.00 = 100
    wholesale_short_term=50.0,     # ASF: 50*0.50 = 25
)
# Total ASF = 100 + 380 + 180 + 100 + 25 = 785

assets_t3 = AssetStructure(
    cash=50.0,               # RSF: 50*0.00 = 0
    hqla_l1=100.0,           # RSF: 100*0.05 = 5
    hqla_l2a=80.0,           # RSF: 80*0.15 = 12
    residential_mortgages=200.0,  # RSF: 200*0.65 = 130
    retail_loans=150.0,      # RSF: 150*0.85 = 127.5
    wholesale_lending=80.0,  # RSF: 80*0.50 = 40
    illiquid_assets=100.0,   # RSF: 100*1.00 = 100
)
# Total RSF = 0 + 5 + 12 + 130 + 127.5 + 40 + 100 = 414.5

asf_t3 = funding_t3.available_stable_funding()
rsf_t3 = assets_t3.required_stable_funding()
ratio_t3 = nsfr_engine.compute(funding_t3, assets_t3)

check("ASF = 785.0", abs(asf_t3 - 785.0) < 0.01, f"ASF={asf_t3:.2f}")
check("RSF = 414.5", abs(rsf_t3 - 414.5) < 0.01, f"RSF={rsf_t3:.2f}")
check("NSFR >= 1.0 (compliant)", nsfr_engine.is_compliant(ratio_t3),
      f"ratio={ratio_t3:.4f} ({ratio_t3*100:.1f}%)")
check("NSFR > 1.5 (well-funded bank)", ratio_t3 > 1.5,
      f"ratio={ratio_t3:.4f}")

# compute_nsfr convenience function
ratio_simple_t3 = compute_nsfr(asf_t3, rsf_t3)
check("compute_nsfr matches class result", abs(ratio_simple_t3 - ratio_t3) < 1e-9)

# NSFR breakdown
bd_t3 = nsfr_engine.breakdown(funding_t3, assets_t3)
check("NSFR breakdown has 'asf_total'", "asf_total" in bd_t3)
check("NSFR breakdown has 'rsf_total'", "rsf_total" in bd_t3)
check("NSFR breakdown compliant=True", bd_t3["compliant"] is True)

# ------------------------------------------------------------------
# TEST 4: Level 2A haircut (15%) applied correctly
# ------------------------------------------------------------------
print("\n[TEST 4] Level 2A haircut application (15%)")

hqla_t4 = HQLAPortfolio(level1=0.0, level2a=100.0, level2b=0.0)
adj_l2a = hqla_t4.adjusted_level2a()
check("Level 2A: 100 * (1-0.15) = 85.0", abs(adj_l2a - 85.0) < 1e-9,
      f"adj_L2A={adj_l2a:.4f}")

# Since level1=0, Level 2 cap also equals 0 * (40/60) = 0,
# so the cap would zero out L2A. Use a real bank with L1 > 0.
hqla_t4b = HQLAPortfolio(level1=200.0, level2a=100.0, level2b=0.0)
adj_l2a_b = hqla_t4b.adjusted_level2a()
total_t4b = hqla_t4b.total_adjusted_hqla()

# adj_l2a = 85. Level2 cap = (40/60)*200 = 133.33. 85 < 133.33, so not capped.
check("Level 2A: 100 adj to 85 (not capped, L1=200)", abs(adj_l2a_b - 85.0) < 1e-9,
      f"adj_L2A={adj_l2a_b:.4f}")
check("Total HQLA = 200 + 85 = 285", abs(total_t4b - 285.0) < 0.01,
      f"Total={total_t4b:.4f}")
check("LEVEL2A_HAIRCUT constant = 0.15", abs(LEVEL2A_HAIRCUT - 0.15) < 1e-9)

# ------------------------------------------------------------------
# TEST 5: Level 2 cap at 40% of total HQLA
# ------------------------------------------------------------------
print("\n[TEST 5] Level 2 cap (max 40% of total HQLA)")

# L1=100, L2A=200 (adj=170). Cap= (40/60)*100 = 66.67. L2A should be capped at 66.67.
hqla_t5 = HQLAPortfolio(level1=100.0, level2a=200.0, level2b=0.0)
total_t5 = hqla_t5.total_adjusted_hqla()

# Expected total: L1=100, L2A capped = 66.67 => total = 166.67
expected_l2_cap = (LEVEL2_CAP_FRACTION / (1.0 - LEVEL2_CAP_FRACTION)) * 100.0  # = 66.67
expected_total = 100.0 + expected_l2_cap
check("Level 2 cap applied: total < uncapped", total_t5 < 270.0,
      f"total={total_t5:.4f}")
check("Total HQLA = 166.67 (L1=100 + L2A_capped=66.67)",
      abs(total_t5 - expected_total) < 0.01,
      f"total={total_t5:.4f}, expected={expected_total:.4f}")

# Level 2 fraction should be ~40%
l2_frac_t5 = hqla_t5.level2_fraction()
check("Level 2 fraction = 40%", abs(l2_frac_t5 - LEVEL2_CAP_FRACTION) < 0.01,
      f"L2 fraction={l2_frac_t5:.4f}")
check("LEVEL2_CAP_FRACTION constant = 0.40", abs(LEVEL2_CAP_FRACTION - 0.40) < 1e-9)

# ------------------------------------------------------------------
# TEST 6: NSFR non-compliant bank
# ------------------------------------------------------------------
print("\n[TEST 6] NSFR non-compliant bank (heavy short-term funding)")

funding_t6 = FundingStructure(
    tier1_capital=10.0,                    # ASF: 10
    retail_deposits_stable=0.0,
    retail_deposits_less_stable=0.0,
    wholesale_short_term=0.0,
    wholesale_long_term=0.0,
    wholesale_short_non_operational=300.0, # ASF: 0 (non-operational, <6mo)
)
assets_t6 = AssetStructure(
    illiquid_assets=500.0,  # RSF: 500
)
ratio_t6 = nsfr_engine.compute(funding_t6, assets_t6)
check("Non-compliant NSFR < 1.0", not nsfr_engine.is_compliant(ratio_t6),
      f"ratio={ratio_t6:.4f}")
check("Non-compliant NSFR is positive", ratio_t6 > 0,
      f"ratio={ratio_t6:.4f}")

# ------------------------------------------------------------------
# TEST 7: compute_lcr() and compute_nsfr() boundary cases
# ------------------------------------------------------------------
print("\n[TEST 7] Boundary cases (zero outflows, zero RSF)")

ratio_inf_lcr = compute_lcr(hqla=100.0, net_cash_outflows=0.0)
check("compute_lcr(100, 0) = inf", math.isinf(ratio_inf_lcr),
      f"got {ratio_inf_lcr}")

ratio_inf_nsfr = compute_nsfr(available_stable_funding=100.0, required_stable_funding=0.0)
check("compute_nsfr(100, 0) = inf", math.isinf(ratio_inf_nsfr),
      f"got {ratio_inf_nsfr}")

ratio_zero_lcr = compute_lcr(hqla=0.0, net_cash_outflows=100.0)
check("compute_lcr(0, 100) = 0.0", abs(ratio_zero_lcr - 0.0) < 1e-9,
      f"got {ratio_zero_lcr}")

# ------------------------------------------------------------------
# TEST 8: NCO formula — inflow cap + outflow floor
# ------------------------------------------------------------------
print("\n[TEST 8] Net Cash Outflow formula validation")

# Outflows = 100, inflows = 80 (> 75 cap => capped at 75)
# NCO = max(100 - 75, 25) = max(25, 25) = 25
stress_t8a = CashFlowStress(
    retail_stable_deposits=3333.333333,  # outflows = 100
    cash_inflows=80.0,
)
nco_t8a = stress_t8a.net_cash_outflows()
check("NCO with inflows>75%: NCO = 25.0", abs(nco_t8a - 25.0) < 0.01,
      f"NCO={nco_t8a:.4f}")

# Outflows = 100, inflows = 20 (< 75 => not capped)
# NCO = max(100 - 20, 25) = max(80, 25) = 80
stress_t8b = CashFlowStress(
    retail_stable_deposits=3333.333333,  # outflows = 100
    cash_inflows=20.0,
)
nco_t8b = stress_t8b.net_cash_outflows()
check("NCO with inflows<75%: NCO = 80.0", abs(nco_t8b - 80.0) < 0.01,
      f"NCO={nco_t8b:.4f}")

# ------------------------------------------------------------------
# TEST 9: Run-off rates applied correctly
# ------------------------------------------------------------------
print("\n[TEST 9] Run-off rates validation")

# retail_stable * 3% + retail_less_stable * 10%
stress_t9 = CashFlowStress(
    retail_stable_deposits=1000.0,        # outflow = 30.0
    retail_less_stable_deposits=1000.0,   # outflow = 100.0
    wholesale_deposits=1000.0,            # outflow = 250.0
    committed_credit_lines=1000.0,        # outflow = 100.0
)
total_out_t9 = stress_t9.total_outflows()
expected_out = 30.0 + 100.0 + 250.0 + 100.0  # = 480.0
check("Run-off totals: 30+100+250+100=480", abs(total_out_t9 - expected_out) < 0.01,
      f"total={total_out_t9:.2f}")

# ------------------------------------------------------------------
# TEST 10: Level 2B haircut (50% default)
# ------------------------------------------------------------------
print("\n[TEST 10] Level 2B haircut (50% default)")

hqla_t10 = HQLAPortfolio(level1=200.0, level2a=0.0, level2b=100.0)
adj_l2b = hqla_t10.adjusted_level2b()
check("Level 2B: 100 * (1-0.50) = 50.0", abs(adj_l2b - 50.0) < 1e-9,
      f"adj_L2B={adj_l2b:.4f}")

# With L1=200, L2B_sub_cap = (15/85)*200 = 35.29, so 50 capped to 35.29
total_t10 = hqla_t10.total_adjusted_hqla()
l2b_sub_cap = (15.0 / 85.0) * 200.0  # = 35.29...
expected_t10 = 200.0 + l2b_sub_cap
check("Level 2B sub-cap (15% of total) applied",
      abs(total_t10 - expected_t10) < 0.01,
      f"total={total_t10:.4f}, expected={expected_t10:.4f}")

# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"[FAIL] dim_135: {len(errors)} check(s) failed: {errors}")
    sys.exit(1)
else:
    print("[PASS] dim_135: Basel III LCR/NSFR engine -- all checks passed")
PYEOF
