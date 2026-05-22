#!/usr/bin/env bash
# dim_136: Counterparty Credit Risk — CVA / DVA / Wrong-Way Risk
# Tests pure computation logic (no network calls required)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# 1. Import verification
# ---------------------------------------------------------------------------
from sentinel.spm.counterparty_risk_v3 import (
    ExposureProfile,
    CreditCurve,
    CVACalculator,
    DVACalculator,
    WrongWayRisk,
    compute_cva,
    compute_dva,
    build_discount_factors,
)
print("[OK] All counterparty risk classes imported")

# ---------------------------------------------------------------------------
# 2. Build 5-year interest-rate swap exposure profile
#    EE rises from 0 to $10M (peak at year 3) then falls back to 0
# ---------------------------------------------------------------------------
times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# Hump-shaped EE typical of an interest-rate swap
ee = [1_000_000, 3_000_000, 5_500_000, 7_500_000, 9_000_000,
      10_000_000, 8_500_000, 6_000_000, 3_500_000, 0]

exposure = ExposureProfile.from_ee(times=times, ee=ee)
assert len(exposure.times) == 10
assert len(exposure.effective_expected_exposure) == 10
# EEE must be non-decreasing then plateau
for i in range(1, len(exposure.effective_expected_exposure)):
    assert exposure.effective_expected_exposure[i] >= exposure.effective_expected_exposure[i-1], \
        f"EEE not non-decreasing at index {i}"
assert exposure.peak_exposure == 10_000_000
print(f"[OK] ExposureProfile: peak EE = ${exposure.peak_exposure:,.0f}, EEE non-decreasing")

# ---------------------------------------------------------------------------
# 3. Build counterparty credit curve — flat 100bps, 40% recovery
# ---------------------------------------------------------------------------
counterparty_curve = CreditCurve.from_flat_spread(
    spread_bps=100.0,  # 100 bps = 1%
    times=times,
    recovery_rate=0.40,
)
assert all(0 < s <= 1.0 for s in counterparty_curve.survival_probabilities)
# Survival must be decreasing
for i in range(1, len(counterparty_curve.survival_probabilities)):
    assert counterparty_curve.survival_probabilities[i] <= counterparty_curve.survival_probabilities[i-1]
h = counterparty_curve.hazard_rate()
expected_h = (0.01 / 0.60)  # spread / LGD
assert abs(h - expected_h) < 1e-6, f"Hazard rate mismatch: {h} vs {expected_h}"
print(f"[OK] CreditCurve: hazard_rate={h:.6f} ({h*10000:.1f} bps), S(5y)={counterparty_curve.survival_probabilities[-1]:.4f}")

# Marginal PD sums to 1 - S(T)
total_pd = sum(
    counterparty_curve.marginal_pd(times[i-1] if i > 0 else 0, times[i])
    for i in range(len(times))
)
expected_total = 1.0 - counterparty_curve.survival_probabilities[-1]
assert abs(total_pd - expected_total) < 1e-9, f"Total PD mismatch: {total_pd} vs {expected_total}"
print(f"[OK] Marginal PDs sum to {total_pd:.6f} = 1 - S(T) = {expected_total:.6f}")

# ---------------------------------------------------------------------------
# 4. Build discount factors (3% flat risk-free rate)
# ---------------------------------------------------------------------------
dfs = build_discount_factors(times, rate=0.03)
assert len(dfs) == len(times)
assert all(0 < df <= 1.0 for df in dfs)
assert dfs[-1] < dfs[0], "Discount factors must be decreasing"
print(f"[OK] Discount factors: DF(0.5)={dfs[0]:.4f}, DF(5.0)={dfs[-1]:.4f}")

# ---------------------------------------------------------------------------
# 5. Compute CVA
# ---------------------------------------------------------------------------
cva_calc = CVACalculator()
lgd_counterparty = 0.60  # 1 - 0.40

cva = cva_calc.compute(exposure, counterparty_curve, dfs)
assert cva > 0, f"CVA must be positive, got {cva}"

max_possible_cva = exposure.peak_exposure * lgd_counterparty
assert cva < max_possible_cva, \
    f"CVA must be < EE_max * LGD = {max_possible_cva:,.0f}, got {cva:,.2f}"

print(f"[OK] CVA = ${cva:,.2f} (positive, < EE_max*LGD=${max_possible_cva:,.0f})")

# Also test via convenience function
cva_fn = compute_cva(exposure, counterparty_curve, lgd_counterparty, dfs)
assert abs(cva_fn - cva) < 0.01, f"compute_cva mismatch: {cva_fn} vs {cva}"
print(f"[OK] compute_cva() convenience function matches: ${cva_fn:,.2f}")

# ---------------------------------------------------------------------------
# 6. Compute DVA (own credit curve — assume same spread for this test)
# ---------------------------------------------------------------------------
own_curve = CreditCurve.from_flat_spread(
    spread_bps=80.0,  # our own credit spread (slightly better)
    times=times,
    recovery_rate=0.40,
)
lgd_own = 0.60

dva_calc = DVACalculator()
dva = dva_calc.compute(exposure, own_curve, dfs)
assert dva > 0, f"DVA must be positive, got {dva}"
print(f"[OK] DVA = ${dva:,.2f} (positive)")

# Also test via convenience function
dva_fn = compute_dva(exposure, own_curve, lgd_own, dfs)
assert abs(dva_fn - dva) < 0.01, f"compute_dva mismatch: {dva_fn} vs {dva}"
print(f"[OK] compute_dva() convenience function matches: ${dva_fn:,.2f}")

# ---------------------------------------------------------------------------
# 7. Compute BCVA = CVA - DVA
# ---------------------------------------------------------------------------
bcva_result = cva_calc.compute_bilateral(exposure, counterparty_curve, own_curve, dfs)
assert "cva" in bcva_result and "dva" in bcva_result and "bcva" in bcva_result
assert abs(bcva_result["cva"] - cva) < 0.01
assert abs(bcva_result["dva"] - dva) < 0.01
bcva = bcva_result["bcva"]
assert abs(bcva - (cva - dva)) < 0.01
print(f"[OK] BCVA = CVA - DVA = ${cva:,.2f} - ${dva:,.2f} = ${bcva:,.2f}")

# ---------------------------------------------------------------------------
# 8. Regulatory simplified CVA
#    notional=$100M, maturity=5y, spread=100bps -> verify in [0.1M, 5M]
# ---------------------------------------------------------------------------
reg_cva = cva_calc.compute_regulatory(
    notional=100_000_000,
    maturity=5.0,
    cds_spread=0.01,   # 100bps in decimal
    lgd=0.60,
    discount_rate=0.03,
)
assert reg_cva > 100_000, \
    f"Regulatory CVA must be > $0.1M, got ${reg_cva:,.0f}"
assert reg_cva < 5_000_000, \
    f"Regulatory CVA must be < $5M, got ${reg_cva:,.0f}"
print(f"[OK] Regulatory CVA: notional=$100M, 5y, 100bps -> ${reg_cva:,.0f} in [$0.1M, $5M]")

# ---------------------------------------------------------------------------
# 9. Wrong-Way Risk
# ---------------------------------------------------------------------------
wwr = WrongWayRisk()

# Positive correlation -> WWR -> multiplier > 1
mult_pos = wwr.compute_wwr_multiplier(0.5)
assert mult_pos > 1.0, f"Positive correlation must give multiplier > 1, got {mult_pos}"
assert wwr.classify(0.5) == "WWR"

# Zero correlation -> multiplier = 1.0
mult_zero = wwr.compute_wwr_multiplier(0.0)
assert abs(mult_zero - 1.0) < 1e-9, f"Zero correlation: multiplier should be 1.0, got {mult_zero}"
assert wwr.classify(0.0) == "NEUTRAL"

# Negative correlation -> RWR -> multiplier < 1
mult_neg = wwr.compute_wwr_multiplier(-0.5)
assert mult_neg < 1.0, f"Negative correlation must give multiplier < 1, got {mult_neg}"
assert wwr.classify(-0.5) == "RWR"

# Maximum WWR (rho=1): multiplier ~ 1 + sqrt(2/pi) ~ 1.7979
max_mult = wwr.compute_wwr_multiplier(1.0)
expected_max = 1.0 + math.sqrt(2.0 / math.pi)
assert abs(max_mult - expected_max) < 1e-9

# Adjusted CVA > base CVA for WWR
adj_cva = wwr.adjust_cva(cva, 0.5)
assert adj_cva > cva, f"WWR-adjusted CVA must exceed base CVA"
print(f"[OK] WWR multiplier(rho=0.5)={mult_pos:.4f} > 1.0")
print(f"[OK] WWR multiplier(rho=0.0)={mult_zero:.4f} = 1.0")
print(f"[OK] WWR multiplier(rho=-0.5)={mult_neg:.4f} < 1.0")
print(f"[OK] WWR-adjusted CVA: ${adj_cva:,.2f} > base ${cva:,.2f}")

# ---------------------------------------------------------------------------
# 10. Edge cases and robustness
# ---------------------------------------------------------------------------

# Single-period CVA — closed form check
single_exp = ExposureProfile.from_ee(times=[1.0], ee=[5_000_000])
single_dfs = [0.97]
single_curve = CreditCurve(times=[1.0], survival_probabilities=[0.99], recovery_rate=0.4)
single_cva = cva_calc.compute(single_exp, single_curve, single_dfs)
# PD = 1 - 0.99 = 0.01; EE = 5M; LGD = 0.6; DF = 0.97
expected_single = 0.6 * 0.01 * 5_000_000 * 0.97
assert abs(single_cva - expected_single) < 0.01, \
    f"Single-period CVA: {single_cva:.2f} vs expected {expected_single:.2f}"
print(f"[OK] Single-period CVA = ${single_cva:,.2f} (expected ${expected_single:,.2f})")

# WWR multiplier boundary: rho=1 and rho=-1
assert abs(wwr.compute_wwr_multiplier(1.0) - (1.0 + math.sqrt(2/math.pi))) < 1e-9
assert abs(wwr.compute_wwr_multiplier(-1.0) - (1.0 - math.sqrt(2/math.pi))) < 1e-9
print(f"[OK] WWR boundary: max={wwr.compute_wwr_multiplier(1.0):.4f}, min={wwr.compute_wwr_multiplier(-1.0):.4f}")

# ExposureProfile ENE is negative of EE
assert exposure.expected_negative_exposure[5] == -10_000_000
print("[OK] ENE = negative of EE confirmed")

# CreditCurve.survival() interpolation
s_mid = counterparty_curve.survival(2.75)
s_lo  = counterparty_curve.survival(2.5)
s_hi  = counterparty_curve.survival(3.0)
assert s_lo >= s_mid >= s_hi, "Survival interpolation must be monotone decreasing"
print(f"[OK] Survival interpolation monotone: S(2.5)={s_lo:.4f} >= S(2.75)={s_mid:.4f} >= S(3.0)={s_hi:.4f}")

print("\n[PASS] dim_136: Counterparty Credit Risk -- CVA/DVA/WWR all checks passed")
PYEOF
