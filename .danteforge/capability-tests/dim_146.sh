#!/usr/bin/env bash
# dim_146: LDI Analytics — duration-gap matching, immunization, pension funding
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from sentinel.spm.ldi_analytics_v3 import (
    CashFlow,
    LiabilityProfile,
    AssetPortfolio,
    FundingStatus,
    LDIAnalyzer,
    ImmunizationEngine,
    macaulay_duration,
    modified_duration,
    dv01,
    funding_ratio,
    duration_gap,
)
print("[OK] All imports successful from sentinel.spm.ldi_analytics_v3")

# ---------------------------------------------------------------------------
# 2. Build liability: 20 annual cash flows of 100k @ 5% discount rate
# ---------------------------------------------------------------------------
cash_flows = [CashFlow(time=float(t), amount=100_000.0) for t in range(1, 21)]
liabilities = LiabilityProfile(cash_flows=cash_flows, discount_rate=0.05)

pv_l = liabilities.pv()
print(f"[OK] Liability PV = {pv_l:,.2f}")
assert 1_200_000 < pv_l < 1_300_000, (
    f"FAIL: PV should be between 1.2M and 1.3M, got {pv_l:,.2f}"
)
print(f"[OK] PV in expected range [1.2M, 1.3M]")

# ---------------------------------------------------------------------------
# 3. Macaulay duration: for an annuity at 5%, expect ~8-12 years
# ---------------------------------------------------------------------------
mac_dur = liabilities.duration()
print(f"[OK] Macaulay duration = {mac_dur:.4f} years")
assert 8.0 < mac_dur < 12.0, (
    f"FAIL: Macaulay duration should be 8-12, got {mac_dur:.4f}"
)
print(f"[OK] Macaulay duration in expected range [8, 12]")

# ---------------------------------------------------------------------------
# 4. Modified duration
# ---------------------------------------------------------------------------
mod_dur = liabilities.mod_duration()
print(f"[OK] Modified duration = {mod_dur:.4f}")
assert mod_dur > 7.0, f"FAIL: Modified duration should be > 7, got {mod_dur:.4f}"
assert mod_dur < mac_dur + 0.01, (
    f"FAIL: Modified duration ({mod_dur:.4f}) should be <= Macaulay ({mac_dur:.4f})"
)
print(f"[OK] Modified duration > 7 and < Macaulay duration")

# Manual formula check
expected_mod = mac_dur / (1 + 0.05 / 1)
assert abs(mod_dur - expected_mod) < 1e-8, (
    f"FAIL: Modified duration formula mismatch: {mod_dur:.8f} vs {expected_mod:.8f}"
)
print(f"[OK] Modified duration formula verified: D_mac/(1+y/n) = {mod_dur:.6f}")

# ---------------------------------------------------------------------------
# 5. DV01 of liabilities
# ---------------------------------------------------------------------------
dv01_l = liabilities.dv01()
expected_dv01 = mod_dur * pv_l / 10_000.0
print(f"[OK] Liability DV01 = {dv01_l:.4f}")
assert abs(dv01_l - expected_dv01) < 1e-4, (
    f"FAIL: DV01 mismatch: {dv01_l:.6f} vs {expected_dv01:.6f}"
)
print(f"[OK] DV01 formula verified: D_mod * PV / 10000 = {dv01_l:.4f}")

# ---------------------------------------------------------------------------
# 6. Asset portfolio: 1.5M value, 10yr duration, 5% YTM
# ---------------------------------------------------------------------------
assets = AssetPortfolio(
    market_value=1_500_000.0,
    duration=10.0,
    yield_to_maturity=0.05,
    dv01=1_500.0,   # 10yr * 1.5M / 10000 = 1500
)

# ---------------------------------------------------------------------------
# 7. LDI Analyzer: funding status
# ---------------------------------------------------------------------------
analyzer = LDIAnalyzer(assets=assets, liabilities=liabilities)
status = analyzer.funding_status()

print(f"[OK] Funding ratio = {status.funding_ratio:.4f}")
assert status.funding_ratio > 1.0, (
    f"FAIL: Should be over-funded (ratio > 1.0), got {status.funding_ratio:.4f}"
)
print(f"[OK] Fully funded: {status.is_fully_funded()}")
assert status.is_fully_funded(), "FAIL: is_fully_funded() should be True"

print(f"[OK] Surplus = {status.surplus:,.2f}")
assert status.surplus > 0, f"FAIL: Surplus should be positive, got {status.surplus:,.2f}"

# FR ≈ 1.5M / 1.247M ≈ 1.20
assert 1.1 < status.funding_ratio < 1.4, (
    f"FAIL: Funding ratio should be ~1.20, got {status.funding_ratio:.4f}"
)
print(f"[OK] Funding ratio ~1.20 as expected (1.5M assets / ~1.247M liabilities)")

# ---------------------------------------------------------------------------
# 8. Duration gap
# ---------------------------------------------------------------------------
dgap = analyzer.duration_gap()
print(f"[OK] Duration gap = {dgap:.4f} years")
assert abs(dgap) < 15.0, (
    f"FAIL: Duration gap magnitude should be < 15, got {abs(dgap):.4f}"
)
print(f"[OK] Duration gap is within expected bounds (abs < 15)")

# Verify formula: D_A - (L/A) * D_L
d_a = assets.duration
d_l = mac_dur
ratio_la = pv_l / assets.market_value
expected_gap = d_a - ratio_la * d_l
assert abs(dgap - expected_gap) < 1e-8, (
    f"FAIL: Duration gap formula: {dgap:.8f} vs {expected_gap:.8f}"
)
print(f"[OK] Duration gap formula: D_A - (L/A)*D_L = {expected_gap:.4f}")

# ---------------------------------------------------------------------------
# 9. Surplus at risk (1% rate shock)
# ---------------------------------------------------------------------------
sar = analyzer.surplus_at_risk(interest_rate_shock=0.01)
print(f"[OK] Surplus at risk (+1% shock) = {sar:,.2f}")
assert isinstance(sar, float) and not np.isnan(sar), (
    "FAIL: surplus_at_risk should return finite float"
)
print(f"[OK] surplus_at_risk returns finite value: {sar:,.2f}")

# ---------------------------------------------------------------------------
# 10. Glide path: equity allocation decreases as funding ratio improves
# ---------------------------------------------------------------------------
funding_ratios = np.array([0.80, 0.90, 1.00])
equity_alloc = analyzer.glide_path(funding_ratios)
print(f"[OK] Glide path equity allocations: {equity_alloc}")
assert len(equity_alloc) == 3, "FAIL: glide_path should return 3 values"
# Equity should be decreasing as FR increases
assert equity_alloc[0] > equity_alloc[1] > equity_alloc[2], (
    f"FAIL: Equity allocation should decrease as FR increases: {equity_alloc}"
)
assert equity_alloc[2] <= 0.5, (  # at FR=1.0 equity should be low/zero
    f"FAIL: At FR=1.0, equity should be near 0, got {equity_alloc[2]}"
)
print(f"[OK] Glide path: equity decreases from {equity_alloc[0]:.1f}% to {equity_alloc[2]:.1f}%")

# ---------------------------------------------------------------------------
# 11. Immunization engine
# ---------------------------------------------------------------------------
engine = ImmunizationEngine()

# Unmatched: asset duration=10, liability duration=~9.7 -- not immunized
gap_result = engine.immunization_gap(assets, liabilities)
print(f"[OK] Immunization gap: {gap_result}")
assert "duration_gap" in gap_result, "FAIL: immunization_gap should have 'duration_gap'"
assert "funding_ratio" in gap_result, "FAIL: immunization_gap should have 'funding_ratio'"
assert "surplus" in gap_result, "FAIL: immunization_gap should have 'surplus'"

# Create a duration-matched asset
matched_assets = AssetPortfolio(
    market_value=pv_l,   # exactly funded
    duration=mac_dur,    # exactly matched
    yield_to_maturity=0.05,
    dv01=mod_dur * pv_l / 10_000.0,
)
is_imm = engine.is_immunized(matched_assets, liabilities, tol=0.05)
print(f"[OK] Duration-matched assets -- is_immunized = {is_imm}")
assert is_imm, (
    f"FAIL: Exactly duration-matched assets should be immunized. "
    f"gap={engine.immunization_gap(matched_assets, liabilities)['duration_gap']:.4f}"
)

# Assets with very different duration should NOT be immunized
mismatched = AssetPortfolio(
    market_value=pv_l,
    duration=2.0,       # very short vs ~9.7yr liabilities
    yield_to_maturity=0.04,
    dv01=200.0,
)
not_imm = engine.is_immunized(mismatched, liabilities, tol=0.01)
assert not not_imm, "FAIL: Mismatched duration should NOT be immunized"
print(f"[OK] Mismatched duration -- is_immunized = {not_imm} (False as expected)")

# ---------------------------------------------------------------------------
# 12. Standalone functions
# ---------------------------------------------------------------------------
# macaulay_duration
mac_check = macaulay_duration(cash_flows, 0.05)
assert abs(mac_check - mac_dur) < 1e-8, "FAIL: standalone macaulay_duration mismatch"
print(f"[OK] standalone macaulay_duration = {mac_check:.4f}")

# modified_duration
mod_check = modified_duration(mac_dur, 0.05, freq=1)
assert abs(mod_check - mod_dur) < 1e-8, "FAIL: standalone modified_duration mismatch"
print(f"[OK] standalone modified_duration = {mod_check:.4f}")

# dv01 alias
dv01_check = dv01(mod_dur, pv_l)
assert abs(dv01_check - dv01_l) < 1e-4, "FAIL: standalone dv01 mismatch"
print(f"[OK] standalone dv01 = {dv01_check:.4f}")

# funding_ratio
fr_check = funding_ratio(assets.market_value, pv_l)
assert abs(fr_check - status.funding_ratio) < 1e-8, "FAIL: standalone funding_ratio mismatch"
print(f"[OK] standalone funding_ratio = {fr_check:.4f}")

# duration_gap
dg_check = duration_gap(d_a, mac_dur, assets.market_value, pv_l)
assert abs(dg_check - dgap) < 1e-8, "FAIL: standalone duration_gap mismatch"
print(f"[OK] standalone duration_gap = {dg_check:.4f}")

# ---------------------------------------------------------------------------
# 13. Summary
# ---------------------------------------------------------------------------
print("\n--- LDI Analytics Summary ---")
print(f"  Cash flows    : {len(cash_flows)} annual payments of $100k")
print(f"  Discount rate : 5%")
print(f"  PV(Liab)      : ${pv_l:,.2f}")
print(f"  Macaulay dur  : {mac_dur:.4f} yr")
print(f"  Modified dur  : {mod_dur:.4f} yr")
print(f"  DV01(Liab)    : ${dv01_l:.4f}")
print(f"  Asset value   : $1,500,000")
print(f"  Funding ratio : {status.funding_ratio:.4f}")
print(f"  Surplus       : ${status.surplus:,.2f}")
print(f"  Duration gap  : {dgap:.4f} yr")
print(f"  Surplus@Risk  : ${sar:,.2f}")
print(f"  Glide path    : {equity_alloc.tolist()}")

print("\n[PASS] dim_146: LDI duration-gap analytics")
PYEOF
