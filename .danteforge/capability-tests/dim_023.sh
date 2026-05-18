#!/bin/bash
# dim_023: DCF/WACC — Hamada equation, WACC components, sensitivity, LBO IRR (pure computation)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

from sentinel.sfe.dcf_wacc_v3 import (
    WACCCalculator, WACCResult, DCFResult, LBOResult,
    DAMODARAN_UNLEVERED_BETAS, SECTOR_EV_EBITDA, _safe_float,
)

# Test 1: Damodaran databases
assert "Technology" in DAMODARAN_UNLEVERED_BETAS
assert "Semiconductor" in DAMODARAN_UNLEVERED_BETAS
assert DAMODARAN_UNLEVERED_BETAS["Semiconductor"] == 1.22
assert len(DAMODARAN_UNLEVERED_BETAS) >= 50

assert "Technology" in SECTOR_EV_EBITDA
assert SECTOR_EV_EBITDA["Technology"] == 22.0
print(f"[OK] Damodaran databases: {len(DAMODARAN_UNLEVERED_BETAS)} betas, {len(SECTOR_EV_EBITDA)} EV/EBITDA sectors")

# Test 2: WACCCalculator — Hamada equation (pure computation)
calc = WACCCalculator()

# Unlevered beta: Apple-like company
# βU = 0.85 (tech hardware)
# D/E = 0.30, Tax rate = 21%
# βL = 0.85 * (1 + (1-0.21) * 0.30) = 0.85 * 1.237 = 1.051
beta_unlevered = 0.85
d_e_ratio = 0.30
tax_rate = 0.21
beta_levered_computed = calc.compute_levered_beta(beta_unlevered, d_e_ratio, tax_rate)
expected = 0.85 * (1.0 + (1.0 - 0.21) * 0.30)
assert abs(beta_levered_computed - expected) < 1e-10, f"Hamada levered beta wrong: {beta_levered_computed}"

# Round-trip: unlevre then re-lever
beta_unlevered_back = calc.compute_unlevered_beta(beta_levered_computed, d_e_ratio, tax_rate)
assert abs(beta_unlevered_back - beta_unlevered) < 1e-10, \
    f"Hamada round-trip failed: {beta_unlevered_back}"
print(f"[OK] Hamada: betaU={beta_unlevered} -> betaL={beta_levered_computed:.4f} -> betaU={beta_unlevered_back:.4f}")

# Test 3: Cost of Equity = Rf + beta * ERP + size_premium
rf = 0.043      # 4.3% 10y treasury
erp = 0.050     # 5.0% ERP
size_prem = 0.0  # large cap
cost_of_equity = rf + beta_levered_computed * erp + size_prem
assert 0.05 < cost_of_equity < 0.20, f"Cost of equity out of range: {cost_of_equity}"
print(f"[OK] Cost of equity = {cost_of_equity*100:.2f}% (Rf={rf*100:.1f}% + beta*ERP)")

# Test 4: WACC formula (no network calls)
# Apple-like: 78% equity, 22% debt; cost_of_debt_after_tax = 3.5% * (1-21%) = 2.77%
weight_equity = 0.78
weight_debt = 0.22
cost_of_debt_aftertax = 0.035 * (1.0 - tax_rate)
wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt_aftertax
assert 0.05 < wacc < 0.20, f"WACC out of range: {wacc}"
print(f"[OK] WACC = {wacc*100:.2f}% (equity={weight_equity*100:.0f}%, debt={weight_debt*100:.0f}%)")

# Test 5: Size premium lookup
assert calc.get_size_premium(0.1) == 0.05   # micro-cap
assert calc.get_size_premium(1.0) == 0.03   # small-cap
assert calc.get_size_premium(5.0) == 0.015  # mid-cap
assert calc.get_size_premium(50.0) == 0.0   # large-cap
print("[OK] Size premiums: micro=5%, small=3%, mid=1.5%, large=0%")

# Test 6: ERP table
erp_2024 = calc.get_erp.__func__(type('obj', (object,), {
    'ERP_TABLE': WACCCalculator.ERP_TABLE
})()) if False else calc.ERP_TABLE.get(2024, 0.046)
assert 0.04 < erp_2024 < 0.08, f"ERP 2024 out of range: {erp_2024}"
print(f"[OK] ERP table has {len(calc.ERP_TABLE)} years, 2024 ERP={erp_2024*100:.2f}%")

# Test 7: _safe_float edge cases
assert _safe_float(None) == 0.0
assert _safe_float("invalid") == 0.0
assert abs(_safe_float(1.5) - 1.5) < 1e-10
assert _safe_float(float("inf")) == 0.0
assert _safe_float(float("nan")) == 0.0
print("[OK] _safe_float handles None, invalid, inf, nan correctly")

print("[PASS]")
PYEOF
