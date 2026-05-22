#!/usr/bin/env bash
# dim_142: Commodity futures roll analysis (contango / backwardation)
# Comprehensive capability verification for sentinel.sfe.commodity_futures_v3
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---------------------------------------------------------------------------
# 1. Imports - all public API
# ---------------------------------------------------------------------------
from sentinel.sfe.commodity_futures_v3 import (
    FuturesContract,
    TermStructure,
    RollAnalytics,
    CostOfCarryModel,
    CommoditySignals,
    roll_yield,
    term_structure_slope,
    implied_convenience_yield,
    total_return_decomposition,
    is_contango,
    calendar_spread,
    # Legacy aliases required by original dim_142.sh
    CommodityRollAnalyzer,
    RollYieldCalculator,
    detect_contango,
    detect_backwardation,
)
print("[OK] All classes and functions imported from commodity_futures_v3")

# ---------------------------------------------------------------------------
# 2. Build a contango term structure: prices rise with maturity
#    contracts at [1, 2, 3, 6, 12] months, prices = [75, 76, 77, 79, 82]
# ---------------------------------------------------------------------------
maturities  = [1, 2, 3, 6, 12]
prices_cng  = [75, 76, 77, 79, 82]

contracts_contango = [
    FuturesContract(commodity="CL", expiry_months_out=m, price=p)
    for m, p in zip(maturities, prices_cng)
]
ts_contango = TermStructure(contracts=contracts_contango, spot_price=74.0)

assert ts_contango.is_contango is True,      "is_contango should be True"
assert ts_contango.is_backwardation is False, "is_backwardation should be False"
print("[OK] is_contango=True, is_backwardation=False for contango term structure")

# slope > 0 in contango
slp = ts_contango.slope()
assert slp > 0, f"slope should be > 0 in contango, got {slp}"
print(f"[OK] term structure slope = {slp:.4f}  (> 0 = contango)")

# curvature is finite
curv = ts_contango.curvature()
assert np.isfinite(curv), f"curvature must be finite, got {curv}"
print(f"[OK] curvature = {curv:.6f}")

# maturities and prices arrays
assert list(ts_contango.maturities) == maturities, "maturities mismatch"
assert list(ts_contango.prices) == prices_cng,     "prices mismatch"
print("[OK] maturities and prices arrays correct")

# ---------------------------------------------------------------------------
# 3. RollAnalytics — contango case
# ---------------------------------------------------------------------------
ra = RollAnalytics()

# roll_yield(near=75, far=76, days=30): should be < 0 (contango drag)
near_c = contracts_contango[0]   # 75 at 1M
far_c  = contracts_contango[1]   # 76 at 2M

ry = ra.roll_yield(near_c, far_c)
assert ry < 0, f"roll_yield should be < 0 in contango, got {ry:.4f}"
print(f"[OK] roll_yield = {ry:.4f}  (< 0 = contango)")

# annualized_roll (explicit inputs matching the spec)
ann_ry = ra.annualized_roll(75.0, 76.0, 30.0)
assert ann_ry < 0, f"annualized_roll should be < 0, got {ann_ry:.4f}"
print(f"[OK] annualized_roll(75, 76, 30) = {ann_ry:.4f}  (less than 0)")

# calendar_spread → regime = 'contango'
cs = ra.calendar_spread(75.0, 76.0)
assert cs["regime"] == "contango", f"Expected 'contango', got {cs['regime']}"
assert cs["absolute"] < 0, f"absolute spread should be < 0 in contango"
print(f"[OK] calendar_spread regime='{cs['regime']}', absolute={cs['absolute']:.2f}")

# ---------------------------------------------------------------------------
# 4. Backwardation case: prices = [82, 81, 80, 78, 75]
# ---------------------------------------------------------------------------
prices_back = [82, 81, 80, 78, 75]
contracts_back = [
    FuturesContract(commodity="CL", expiry_months_out=m, price=p)
    for m, p in zip(maturities, prices_back)
]
ts_back = TermStructure(contracts=contracts_back, spot_price=83.0)

assert ts_back.is_contango is False,     "is_contango should be False in backwardation"
assert ts_back.is_backwardation is True, "is_backwardation should be True"
print("[OK] Backwardation term structure: is_backwardation=True")

ry_back = ra.roll_yield(contracts_back[0], contracts_back[1])
assert ry_back > 0, f"roll_yield > 0 in backwardation, got {ry_back:.4f}"
print(f"[OK] roll_yield in backwardation = {ry_back:.4f}  (> 0)")

cs_back = ra.calendar_spread(82.0, 81.0)
assert cs_back["regime"] == "backwardation", f"Expected 'backwardation', got {cs_back['regime']}"
print(f"[OK] calendar_spread regime='{cs_back['regime']}'")

# ---------------------------------------------------------------------------
# 5. optimal_roll_timing
# ---------------------------------------------------------------------------
opt_idx = ra.optimal_roll_timing(ts_back)
assert 0 <= opt_idx < len(contracts_back) - 1, f"optimal_roll_timing index OOB: {opt_idx}"
print(f"[OK] optimal_roll_timing = {opt_idx}")

# roll_cost_total_return
total_ret = ra.roll_cost_total_return(
    spot_return=0.05, roll_yield=-0.03, collateral_rate=0.05, T=1.0
)
assert np.isfinite(total_ret), "roll_cost_total_return must be finite"
print(f"[OK] roll_cost_total_return = {total_ret:.4f}")

# ---------------------------------------------------------------------------
# 6. CostOfCarryModel
# ---------------------------------------------------------------------------
ccm = CostOfCarryModel()

# Theoretical price: spot=75, r=0.05, q=0.02, u=0.01, T=1/12
tp = ccm.theoretical_price(spot=75.0, r=0.05, convenience_yield=0.02,
                            storage_cost=0.01, T=1/12)
assert tp > 0, f"theoretical_price must be > 0, got {tp}"
print(f"[OK] theoretical_price = {tp:.4f}")

# implied_convenience_yield: spot=75, futures=76, r=0.05, storage=0.02, T=1/12
icy = ccm.implied_convenience_yield(spot=75.0, futures=76.0, r=0.05,
                                     storage_cost=0.02, T=1/12)
assert np.isfinite(icy), f"implied_convenience_yield must be finite, got {icy}"
print(f"[OK] implied_convenience_yield = {icy:.4f}")

# carry_return
cr = ccm.carry_return(spot=75.0, futures=76.0, T=1/12, r=0.05)
assert np.isfinite(cr), f"carry_return must be finite, got {cr}"
print(f"[OK] carry_return = {cr:.4f}")

# ---------------------------------------------------------------------------
# 7. Standalone functions
# ---------------------------------------------------------------------------
ry_sa = roll_yield(75.0, 76.0, 30.0)
assert ry_sa < 0, f"standalone roll_yield should be < 0, got {ry_sa}"
print(f"[OK] standalone roll_yield = {ry_sa:.4f}")

slope_sa = term_structure_slope(
    np.array([1., 2., 3., 6., 12.]),
    np.array([75., 76., 77., 79., 82.])
)
assert slope_sa > 0, f"term_structure_slope should be > 0, got {slope_sa}"
print(f"[OK] standalone term_structure_slope = {slope_sa:.4f}")

icy_sa = implied_convenience_yield(75.0, 76.0, 0.05, 0.02, 1/12)
assert np.isfinite(icy_sa), f"standalone implied_convenience_yield not finite: {icy_sa}"
print(f"[OK] standalone implied_convenience_yield = {icy_sa:.4f}")

# total_return_decomposition: spot=0.05, roll=-0.03, collateral=0.05, T=1 -> total > 0
trd = total_return_decomposition(0.05, -0.03, 0.05, 1.0)
assert "total" in trd,  "total_return_decomposition missing 'total' key"
assert trd["total"] > 0, f"total return should be > 0, got {trd['total']}"
print(f"[OK] total_return_decomposition total = {trd['total']:.4f}")

cs_sa = calendar_spread(75.0, 76.0)
assert cs_sa < 0, f"standalone calendar_spread should be < 0 in contango, got {cs_sa}"
print(f"[OK] standalone calendar_spread = {cs_sa:.4f}")

is_c = is_contango(ts_contango)
assert is_c is True, "is_contango function should return True"
print(f"[OK] is_contango function returns True")

# ---------------------------------------------------------------------------
# 8. CommoditySignals
# ---------------------------------------------------------------------------
cs_obj = CommoditySignals()

# momentum on 14 prices
price_series = np.linspace(70.0, 82.0, 14)
mom = cs_obj.momentum(price_series, window=12)
assert np.isfinite(mom), f"momentum must be finite, got {mom}"
print(f"[OK] CommoditySignals.momentum = {mom:.4f}")

ts_sig = cs_obj.term_structure_signal(ts_contango)
assert np.isfinite(ts_sig), f"term_structure_signal must be finite"
print(f"[OK] CommoditySignals.term_structure_signal = {ts_sig:.6f}")

carry_sig = cs_obj.carry_signal(75.0, 76.0, 30.0)
assert np.isfinite(carry_sig), "carry_signal must be finite"
print(f"[OK] CommoditySignals.carry_signal = {carry_sig:.4f}")

combined = cs_obj.combined_signal(mom, carry_sig, ts_sig)
assert np.isfinite(combined), "combined_signal must be finite"
print(f"[OK] CommoditySignals.combined_signal = {combined:.6f}")

# ---------------------------------------------------------------------------
# 9. Legacy aliases (original dim_142.sh requirements)
# ---------------------------------------------------------------------------
cra = CommodityRollAnalyzer()
assert cra is not None, "CommodityRollAnalyzer must be instantiable"

ryc = RollYieldCalculator()
assert ryc is not None, "RollYieldCalculator must be instantiable"
ry_ryc = ryc(75.0, 76.0, 30.0)
assert np.isfinite(ry_ryc), "RollYieldCalculator must return finite value"

assert detect_contango(ts_contango) is True, "detect_contango should be True"
assert detect_backwardation(ts_back) is True, "detect_backwardation should be True"
print("[OK] Legacy aliases: CommodityRollAnalyzer, RollYieldCalculator, detect_contango, detect_backwardation")

print("\n[PASS] dim_142: Commodity futures roll analysis -- all checks passed")
PYEOF
