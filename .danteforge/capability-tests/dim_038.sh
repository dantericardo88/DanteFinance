#!/bin/bash
# dim_038: Bond analytics engine -- DV01, OAS, Z-spread, duration, convexity
# Verifies comprehensive pure-Python bond math in sentinel/sfe/bond_analytics_v3.py
# No network calls, no QuantLib required.
set -e

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# 0. Legacy shim still imports cleanly
# ---------------------------------------------------------------------------
import sentinel.sbx.bond_analytics as ba_shim
print(f"[OK] sbx.bond_analytics imported (QL available: {ba_shim._QL_AVAILABLE})")

# ---------------------------------------------------------------------------
# 1. v3 engine imports and exposes all required classes
# ---------------------------------------------------------------------------
from sentinel.sfe.bond_analytics_v3 import (
    BondCashFlows,
    YieldCalculator,
    DurationConvexity,
    StandaloneSpreadCalculator,
    BondScenarioAnalyzer,
    KeyRateDuration,
    BondPricer,
)
print("[OK] All dim_038 analytics classes imported from bond_analytics_v3")

# ---------------------------------------------------------------------------
# 2. BondCashFlows -- generate coupon + principal schedule
# ---------------------------------------------------------------------------
from datetime import date, timedelta

settle = date(2026, 1, 1)
maturity = date(2031, 1, 1)   # exactly 5 years
cfs = BondCashFlows.generate(
    coupon_rate=0.05,
    face=100.0,
    maturity_date=maturity,
    settlement=settle,
    freq=2,
)
assert len(cfs) == 10, f"Expected 10 semi-annual coupons for 5Y bond, got {len(cfs)}"
last_cf = cfs[-1][1]
assert abs(last_cf - 102.5) < 0.01, f"Last CF (coupon+face) should be ~102.5, got {last_cf}"
print(f"[OK] BondCashFlows: 10 cash flows, last={last_cf:.2f} (coupon+face)")

# ---------------------------------------------------------------------------
# 3. YieldCalculator.ytm -- price=95, coupon=5%, 5Y => YTM ~6.1%
# ---------------------------------------------------------------------------
ytm_val = YieldCalculator.ytm(
    price=95.0,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    freq=2,
)
assert 5.9 <= ytm_val * 100 <= 6.3, f"YTM should be ~6.1%, got {ytm_val*100:.4f}%"
print(f"[OK] YTM solver: price=95, 5Y 5% coupon => YTM={ytm_val*100:.4f}% (expected ~6.1%)")

# ---------------------------------------------------------------------------
# 4. YieldCalculator.price_from_ytm -- round-trip consistency
# ---------------------------------------------------------------------------
recovered_price = YieldCalculator.price_from_ytm(
    ytm=ytm_val,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    freq=2,
)
assert abs(recovered_price - 95.0) < 0.01, \
    f"Round-trip price should recover 95.0, got {recovered_price:.4f}"
print(f"[OK] Price round-trip: YTM=>price={recovered_price:.4f} (expected 95.0)")

# ---------------------------------------------------------------------------
# 5. Par bond: YTM = coupon rate
# ---------------------------------------------------------------------------
ytm_par = YieldCalculator.ytm(
    price=100.0,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    freq=2,
)
assert abs(ytm_par * 100 - 5.0) < 0.02, \
    f"Par bond YTM should ~= coupon rate 5%, got {ytm_par*100:.4f}%"
print(f"[OK] Par bond: YTM={ytm_par*100:.4f}% ~= 5.00% coupon rate")

# ---------------------------------------------------------------------------
# 6. DurationConvexity.macaulay -- 5Y par bond Macaulay ~4.3-4.5Y
# ---------------------------------------------------------------------------
mac = DurationConvexity.macaulay(
    price=100.0,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    freq=2,
)
assert 4.0 <= mac <= 5.0, f"Macaulay duration for 5Y par bond should be 4-5Y, got {mac:.4f}"
print(f"[OK] Macaulay duration: 5Y 5% par bond => {mac:.4f}Y (expected ~4.3-4.5Y)")

# ---------------------------------------------------------------------------
# 7. DurationConvexity.modified -- mod = mac / (1 + ytm/freq)
# ---------------------------------------------------------------------------
mod = DurationConvexity.modified(ytm=ytm_par, macaulay=mac, freq=2)
expected_mod = mac / (1.0 + ytm_par / 2)
assert abs(mod - expected_mod) < 1e-8, \
    f"Modified duration formula error: got {mod:.6f}, expected {expected_mod:.6f}"
print(f"[OK] Modified duration: {mod:.4f} (= mac/(1+ytm/freq))")

# ---------------------------------------------------------------------------
# 8. DurationConvexity.dv01 -- 5Y par bond DV01 ~4-5 cents per $100 face
# ---------------------------------------------------------------------------
dv01_val = DurationConvexity.dv01(
    modified_duration=mod,
    price=100.0,
    face=100.0,
)
# Standard: 5Y par bond DV01 = mod_dur * (price/face*100) * 0.0001
# With price=face=100: dv01 = 4.375 * 100 * 0.0001 = 0.04375 (dollars per $100 face)
# i.e. ~4.375 cents expressed as $0.04375
assert 0.035 <= dv01_val <= 0.060, \
    f"DV01 for 5Y par bond should be $0.035-$0.060 per $100 face (~4-5 cents), got {dv01_val:.4f}"
print(f"[OK] DV01: 5Y par bond = ${dv01_val:.4f} per $100 face ({dv01_val*100:.2f} cents, expected ~4-5 cents)")

# ---------------------------------------------------------------------------
# 9. DurationConvexity.convexity -- must be non-negative for standard bonds
# ---------------------------------------------------------------------------
conv = DurationConvexity.convexity(
    price=100.0,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    freq=2,
)
assert conv >= 0, f"Convexity must be non-negative for standard bond, got {conv:.6f}"
assert conv > 10, f"Convexity for 5Y bond should be >10, got {conv:.4f}"
print(f"[OK] Convexity: {conv:.4f} (non-negative, >10 for 5Y bond)")

# ---------------------------------------------------------------------------
# 10. StandaloneSpreadCalculator.g_spread
# ---------------------------------------------------------------------------
# When ytm = benchmark, G-spread = 0
gs_zero = StandaloneSpreadCalculator.g_spread(ytm=0.06, benchmark_ytm=0.06)
assert abs(gs_zero) < 1e-10, f"G-spread should be 0 when ytm==benchmark, got {gs_zero}"
# When ytm > benchmark
gs = StandaloneSpreadCalculator.g_spread(ytm=0.06, benchmark_ytm=0.045)
assert abs(gs - 0.015) < 1e-10, f"G-spread 6%%-4.5%% should be 0.015 (150bps), got {gs}"
print(f"[OK] G-spread: 6%% vs 4.5%% benchmark => {gs*10000:.1f} bps (expected 150 bps)")

# ---------------------------------------------------------------------------
# 11. StandaloneSpreadCalculator.z_spread -- flat curve: z_spread ~= g_spread
# ---------------------------------------------------------------------------
# Flat treasury curve at 4.5% for all tenors
flat_curve = {t: 4.5 for t in [0.5, 1, 2, 3, 5, 7, 10, 20, 30]}
cfs_for_zspread = BondCashFlows.generate(0.05, 100.0, maturity, settle, 2)

# Get clean price at YTM=6%
price_at_6pct = YieldCalculator.price_from_ytm(
    ytm=0.06, coupon_rate=0.05, face=100.0,
    maturity=maturity, settlement=settle, freq=2,
)
# On a flat curve at 4.5%, Z-spread should ~= 150 bps (= 6% - 4.5%)
z = StandaloneSpreadCalculator.z_spread(
    price=price_at_6pct,
    cash_flows=cfs_for_zspread,
    treasury_curve_fn=flat_curve,
    settlement=settle,
    face=100.0,
    freq=2,
    guess=0.015,
)
assert 0.010 <= z <= 0.020, \
    f"Z-spread on flat curve should ~= 0.015 (150 bps), got {z:.6f} ({z*10000:.1f} bps)"
print(f"[OK] Z-spread (flat curve): {z*10000:.1f} bps (expected ~150 bps; G-spread ~= Z-spread on flat curve)")

# ---------------------------------------------------------------------------
# 12. StandaloneSpreadCalculator.oas
# ---------------------------------------------------------------------------
# OAS = z_spread - option_cost; for a non-callable (option_cost=0) OAS = Z-spread
oas_no_option = StandaloneSpreadCalculator.oas(z_spread=z, option_adjusted_bps=0.0)
assert abs(oas_no_option - z) < 1e-10, "OAS with 0 option cost must equal Z-spread"
# Callable: option costs 30 bps
oas_callable = StandaloneSpreadCalculator.oas(z_spread=z, option_adjusted_bps=30.0)
assert abs(oas_callable - (z - 0.003)) < 1e-10, \
    f"OAS with 30bp option cost error: got {oas_callable:.6f}"
print(f"[OK] OAS: non-callable OAS={oas_no_option*10000:.1f}bps; callable (30bp opt)={oas_callable*10000:.1f}bps")

# ---------------------------------------------------------------------------
# 13. BondScenarioAnalyzer.rate_shock_grid -- 9 rows, correct structure
# ---------------------------------------------------------------------------
import pandas as pd

grid = BondScenarioAnalyzer.rate_shock_grid(
    price=95.0,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    freq=2,
    shocks_bps=[-200, -100, -50, -25, 0, 25, 50, 100, 200],
)
assert isinstance(grid, pd.DataFrame), "rate_shock_grid must return a DataFrame"
assert len(grid) == 9, f"Grid should have 9 rows (one per shock), got {len(grid)}"
assert list(grid.columns) == [
    "shock_bps", "price", "ytm_pct", "mod_duration",
    "convexity", "dv01", "pnl_dollar", "pnl_pct"
], f"Unexpected columns: {list(grid.columns)}"

# Prices should decrease monotonically as shock increases (inverse relationship)
prices = list(grid["price"])
for i in range(len(prices) - 1):
    assert prices[i] >= prices[i + 1], \
        f"Price should decrease as yield rises: {prices[i]:.4f} >= {prices[i+1]:.4f}"

# Zero-shock row should match base price closely
zero_row = grid[grid["shock_bps"] == 0].iloc[0]
assert abs(zero_row["price"] - 95.0) < 0.5, \
    f"Zero-shock price should be ~95.0, got {zero_row['price']:.4f}"

# All convexity values should be non-negative
assert (grid["convexity"] >= 0).all(), "All convexity values must be non-negative"

# P&L should be negative for positive shocks (rates up => price down)
pos_shock = grid[grid["shock_bps"] == 100].iloc[0]
assert pos_shock["pnl_pct"] < 0, \
    f"+100bp shock should give negative P&L, got {pos_shock['pnl_pct']:.4f}"

neg_shock = grid[grid["shock_bps"] == -100].iloc[0]
assert neg_shock["pnl_pct"] > 0, \
    f"-100bp shock should give positive P&L, got {neg_shock['pnl_pct']:.4f}"

print(f"[OK] Rate shock grid: {len(grid)} rows, monotone prices, correct P&L signs")
print(f"     Zero shock price={zero_row['price']:.4f}, +100bp P&L={pos_shock['pnl_pct']:.4f}%%,"
      f" -100bp P&L={neg_shock['pnl_pct']:.4f}%%")

# ---------------------------------------------------------------------------
# 14. KeyRateDuration.compute -- curve-based and approximation modes
# ---------------------------------------------------------------------------
# Mode 1: no curve (triangular approximation)
krd_approx = KeyRateDuration.compute(
    price=100.0,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    treasury_curve=None,
    freq=2,
)
assert set(krd_approx.keys()) == {"2Y", "5Y", "10Y", "30Y", "total"}, \
    f"KRD keys mismatch: {set(krd_approx.keys())}"
assert abs(krd_approx["total"] - mod) < 0.1, \
    f"KRD total should ~= modified duration {mod:.4f}, got {krd_approx['total']:.4f}"
# For a 5Y bond, 5Y node should dominate
assert krd_approx["5Y"] >= krd_approx["2Y"], "5Y KRD should be >= 2Y for a 5Y bond"
assert krd_approx["5Y"] >= krd_approx["10Y"], "5Y KRD should be >= 10Y for a 5Y bond"

# Mode 2: with flat treasury curve (spot-curve repricing)
krd_curve = KeyRateDuration.compute(
    price=100.0,
    coupon_rate=0.05,
    face=100.0,
    maturity=maturity,
    settlement=settle,
    treasury_curve=flat_curve,
    freq=2,
)
assert set(krd_curve.keys()) == {"2Y", "5Y", "10Y", "30Y", "total"}, \
    f"KRD curve-mode keys mismatch: {set(krd_curve.keys())}"
assert krd_curve["total"] > 0, "Total KRD should be positive"

print(f"[OK] KeyRateDuration: approx 2Y={krd_approx['2Y']:.4f}, 5Y={krd_approx['5Y']:.4f},"
      f" 10Y={krd_approx['10Y']:.4f}, 30Y={krd_approx['30Y']:.4f}, total={krd_approx['total']:.4f}")
print(f"     Curve-mode: 2Y={krd_curve['2Y']:.4f}, 5Y={krd_curve['5Y']:.4f},"
      f" 10Y={krd_curve['10Y']:.4f}, 30Y={krd_curve['30Y']:.4f}")

# ---------------------------------------------------------------------------
# 15. BondPricer (existing class) integration -- cross-check with standalone
# ---------------------------------------------------------------------------
pricer = BondPricer()
metrics = pricer.price_from_ytm(
    face=100.0,
    coupon_pct=5.0,
    maturity=maturity,
    settle=settle,
    ytm_pct=5.0,
    freq=2,
)
assert abs(metrics.clean_price - 100.0) < 0.5, \
    f"BondPricer par bond clean price should ~= 100, got {metrics.clean_price:.4f}"
assert metrics.convexity >= 0, "BondPricer convexity must be non-negative"
assert metrics.dv01 > 0, "BondPricer DV01 must be positive"
print(f"[OK] BondPricer integration: par bond price={metrics.clean_price:.4f},"
      f" DV01={metrics.dv01:.6f}, convexity={metrics.convexity:.4f}")

# ---------------------------------------------------------------------------
# 16. YTW callable -- yields to worst
# ---------------------------------------------------------------------------
call_date_early = date(2028, 1, 1)   # 2Y call
ytw = YieldCalculator.ytw_callable(
    price=95.0,
    coupon_rate=0.05,
    face=100.0,
    call_dates=[call_date_early, maturity],
    call_prices=[100.0, 100.0],
    settlement=settle,
    freq=2,
)
assert not math.isnan(ytw), "YTW should not be NaN"
print(f"[OK] YTW callable: price=95, 2Y call at par => YTW={ytw*100:.4f}%%")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("[PASS] dim_038: Comprehensive bond analytics engine -- all 16 checks passed")
print("       Pure Python: no QuantLib, no network calls")
print(f"       YTM={ytm_val*100:.4f}%%, DV01={dv01_val:.4f}c, conv={conv:.2f},"
      f" Z-spread~={z*10000:.1f}bps, grid={len(grid)}rows, KRD nodes=4")
PYEOF
