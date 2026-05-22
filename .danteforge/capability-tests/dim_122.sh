#!/usr/bin/env bash
# dim_122: Trade flow analysis (VWAP, TWAP, Kyle lambda, VPIN)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

try:
    from sentinel.sma.trade_flow_v3 import (
        TradeFlowMetrics, TradeClassifier,
        VWAPEngine, TWAPEngine, VPINCalculator, FuturesBasis,
        classify_trades_tick_rule, compute_vwap, compute_vpin,
        kyle_lambda, roll_yield,
    )
except ImportError as e:
    print(f"[NOT-BUILT] dim_122: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sma.trade_flow_v3")
    sys.exit(0)

# ── 1. Existence checks ───────────────────────────────────────────────────────
assert TradeFlowMetrics is not None, "Missing TradeFlowMetrics"
assert TradeClassifier is not None, "Missing TradeClassifier"
assert VWAPEngine is not None, "Missing VWAPEngine"
assert TWAPEngine is not None, "Missing TWAPEngine"
assert VPINCalculator is not None, "Missing VPINCalculator"
assert FuturesBasis is not None, "Missing FuturesBasis"
assert classify_trades_tick_rule is not None, "Missing classify_trades_tick_rule"
assert compute_vwap is not None, "Missing compute_vwap"
assert compute_vpin is not None, "Missing compute_vpin"
assert kyle_lambda is not None, "Missing kyle_lambda"
assert roll_yield is not None, "Missing roll_yield"
print("[OK] All classes and convenience functions present")

# ── 2. Reference data ─────────────────────────────────────────────────────────
prices  = np.array([100.00, 100.10, 100.05, 100.20, 100.15, 100.30])
volumes = np.array([100.0,  200.0,  150.0,  300.0,  200.0,  400.0])

# ── 3. Tick rule ─────────────────────────────────────────────────────────────
clf = TradeClassifier()
tick = clf.tick_rule(prices)

assert tick.shape == prices.shape, (
    f"tick_rule result shape {tick.shape} != prices shape {prices.shape}"
)
assert set(tick).issubset({1, -1}), f"tick_rule must contain only +1/-1, got {set(tick)}"
assert 1 in tick, "tick_rule result must contain at least one +1"
assert -1 in tick, "tick_rule result must contain at least one -1"
print(f"[OK] tick_rule: shape={tick.shape}, values={set(int(x) for x in tick)}")

# Verify specific transitions: prices go up, then down, then up, then down, then up
# [100.00, 100.10, 100.05, 100.20, 100.15, 100.30]
#  idx0:+1  idx1:+1  idx2:-1  idx3:+1  idx4:-1  idx5:+1
assert int(tick[1]) == 1,  f"tick[1] should be +1 (price up 100.00->100.10), got {tick[1]}"
assert int(tick[2]) == -1, f"tick[2] should be -1 (price down 100.10->100.05), got {tick[2]}"
assert int(tick[3]) == 1,  f"tick[3] should be +1 (price up 100.05->100.20), got {tick[3]}"
print("[OK] tick_rule specific transitions correct")

# ── 4. Lee-Ready ─────────────────────────────────────────────────────────────
# quotes at midpoint - 0.02 so most trades are above the quote
quotes = prices - 0.02
lr = clf.lee_ready(prices, quotes)
assert lr.shape == prices.shape, (
    f"lee_ready result shape {lr.shape} != prices shape {prices.shape}"
)
assert set(lr).issubset({1, -1}), f"lee_ready must return only +1/-1, got {set(lr)}"
print(f"[OK] lee_ready: shape={lr.shape}, values={set(int(x) for x in lr)}")

# ── 5. Bulk volume classification ────────────────────────────────────────────
price_changes = np.diff(prices, prepend=prices[0])
bvc = clf.bulk_volume_classification(price_changes, volumes)
assert bvc.shape == prices.shape, f"BVC shape mismatch: {bvc.shape}"
assert set(bvc).issubset({1, -1}), f"BVC must return only +1/-1, got {set(bvc)}"
print(f"[OK] bulk_volume_classification: shape={bvc.shape}")

# ── 6. VWAP ──────────────────────────────────────────────────────────────────
engine_v = VWAPEngine()
computed_vwap = engine_v.compute_vwap(prices, volumes)

# Expected: (100.00*100 + 100.10*200 + 100.05*150 + 100.20*300 + 100.15*200 + 100.30*400) / 1350
numerator = (100.00*100 + 100.10*200 + 100.05*150 + 100.20*300 + 100.15*200 + 100.30*400)
expected_v = numerator / float(np.sum(volumes))
assert abs(computed_vwap - expected_v) < 1e-6, (
    f"VWAP mismatch: {computed_vwap:.6f} vs expected {expected_v:.6f}"
)
# Verify VWAP is between price min and max (reasonable range check)
assert prices.min() <= computed_vwap <= prices.max(), (
    f"VWAP {computed_vwap:.6f} should be in price range [{prices.min():.4f}, {prices.max():.4f}]"
)
print(f"[OK] VWAPEngine.compute_vwap = {computed_vwap:.6f} (in price range, weighted toward high-vol)")

# Standalone function matches
assert abs(compute_vwap(prices, volumes) - computed_vwap) < 1e-9, \
    "compute_vwap() standalone must match VWAPEngine.compute_vwap()"
print("[OK] compute_vwap() standalone matches VWAPEngine")

# ── 7. VWAP schedule ────────────────────────────────────────────────────────
hist_vols = np.array([100.0, 200.0, 150.0, 300.0, 200.0, 400.0])
weights = engine_v.vwap_schedule(hist_vols, n_intervals=6)
assert weights.shape == (6,), f"vwap_schedule shape should be (6,), got {weights.shape}"
assert abs(float(np.sum(weights)) - 1.0) < 1e-9, (
    f"vwap_schedule weights must sum to 1.0, got {np.sum(weights):.10f}"
)
assert np.all(weights >= 0), "vwap_schedule weights must be non-negative"
print(f"[OK] vwap_schedule: sum={np.sum(weights):.6f}, shape={weights.shape}")

# ── 8. VWAP tracking error ───────────────────────────────────────────────────
exec_prices = np.array([100.15, 100.25])
exec_sizes = np.array([500.0, 500.0])
te = engine_v.tracking_error(exec_prices, exec_sizes, market_vwap=computed_vwap)
assert isinstance(te, float), "tracking_error must return float"
print(f"[OK] VWAP tracking error = {te:.4f} bps")

# ── 9. TWAP ──────────────────────────────────────────────────────────────────
engine_t = TWAPEngine()
computed_twap = engine_t.compute_twap(prices)
expected_twap = float(np.mean(prices))
assert abs(computed_twap - expected_twap) < 1e-9, (
    f"TWAP mismatch: {computed_twap:.6f} vs {expected_twap:.6f}"
)
# Verify TWAP is within price range
assert prices.min() <= computed_twap <= prices.max(), (
    f"TWAP {computed_twap:.6f} should be in price range [{prices.min():.4f}, {prices.max():.4f}]"
)
print(f"[OK] TWAPEngine.compute_twap = {computed_twap:.6f} (simple mean of 6 prices)")

# ── 10. TWAP schedule ────────────────────────────────────────────────────────
schedule = engine_t.twap_schedule(total_size=6000.0, n_intervals=6)
assert schedule.shape == (6,), f"twap_schedule shape should be (6,), got {schedule.shape}"
assert abs(float(np.sum(schedule)) - 6000.0) < 1e-6, (
    f"twap_schedule must sum to total_size, got {np.sum(schedule):.4f}"
)
assert np.allclose(schedule, 1000.0), f"TWAP schedule should be uniform 1000/interval, got {schedule}"
print(f"[OK] twap_schedule: sum={np.sum(schedule):.1f}, uniform={np.allclose(schedule, 1000.0)}")

# ── 11. Adaptive TWAP ────────────────────────────────────────────────────────
vol_levels = np.array([0.02, 0.04, 0.01, 0.03, 0.02, 0.05])
adaptive = engine_t.adaptive_twap(prices, vol_levels, total_size=6000.0)
assert adaptive.shape == (6,), f"adaptive_twap shape should be (6,), got {adaptive.shape}"
assert abs(float(np.sum(adaptive)) - 6000.0) < 1e-4, (
    f"adaptive_twap must sum to total_size, got {np.sum(adaptive):.4f}"
)
# Low vol interval (index 2, vol=0.01) should get more allocation than high vol (index 1, vol=0.04)
assert adaptive[2] > adaptive[1], (
    f"Low-vol interval (idx=2) should have larger allocation than high-vol (idx=1), "
    f"got {adaptive[2]:.4f} vs {adaptive[1]:.4f}"
)
assert adaptive[2] > adaptive[5], (
    f"Low-vol interval (idx=2, vol=0.01) should have more allocation than idx=5 (vol=0.05), "
    f"got {adaptive[2]:.4f} vs {adaptive[5]:.4f}"
)
print(f"[OK] adaptive_twap: sum={np.sum(adaptive):.1f}, "
      f"low-vol gets more: {adaptive[2]:.1f} > {adaptive[1]:.1f}")

# ── 12. VPIN ─────────────────────────────────────────────────────────────────
calc = VPINCalculator()
vpin_val = calc.compute(prices, volumes, bucket_size=200.0)
assert 0.0 <= vpin_val <= 1.0, f"VPIN must be in [0, 1], got {vpin_val:.6f}"
print(f"[OK] VPINCalculator.compute = {vpin_val:.6f} in [0, 1]")

# VPIN series
vpin_series = calc.vpin_series(prices, volumes, bucket_size=200.0, window=3)
assert vpin_series.shape == prices.shape, (
    f"vpin_series shape {vpin_series.shape} != prices shape {prices.shape}"
)
# Last elements should be valid floats in [0,1]
valid_vals = vpin_series[~np.isnan(vpin_series)]
assert len(valid_vals) > 0, "vpin_series should have some non-NaN values"
assert np.all((valid_vals >= 0) & (valid_vals <= 1)), (
    f"All valid VPIN series values must be in [0,1], got {valid_vals}"
)
print(f"[OK] vpin_series: shape={vpin_series.shape}, valid_count={len(valid_vals)}")

# Standalone compute_vpin
vpin_standalone = compute_vpin(prices, volumes, bucket_size=200.0)
assert 0.0 <= vpin_standalone <= 1.0, f"compute_vpin standalone out of range: {vpin_standalone}"
print(f"[OK] compute_vpin() standalone = {vpin_standalone:.6f}")

# ── 13. Futures basis ────────────────────────────────────────────────────────
fb = FuturesBasis()
spot, near, far, days = 4500.0, 4505.0, 4510.0, 30.0

basis_val = fb.basis(spot, near)
assert basis_val > 0, f"basis(spot, near) should be > 0 (near > spot), got {basis_val:.4f}"
assert abs(basis_val - 5.0) < 1e-9, f"basis should be 5.0, got {basis_val:.4f}"
print(f"[OK] FuturesBasis.basis = {basis_val:.4f} (> 0)")

basis_bps_val = fb.basis_bps(spot, near)
assert basis_bps_val > 0, f"basis_bps should be > 0, got {basis_bps_val:.4f}"
assert abs(basis_bps_val - (5.0 / 4500.0 * 10000)) < 1e-6, (
    f"basis_bps mismatch: {basis_bps_val:.6f}"
)
print(f"[OK] FuturesBasis.basis_bps = {basis_bps_val:.4f} bps")

ry = fb.roll_yield(near, far, days)
assert np.isfinite(ry), f"roll_yield must be finite, got {ry}"
# near < far (contango) = negative roll yield
assert ry < 0, f"roll_yield should be < 0 in contango (near < far), got {ry:.6f}"
print(f"[OK] FuturesBasis.roll_yield = {ry:.6f} (< 0, contango)")

ir = fb.implied_repo(spot, near, T=30.0/365.0)
assert np.isfinite(ir), f"implied_repo must be finite, got {ir}"
print(f"[OK] FuturesBasis.implied_repo = {ir:.6f}")

# Standalone roll_yield convenience function
ry_standalone = roll_yield(near, far, days)
assert abs(ry_standalone - ry) < 1e-12, (
    f"roll_yield() standalone mismatch: {ry_standalone:.8f} vs {ry:.8f}"
)
print(f"[OK] roll_yield() standalone matches FuturesBasis.roll_yield()")

# ── 14. Kyle lambda ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
lam_true = 0.002  # $0.002 per share of signed flow
n = 500
signed_flow = rng.normal(0, 10_000, n)
noise = rng.normal(0, 0.001, n)
price_changes_synth = lam_true * signed_flow + noise

lam_hat = kyle_lambda(price_changes_synth, signed_flow)
assert lam_hat > 0, f"kyle_lambda should be > 0 for positively correlated data, got {lam_hat:.8f}"
# Should be reasonably close to the true value
assert abs(lam_hat - lam_true) < 0.0005, (
    f"kyle_lambda estimate {lam_hat:.6f} should be close to {lam_true:.6f}"
)
print(f"[OK] kyle_lambda = {lam_hat:.6f} (true = {lam_true:.6f})")

# ── 15. TradeFlowMetrics container ────────────────────────────────────────────
tfm = TradeFlowMetrics(
    kyle_lambda=lam_hat,
    vpin=vpin_val,
    buy_sell_ratio=1.2,
    net_order_flow=50000.0,
    price_impact_bps_per_M=2.5,
    informed_trading_probability=0.15,
)
assert tfm.kyle_lambda == lam_hat, "TradeFlowMetrics.kyle_lambda mismatch"
assert tfm.vpin == vpin_val, "TradeFlowMetrics.vpin mismatch"
print(f"[OK] TradeFlowMetrics container: lambda={tfm.kyle_lambda:.6f}, vpin={tfm.vpin:.6f}")

print("\n[PASS] dim_122: Trade flow analytics (VWAP/TWAP/VPIN) -- all checks passed")
PYEOF
