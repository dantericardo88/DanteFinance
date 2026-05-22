#!/usr/bin/env bash
# dim_132: HF microstructure signals (roll yield, futures basis, VPIN, realized vol)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ------------------------------------------------------------------ #
# 1. Import from sfe layer (legacy test interface)
# ------------------------------------------------------------------ #
try:
    from sentinel.sfe.hf_signals_v3 import (
        RollYieldSignal, FuturesBasisSignal, compute_roll_yield, compute_futures_basis
    )
    assert RollYieldSignal is not None, "Missing RollYieldSignal"
    assert FuturesBasisSignal is not None, "Missing FuturesBasisSignal"
    assert compute_roll_yield is not None, "Missing compute_roll_yield"
    assert compute_futures_basis is not None, "Missing compute_futures_basis"
    print("[OK] RollYieldSignal present")
    print("[OK] FuturesBasisSignal present")
    print("[OK] compute_roll_yield present")
    print("[OK] compute_futures_basis present")
except ImportError as e:
    print(f"[FAIL] Import from sentinel.sfe.hf_signals_v3 failed: {e}")
    sys.exit(1)

# ------------------------------------------------------------------ #
# 2. Import full API from sma layer
# ------------------------------------------------------------------ #
from sentinel.sma.hf_microstructure_v3 import (
    FuturesMicrostructure, RealizedVolAnalytics, OrderFlowImbalance,
    HFSignalGenerator, futures_basis, roll_yield as ry_fn,
    realized_vol_decomposition, vpin, ofi_lambda,
)

# ------------------------------------------------------------------ #
# 3. futures_basis: spot=4500, futures=4515, T=0.25
# ------------------------------------------------------------------ #
fm = FuturesMicrostructure()
bm = fm.basis(spot=4500, futures=4515, T=0.25)
assert abs(bm.basis - 15.0) < 1e-9, f"basis={bm.basis}, expected 15"
assert abs(bm.basis_pct - 15/4500*100) < 1e-6, f"basis_pct={bm.basis_pct}"
assert bm.market_regime == "contango", f"regime={bm.market_regime}"
print(f"[OK] futures_basis: basis={bm.basis}, basis_pct={bm.basis_pct:.4f}%, regime={bm.market_regime}")

# convenience function
bm2 = futures_basis(4500, 4515, 0.25)
assert abs(bm2.basis - 15.0) < 1e-9, "convenience futures_basis mismatch"
print("[OK] convenience futures_basis matches")

# ------------------------------------------------------------------ #
# 4. roll_yield: near=4505, far=4515, days=30 → negative (contango)
# ------------------------------------------------------------------ #
ry = fm.roll_yield(near=4505, far=4515, days_to_roll=30)
assert ry < 0, f"roll_yield should be negative in contango, got {ry}"
print(f"[OK] roll_yield: {ry:.6f} (negative = contango)")

ry2 = ry_fn(near=4505, far=4515, days_to_roll=30)
assert abs(ry - ry2) < 1e-12, "convenience roll_yield mismatch"
print("[OK] convenience roll_yield matches")

# backwardation: near > far → positive
ry_back = fm.roll_yield(near=4515, far=4505, days_to_roll=30)
assert ry_back > 0, f"roll_yield in backwardation should be positive, got {ry_back}"
print(f"[OK] backwardation roll_yield: {ry_back:.6f} (positive)")

# ------------------------------------------------------------------ #
# 5. RealizedVolComponents: 100 normal returns
# ------------------------------------------------------------------ #
rng = np.random.default_rng(42)
returns_100 = rng.normal(0, 0.01, 100)
rva = RealizedVolAnalytics()
comps = rva.decompose(returns_100)

assert comps.realized_vol > 0, f"realized_vol={comps.realized_vol}"
assert 0.0 <= comps.jump_ratio <= 1.0, f"jump_ratio={comps.jump_ratio}"
# continuous + jump ≈ realized_variance (within 1e-6)
recon = comps.continuous_component + comps.jump_component
assert abs(recon - comps.realized_variance) < 1e-6, \
    f"decomp error: cont+jump={recon}, RV={comps.realized_variance}"
print(f"[OK] RealizedVolComponents: RV={comps.realized_variance:.8f}, "
      f"jump_ratio={comps.jump_ratio:.4f}, vol={comps.realized_vol:.6f}")

# convenience function
comps2 = realized_vol_decomposition(returns_100)
assert abs(comps2.realized_variance - comps.realized_variance) < 1e-12
print("[OK] convenience realized_vol_decomposition matches")

# ------------------------------------------------------------------ #
# 6. jump_test
# ------------------------------------------------------------------ #
jt = rva.jump_test(returns_100, confidence=0.99)
assert "jump_detected" in jt and "jump_ratio" in jt and "z_stat" in jt
assert isinstance(jt["jump_detected"], bool)
assert 0.0 <= jt["jump_ratio"] <= 1.0
print(f"[OK] jump_test: detected={jt['jump_detected']}, ratio={jt['jump_ratio']:.4f}, z={jt['z_stat']:.4f}")

# ------------------------------------------------------------------ #
# 7. realized_kernel
# ------------------------------------------------------------------ #
rk = rva.realized_kernel(returns_100, bandwidth=5)
assert np.isfinite(rk), f"realized_kernel not finite: {rk}"
print(f"[OK] realized_kernel: {rk:.8f}")

# ------------------------------------------------------------------ #
# 8. VPIN: assert 0 <= vpin <= 1
# ------------------------------------------------------------------ #
prices_v = np.cumsum(rng.normal(0, 1, 200)) + 100
volumes_v = rng.exponential(1000, 200)
vpin_val = vpin(prices_v, volumes_v, bucket_size=2000.0)
assert 0.0 <= vpin_val <= 1.0, f"VPIN={vpin_val} out of [0,1]"
print(f"[OK] VPIN: {vpin_val:.4f} in [0,1]")

# ------------------------------------------------------------------ #
# 9. OFI lambda: synthetic OFI and price changes → finite
# ------------------------------------------------------------------ #
ofi_arr = rng.normal(0, 10, 100)
price_ch = 0.5 * ofi_arr + rng.normal(0, 1, 100)
lam = ofi_lambda(price_ch, ofi_arr)
assert np.isfinite(lam), f"ofi_lambda not finite: {lam}"
print(f"[OK] ofi_lambda: {lam:.6f}")

# ------------------------------------------------------------------ #
# 10. OrderFlowImbalance.ofi
# ------------------------------------------------------------------ #
ofi_calc = OrderFlowImbalance()
n_ticks = 50
bid_p = np.cumsum(rng.normal(0, 0.01, n_ticks)) + 100
ask_p = bid_p + 0.05
bid_s = rng.uniform(100, 500, n_ticks)
ask_s = rng.uniform(100, 500, n_ticks)
ofi_series = ofi_calc.ofi(bid_p, ask_p, bid_s, ask_s)
assert len(ofi_series) == n_ticks - 1, f"OFI length={len(ofi_series)}, expected {n_ticks-1}"
print(f"[OK] OrderFlowImbalance.ofi: shape={ofi_series.shape}")

# ------------------------------------------------------------------ #
# 11. HFSignalGenerator.composite_signal returns dict with 'carry' key
# ------------------------------------------------------------------ #
gen = HFSignalGenerator()
prices_g = np.cumsum(rng.normal(0.001, 0.01, 100)) + 100
volumes_g = rng.exponential(500, 100)
sig = gen.composite_signal(
    prices=prices_g, volumes=volumes_g,
    near_future=101.0, far_future=102.0, days=30
)
assert isinstance(sig, dict), "composite_signal must return dict"
assert "carry" in sig, f"'carry' not in composite_signal keys: {list(sig.keys())}"
for k in ["vpin", "momentum", "mean_reversion", "carry", "jump_ratio", "realized_vol"]:
    assert k in sig, f"'{k}' missing from composite_signal"
    assert np.isfinite(sig[k]), f"'{k}'={sig[k]} not finite"
print(f"[OK] composite_signal: {list(sig.keys())}")

# ------------------------------------------------------------------ #
# 12. term_structure_slope: prices growing with maturity → positive
# ------------------------------------------------------------------ #
prices_ts = [100, 101, 102, 103, 104]
mats = [0.25, 0.5, 0.75, 1.0, 1.25]
slope = fm.term_structure_slope(prices_ts, mats)
assert slope > 0, f"term_structure_slope should be positive for contango, got {slope}"
print(f"[OK] term_structure_slope: {slope:.4f} (positive = contango)")

# ------------------------------------------------------------------ #
# 13. RollYieldSignal and FuturesBasisSignal via sfe layer
# ------------------------------------------------------------------ #
ry_sig = RollYieldSignal()
ry_val = ry_sig.compute(near=4505, far=4515, days_to_roll=30)
assert ry_val < 0, f"RollYieldSignal.compute should be negative in contango"
print(f"[OK] RollYieldSignal.compute: {ry_val:.6f}")

fb_sig = FuturesBasisSignal()
fb_met = fb_sig.compute(spot=4500, futures_price=4515, T=0.25)
assert abs(fb_met.basis - 15.0) < 1e-9
assert fb_sig.regime(4500, 4515) == "contango"
print(f"[OK] FuturesBasisSignal.compute: basis={fb_met.basis}, regime={fb_sig.regime(4500,4515)}")

print("\n[PASS] dim_132: HF microstructure signals -- all checks passed")
PYEOF
