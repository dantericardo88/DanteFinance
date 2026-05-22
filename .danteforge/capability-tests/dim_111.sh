#!/usr/bin/env bash
# dim_111: Options Greeks surface (delta/gamma/vega/theta/rho)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import math
import numpy as np

try:
    from sentinel.sbx.greeks_surface_v3 import (
        GreeksSurface, compute_delta, compute_gamma, compute_vega, compute_theta
    )
except ImportError as e:
    print(f"[NOT-BUILT] dim_111: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sbx.greeks_surface_v3")
    sys.exit(0)

# ── 1. Existence checks ──────────────────────────────────────────────────────
assert GreeksSurface is not None, "Missing GreeksSurface"
assert compute_delta is not None, "Missing compute_delta"
assert compute_gamma is not None, "Missing compute_gamma"
assert compute_vega is not None,  "Missing compute_vega"
assert compute_theta is not None, "Missing compute_theta"
print("[OK] GreeksSurface present")
print("[OK] compute_delta present")
print("[OK] compute_gamma present")
print("[OK] compute_vega present")
print("[OK] compute_theta present")

# ── 2. ATM exact values: S=100, K=100, T=1, r=0.05, sigma=0.2 ───────────────
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

dc = compute_delta(S, K, T, r, sigma, "call")
dp = compute_delta(S, K, T, r, sigma, "put")
gc = compute_gamma(S, K, T, r, sigma, "call")
gp = compute_gamma(S, K, T, r, sigma, "put")
vc = compute_vega(S, K, T, r, sigma, "call")
tc = compute_theta(S, K, T, r, sigma, "call")
tp = compute_theta(S, K, T, r, sigma, "put")

assert abs(dc - 0.6368) < 0.001, f"delta_call {dc:.6f} deviates from 0.6368"
assert abs(dp - (-0.3632)) < 0.001, f"delta_put {dp:.6f} deviates from -0.3632"
print(f"[OK] delta_call = {dc:.6f}  (expected ~0.6368)")
print(f"[OK] delta_put  = {dp:.6f}  (expected ~-0.3632)")

assert gc > 0, "gamma_call must be > 0"
assert gp > 0, "gamma_put must be > 0"
print(f"[OK] gamma > 0  ({gc:.8f})")

assert vc > 0, "vega must be > 0"
print(f"[OK] vega > 0   ({vc:.6f})")

assert tc < 0, "theta_call must be < 0"
assert tp < 0, "theta_put must be < 0"
print(f"[OK] theta_call < 0  ({tc:.6f})")
print(f"[OK] theta_put  < 0  ({tp:.6f})")

# ── 3. Put-call parity: delta_call - delta_put = 1.0 ────────────────────────
parity = dc - dp
assert abs(parity - 1.0) < 1e-10, f"Put-call delta parity failed: {parity}"
print(f"[OK] Put-call parity: delta_call - delta_put = {parity:.10f}")

# ── 4. GreeksSurface class — scalar round-trip ───────────────────────────────
gs_call = GreeksSurface(S, K, T, r, sigma, "call")
gs_put  = GreeksSurface(S, K, T, r, sigma, "put")

assert abs(gs_call.delta() - dc) < 1e-12, "GreeksSurface.delta() mismatch"
assert abs(gs_call.gamma() - gc) < 1e-12, "GreeksSurface.gamma() mismatch"
assert abs(gs_call.vega()  - vc) < 1e-12, "GreeksSurface.vega() mismatch"
assert abs(gs_call.theta() - tc) < 1e-12, "GreeksSurface.theta() mismatch"
print("[OK] GreeksSurface scalar agrees with standalone functions")

all_g = gs_call.all_greeks()
assert set(all_g.keys()) >= {"delta", "gamma", "vega", "theta", "rho"}, \
    f"all_greeks missing keys: {set(all_g.keys())}"
print("[OK] all_greeks() returns expected keys")

# ── 5. Surface grid: 5 strikes x 3 expiries — all gammas > 0 ────────────────
strikes  = [80.0, 90.0, 100.0, 110.0, 120.0]
expiries = [0.25, 0.5, 1.0]

surface = GreeksSurface.surface(S, strikes, expiries, r, sigma, "call")
gamma_grid = surface["gamma"]
delta_grid = surface["delta"]
vega_grid  = surface["vega"]
theta_grid = surface["theta"]

assert gamma_grid.shape == (5, 3), f"Wrong surface shape: {gamma_grid.shape}"
assert np.all(gamma_grid > 0), f"Not all gammas > 0:\n{gamma_grid}"
assert np.all(vega_grid  > 0), "Not all vegas > 0 on grid"
assert np.all(delta_grid > 0) and np.all(delta_grid < 1), \
    "Call deltas out of (0,1) on grid"
assert np.all(theta_grid < 0), "Not all thetas < 0 on call surface"
print(f"[OK] Surface grid {gamma_grid.shape}: all gammas > 0, deltas in (0,1), thetas < 0")

# ── 6. Put surface: gamma same as call ───────────────────────────────────────
surface_put = GreeksSurface.surface(S, strikes, expiries, r, sigma, "put")
assert np.allclose(gamma_grid, surface_put["gamma"]), \
    "Gamma differs between call/put surfaces"
print("[OK] Call and put gamma surfaces are identical")

print("\n[PASS] dim_111: Options Greeks surface (delta/gamma/vega/theta/rho) -- all checks passed")
PYEOF
