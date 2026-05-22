"""Inline verification test for dim_111: Options Greeks surface engine."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import numpy as np

# ── 1. Import checks ──────────────────────────────────────────────────────────
from sentinel.sbx.greeks_surface_v3 import (
    GreeksSurface, compute_delta, compute_gamma, compute_vega, compute_theta
)

assert GreeksSurface is not None,   "Missing GreeksSurface"
assert compute_delta is not None,   "Missing compute_delta"
assert compute_gamma is not None,   "Missing compute_gamma"
assert compute_vega is not None,    "Missing compute_vega"
assert compute_theta is not None,   "Missing compute_theta"
print("[OK] All imports present")

# ── 2. ATM scalar values: S=100, K=100, T=1, r=0.05, sigma=0.2 ───────────────
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

dc = compute_delta(S, K, T, r, sigma, "call")
dp = compute_delta(S, K, T, r, sigma, "put")
gamma_c = compute_gamma(S, K, T, r, sigma, "call")
gamma_p = compute_gamma(S, K, T, r, sigma, "put")
vega_c  = compute_vega(S, K, T, r, sigma, "call")
theta_c = compute_theta(S, K, T, r, sigma, "call")
theta_p = compute_theta(S, K, T, r, sigma, "put")

print(f"\nATM scalars: S={S} K={K} T={T} r={r} sigma={sigma}")
print(f"  delta_call = {dc:.6f}  (expect ~0.6368)")
print(f"  delta_put  = {dp:.6f}  (expect ~-0.3632)")
print(f"  gamma      = {gamma_c:.8f}  (>0)")
print(f"  vega       = {vega_c:.6f}  (>0)")
print(f"  theta_call = {theta_c:.6f}  (<0)")
print(f"  theta_put  = {theta_p:.6f}  (<0)")

assert abs(dc - 0.6368) < 0.001, f"delta_call {dc:.6f} deviates from 0.6368"
assert abs(dp - (-0.3632)) < 0.001, f"delta_put {dp:.6f} deviates from -0.3632"
assert gamma_c > 0, "gamma must be positive"
assert gamma_p > 0, "gamma_put must be positive"
assert math.isclose(gamma_c, gamma_p, rel_tol=1e-9), "gamma call != gamma put"
assert vega_c > 0, "vega must be positive"
assert theta_c < 0, "theta_call must be negative"
assert theta_p < 0, "theta_put must be negative"
print("[OK] ATM scalar Greeks within tolerance")

# ── 3. Put-call parity: delta_call - delta_put ≈ 1.0 ────────────────────────
parity = dc - dp
print(f"\nPut-call parity: delta_call - delta_put = {parity:.6f}  (expect ~1.0)")
assert abs(parity - 1.0) < 1e-10, f"Put-call delta parity failed: {parity}"
print("[OK] Put-call parity holds")

# ── 4. GreeksSurface class — scalar ──────────────────────────────────────────
gs_call = GreeksSurface(S, K, T, r, sigma, "call")
gs_put  = GreeksSurface(S, K, T, r, sigma, "put")

assert abs(gs_call.delta() - dc) < 1e-12, "GreeksSurface.delta() mismatch"
assert abs(gs_call.gamma() - gamma_c) < 1e-12, "GreeksSurface.gamma() mismatch"
assert abs(gs_call.vega()  - vega_c)  < 1e-12, "GreeksSurface.vega() mismatch"
assert abs(gs_call.theta() - theta_c) < 1e-12, "GreeksSurface.theta() mismatch"
print("[OK] GreeksSurface scalar agrees with standalone functions")

all_g = gs_call.all_greeks()
assert set(all_g.keys()) >= {"delta", "gamma", "vega", "theta", "rho"}, \
    f"all_greeks missing keys: {all_g.keys()}"
print("[OK] all_greeks() returns expected keys")

# ── 5. Surface grid: 5 strikes × 3 expiries — all gammas > 0 ────────────────
strikes  = [80.0, 90.0, 100.0, 110.0, 120.0]
expiries = [0.25, 0.5, 1.0]

surface = GreeksSurface.surface(S, strikes, expiries, r, sigma, "call")
gamma_grid = surface["gamma"]
delta_grid = surface["delta"]
vega_grid  = surface["vega"]
theta_grid = surface["theta"]

print(f"\nSurface grid shape: {gamma_grid.shape}  (expect (5, 3))")
assert gamma_grid.shape == (5, 3), f"Wrong surface shape: {gamma_grid.shape}"
assert np.all(gamma_grid > 0), f"Not all gammas > 0:\n{gamma_grid}"
assert np.all(vega_grid  > 0), "Not all vegas > 0 on grid"
assert np.all(delta_grid > 0) and np.all(delta_grid < 1), "Call deltas out of (0,1)"
assert np.all(theta_grid < 0), "Not all thetas < 0 on call surface"
print("[OK] Surface grid: all gammas > 0, deltas in (0,1), thetas < 0")

# ── 6. Put surface ────────────────────────────────────────────────────────────
surface_put = GreeksSurface.surface(S, strikes, expiries, r, sigma, "put")
gamma_grid_put = surface_put["gamma"]
assert np.allclose(gamma_grid, gamma_grid_put), "Gamma differs between call/put surfaces"
print("[OK] Call and put gamma surfaces are identical")

print("\n[PASS] dim_111: Options Greeks surface — all checks passed")
