#!/usr/bin/env bash
# dim_115: IV term structure — Nelson-Siegel, SVI calibration, IVSurface arbitrage check
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
from sentinel.sbx.iv_term_structure_v3 import (
    IVTermStructure, NelsonSiegelParams,
    SVICalibrator, SVIParams,
    IVSurface,
    fit_svi, fit_nelson_siegel, forward_vol, iv_surface_from_svi,
)

# ── 1. IVTermStructure construction ──────────────────────────────────────────
expiries = [0.25, 0.5, 1.0, 2.0]
atm_vols = [0.18, 0.20, 0.22, 0.24]
ts = IVTermStructure(expiries, atm_vols)
assert len(ts.expiries) == 4
print("[OK] IVTermStructure constructed with 4 expiries")

# ── 2. Nelson-Siegel fit within 3% of market ─────────────────────────────────
ns = ts.fit_nelson_siegel()
assert isinstance(ns, NelsonSiegelParams)
for T, mv in zip(expiries, atm_vols):
    fitted = ns.vol(T)
    err = abs(fitted - mv) / mv
    assert err < 0.03, f"NS fitted vol {fitted:.4f} vs market {mv:.4f} at T={T}: err={err:.4f} >= 3%"
print(f"[OK] Nelson-Siegel fitted within 3%: beta0={ns.beta0:.4f} beta1={ns.beta1:.4f} tau={ns.tau:.4f}")

# ── 3. Interpolation at T=0.75 ────────────────────────────────────────────────
v075 = ts.interpolate(0.75)
assert 0.20 <= v075 <= 0.22, f"interpolate(0.75)={v075:.4f} not in [0.20, 0.22]"
print(f"[OK] interpolate(T=0.75) = {v075:.4f} (in [0.20, 0.22])")

# ── 4. Forward vol (T1=0.5, T2=1.0) should exceed atm_vol(0.5) ───────────────
fv = ts.forward_vol(0.5, 1.0)
atm_half = ts.interpolate(0.5)
assert fv > atm_half, f"forward vol {fv:.4f} should be > atm_vol(0.5)={atm_half:.4f}"
print(f"[OK] forward_vol(0.5->1.0) = {fv:.4f} > atm_vol(0.5)={atm_half:.4f}")

# ── 5. SVI calibration — symmetric smile ─────────────────────────────────────
strikes = np.array([90., 95., 100., 105., 110.])
vols    = np.array([0.22, 0.21, 0.20, 0.21, 0.22])
F = 100.0
T_svi = 0.5
svi = fit_svi(strikes, vols, F, T_svi)
assert isinstance(svi, SVIParams)
w_atm = svi.total_variance(0.0)
assert w_atm > 0, f"SVI total variance at ATM should be > 0, got {w_atm}"
print(f"[OK] SVI calibrated; w(ATM)={w_atm:.6f} > 0")

# ── 6. vol at 90 vs vol at 110 (smile or skew) ───────────────────────────────
k90  = float(np.log(90. / F))
k110 = float(np.log(110. / F))
vol90  = svi.implied_vol(k90,  T_svi)
vol110 = svi.implied_vol(k110, T_svi)
vol_atm = svi.implied_vol(0.0, T_svi)
assert vol90 > 0 and vol110 > 0
# At least one wing higher than ATM (smile/skew present)
assert vol90 >= vol_atm or vol110 >= vol_atm, \
    f"At least one wing should be >= ATM. vol90={vol90:.4f}, vol_atm={vol_atm:.4f}, vol110={vol110:.4f}"
print(f"[OK] SVI smile: vol90={vol90:.4f} vol_atm={vol_atm:.4f} vol110={vol110:.4f}")

# ── 7. IVSurface: build 5x4 surface and interpolate ──────────────────────────
multi_expiries = np.array([0.25, 0.5, 1.0, 2.0])
base_vols = np.array([0.18, 0.20, 0.22, 0.24])
vol_surf_in = np.array([
    np.array([0.20, 0.19, bv, 0.19, 0.20])
    for bv in base_vols
])
ivsurface = IVSurface(strikes, multi_expiries, vol_surf_in)
v_int = ivsurface.interpolate(100.0, 0.75)
assert v_int > 0, f"Interpolated vol must be positive, got {v_int}"
assert 0.10 < v_int < 0.50, f"Interpolated vol {v_int:.4f} outside plausible range"
print(f"[OK] IVSurface.interpolate(K=100, T=0.75) = {v_int:.4f}")

# ── 8. Arbitrage-free check ───────────────────────────────────────────────────
arb_free = ivsurface.is_arbitrage_free()
assert isinstance(arb_free, bool), "is_arbitrage_free() must return bool"
print(f"[OK] is_arbitrage_free() = {arb_free}")

# ── 9. smile_at_expiry and term_structure_at_strike shapes ───────────────────
smile = ivsurface.smile_at_expiry(0.5)
assert smile.shape == (5,)
ts_strike = ivsurface.term_structure_at_strike(100.0)
assert ts_strike.shape == (4,)
print(f"[OK] smile_at_expiry shape={smile.shape}, term_structure_at_strike shape={ts_strike.shape}")

# ── 10. module-level forward_vol helper ──────────────────────────────────────
fv2 = forward_vol(0.5, 1.0, 0.20, 0.22)
assert fv2 > 0.22, f"forward_vol(0.5,1.0,0.20,0.22)={fv2:.4f} should be > 0.22"
print(f"[OK] module-level forward_vol(0.5,1.0,0.20,0.22) = {fv2:.4f}")

print("\n[PASS] dim_115: IV term structure fitting")
PYEOF
