#!/usr/bin/env bash
# dim_112: SABR volatility smile and skew analytics
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

import numpy as np

try:
    from sentinel.sbx.vol_smile_v3 import (
        SABRParams, SABRModel, VolSmile,
        sabr_implied_vol, fit_sabr, dupire_local_vol,
    )
except ImportError as e:
    print(f"[NOT-BUILT] dim_112: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sbx.vol_smile_v3")
    sys.exit(0)

# ── 1. Existence checks ──────────────────────────────────────────────────────
assert SABRParams is not None, "Missing SABRParams"
assert SABRModel is not None, "Missing SABRModel"
assert VolSmile is not None, "Missing VolSmile"
assert sabr_implied_vol is not None, "Missing sabr_implied_vol"
assert fit_sabr is not None, "Missing fit_sabr"
assert dupire_local_vol is not None, "Missing dupire_local_vol"
print("[OK] All public symbols present")

# ── 2. SABRParams construction ───────────────────────────────────────────────
params = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
assert params.alpha == 0.3
assert params.beta == 0.5
assert params.rho == -0.3
assert params.nu == 0.4
print(f"[OK] SABRParams: alpha={params.alpha}, beta={params.beta}, rho={params.rho}, nu={params.nu}")

# ── 3. Implied vol smile across strikes ──────────────────────────────────────
F = 100.0
T = 1.0
strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
model = SABRModel()

vols = []
for K in strikes:
    v = model.implied_vol(F, K, T, params)
    assert v > 0, f"Non-positive vol at K={K}: {v}"
    assert not math.isnan(v), f"NaN vol at K={K}"
    vols.append(v)
    print(f"  K={K:6.1f} -> sigma={v:.6f}")

# ── 4. Negative skew: vol at 80 > vol at 120 (rho = -0.3) ───────────────────
assert vols[0] > vols[-1], (
    f"Expected negative skew (vol[80]={vols[0]:.6f} > vol[120]={vols[-1]:.6f}) "
    f"but got the opposite"
)
print(f"[OK] Negative skew confirmed: vol[80]={vols[0]:.6f} > vol[120]={vols[-1]:.6f}")

# ── 5. ATM vol is positive and finite ────────────────────────────────────────
# ATM SABR vol ~ alpha / F^(1-beta) for beta=0.5, F=100: alpha/10 = 0.03
atm = model.atm_vol(F, T, params)
atm_approx = params.alpha / (F ** (1.0 - params.beta))
assert atm > 0, f"ATM vol must be positive, got {atm}"
assert not math.isnan(atm), "ATM vol must not be NaN"
assert abs(atm - atm_approx) / atm_approx < 0.5, (
    f"ATM vol {atm:.6f} deviates from approx {atm_approx:.6f} by more than 50%"
)
print(f"[OK] ATM vol = {atm:.6f}, approx = {atm_approx:.6f} (within 50%)")

# ── 6. Skew is negative (dsigma/dK < 0 for rho = -0.3) ─────────────────────
skew_val = model.skew(F, T, params, dk=0.005)
assert skew_val < 0, f"Expected negative skew (dsigma/dK < 0), got {skew_val:.8f}"
print(f"[OK] ATM skew dsigma/dK = {skew_val:.8f} < 0")

# ── 7. Calibrate alpha to flat vol surface, assert convergence ───────────────
flat_vol = 0.25
market_vols_flat = np.full(len(strikes), flat_vol)
alpha_fit = model.calibrate_alpha(
    F, T, market_vols_flat, np.array(strikes),
    beta=0.5, rho=0.0, nu=0.01,
)
assert alpha_fit > 0, f"Fitted alpha must be positive, got {alpha_fit}"
# Verify fitted alpha reproduces vols reasonably
fitted_vols = [sabr_implied_vol(F, K, T, alpha_fit, 0.5, 0.0, 0.01) for K in strikes]
for mv, fv in zip(market_vols_flat, fitted_vols):
    assert abs(mv - fv) < 0.05, f"Calibration residual too large: {abs(mv-fv):.6f}"
print(f"[OK] Alpha calibration: alpha_fit={alpha_fit:.6f}, max_residual={max(abs(mv-fv) for mv,fv in zip(market_vols_flat,fitted_vols)):.6f}")

# ── 8. Full SABR calibration (fit_sabr) ─────────────────────────────────────
# Generate target vols from known params, then recover them
target_params = SABRParams(alpha=0.25, beta=0.5, rho=-0.2, nu=0.35)
target_strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
target_vols = np.array([model.implied_vol(F, K, T, target_params) for K in target_strikes])

fitted = fit_sabr(F, T, target_strikes, target_vols, beta=0.5)
assert fitted.alpha > 0, "fit_sabr returned non-positive alpha"
# Check residuals
resid_vols = np.array([model.implied_vol(F, K, T, fitted) for K in target_strikes])
max_resid = float(np.max(np.abs(resid_vols - target_vols)))
assert max_resid < 0.01, f"fit_sabr max residual {max_resid:.6f} > 0.01"
print(f"[OK] fit_sabr: alpha={fitted.alpha:.4f}, rho={fitted.rho:.4f}, nu={fitted.nu:.4f}, max_resid={max_resid:.6f}")

# ── 9. VolSmile: interpolation within [min, max] bounds ─────────────────────
smile_strikes = np.array(strikes, dtype=float)
smile_vols = np.array(vols)
vs = VolSmile(smile_strikes, smile_vols, F, T)

# Interpolate at several mid-points
test_Ks = [85.0, 92.5, 97.5, 102.5, 107.5, 115.0]
vol_min = float(smile_vols.min())
vol_max = float(smile_vols.max())
for tK in test_Ks:
    iv = vs.interpolate(tK)
    # Allow slight extrapolation outside min/max due to spline
    assert vol_min * 0.8 < iv < vol_max * 1.2, (
        f"Interpolated vol at K={tK} out of range: {iv:.6f} not in [{vol_min*0.8:.4f}, {vol_max*1.2:.4f}]"
    )
print(f"[OK] VolSmile interpolation within expected range for {len(test_Ks)} test strikes")

# ── 10. Risk reversal, strangle, butterfly metrics ──────────────────────────
rr = vs.risk_reversal(0.25)
stg = vs.strangle(0.25)
bfly = vs.butterfly(0.25)
print(f"[OK] Risk reversal (25d): {rr:.6f}")
print(f"[OK] Strangle (25d):      {stg:.6f}")
print(f"[OK] Butterfly (25d):     {bfly:.6f}")

# With negative rho smile, risk reversal should be negative (put IV > call IV)
assert rr < 0, f"Expected negative risk reversal for negative-rho smile, got {rr:.6f}"
print(f"[OK] Risk reversal is negative ({rr:.6f}) consistent with negative skew")

# ── 11. Butterfly (strangle) >= 0 for convex smile ───────────────────────────
assert bfly >= 0, f"Butterfly must be >= 0 (no arbitrage), got {bfly:.6f}"
print(f"[OK] Butterfly >= 0 ({bfly:.6f})")

# ── 12. Arbitrage-free check ─────────────────────────────────────────────────
arb_free = vs.is_arbitrage_free()
assert arb_free, "SABR smile should be arbitrage-free"
print(f"[OK] Smile is arbitrage-free: {arb_free}")

# ── 13. Dupire local vol ─────────────────────────────────────────────────────
# Build a smile_func closure from fitted SABR
fitted_sabr = fit_sabr(F, T, smile_strikes, smile_vols, beta=0.5)
def smile_func(K_arg, T_arg):
    return sabr_implied_vol(F, K_arg, T_arg, fitted_sabr.alpha, fitted_sabr.beta,
                            fitted_sabr.rho, fitted_sabr.nu)

loc_v = dupire_local_vol(F, F, T, smile_func, dK=0.5, dT=0.005)
# Local vol at ATM should be in a reasonable range
assert not math.isnan(loc_v), "Dupire local vol returned NaN at ATM"
assert 0 < loc_v < 2.0, f"Dupire local vol out of range: {loc_v:.6f}"
print(f"[OK] Dupire local vol at ATM: {loc_v:.6f}")

# ── 14. Implied vol surface shape ────────────────────────────────────────────
surf = model.implied_vol_surface(F, strikes, [0.25, 0.5, 1.0, 2.0], params)
assert surf.shape == (7, 4), f"Surface shape mismatch: {surf.shape}"
assert np.all(surf > 0), "All surface vols must be positive"
print(f"[OK] Implied vol surface shape: {surf.shape}, all positive")

print("\n[PASS] dim_112: SABR vol smile analytics -- all checks passed")
PYEOF
