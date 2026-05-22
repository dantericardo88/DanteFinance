#!/usr/bin/env bash
# dim_118: Higher-order Greeks — vanna, volga, charm, speed, dual delta/gamma, numerical verification
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sbx.higher_greeks_v3 import (
    GreeksBundle, HigherGreeksEngine, compute_all_greeks,
    vanna, volga, charm, speed, ultima, color, veta, dual_delta, dual_gamma,
)

# ── Parameters ────────────────────────────────────────────────────────────────
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
option_type = "call"

# ── 1. compute_all_greeks returns GreeksBundle ────────────────────────────────
bundle = compute_all_greeks(S, K, T, r, sigma, q=0.0, option_type=option_type)
assert isinstance(bundle, GreeksBundle), f"Expected GreeksBundle, got {type(bundle)}"
print("[OK] compute_all_greeks returned GreeksBundle")

# ── 2. all() returns dict with >= 12 keys ────────────────────────────────────
d = bundle.all()
assert isinstance(d, dict), f"all() must return dict, got {type(d)}"
assert len(d) >= 12, f"all() returned only {len(d)} keys (expected >= 12): {list(d.keys())}"
print(f"[OK] GreeksBundle.all() has {len(d)} keys: {sorted(d.keys())}")

# ── 3. vanna != 0 ─────────────────────────────────────────────────────────────
vanna_val = bundle.vanna
assert vanna_val != 0, f"vanna should be non-zero for ATM call; got {vanna_val}"
print(f"[OK] vanna = {vanna_val:.6f} (non-zero)")

# ── 4. volga > 0 ──────────────────────────────────────────────────────────────
volga_val = bundle.volga
assert volga_val > 0, f"volga must be > 0 (long vol convexity); got {volga_val}"
print(f"[OK] volga = {volga_val:.6f} > 0")

# ── 5. speed < 0 ──────────────────────────────────────────────────────────────
speed_val = bundle.speed
assert speed_val < 0, f"speed must be < 0 (gamma decreases as S increases); got {speed_val}"
print(f"[OK] speed = {speed_val:.8f} < 0")

# ── 6. dual_delta < 0 (call price decreases as K increases) ──────────────────
dd_val = bundle.dual_delta
assert dd_val < 0, f"dual_delta (call) must be < 0; got {dd_val}"
print(f"[OK] dual_delta = {dd_val:.6f} < 0")

# ── 7. dual_gamma > 0 ─────────────────────────────────────────────────────────
dg_val = bundle.dual_gamma
assert dg_val > 0, f"dual_gamma must be > 0 (convex in strike); got {dg_val}"
print(f"[OK] dual_gamma = {dg_val:.6f} > 0")

# ── 8. Numerical verification of vanna: rel_error < 1% ───────────────────────
engine = HigherGreeksEngine()
vanna_check = engine.numerical_verify("vanna", S, K, T, r, sigma)
assert set(vanna_check.keys()) >= {"analytical", "numerical", "diff", "rel_error"}, \
    f"numerical_verify must return dict with 4 keys; got {list(vanna_check.keys())}"
re_vanna = vanna_check["rel_error"]
assert re_vanna < 0.01, \
    f"vanna numerical verification: rel_error={re_vanna:.6f} >= 1%  (analytical={vanna_check['analytical']:.6f}, numerical={vanna_check['numerical']:.6f})"
print(f"[OK] vanna numerical verify: analytical={vanna_check['analytical']:.6f}, "
      f"numerical={vanna_check['numerical']:.6f}, rel_error={re_vanna:.6f}")

# ── 9. Numerical verification of volga: rel_error < 1% ───────────────────────
volga_check = engine.numerical_verify("volga", S, K, T, r, sigma)
re_volga = volga_check["rel_error"]
assert re_volga < 0.01, \
    f"volga numerical verification: rel_error={re_volga:.6f} >= 1%  (analytical={volga_check['analytical']:.6f}, numerical={volga_check['numerical']:.6f})"
print(f"[OK] volga numerical verify: analytical={volga_check['analytical']:.6f}, "
      f"numerical={volga_check['numerical']:.6f}, rel_error={re_volga:.6f}")

# ── 10. Individual module-level functions consistent with bundle ──────────────
assert abs(vanna(S, K, T, r, sigma) - bundle.vanna) < 1e-10
assert abs(volga(S, K, T, r, sigma) - bundle.volga) < 1e-10
assert abs(speed(S, K, T, r, sigma) - bundle.speed) < 1e-10
assert abs(dual_delta(S, K, T, r, sigma, option_type) - bundle.dual_delta) < 1e-10
assert abs(dual_gamma(S, K, T, r, sigma) - bundle.dual_gamma) < 1e-10
print("[OK] Module-level functions consistent with GreeksBundle")

# ── 11. Ultima, color, veta are finite ────────────────────────────────────────
import math
assert math.isfinite(bundle.ultima), f"ultima must be finite, got {bundle.ultima}"
assert math.isfinite(bundle.color),  f"color must be finite, got {bundle.color}"
assert math.isfinite(bundle.veta),   f"veta must be finite, got {bundle.veta}"
print(f"[OK] ultima={bundle.ultima:.4f}, color={bundle.color:.6f}, veta={bundle.veta:.4f}")

print("\n[PASS] dim_118: Higher-order Greeks")
PYEOF
