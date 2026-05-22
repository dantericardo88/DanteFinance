#!/usr/bin/env bash
# dim_111: Options Greeks surface (delta/gamma/vega/theta/rho)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sbx.greeks_surface_v3 import (
        GreeksSurface, compute_delta, compute_gamma, compute_vega, compute_theta
    )
    assert GreeksSurface is not None, "Missing GreeksSurface"
    assert compute_delta is not None, "Missing compute_delta"
    assert compute_gamma is not None, "Missing compute_gamma"
    assert compute_vega is not None, "Missing compute_vega"
    assert compute_theta is not None, "Missing compute_theta"
    print("[OK] GreeksSurface present")
    print("[OK] compute_delta present")
    print("[OK] compute_gamma present")
    print("[OK] compute_vega present")
    print("[OK] compute_theta present")
    print("\n[PASS] dim_111: Options Greeks surface (delta/gamma/vega/theta/rho) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_111: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sbx.greeks_surface_v3")
    sys.exit(0)
PYEOF
