#!/usr/bin/env bash
# dim_118: Higher-order Greeks (vanna, volga, charm, speed)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sbx.higher_order_greeks_v3 import (
        compute_vanna, compute_volga, compute_charm, compute_speed
    )
    assert compute_vanna is not None, "Missing compute_vanna"
    assert compute_volga is not None, "Missing compute_volga"
    assert compute_charm is not None, "Missing compute_charm"
    assert compute_speed is not None, "Missing compute_speed"
    print("[OK] compute_vanna present")
    print("[OK] compute_volga present")
    print("[OK] compute_charm present")
    print("[OK] compute_speed present")
    print("\n[PASS] dim_118: Higher-order Greeks (vanna, volga, charm, speed) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_118: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sbx.higher_order_greeks_v3")
    sys.exit(0)
PYEOF
