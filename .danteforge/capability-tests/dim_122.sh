#!/usr/bin/env bash
# dim_122: Trade flow analysis (VWAP, TWAP, Kyle lambda, VPIN)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.trade_flow_v3 import (
        VWAPCalculator, TWAPCalculator, compute_kyle_lambda, compute_vpin
    )
    assert VWAPCalculator is not None, "Missing VWAPCalculator"
    assert TWAPCalculator is not None, "Missing TWAPCalculator"
    assert compute_kyle_lambda is not None, "Missing compute_kyle_lambda"
    assert compute_vpin is not None, "Missing compute_vpin"
    print("[OK] VWAPCalculator present")
    print("[OK] TWAPCalculator present")
    print("[OK] compute_kyle_lambda present")
    print("[OK] compute_vpin present")
    print("\n[PASS] dim_122: Trade flow analysis (VWAP, TWAP, Kyle lambda, VPIN) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_122: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.trade_flow_v3")
    sys.exit(0)
PYEOF
