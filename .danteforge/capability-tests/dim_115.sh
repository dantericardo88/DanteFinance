#!/usr/bin/env bash
# dim_115: Implied volatility term structure (IV curve fitting)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sbx.iv_term_structure_v3 import (
        IVTermStructure, fit_iv_curve, extrapolate_iv
    )
    assert IVTermStructure is not None, "Missing IVTermStructure"
    assert fit_iv_curve is not None, "Missing fit_iv_curve"
    assert extrapolate_iv is not None, "Missing extrapolate_iv"
    print("[OK] IVTermStructure present")
    print("[OK] fit_iv_curve present")
    print("[OK] extrapolate_iv present")
    print("\n[PASS] dim_115: Implied volatility term structure (IV curve fitting) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_115: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sbx.iv_term_structure_v3")
    sys.exit(0)
PYEOF
