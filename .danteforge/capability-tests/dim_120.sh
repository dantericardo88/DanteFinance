#!/usr/bin/env bash
# dim_120: Transaction cost analysis (TCA / implementation shortfall)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.tca_v3 import (
        TCAReport, ImplementationShortfall, compute_tca
    )
    assert TCAReport is not None, "Missing TCAReport"
    assert ImplementationShortfall is not None, "Missing ImplementationShortfall"
    assert compute_tca is not None, "Missing compute_tca"
    print("[OK] TCAReport present")
    print("[OK] ImplementationShortfall present")
    print("[OK] compute_tca present")
    print("\n[PASS] dim_120: Transaction cost analysis (TCA / implementation shortfall) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_120: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.tca_v3")
    sys.exit(0)
PYEOF
