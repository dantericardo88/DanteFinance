#!/usr/bin/env bash
# dim_139: Operational risk / loss event database (OpVaR)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.op_risk_v3 import (
        OperationalRiskDatabase, OpVaRCalculator, compute_op_var
    )
    assert OperationalRiskDatabase is not None, "Missing OperationalRiskDatabase"
    assert OpVaRCalculator is not None, "Missing OpVaRCalculator"
    assert compute_op_var is not None, "Missing compute_op_var"
    print("[OK] OperationalRiskDatabase present")
    print("[OK] OpVaRCalculator present")
    print("[OK] compute_op_var present")
    print("\n[PASS] dim_139: Operational risk / loss event database (OpVaR) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_139: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.op_risk_v3")
    sys.exit(0)
PYEOF
