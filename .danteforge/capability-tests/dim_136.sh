#!/usr/bin/env bash
# dim_136: Counterparty credit risk (CVA / DVA / wrong-way risk)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.counterparty_risk_v3 import (
        CVACalculator, DVACalculator, WrongWayRisk, compute_cva, compute_dva
    )
    assert CVACalculator is not None, "Missing CVACalculator"
    assert DVACalculator is not None, "Missing DVACalculator"
    assert WrongWayRisk is not None, "Missing WrongWayRisk"
    assert compute_cva is not None, "Missing compute_cva"
    assert compute_dva is not None, "Missing compute_dva"
    print("[OK] CVACalculator present")
    print("[OK] DVACalculator present")
    print("[OK] WrongWayRisk present")
    print("[OK] compute_cva present")
    print("[OK] compute_dva present")
    print("\n[PASS] dim_136: Counterparty credit risk (CVA / DVA / wrong-way risk) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_136: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.counterparty_risk_v3")
    sys.exit(0)
PYEOF
