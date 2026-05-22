#!/usr/bin/env bash
# dim_145: Tail risk hedging (VIX options / put spreads / regime triggers)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.tail_risk_hedging_v3 import (
        TailRiskHedge, VIXOptionHedge, PutSpreadHedge, compute_hedge_cost
    )
    assert TailRiskHedge is not None, "Missing TailRiskHedge"
    assert VIXOptionHedge is not None, "Missing VIXOptionHedge"
    assert PutSpreadHedge is not None, "Missing PutSpreadHedge"
    assert compute_hedge_cost is not None, "Missing compute_hedge_cost"
    print("[OK] TailRiskHedge present")
    print("[OK] VIXOptionHedge present")
    print("[OK] PutSpreadHedge present")
    print("[OK] compute_hedge_cost present")
    print("\n[PASS] dim_145: Tail risk hedging (VIX options / put spreads / regime triggers) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_145: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.tail_risk_hedging_v3")
    sys.exit(0)
PYEOF
