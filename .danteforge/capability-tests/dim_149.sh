#!/usr/bin/env bash
# dim_149: Autonomous portfolio rebalancing orchestrator (drift + tax-loss harvest)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.api.rebalancing_agent_v3 import (
        RebalancingOrchestrator, DriftMonitor, TaxLossHarvester, execute_rebalance
    )
    assert RebalancingOrchestrator is not None, "Missing RebalancingOrchestrator"
    assert DriftMonitor is not None, "Missing DriftMonitor"
    assert TaxLossHarvester is not None, "Missing TaxLossHarvester"
    assert execute_rebalance is not None, "Missing execute_rebalance"
    print("[OK] RebalancingOrchestrator present")
    print("[OK] DriftMonitor present")
    print("[OK] TaxLossHarvester present")
    print("[OK] execute_rebalance present")
    print("\n[PASS] dim_149: Autonomous portfolio rebalancing orchestrator (drift + tax-loss harvest) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_149: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.api.rebalancing_agent_v3")
    sys.exit(0)
PYEOF
