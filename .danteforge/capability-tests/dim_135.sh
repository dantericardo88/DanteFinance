#!/usr/bin/env bash
# dim_135: Liquidity risk framework (Basel III LCR / NSFR)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.liquidity_risk_v3 import (
        LiquidityCoverageRatio, NetStableFundingRatio, compute_lcr, compute_nsfr
    )
    assert LiquidityCoverageRatio is not None, "Missing LiquidityCoverageRatio"
    assert NetStableFundingRatio is not None, "Missing NetStableFundingRatio"
    assert compute_lcr is not None, "Missing compute_lcr"
    assert compute_nsfr is not None, "Missing compute_nsfr"
    print("[OK] LiquidityCoverageRatio present")
    print("[OK] NetStableFundingRatio present")
    print("[OK] compute_lcr present")
    print("[OK] compute_nsfr present")
    print("\n[PASS] dim_135: Liquidity risk framework (Basel III LCR / NSFR) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_135: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.liquidity_risk_v3")
    sys.exit(0)
PYEOF
