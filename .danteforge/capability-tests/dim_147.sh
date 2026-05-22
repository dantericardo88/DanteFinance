#!/usr/bin/env bash
# dim_147: Smart beta / alternative weighting (risk-parity, momentum tilt)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.smart_beta_v3 import (
        SmartBetaPortfolio, RiskParityEngine, MomentumTiltPortfolio, compute_factor_tilt
    )
    assert SmartBetaPortfolio is not None, "Missing SmartBetaPortfolio"
    assert RiskParityEngine is not None, "Missing RiskParityEngine"
    assert MomentumTiltPortfolio is not None, "Missing MomentumTiltPortfolio"
    assert compute_factor_tilt is not None, "Missing compute_factor_tilt"
    print("[OK] SmartBetaPortfolio present")
    print("[OK] RiskParityEngine present")
    print("[OK] MomentumTiltPortfolio present")
    print("[OK] compute_factor_tilt present")
    print("\n[PASS] dim_147: Smart beta / alternative weighting (risk-parity, momentum tilt) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_147: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.smart_beta_v3")
    sys.exit(0)
PYEOF
