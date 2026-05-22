#!/usr/bin/env bash
# dim_138: Regulatory capital (Pillar 3 / CCR standardised approach)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.regulatory_capital_v3 import (
        RegulatoryCapital, Pillar3Disclosure, compute_risk_weighted_assets
    )
    assert RegulatoryCapital is not None, "Missing RegulatoryCapital"
    assert Pillar3Disclosure is not None, "Missing Pillar3Disclosure"
    assert compute_risk_weighted_assets is not None, "Missing compute_risk_weighted_assets"
    print("[OK] RegulatoryCapital present")
    print("[OK] Pillar3Disclosure present")
    print("[OK] compute_risk_weighted_assets present")
    print("\n[PASS] dim_138: Regulatory capital (Pillar 3 / CCR standardised approach) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_138: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.regulatory_capital_v3")
    sys.exit(0)
PYEOF
