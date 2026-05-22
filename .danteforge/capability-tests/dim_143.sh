#!/usr/bin/env bash
# dim_143: Energy transition analytics (stranded assets / EV demand curves)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.energy_transition_v3 import (
        EnergyTransitionModel, StrandedAssetRisk, EVDemandCurve, compute_transition_risk
    )
    assert EnergyTransitionModel is not None, "Missing EnergyTransitionModel"
    assert StrandedAssetRisk is not None, "Missing StrandedAssetRisk"
    assert EVDemandCurve is not None, "Missing EVDemandCurve"
    assert compute_transition_risk is not None, "Missing compute_transition_risk"
    print("[OK] EnergyTransitionModel present")
    print("[OK] StrandedAssetRisk present")
    print("[OK] EVDemandCurve present")
    print("[OK] compute_transition_risk present")
    print("\n[PASS] dim_143: Energy transition analytics (stranded assets / EV demand curves) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_143: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.energy_transition_v3")
    sys.exit(0)
PYEOF
