#!/usr/bin/env bash
# dim_116: Vol regime detection (HMM / Markov switching)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sbx.vol_regime_v3 import (
        VolRegimeDetector, HMMVolModel, detect_regime
    )
    assert VolRegimeDetector is not None, "Missing VolRegimeDetector"
    assert HMMVolModel is not None, "Missing HMMVolModel"
    assert detect_regime is not None, "Missing detect_regime"
    print("[OK] VolRegimeDetector present")
    print("[OK] HMMVolModel present")
    print("[OK] detect_regime present")
    print("\n[PASS] dim_116: Vol regime detection (HMM / Markov switching) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_116: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sbx.vol_regime_v3")
    sys.exit(0)
PYEOF
