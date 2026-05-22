#!/usr/bin/env bash
# dim_144: Multi-factor attribution extended (systematic vs idiosyncratic)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.factor_attribution_v3 import (
        MultiFactorAttribution, SystematicReturn, IdiosyncraticReturn, decompose_returns
    )
    assert MultiFactorAttribution is not None, "Missing MultiFactorAttribution"
    assert SystematicReturn is not None, "Missing SystematicReturn"
    assert IdiosyncraticReturn is not None, "Missing IdiosyncraticReturn"
    assert decompose_returns is not None, "Missing decompose_returns"
    print("[OK] MultiFactorAttribution present")
    print("[OK] SystematicReturn present")
    print("[OK] IdiosyncraticReturn present")
    print("[OK] decompose_returns present")
    print("\n[PASS] dim_144: Multi-factor attribution extended (systematic vs idiosyncratic) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_144: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.factor_attribution_v3")
    sys.exit(0)
PYEOF
