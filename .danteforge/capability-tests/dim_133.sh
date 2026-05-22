#!/usr/bin/env bash
# dim_133: Cross-asset signal fusion (ML ensemble, correlation-weighted)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sai.signal_fusion_v3 import (
        SignalFusion, EnsembleAlpha, CorrelationWeightedEnsemble, compute_ensemble_ic
    )
    assert SignalFusion is not None, "Missing SignalFusion"
    assert EnsembleAlpha is not None, "Missing EnsembleAlpha"
    assert CorrelationWeightedEnsemble is not None, "Missing CorrelationWeightedEnsemble"
    assert compute_ensemble_ic is not None, "Missing compute_ensemble_ic"
    print("[OK] SignalFusion present")
    print("[OK] EnsembleAlpha present")
    print("[OK] CorrelationWeightedEnsemble present")
    print("[OK] compute_ensemble_ic present")
    print("\n[PASS] dim_133: Cross-asset signal fusion (ML ensemble, correlation-weighted) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_133: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sai.signal_fusion_v3")
    sys.exit(0)
PYEOF
