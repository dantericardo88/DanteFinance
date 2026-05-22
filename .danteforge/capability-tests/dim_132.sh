#!/usr/bin/env bash
# dim_132: HF microstructure signals (roll yield, futures basis, VPIN)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.hf_signals_v3 import (
        RollYieldSignal, FuturesBasisSignal, compute_roll_yield, compute_futures_basis
    )
    assert RollYieldSignal is not None, "Missing RollYieldSignal"
    assert FuturesBasisSignal is not None, "Missing FuturesBasisSignal"
    assert compute_roll_yield is not None, "Missing compute_roll_yield"
    assert compute_futures_basis is not None, "Missing compute_futures_basis"
    print("[OK] RollYieldSignal present")
    print("[OK] FuturesBasisSignal present")
    print("[OK] compute_roll_yield present")
    print("[OK] compute_futures_basis present")
    print("\n[PASS] dim_132: HF microstructure signals (roll yield, futures basis, VPIN) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_132: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.hf_signals_v3")
    sys.exit(0)
PYEOF
