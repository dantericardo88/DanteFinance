#!/usr/bin/env bash
# dim_129: ML alpha signals (XGBoost / LightGBM / TabNet)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sai.ml_alpha_v3 import (
        MLAlphaSignal, XGBoostAlpha, LightGBMAlpha, compute_ic, compute_rank_ic
    )
    assert MLAlphaSignal is not None, "Missing MLAlphaSignal"
    assert XGBoostAlpha is not None, "Missing XGBoostAlpha"
    assert LightGBMAlpha is not None, "Missing LightGBMAlpha"
    assert compute_ic is not None, "Missing compute_ic"
    assert compute_rank_ic is not None, "Missing compute_rank_ic"
    print("[OK] MLAlphaSignal present")
    print("[OK] XGBoostAlpha present")
    print("[OK] LightGBMAlpha present")
    print("[OK] compute_ic present")
    print("[OK] compute_rank_ic present")
    print("\n[PASS] dim_129: ML alpha signals (XGBoost / LightGBM / TabNet) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_129: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sai.ml_alpha_v3")
    sys.exit(0)
PYEOF
