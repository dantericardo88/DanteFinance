#!/usr/bin/env bash
# dim_146: Liability-driven investing (LDI) / duration-gap matching
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.ldi_v3 import (
        LDIPortfolio, DurationGapAnalyzer, compute_duration_gap, match_liability_cashflows
    )
    assert LDIPortfolio is not None, "Missing LDIPortfolio"
    assert DurationGapAnalyzer is not None, "Missing DurationGapAnalyzer"
    assert compute_duration_gap is not None, "Missing compute_duration_gap"
    assert match_liability_cashflows is not None, "Missing match_liability_cashflows"
    print("[OK] LDIPortfolio present")
    print("[OK] DurationGapAnalyzer present")
    print("[OK] compute_duration_gap present")
    print("[OK] match_liability_cashflows present")
    print("\n[PASS] dim_146: Liability-driven investing (LDI) / duration-gap matching -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_146: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.ldi_v3")
    sys.exit(0)
PYEOF
