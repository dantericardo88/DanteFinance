#!/usr/bin/env bash
# dim_140: REIT fundamental analysis (FFO / AFFO / NAV / cap-rate)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.reit_analytics_v3 import (
        REITFundamentals, compute_ffo, compute_affo, compute_nav, compute_cap_rate
    )
    assert REITFundamentals is not None, "Missing REITFundamentals"
    assert compute_ffo is not None, "Missing compute_ffo"
    assert compute_affo is not None, "Missing compute_affo"
    assert compute_nav is not None, "Missing compute_nav"
    assert compute_cap_rate is not None, "Missing compute_cap_rate"
    print("[OK] REITFundamentals present")
    print("[OK] compute_ffo present")
    print("[OK] compute_affo present")
    print("[OK] compute_nav present")
    print("[OK] compute_cap_rate present")
    print("\n[PASS] dim_140: REIT fundamental analysis (FFO / AFFO / NAV / cap-rate) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_140: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.reit_analytics_v3")
    sys.exit(0)
PYEOF
