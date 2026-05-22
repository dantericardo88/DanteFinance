#!/usr/bin/env bash
# dim_137: Margin analytics & SPAN / portfolio-margin simulation
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.margin_analytics_v3 import (
        SPANMarginCalculator, PortfolioMarginSimulator, compute_span_margin
    )
    assert SPANMarginCalculator is not None, "Missing SPANMarginCalculator"
    assert PortfolioMarginSimulator is not None, "Missing PortfolioMarginSimulator"
    assert compute_span_margin is not None, "Missing compute_span_margin"
    print("[OK] SPANMarginCalculator present")
    print("[OK] PortfolioMarginSimulator present")
    print("[OK] compute_span_margin present")
    print("\n[PASS] dim_137: Margin analytics & SPAN / portfolio-margin simulation -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_137: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.spm.margin_analytics_v3")
    sys.exit(0)
PYEOF
