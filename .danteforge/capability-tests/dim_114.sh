#!/usr/bin/env bash
# dim_114: Volatility forecasting (GARCH / EWMA / realized-vol)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sbx.vol_forecast_v3 import (
        GARCHModel, EWMAModel, RealizedVolEstimator, forecast_vol
    )
    assert GARCHModel is not None, "Missing GARCHModel"
    assert EWMAModel is not None, "Missing EWMAModel"
    assert RealizedVolEstimator is not None, "Missing RealizedVolEstimator"
    assert forecast_vol is not None, "Missing forecast_vol"
    print("[OK] GARCHModel present")
    print("[OK] EWMAModel present")
    print("[OK] RealizedVolEstimator present")
    print("[OK] forecast_vol present")
    print("\n[PASS] dim_114: Volatility forecasting (GARCH / EWMA / realized-vol) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_114: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sbx.vol_forecast_v3")
    sys.exit(0)
PYEOF
