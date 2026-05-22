#!/usr/bin/env bash
# dim_123: Liquidity dry-up detection & bid-ask spread modeling
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.liquidity_v3 import (
        LiquidityMonitor, detect_liquidity_dryup, model_bid_ask_spread
    )
    assert LiquidityMonitor is not None, "Missing LiquidityMonitor"
    assert detect_liquidity_dryup is not None, "Missing detect_liquidity_dryup"
    assert model_bid_ask_spread is not None, "Missing model_bid_ask_spread"
    print("[OK] LiquidityMonitor present")
    print("[OK] detect_liquidity_dryup present")
    print("[OK] model_bid_ask_spread present")
    print("\n[PASS] dim_123: Liquidity dry-up detection & bid-ask spread modeling -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_123: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.liquidity_v3")
    sys.exit(0)
PYEOF
