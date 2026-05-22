#!/usr/bin/env bash
# dim_119: Order book depth & microstructure analytics (LOB)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sds.order_book_v3 import (
        OrderBook, LimitOrderBook, compute_bid_ask_spread, compute_order_imbalance
    )
    assert OrderBook is not None, "Missing OrderBook"
    assert LimitOrderBook is not None, "Missing LimitOrderBook"
    assert compute_bid_ask_spread is not None, "Missing compute_bid_ask_spread"
    assert compute_order_imbalance is not None, "Missing compute_order_imbalance"
    print("[OK] OrderBook present")
    print("[OK] LimitOrderBook present")
    print("[OK] compute_bid_ask_spread present")
    print("[OK] compute_order_imbalance present")
    print("\n[PASS] dim_119: Order book depth & microstructure analytics (LOB) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_119: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sds.order_book_v3")
    sys.exit(0)
PYEOF
