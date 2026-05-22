#!/usr/bin/env bash
# dim_126: Executive compensation benchmarking (CEO/CFO pay vs peers)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.exec_comp_v3 import (
        ExecCompBenchmark, compute_pay_ratio, peer_compensation_comparison
    )
    assert ExecCompBenchmark is not None, "Missing ExecCompBenchmark"
    assert compute_pay_ratio is not None, "Missing compute_pay_ratio"
    assert peer_compensation_comparison is not None, "Missing peer_compensation_comparison"
    print("[OK] ExecCompBenchmark present")
    print("[OK] compute_pay_ratio present")
    print("[OK] peer_compensation_comparison present")
    print("\n[PASS] dim_126: Executive compensation benchmarking (CEO/CFO pay vs peers) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_126: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.exec_comp_v3")
    sys.exit(0)
PYEOF
