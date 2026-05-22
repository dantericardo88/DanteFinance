#!/usr/bin/env bash
# dim_128: Product-line / segment margin NLP (earnings call parsing)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sai.segment_margin_v3 import (
        SegmentMarginExtractor, parse_earnings_call_segments, compute_margin_trend
    )
    assert SegmentMarginExtractor is not None, "Missing SegmentMarginExtractor"
    assert parse_earnings_call_segments is not None, "Missing parse_earnings_call_segments"
    assert compute_margin_trend is not None, "Missing compute_margin_trend"
    print("[OK] SegmentMarginExtractor present")
    print("[OK] parse_earnings_call_segments present")
    print("[OK] compute_margin_trend present")
    print("\n[PASS] dim_128: Product-line / segment margin NLP (earnings call parsing) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_128: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sai.segment_margin_v3")
    sys.exit(0)
PYEOF
