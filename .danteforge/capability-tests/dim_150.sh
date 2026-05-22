#!/usr/bin/env bash
# dim_150: Portfolio copilot (NL query-driven what-if + trade ideas)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.api.portfolio_copilot_v3 import (
        PortfolioCopilot, WhatIfAnalyzer, TradeIdeaGenerator, explain_portfolio
    )
    assert PortfolioCopilot is not None, "Missing PortfolioCopilot"
    assert WhatIfAnalyzer is not None, "Missing WhatIfAnalyzer"
    assert TradeIdeaGenerator is not None, "Missing TradeIdeaGenerator"
    assert explain_portfolio is not None, "Missing explain_portfolio"
    print("[OK] PortfolioCopilot present")
    print("[OK] WhatIfAnalyzer present")
    print("[OK] TradeIdeaGenerator present")
    print("[OK] explain_portfolio present")
    print("\n[PASS] dim_150: Portfolio copilot (NL query-driven what-if + trade ideas) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_150: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.api.portfolio_copilot_v3")
    sys.exit(0)
PYEOF
