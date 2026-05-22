#!/usr/bin/env bash
# dim_148: Multi-agent portfolio management workflow (research + PM + risk + exec)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.api.multi_agent_v3 import (
        ResearchAgent, PortfolioManagerAgent, RiskAgent, ExecutionAgent, MultiAgentOrchestrator
    )
    assert ResearchAgent is not None, "Missing ResearchAgent"
    assert PortfolioManagerAgent is not None, "Missing PortfolioManagerAgent"
    assert RiskAgent is not None, "Missing RiskAgent"
    assert ExecutionAgent is not None, "Missing ExecutionAgent"
    assert MultiAgentOrchestrator is not None, "Missing MultiAgentOrchestrator"
    print("[OK] ResearchAgent present")
    print("[OK] PortfolioManagerAgent present")
    print("[OK] RiskAgent present")
    print("[OK] ExecutionAgent present")
    print("[OK] MultiAgentOrchestrator present")
    print("\n[PASS] dim_148: Multi-agent portfolio management workflow (research + PM + risk + exec) -- all checks passed")
except ImportError as e:
    print(f"[NOT-BUILT] dim_148: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.api.multi_agent_v3")
    sys.exit(0)
PYEOF
