#!/bin/bash
# dim_060: Research Agent v3 — class structure, TOOL_REGISTRY, ResearchMemo
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.research_agent_v3 import (
    TOOL_REGISTRY, ToolDef, ToolCall, ResearchMemo,
    _ANTHROPIC_AVAILABLE, _MAX_REACT_STEPS,
)

# 1. TOOL_REGISTRY has 22 tools
assert len(TOOL_REGISTRY) == 22, f"Expected 22 tools, got {len(TOOL_REGISTRY)}"
print(f"[OK] TOOL_REGISTRY has {len(TOOL_REGISTRY)} tools")

# 2. All tools are ToolDef instances with required fields
assert all(isinstance(t, ToolDef) for t in TOOL_REGISTRY)
tool_names = [t.name for t in TOOL_REGISTRY]
assert "edgar_search" in tool_names, f"edgar_search missing: {tool_names}"
assert "get_price_history" in tool_names
assert "get_financials" in tool_names
print(f"[OK] TOOL_REGISTRY: edgar_search, get_price_history, get_financials present")

# 3. ToolDef structure
first = TOOL_REGISTRY[0]
assert hasattr(first, 'name') and hasattr(first, 'description')
assert hasattr(first, 'params') and hasattr(first, 'required')
assert isinstance(first.params, dict)
assert isinstance(first.required, list)
print(f"[OK] ToolDef.name={first.name}, params keys={list(first.params.keys())[:3]}")

# 4. Expected tools coverage (use actual tool names)
expected = {"edgar_search", "get_price_history", "get_financials",
            "get_news_sentiment", "get_dcf_valuation", "screen_peers"}
found = set(tool_names) & expected
assert len(found) >= 5, f"Expected 5+ from {expected}, found {found}"
print(f"[OK] Tool coverage ({len(found)}/6 expected): {sorted(found)}")

# 5. ResearchMemo construction
memo = ResearchMemo(ticker="AAPL", question="Is Apple undervalued relative to peers?")
assert memo.ticker == "AAPL"
assert memo.findings == {}
assert memo.tool_calls == []
assert memo.reasoning_steps == []
print("[OK] ResearchMemo constructed with empty state")

# 6. add_finding() and to_context_string()
memo.add_finding("price", {"last": 185.0, "52w_high": 220.0})
memo.add_finding("fundamentals", {"pe_ratio": 28.5, "eps": 6.50})
memo.reasoning_steps.append("Analyzed price vs fundamentals")
assert memo.findings["fundamentals"]["pe_ratio"] == 28.5
ctx = memo.to_context_string(max_chars=2000)
assert "AAPL" in ctx
assert len(ctx) <= 2000
print(f"[OK] ResearchMemo.to_context_string() -> {len(ctx)} chars, contains AAPL")

# 7. ToolCall dataclass
tc = ToolCall(
    tool_name="edgar_search", params={"query": "AAPL 10-K"},
    result={"count": 5}, success=True, cached=False, latency_ms=250.0,
)
assert tc.tool_name == "edgar_search"
assert tc.success == True
assert tc.latency_ms == 250.0
print(f"[OK] ToolCall: tool={tc.tool_name}, success={tc.success}, latency={tc.latency_ms}ms")

# 8. Constants
assert _MAX_REACT_STEPS == 20, f"Expected 20, got {_MAX_REACT_STEPS}"
print(f"[OK] _MAX_REACT_STEPS={_MAX_REACT_STEPS}, _ANTHROPIC_AVAILABLE={_ANTHROPIC_AVAILABLE}")

print("\n[PASS] dim_060: Research Agent v3 — class structure and tool registry verified")
PYEOF
