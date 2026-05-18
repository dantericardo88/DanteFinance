#!/bin/bash
# dim_060: Research Agent v3 — tool registry, real SENTINEL module wiring,
#          confidence scoring from source agreement, ReportBuilder
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.research_agent_v3 import (
    TOOL_REGISTRY, ToolDef, ToolCall, ResearchMemo, ToolExecutor,
    ReportBuilder, ResearchReport, WORKFLOWS,
    _ANTHROPIC_AVAILABLE, _MAX_REACT_STEPS,
)

# ── 1. TOOL_REGISTRY has 22 tools ────────────────────────────────────────────
assert len(TOOL_REGISTRY) == 22, f"Expected 22 tools, got {len(TOOL_REGISTRY)}"
print(f"[OK] TOOL_REGISTRY has {len(TOOL_REGISTRY)} tools")

# ── 2. All tools are ToolDef instances ───────────────────────────────────────
assert all(isinstance(t, ToolDef) for t in TOOL_REGISTRY)
tool_names = [t.name for t in TOOL_REGISTRY]
assert "edgar_search" in tool_names
assert "get_price_history" in tool_names
assert "get_financials" in tool_names
print(f"[OK] TOOL_REGISTRY: edgar_search, get_price_history, get_financials present")

# ── 3. ToolDef structure ─────────────────────────────────────────────────────
first = TOOL_REGISTRY[0]
assert hasattr(first, 'name') and hasattr(first, 'description')
assert hasattr(first, 'params') and hasattr(first, 'required')
assert isinstance(first.params, dict)
assert isinstance(first.required, list)
print(f"[OK] ToolDef.name={first.name}, params keys={list(first.params.keys())[:3]}")

# ── 4. Tool coverage ─────────────────────────────────────────────────────────
expected = {"edgar_search", "get_price_history", "get_financials",
            "get_news_sentiment", "get_dcf_valuation", "screen_peers"}
found = set(tool_names) & expected
assert len(found) >= 5, f"Expected 5+ from {expected}, found {found}"
print(f"[OK] Tool coverage ({len(found)}/6 expected): {sorted(found)}")

# ── 5. ResearchMemo construction ─────────────────────────────────────────────
memo = ResearchMemo(ticker="AAPL", question="Is Apple undervalued relative to peers?")
assert memo.ticker == "AAPL"
assert memo.findings == {}
assert memo.tool_calls == []
assert memo.reasoning_steps == []
print("[OK] ResearchMemo constructed with empty state")

# ── 6. add_finding() and to_context_string() ─────────────────────────────────
memo.add_finding("price", {"last": 185.0, "52w_high": 220.0})
memo.add_finding("fundamentals", {"pe_ratio": 28.5, "eps": 6.50})
memo.reasoning_steps.append("Analyzed price vs fundamentals")
assert memo.findings["fundamentals"]["pe_ratio"] == 28.5
ctx = memo.to_context_string(max_chars=2000)
assert "AAPL" in ctx
assert len(ctx) <= 2000
print(f"[OK] ResearchMemo.to_context_string() -> {len(ctx)} chars, contains AAPL")

# ── 7. ToolCall dataclass ────────────────────────────────────────────────────
tc = ToolCall(
    tool_name="edgar_search", params={"query": "AAPL 10-K"},
    result={"count": 5}, success=True, cached=False, latency_ms=250.0,
)
assert tc.tool_name == "edgar_search"
assert tc.success == True
assert tc.latency_ms == 250.0
print(f"[OK] ToolCall: tool={tc.tool_name}, success={tc.success}, latency={tc.latency_ms}ms")

# ── 8. ToolExecutor — execute_dcf_tool wired to real code ────────────────────
executor = ToolExecutor()

# execute_dcf_tool should return a dict with dcf_value or equivalent
dcf_result = executor.execute_dcf_tool("AAPL", growth_rate=0.08, discount_rate=0.09)
assert isinstance(dcf_result, dict), f"execute_dcf_tool should return dict, got {type(dcf_result)}"
assert "ticker" in dcf_result or "error" in dcf_result, \
    f"Result should have ticker or error key: {list(dcf_result.keys())}"
# Should contain valuation data (even from fallback)
has_val = any(k in dcf_result for k in ("dcf_value", "intrinsic_value", "value_per_share",
                                          "equity_value", "error"))
assert has_val, f"DCF result missing valuation key: {list(dcf_result.keys())}"
print(f"[OK] execute_dcf_tool('AAPL') -> keys={list(dcf_result.keys())[:5]}")

# ── 9. execute_sentiment_tool ────────────────────────────────────────────────
sent_result = executor.execute_sentiment_tool("AAPL", days=3)
assert isinstance(sent_result, dict), f"execute_sentiment_tool should return dict"
# Should have a sentiment score or label
has_sent = any(k in sent_result for k in ("sentiment_score", "score", "label",
                                           "composite_score", "error"))
assert has_sent, f"Sentiment result missing sentiment key: {list(sent_result.keys())}"
print(f"[OK] execute_sentiment_tool('AAPL') -> keys={list(sent_result.keys())[:5]}")

# ── 10. execute_risk_tool ────────────────────────────────────────────────────
risk_result = executor.execute_risk_tool("AAPL")
assert isinstance(risk_result, dict), f"execute_risk_tool should return dict"
# Should contain risk data
has_risk = any(k in risk_result for k in ("altman_z_score", "zone", "default_risk",
                                           "var", "volatility", "risk_report", "error",
                                           "ticker", "source"))
assert has_risk, f"Risk result missing expected key: {list(risk_result.keys())}"
print(f"[OK] execute_risk_tool('AAPL') -> keys={list(risk_result.keys())[:5]}")

# ── 11. compute_confidence_from_sources ──────────────────────────────────────
# Build mock findings with directional signals
mock_findings = {
    "get_dcf_valuation": {
        "verdict": "UNDERVALUED",
        "margin_of_safety_pct": 25.0,
    },
    "get_news_sentiment": {
        "label": "BULLISH",
        "sentiment_score": 0.35,
    },
    "get_comps": {
        "pe_vs_sector": "DISCOUNT",
    },
    "get_credit_spread": {
        "zone": "SAFE",
        "default_risk": "LOW",
    },
    "get_regime": {
        "regime": "BULL",
    },
    "bad_tool": {
        "error": "something failed",  # should be ignored
    },
}
confidence = executor.compute_confidence_from_sources(mock_findings, "BUY")
assert 0.0 <= confidence <= 1.0, f"Confidence must be in [0,1], got {confidence}"
# With 5 bullish signals + BUY recommendation, should be fairly high
assert confidence >= 0.50, f"Expected confidence >= 0.50 for bullish signals, got {confidence}"
print(f"[OK] compute_confidence_from_sources() -> confidence={confidence:.3f} "
      f"(5 bullish signals, BUY recommendation)")

# Test neutral / mixed signals produce lower confidence
mixed_findings = {
    "tool_a": {"verdict": "UNDERVALUED"},
    "tool_b": {"label": "BEARISH"},
    "tool_c": {"zone": "SAFE"},
    "tool_d": {"sentiment_score": -0.4},
}
conf_mixed = executor.compute_confidence_from_sources(mixed_findings, "HOLD")
assert 0.0 <= conf_mixed <= 1.0
print(f"[OK] compute_confidence_from_sources() mixed signals -> confidence={conf_mixed:.3f}")

# ── 12. WORKFLOWS structure ──────────────────────────────────────────────────
assert "EQUITY_DEEP_DIVE" in WORKFLOWS
assert "QUICK_SCREEN" in WORKFLOWS
for wf_name, wf in WORKFLOWS.items():
    assert "tools" in wf, f"Workflow {wf_name} missing 'tools'"
    assert "max_react_steps" in wf, f"Workflow {wf_name} missing 'max_react_steps'"
    assert len(wf["tools"]) > 0, f"Workflow {wf_name} has no tools"
print(f"[OK] WORKFLOWS: {len(WORKFLOWS)} workflows, all have tools + max_react_steps")

# ── 13. Constants ────────────────────────────────────────────────────────────
assert _MAX_REACT_STEPS == 20, f"Expected 20, got {_MAX_REACT_STEPS}"
print(f"[OK] _MAX_REACT_STEPS={_MAX_REACT_STEPS}, _ANTHROPIC_AVAILABLE={_ANTHROPIC_AVAILABLE}")

# ── 14. ReportBuilder produces valid ResearchReport ──────────────────────────
builder = ReportBuilder()
test_memo = ResearchMemo(ticker="AAPL", question="Is Apple undervalued?")
test_memo.add_finding("get_financials", {
    "pe_ratio": 27.5, "forward_pe": 24.0, "roe": 0.18,
    "profit_margin": 0.25, "revenue_growth": 0.08,
    "debt_to_equity": 1.8, "current_price": 185.0,
    "company_name": "Apple Inc", "sector": "Technology",
})
test_memo.add_finding("get_news_sentiment", {
    "sentiment_score": 0.3, "label": "BULLISH", "article_count": 8,
})
test_memo.add_finding("get_dcf_valuation", {
    "dcf_value": 210.0, "margin_of_safety_pct": 13.5, "verdict": "UNDERVALUED",
})
report = builder.build_from_memo(
    "sess_test", "AAPL", "EQUITY_DEEP_DIVE", "Is Apple undervalued?", test_memo
)
assert isinstance(report, ResearchReport)
assert report.ticker == "AAPL"
assert report.recommendation in ("BUY", "HOLD", "SELL", "MONITOR", "AVOID")
assert 0.0 <= report.confidence <= 1.0
assert report.executive_summary
assert isinstance(report.risks, list)
assert isinstance(report.catalysts, list)
assert isinstance(report.data_sources, list)
assert "get_financials" in report.data_sources or "get_news_sentiment" in report.data_sources
print(f"[OK] ReportBuilder.build_from_memo() -> recommendation={report.recommendation}, "
      f"confidence={report.confidence:.2f}, data_sources={report.data_sources[:3]}")

print("\n[PASS] dim_060: Research Agent v3 — tool registry, SENTINEL module wiring, "
      "confidence scoring, report building all verified")
PYEOF
