#!/bin/bash
# dim_059: MCP Server v3 — ToolRegistry, static method dispatch, AND real SENTINEL module invocations
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.api.mcp_server_v3 import (
    ToolRegistry, ToolDefinition, MCPToolHandler,
    SENTINEL_VERSION, SERVER_NAME, MCP_SDK_AVAILABLE,
    SentinelMCPServer,
)

# 1. Constants
assert SENTINEL_VERSION == "3.0.0", f"Expected 3.0.0, got {SENTINEL_VERSION}"
assert SERVER_NAME == "sentinel-mcp-server"
print(f"[OK] Constants: SENTINEL_VERSION={SENTINEL_VERSION}, SERVER_NAME={SERVER_NAME}")
print(f"[OK] MCP_SDK_AVAILABLE={MCP_SDK_AVAILABLE} (optional)")

# 2. ToolDefinition dataclass
tool = ToolDefinition(
    name="get_stock_quote",
    description="Get current stock price for a ticker symbol",
    parameters={"properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    handler=lambda ticker: {"ticker": ticker, "price": 150.0},
    category="market_data",
    requires_ticker=True,
    tags=["price", "realtime"],
)
assert tool.name == "get_stock_quote"
assert tool.category == "market_data"
assert tool.requires_ticker == True
assert "price" in tool.tags
print("[OK] ToolDefinition dataclass fields correct")

# 3. to_mcp_dict() format
mcp_dict = tool.to_mcp_dict()
assert "name" in mcp_dict
assert "description" in mcp_dict
assert "inputSchema" in mcp_dict
assert mcp_dict["name"] == "get_stock_quote"
assert mcp_dict["inputSchema"]["type"] == "object"
print(f"[OK] ToolDefinition.to_mcp_dict() -> keys={list(mcp_dict.keys())}")

# 4. ToolRegistry CRUD
reg = ToolRegistry()
assert reg.count == 0
reg.register(tool)
assert reg.count == 1
fetched = reg.get_tool("get_stock_quote")
assert fetched is not None
assert fetched.name == "get_stock_quote"
print("[OK] ToolRegistry.register() and get_tool() work")

# 5. list_tools() and category filter
tool2 = ToolDefinition(
    name="get_fundamentals",
    description="Get fundamental data",
    parameters={"properties": {}},
    handler=lambda ticker: {},
    category="fundamental",
)
reg.register(tool2)
all_tools = reg.list_tools()
assert len(all_tools) == 2
market_tools = reg.list_tools(category="market_data")
assert len(market_tools) == 1
assert market_tools[0].name == "get_stock_quote"
print(f"[OK] ToolRegistry.list_tools() total={len(all_tools)}, market_data={len(market_tools)}")

# 6. categories()
cats = reg.categories()
assert "market_data" in cats
assert "fundamental" in cats
assert cats == sorted(cats)  # sorted
print(f"[OK] ToolRegistry.categories() = {cats}")

# 7. get_mcp_schema()
schema = reg.get_mcp_schema()
assert len(schema) == 2
assert all("name" in s and "description" in s and "inputSchema" in s for s in schema)
print(f"[OK] ToolRegistry.get_mcp_schema() returns {len(schema)} tool schemas")

# 8. execute() dispatches to handler
result = reg.execute("get_stock_quote", {"ticker": "AAPL"})
assert isinstance(result, dict)
assert "ticker" in result or "error" not in result
print(f"[OK] ToolRegistry.execute() dispatches to handler -> {result}")

# 9. execute() unknown tool returns error dict
err = reg.execute("nonexistent_tool", {})
assert "error" in err
assert "available" in err
print(f"[OK] ToolRegistry.execute() on unknown tool returns error dict")

# 10. MCPToolHandler class exists
assert hasattr(MCPToolHandler, 'get_stock_quote')
assert callable(MCPToolHandler.get_stock_quote)
print("[OK] MCPToolHandler.get_stock_quote is a callable static method")

# 11. SentinelMCPServer registers >= 124 tools across 13 categories (Wave 36)
server = SentinelMCPServer()
assert server.registry.count >= 124, f"Expected >= 124 tools, got {server.registry.count}"
cats = server.registry.categories()
expected_cats = {
    # Original 8
    "market_data", "fundamental", "technical", "sec_regulatory",
    "portfolio_risk", "backtesting", "ai_nlp", "alternative_data",
    # Wave 36 — Expanded agentic surface
    "advanced_analytics", "alt_data", "macro", "crypto_onchain", "private_markets",
}
assert expected_cats.issubset(set(cats)), f"Missing categories: {expected_cats - set(cats)}"
print(f"[OK] SentinelMCPServer has {server.registry.count} tools in {len(cats)} categories: {cats}")

# 12. New v3 tools are registered
new_tools = ["check_backtest_overfitting", "detect_market_regime", "run_portfolio_risk_full", "compute_position_size"]
for t in new_tools:
    td = server.registry.get_tool(t)
    assert td is not None, f"Tool '{t}' not registered"
print(f"[OK] New v3 tools registered: {new_tools}")

# 12b. Wave-36 expanded tools (50 new): verify every one is registered
wave36_tools = [
    # Advanced analytics (10)
    "get_brinson_attribution", "get_factor_loading", "get_kelly_size", "get_risk_parity",
    "get_monte_carlo_var", "get_overfitting_score", "get_walk_forward_results",
    "get_paper_trading_pnl", "get_strategy_promotion_status", "get_factor_decay_curve",
    # Alt-data (10)
    "get_social_sentiment_v3", "get_news_pipeline_signal", "get_congress_clusters",
    "get_insider_clusters", "get_short_squeeze_score", "get_options_skew",
    "get_vol_term_structure", "get_fear_greed_v2", "get_labor_market_tightness",
    "get_central_bank_tone",
    # Macro (10)
    "get_country_macro", "get_central_bank_speech_score", "get_treasury_auction_schedule",
    "get_cot_market_position", "get_fred_series", "get_econ_calendar_today",
    "get_yield_spread_recession_prob", "get_inflation_regime",
    "get_global_pmi_dashboard", "get_credit_spreads_dashboard",
    # Crypto / onchain (10)
    "get_dex_pool_metrics", "get_lp_returns_attribution", "get_impermanent_loss_risk",
    "get_rugpull_risk_score", "get_mvrv_zone", "get_nvt_signal",
    "get_btc_whale_alerts", "get_eth_mempool_pressure", "get_btc_network_health",
    "get_stablecoin_health",
    # Private markets + corporate (10)
    "get_form_d_filing", "get_ria_profile_v2", "get_nport_holdings",
    "get_berkus_valuation", "get_scorecard_valuation", "get_vc_method_valuation",
    "get_fund_metrics", "get_lbo_valuation", "get_activist_campaigns_live",
    "get_ipo_pop_prediction",
]
assert len(wave36_tools) == 50, f"Wave 36 should list 50 tools, found {len(wave36_tools)}"
for tname in wave36_tools:
    td = server.registry.get_tool(tname)
    assert td is not None, f"Wave-36 tool '{tname}' not registered"
print(f"[OK] Wave 36: all {len(wave36_tools)} expanded agentic tools registered")

# 12c. Real-invocation spot-checks on Wave-36 handlers (offline-safe)
print("\n[REAL Wave36] Spot-checking Wave-36 handlers against real SENTINEL modules...")

r = MCPToolHandler.get_kelly_size(0.55, 0.10, 0.07)
assert r.get("source") == "sentinel.spm.position_sizing_v3", f"Bad source: {r}"
assert "full_kelly" in r and 0.0 < r["full_kelly"] < 1.0, r
print(f"[OK] get_kelly_size -> {r['source']} (full_kelly={r['full_kelly']})")

r = MCPToolHandler.get_berkus_valuation(sound_idea=500_000, prototype=400_000, mgmt_quality=300_000)
assert r.get("source") == "sentinel.sfe.private_company_profiles", f"Bad source: {r}"
assert r["valuation"]["mid"] > 0, r
print(f"[OK] get_berkus_valuation -> {r['source']} (mid=${r['valuation']['mid']:,.0f})")

r = MCPToolHandler.get_vc_method_valuation(
    projected_exit_revenue=100_000_000, projected_exit_multiple=5.0,
    years_to_exit=5, investment_amount=2_000_000,
)
assert r.get("source") == "sentinel.sfe.private_company_profiles", f"Bad source: {r}"
assert r["valuation"]["post_money"] > 0, r
print(f"[OK] get_vc_method_valuation -> {r['source']} (post_money=${r['valuation']['post_money']:,.0f})")

r = MCPToolHandler.get_brinson_attribution({"AAPL": 0.6, "MSFT": 0.4})
assert r.get("source") == "sentinel.spm.attribution_v3", f"Bad source: {r}"
assert "attribution" in r, r
print(f"[OK] get_brinson_attribution -> {r['source']}")

r = MCPToolHandler.get_impermanent_loss_risk("ETH", "USDC", vol_a=0.8, vol_b=0.01, correlation=0.0)
assert r.get("source") == "sentinel.sfe.defi_analytics_v3", f"Bad source: {r}"
assert "implied_il_30d" in r, r
print(f"[OK] get_impermanent_loss_risk -> {r['source']} (il_30d={r['implied_il_30d']})")

# -----------------------------------------------------------------------
# REAL INVOCATION TESTS — call into actual SENTINEL modules (no mocks)
# -----------------------------------------------------------------------

# 13. Kelly Criterion via sentinel.spm.position_sizing_v3
print("\n[REAL] Testing Kelly Criterion (sentinel.spm.position_sizing_v3)...")
r = MCPToolHandler.compute_kelly_size(0.55, 0.10, 0.07)
assert r.get("source") == "sentinel.spm.position_sizing_v3", f"Expected v3 source, got: {r.get('source')}"
assert "full_kelly" in r and "quarter_kelly" in r, f"Missing kelly fields: {list(r.keys())}"
assert 0.0 < r["full_kelly"] < 1.0, f"Kelly fraction out of range: {r['full_kelly']}"
print(f"[OK] Kelly full={r['full_kelly']}, quarter={r['quarter_kelly']} — source={r['source']}")

# 14. Position sizing via sentinel.spm.position_sizing_v3
print("\n[REAL] Testing position sizing (sentinel.spm.position_sizing_v3)...")
r = MCPToolHandler.compute_position_size(
    method="kelly", win_rate=0.55, avg_win=0.10, avg_loss=0.07,
    portfolio_equity=100_000.0, entry_price=150.0
)
assert r.get("source") == "sentinel.spm.position_sizing_v3", f"Expected v3 source: {r}"
assert "notional" in r and "shares" in r, f"Missing position fields: {list(r.keys())}"
assert r["shares"] > 0, f"Expected shares > 0: {r['shares']}"
print(f"[OK] Position size: {r['shares']} shares, notional=${r['notional']} — source={r['source']}")

# 15. PBO + DSR overfitting detection via sentinel.sbx.overfitting_detection_v3
print("\n[REAL] Testing overfitting detection (sentinel.sbx.overfitting_detection_v3)...")
import numpy as np
rng = np.random.default_rng(42)
returns_list = [rng.normal(0.0, 0.01, 252).tolist() for _ in range(10)]
r = MCPToolHandler.check_backtest_overfitting(returns_list)
assert r.get("source") == "sentinel.sbx.overfitting_detection_v3", f"Expected v3 source: {r}"
assert "pbo" in r and "deflated_sharpe" in r, f"Missing fields: {list(r.keys())}"
assert 0.0 <= r["pbo"] <= 1.0, f"PBO out of range: {r['pbo']}"
print(f"[OK] PBO={r['pbo']}, DSR={r['deflated_sharpe']}, significant={r['is_significant']} — source={r['source']}")

# 16. Portfolio optimizer v3 (HRP) via sentinel.spm.portfolio_optimizer_v3
print("\n[REAL] Testing portfolio optimizer HRP (sentinel.spm.portfolio_optimizer_v3)...")
r = MCPToolHandler.optimize_portfolio(["AAPL", "MSFT", "JPM"], method="hrp")
assert r.get("source") == "sentinel.spm.portfolio_optimizer_v3", f"Expected v3 source: {r}"
assert "weights" in r, f"Missing weights: {list(r.keys())}"
w = r["weights"]
assert set(w.keys()) == {"AAPL", "MSFT", "JPM"}, f"Wrong tickers in weights: {list(w.keys())}"
total_w = sum(w.values())
assert abs(total_w - 1.0) < 0.01, f"Weights don't sum to 1: {total_w}"
print(f"[OK] HRP weights={w}, sum={round(total_w,4)} — source={r['source']}")

# 17. Portfolio risk full report via sentinel.spm.portfolio_risk_v3
print("\n[REAL] Testing portfolio risk full report (sentinel.spm.portfolio_risk_v3)...")
r = MCPToolHandler.run_portfolio_risk_full({"AAPL": 0.6, "MSFT": 0.4}, portfolio_value=500_000.0)
assert r.get("source") == "sentinel.spm.portfolio_risk_v3", f"Expected v3 source: {r.get('source')}"
assert "report" in r, f"Missing report key: {list(r.keys())}"
print(f"[OK] Risk report returned with keys={list(r['report'].keys())[:5]}... — source={r['source']}")

print(f"\n[PASS] dim_059: MCP Server v3 — ToolRegistry, dispatch, {server.registry.count}+ tools across {len(cats)} categories, and 10 real SENTINEL module invocations all verified")
PYEOF
