#!/bin/bash
# dim_059: MCP Server v3 — class existence, ToolRegistry, static method dispatch
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.api.mcp_server_v3 import (
    ToolRegistry, ToolDefinition, MCPToolHandler,
    SENTINEL_VERSION, SERVER_NAME, MCP_SDK_AVAILABLE,
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

print("\n[PASS] dim_059: MCP Server v3 — ToolRegistry and tool dispatch verified")
PYEOF
