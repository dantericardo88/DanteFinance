#!/usr/bin/env bash
# dim_090: Bloomberg bar — CommandParser.parse() logic and FunctionRegistry
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.api.bloomberg_bar_v3 import (
    CommandParser,
    FunctionRegistry,
    FunctionDispatcher,
    ParsedCommand,
    ASSET_CLASSES,
    VERSION,
)

# Test ASSET_CLASSES constant
assert "EQUITY" in ASSET_CLASSES, "ASSET_CLASSES should contain 'EQUITY'"
assert "GOVT" in ASSET_CLASSES, "ASSET_CLASSES should contain 'GOVT'"
assert "CORP" in ASSET_CLASSES, "ASSET_CLASSES should contain 'CORP'"
assert len(ASSET_CLASSES) >= 6, f"Expected >= 6 asset classes: {len(ASSET_CLASSES)}"
print(f"[OK] ASSET_CLASSES: {sorted(ASSET_CLASSES)}")

# Test VERSION
assert VERSION, "VERSION should be defined"
assert VERSION.startswith("3"), f"Expected version 3.x: {VERSION}"
print(f"[OK] VERSION: {VERSION}")

# Build the dispatcher (registers all 102 functions, teaches parser known codes)
dispatcher = FunctionDispatcher()
registry = dispatcher.registry

# Verify 102 function codes registered
n_codes = len(registry)
assert n_codes >= 90, f"Expected >= 90 function codes registered: {n_codes}"
print(f"[OK] FunctionDispatcher: {n_codes} function codes registered")

# Verify specific essential codes exist
for code in ["DES", "GP", "FA", "YAS", "ECO", "PORT", "VAR", "OMON", "WACC", "SRCH"]:
    fh = registry.lookup(code)
    assert fh is not None, f"Function code '{code}' should be registered"
    assert fh.description, f"Handler for '{code}' should have description"
print("[OK] Essential function codes (DES, GP, FA, YAS, ECO, PORT, VAR, OMON, WACC, SRCH) all registered")

# Test CommandParser.parse(): "AAPL DES"
parser = CommandParser()
cmd = parser.parse("AAPL DES")
assert cmd.ticker == "AAPL", f"Ticker should be AAPL: {cmd.ticker}"
assert cmd.function_code == "DES", f"Function code should be DES: {cmd.function_code}"
assert cmd.raw == "AAPL DES"
print(f"[OK] parse('AAPL DES'): ticker={cmd.ticker} func={cmd.function_code}")

# Test "MSFT GP" — price graph
cmd2 = parser.parse("MSFT GP")
assert cmd2.ticker == "MSFT", f"Ticker should be MSFT: {cmd2.ticker}"
assert cmd2.function_code == "GP", f"Function code should be GP: {cmd2.function_code}"
print(f"[OK] parse('MSFT GP'): ticker={cmd2.ticker} func={cmd2.function_code}")

# Test "AAPL US EQUITY" — asset class detection
cmd3 = parser.parse("AAPL US EQUITY")
assert cmd3.asset_class == "EQUITY", f"Asset class should be EQUITY: {cmd3.asset_class}"
assert "AAPL" in (cmd3.ticker or ""), f"Ticker should contain AAPL: {cmd3.ticker}"
print(f"[OK] parse('AAPL US EQUITY'): ticker={cmd3.ticker} asset_class={cmd3.asset_class}")

# Test "ECO" — single function code with no ticker
cmd4 = parser.parse("ECO")
assert cmd4.function_code == "ECO", f"Function code should be ECO: {cmd4.function_code}"
print(f"[OK] parse('ECO'): func={cmd4.function_code} ticker={cmd4.ticker}")

# Test empty command
cmd_empty = parser.parse("")
assert cmd_empty.function_code is None, "Empty command should have no function code"
print(f"[OK] parse(''): returns empty ParsedCommand")

# Test "AAPL FA PERIOD=Q" — params parsing
cmd5 = parser.parse("AAPL FA PERIOD=Q")
assert cmd5.ticker == "AAPL", f"Ticker should be AAPL: {cmd5.ticker}"
assert cmd5.function_code == "FA", f"Function code should be FA: {cmd5.function_code}"
assert "PERIOD" in cmd5.params, f"Should parse PERIOD param: {cmd5.params}"
assert cmd5.params["PERIOD"] == "Q", f"PERIOD should be Q: {cmd5.params}"
print(f"[OK] parse('AAPL FA PERIOD=Q'): params={cmd5.params}")

# Test FunctionRegistry.lookup() with alias (if any alias exists)
# At least verify lookup is case-insensitive
fh_lower = registry.lookup("des")
fh_upper = registry.lookup("DES")
assert fh_lower is not None and fh_upper is not None, "Lookup should work case-insensitively"
assert fh_lower.code == fh_upper.code, "lookup('des') and lookup('DES') should return same handler"
print(f"[OK] FunctionRegistry.lookup(): case-insensitive, returns handler code={fh_upper.code}")

# Test FunctionRegistry.list_by_category() — verify categories exist
cats = registry.list_by_category()
assert isinstance(cats, dict), f"list_by_category should return dict: {type(cats)}"
assert len(cats) >= 4, f"Expected >= 4 categories: {list(cats.keys())}"
assert "Equity Analysis" in cats, f"Should have 'Equity Analysis' category: {list(cats.keys())}"
for cat_name, codes in cats.items():
    assert len(codes) >= 1, f"Category '{cat_name}' should have >= 1 code"
print(f"[OK] FunctionRegistry.list_by_category(): {len(cats)} categories")

# Test CommandParser.suggest()
suggestions = parser.suggest("D")
assert isinstance(suggestions, list), "suggest() should return a list"
assert "DES" in suggestions or len(suggestions) > 0, f"suggest('D') should return results"
print(f"[OK] CommandParser.suggest('D'): {len(suggestions)} suggestions (includes DES={'DES' in suggestions})")

# Test FunctionDispatcher.dispatch() on unknown code
parsed_unknown = ParsedCommand(raw="AAPL UNKNOWN", ticker="AAPL", function_code="UNKNOWXXX")
result = dispatcher.dispatch(parsed_unknown)
assert result.success is False, "Unknown function code dispatch should return success=False"
print(f"[OK] dispatch(unknown code): success=False text='{result.text[:60]}'")

# Test dispatch on empty function code
parsed_no_func = ParsedCommand(raw="AAPL", ticker="AAPL", function_code=None)
result_nf = dispatcher.dispatch(parsed_no_func)
assert result_nf.success is False, "Missing function code should return success=False"
print(f"[OK] dispatch(no function code): success=False")

print("\n[PASS] dim_090: Bloomberg bar")
PYEOF
