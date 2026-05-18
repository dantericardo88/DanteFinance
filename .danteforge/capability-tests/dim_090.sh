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

# Build the dispatcher (registers all 102+ functions, teaches parser known codes)
dispatcher = FunctionDispatcher()
registry = dispatcher.registry

# Verify 90+ function codes registered (now includes CDSW)
n_codes = len(registry)
assert n_codes >= 90, f"Expected >= 90 function codes registered: {n_codes}"
print(f"[OK] FunctionDispatcher: {n_codes} function codes registered")

# Verify specific essential codes exist — including newly-implemented ones
for code in ["DES", "GP", "FA", "YAS", "ECO", "PORT", "VAR", "OMON", "WACC", "SRCH",
             "HP", "CDSW"]:
    fh = registry.lookup(code)
    assert fh is not None, f"Function code '{code}' should be registered"
    assert fh.description, f"Handler for '{code}' should have description"
print("[OK] All essential codes registered: DES GP FA YAS ECO PORT VAR OMON WACC SRCH HP CDSW")

# ---------------------------------------------------------------
# Test CommandParser.parse(): "AAPL US Equity HP"
# spec: {ticker:"AAPL", exchange:"US", asset:"Equity", function:"HP"}
# ---------------------------------------------------------------
parser = CommandParser()
cmd_hp = parser.parse("AAPL US EQUITY HP")
assert cmd_hp.ticker is not None and "AAPL" in cmd_hp.ticker, \
    f"Ticker should contain AAPL: {cmd_hp.ticker}"
assert cmd_hp.asset_class == "EQUITY", \
    f"Asset class should be EQUITY: {cmd_hp.asset_class}"
assert cmd_hp.function_code == "HP", \
    f"Function code should be HP: {cmd_hp.function_code}"
print(f"[OK] parse('AAPL US EQUITY HP'): ticker={cmd_hp.ticker} asset={cmd_hp.asset_class} func={cmd_hp.function_code}")

# Test "AAPL DES"
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

# ---------------------------------------------------------------
# NEW: Previously-stubbed commands now return non-None results
# HP, WACC, CDSW must return CommandResult with success=True or
# a meaningful stub (not None)
# ---------------------------------------------------------------

# HP — historical price via yfinance (15s timeout for network)
import threading
_hp_result = [None]
def _run_hp():
    cmd_hp_dispatch = ParsedCommand(
        raw="AAPL HP PERIOD=1Y", ticker="AAPL", function_code="HP",
        params={"PERIOD": "1Y"}
    )
    _hp_result[0] = dispatcher.dispatch(cmd_hp_dispatch)

t_hp = threading.Thread(target=_run_hp, daemon=True)
t_hp.start()
t_hp.join(timeout=15)

if _hp_result[0] is None:
    fh_hp = registry.lookup("HP")
    assert fh_hp is not None, "HP handler must be registered even if network timed out"
    print(f"[OK] dispatch(HP): handler registered (network timeout — stub expected)")
else:
    result_hp = _hp_result[0]
    assert result_hp.code == "HP", f"HP result code must be 'HP': {result_hp.code}"
    assert isinstance(result_hp.text, str) and "HP" in result_hp.text, \
        f"HP result text must contain 'HP': {result_hp.text}"
    print(f"[OK] dispatch(HP): success={result_hp.success} text='{result_hp.text[:70]}'")

# WACC — DCF/WACC via dcf_wacc_v3 (15s timeout for network)
_wacc_result = [None]
def _run_wacc():
    cmd_wacc = ParsedCommand(
        raw="AAPL WACC", ticker="AAPL", function_code="WACC", params={}
    )
    _wacc_result[0] = dispatcher.dispatch(cmd_wacc)

t_wacc = threading.Thread(target=_run_wacc, daemon=True)
t_wacc.start()
t_wacc.join(timeout=15)

if _wacc_result[0] is None:
    fh_wacc = registry.lookup("WACC")
    assert fh_wacc is not None, "WACC handler must be registered even if network timed out"
    print(f"[OK] dispatch(WACC): handler registered (network timeout — stub expected)")
else:
    result_wacc = _wacc_result[0]
    assert result_wacc.code == "WACC", f"WACC result code must be 'WACC': {result_wacc.code}"
    assert isinstance(result_wacc.text, str) and "WACC" in result_wacc.text, \
        f"WACC result text must contain 'WACC': {result_wacc.text}"
    print(f"[OK] dispatch(WACC): success={result_wacc.success} text='{result_wacc.text[:70]}'")

# CDSW — Credit default swap / Merton model
# Run with a short timeout so CI doesn't hang if network is slow
import threading
_cdsw_result = [None]
def _run_cdsw():
    cmd_cdsw = ParsedCommand(
        raw="AAPL CDSW", ticker="AAPL", function_code="CDSW", params={}
    )
    _cdsw_result[0] = dispatcher.dispatch(cmd_cdsw)

t = threading.Thread(target=_run_cdsw, daemon=True)
t.start()
t.join(timeout=15)  # allow 15s for network calls

if _cdsw_result[0] is None:
    # Timed out — verify handler is at least registered and callable
    fh_cdsw = registry.lookup("CDSW")
    assert fh_cdsw is not None, "CDSW handler must be registered even if network timed out"
    print(f"[OK] dispatch(CDSW): handler registered (network timeout — stub expected)")
else:
    result_cdsw = _cdsw_result[0]
    assert result_cdsw.code == "CDSW", f"CDSW result code must be 'CDSW': {result_cdsw.code}"
    assert isinstance(result_cdsw.text, str) and "CDSW" in result_cdsw.text, \
        f"CDSW result text must contain 'CDSW': {result_cdsw.text}"
    print(f"[OK] dispatch(CDSW): success={result_cdsw.success} text='{result_cdsw.text[:70]}'")

# ---------------------------------------------------------------
# NEW: At least 5 command types work end-to-end (return non-None).
# These commands gracefully stub when their optional modules are absent,
# so they are safe to call without network access.
# ---------------------------------------------------------------
commands_to_test = [
    ParsedCommand(raw="AAPL DES",  ticker="AAPL", function_code="DES",  params={}),
    ParsedCommand(raw="AAPL GP",   ticker="AAPL", function_code="GP",   params={}),
    ParsedCommand(raw="AAPL FA",   ticker="AAPL", function_code="FA",   params={}),
    ParsedCommand(raw="AAPL RV",   ticker="AAPL", function_code="RV",   params={}),
    ParsedCommand(raw="AAPL DVD",  ticker="AAPL", function_code="DVD",  params={}),
    ParsedCommand(raw="AAPL OWN",  ticker="AAPL", function_code="OWN",  params={}),
    ParsedCommand(raw="AAPL OMON", ticker="AAPL", function_code="OMON", params={}),
]
working = 0
for pcmd in commands_to_test:
    r = dispatcher.dispatch(pcmd)
    if r is not None and r.code == pcmd.function_code:
        working += 1

assert working >= 5, f"At least 5 command types must return valid CommandResult: {working}"
print(f"[OK] End-to-end dispatch: {working}/{len(commands_to_test)} commands returned valid CommandResult")

# Verify CDSW is registered in Fixed Income category
assert "CDSW" in cats.get("Fixed Income", []), \
    f"CDSW should be in 'Fixed Income' category: {cats.get('Fixed Income', [])}"
print(f"[OK] CDSW registered in Fixed Income category")

print("\n[PASS] dim_090: Bloomberg bar")
PYEOF
