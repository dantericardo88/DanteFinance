import sys, os
sys.path.insert(0, os.getcwd())
from sentinel.api.bloomberg_bar_v3 import FunctionDispatcher, ParsedCommand

dispatcher = FunctionDispatcher()
registry = dispatcher.registry
n_codes = len(registry)
print(f'Function codes registered: {n_codes}')

cats = registry.list_by_category()
print(f'Categories: {list(cats.keys())}')
fi_codes = cats.get("Fixed Income", [])
print(f'Fixed Income codes: {fi_codes}')
print(f'CDSW in Fixed Income: {"CDSW" in fi_codes}')

# Test dispatch
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
        print(f'  [OK] {pcmd.function_code}: success={r.success}')
    else:
        print(f'  [FAIL] {pcmd.function_code}')

print(f'Working: {working}/7')

# Check unknown dispatch
from sentinel.api.bloomberg_bar_v3 import ParsedCommand
parsed_unknown = ParsedCommand(raw="AAPL UNKNOWN", ticker="AAPL", function_code="UNKNOWXXX")
result = dispatcher.dispatch(parsed_unknown)
print(f'Unknown dispatch: success={result.success}')

# Check no function code
parsed_no_func = ParsedCommand(raw="AAPL", ticker="AAPL", function_code=None)
result_nf = dispatcher.dispatch(parsed_no_func)
print(f'No func dispatch: success={result_nf.success}')
