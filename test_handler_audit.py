"""Audit all handlers to see which ones return real data vs stubs."""
import sys, os
sys.path.insert(0, os.getcwd())
from sentinel.api.bloomberg_bar_v3 import FunctionDispatcher, ParsedCommand

dispatcher = FunctionDispatcher()
registry = dispatcher.registry

# Test all handlers that don't need network (test with stub-safe inputs)
test_cases = [
    # Equity Analysis
    ("DES", "AAPL", {}),
    ("GP", "AAPL", {}),
    ("FA", "AAPL", {}),
    ("RV", "AAPL", {}),
    ("EE", "AAPL", {}),
    ("DVD", "AAPL", {}),
    ("CF", "AAPL", {}),
    ("RELS", "AAPL", {}),
    ("CH", "AAPL", {}),
    ("MGMT", "AAPL", {}),
    ("OWN", "AAPL", {}),
    ("SCHD", "AAPL", {}),
    ("BRC", "AAPL", {}),
    # Fixed Income
    ("YAS", "AAPL", {}),
    ("CRVD", "AAPL", {}),
    ("VCUB", "AAPL", {}),
    ("FWCM", "AAPL", {}),
    ("SRCH", None, {}),
    ("TRA", "AAPL", {}),
    ("MUNI", "AAPL", {}),
    ("ZV", "AAPL", {}),
    ("DUR", "AAPL", {}),
    ("CSHF", "AAPL", {}),
    ("ALLX", "AAPL", {}),
    ("RATD", "AAPL", {}),
    # Macro
    ("ECO", None, {}),
    ("ECST", None, {}),
    ("WIRP", None, {}),
    ("WEI", None, {}),
    ("WCRS", None, {}),
    ("GLCO", None, {}),
    ("GFUT", None, {}),
    ("YCRV", None, {}),
    ("BYFC", None, {}),
    ("BI", "AAPL", {}),
    ("WBDI", None, {}),
    ("IMFP", None, {}),
    # Portfolio & Risk
    ("PORT", None, {}),
    ("PRTU", None, {}),
    ("RISK", "AAPL", {}),
    ("STRS", "AAPL", {}),
    ("ALLC", None, {}),
    ("CORR", "AAPL", {}),
    ("BETA", "AAPL", {}),
    ("VAR", "AAPL", {}),
    ("PFCH", None, {}),
    ("ATTR", None, {}),
    # Options
    ("OMON", "AAPL", {}),
    ("OVME", "AAPL", {}),
    ("SKEW", "AAPL", {}),
    ("IRTS", None, {}),
    ("OPTA", "AAPL", {}),
    ("VOLCONE", "AAPL", {}),
    ("HVG", "AAPL", {}),
    ("IVG", "AAPL", {}),
    ("OSA", "AAPL", {}),
    ("PCRA", "AAPL", {}),
    # News
    ("N", "AAPL", {}),
    ("NI", "AAPL", {}),
    ("TNI", None, {}),
    ("BRIEF", "AAPL", {}),
    ("FIRST", None, {}),
    ("BN", "AAPL", {}),
    ("SRCH2", "AAPL", {}),
    ("RATD2", "AAPL", {}),
    # Market Data
    ("MKTX", "AAPL", {}),
    ("QMG", "AAPL", {}),
    ("MOV", None, {}),
    ("WMG", None, {}),
    ("GMM", None, {}),
    ("MOST", None, {}),
    ("BTMM", None, {}),
    ("FXFX", None, {}),
    ("COMP", "AAPL", {}),
    ("BSKT", "AAPL", {}),
    # Alternative Data
    ("SENT", "AAPL", {}),
    ("SURV", "AAPL", {}),
    ("SHRT", "AAPL", {}),
    ("INSDR", "AAPL", {}),
    ("ACTV", "AAPL", {}),
    ("FORM", "AAPL", {}),
    ("SCRS", None, {}),
    ("ALTS", "AAPL", {}),
    # Crypto
    ("COIN", "BTC", {}),
    ("DEFI", "AAVE", {}),
    ("XBTS", "BTC", {}),
    ("NFTS", None, {}),
    ("HASH", "BTC", {}),
    ("MVRV", "BTC", {}),
    ("DFLOW", "AAVE", {}),
    # Backtesting
    ("BT", "AAPL", {}),
    ("PROM", "AAPL", {}),
    ("OPTIM", None, {}),
    ("SRTS", None, {}),
    ("WFT", "AAPL", {}),
    # Charting
    ("G", "AAPL", {}),
    ("GPC", "AAPL", {}),
    ("COMP2", "AAPL", {}),
    ("TABT", "AAPL", {}),
    ("DRAW", "AAPL", {}),
]

real_count = 0
stub_count = 0
for code, ticker, params in test_cases:
    cmd = ParsedCommand(raw=f"{ticker} {code}", ticker=ticker, function_code=code, params=params)
    r = dispatcher.dispatch(cmd)
    is_stub = "module unavailable" in r.text or "_runtime_note" in r.data or "_api_note" in r.data
    if is_stub:
        stub_count += 1
        print(f"  [STUB] {code}: {r.text[:80]}")
    else:
        real_count += 1
        print(f"  [REAL] {code}: {r.text[:80]}")

print(f"\nSummary: {real_count} real, {stub_count} stub")
