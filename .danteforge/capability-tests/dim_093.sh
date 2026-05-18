#!/usr/bin/env bash
# dim_093: Excel/Sheets plugin — constants, formula parser, RTD server, named ranges, template
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math
import numpy as np
import pandas as pd

from sentinel.api.excel_sheets_plugin_v3 import (
    _clean_val,
    _df_to_sheet_values,
    _DEFAULT_UNIVERSE,
    _DEFAULT_WATCHLIST,
    _TREASURY_SERIES,
    _SENTINEL_BLUE_DARK,
    _SENTINEL_GREEN,
    _SENTINEL_RED,
    _RTD_INTERVAL_SECONDS,
    _CACHE_TTL_PRICE,
    _CACHE_TTL_FUNDAMENTAL,
    PortfolioHolding,
    YieldCurveResult,
    WorkbookCreateRequest,
    # New additions
    parse_sentinel_formula,
    RTDPollingServer,
    NamedRangeManager,
    create_valuation_workbook,
)

# ── Constants ────────────────────────────────────────────────────────────────
assert len(_DEFAULT_UNIVERSE) >= 10
assert "AAPL" in _DEFAULT_UNIVERSE and "MSFT" in _DEFAULT_UNIVERSE
print(f"[OK] _DEFAULT_UNIVERSE: {len(_DEFAULT_UNIVERSE)} tickers")

assert len(_DEFAULT_WATCHLIST) >= 3 and "AAPL" in _DEFAULT_WATCHLIST
print(f"[OK] _DEFAULT_WATCHLIST: {len(_DEFAULT_WATCHLIST)} tickers")

assert len(_TREASURY_SERIES) >= 8
assert _TREASURY_SERIES["10Y"] == "DGS10" and _TREASURY_SERIES["2Y"] == "DGS2"
print(f"[OK] _TREASURY_SERIES: {len(_TREASURY_SERIES)} tenors")

for name, color in [("_SENTINEL_BLUE_DARK", _SENTINEL_BLUE_DARK),
                     ("_SENTINEL_GREEN", _SENTINEL_GREEN),
                     ("_SENTINEL_RED", _SENTINEL_RED)]:
    assert len(color) == 6 and all(c in "0123456789ABCDEFabcdef" for c in color)
print(f"[OK] Color constants valid hex")

assert _RTD_INTERVAL_SECONDS > 0
assert _CACHE_TTL_FUNDAMENTAL > _CACHE_TTL_PRICE
print(f"[OK] Cache TTLs: price={_CACHE_TTL_PRICE}s fundamental={_CACHE_TTL_FUNDAMENTAL}s")

# ── _clean_val and _df_to_sheet_values ──────────────────────────────────────
assert _clean_val(None) == ""
assert _clean_val(float('nan')) == ""
assert _clean_val(42) == 42
assert isinstance(_clean_val({"a": 1}), str)
print("[OK] _clean_val: None->'' NaN->'' int->int dict->str")

df = pd.DataFrame({"ticker": ["AAPL","MSFT","GOOGL"], "price": [185.5, 420.25, 175.8], "volume": [55e6, 32e6, 18e6]})
sheet_vals = _df_to_sheet_values(df)
assert len(sheet_vals) == 4 and sheet_vals[0] == ["ticker","price","volume"]
print(f"[OK] _df_to_sheet_values: {len(sheet_vals)} rows (header+data)")

# ── Formula parser ───────────────────────────────────────────────────────────
# Valid SENTINEL formulas
result = parse_sentinel_formula('=SENTINEL.PRICE("AAPL")')
assert result is not None, "Should parse =SENTINEL.PRICE(\"AAPL\")"
assert result["ticker"] == "AAPL", f"ticker should be AAPL: {result['ticker']}"
assert result["function"] == "PRICE", f"function should be PRICE: {result['function']}"
assert result["arg2"] is None, "arg2 should be None for single-arg formula"
print(f"[OK] parse_sentinel_formula: PRICE(AAPL) -> ticker={result['ticker']} func={result['function']}")

result2 = parse_sentinel_formula('=SENTINEL.CHANGE_PCT("MSFT")')
assert result2 is not None
assert result2["ticker"] == "MSFT"
assert result2["function"] == "CHANGE_PCT"
print(f"[OK] parse_sentinel_formula: CHANGE_PCT(MSFT) -> ticker={result2['ticker']} func={result2['function']}")

result3 = parse_sentinel_formula('=SENTINEL.FINANCIALS("AAPL","eps")')
assert result3 is not None
assert result3["ticker"] == "AAPL"
assert result3["function"] == "FINANCIALS"
assert result3["arg2"] == "eps", f"arg2 should be 'eps': {result3['arg2']}"
print(f"[OK] parse_sentinel_formula: FINANCIALS(AAPL,eps) -> arg2={result3['arg2']}")

result4 = parse_sentinel_formula('=SENTINEL.VOLUME("TSLA")')
assert result4 is not None and result4["function"] == "VOLUME"
print(f"[OK] parse_sentinel_formula: VOLUME(TSLA) -> func={result4['function']}")

# Non-SENTINEL formulas return None
assert parse_sentinel_formula("=SUM(A1:A10)") is None
assert parse_sentinel_formula("=VLOOKUP(A1,B:C,2)") is None
assert parse_sentinel_formula("") is None
assert parse_sentinel_formula("not a formula") is None
print("[OK] parse_sentinel_formula: non-SENTINEL formulas return None")

# ── RTD Polling Server ───────────────────────────────────────────────────────
rtd = RTDPollingServer(interval_seconds=5)

# Subscribe 3 cells
key1 = rtd.subscribe("Sheet1", "B2", "AAPL", "price")
key2 = rtd.subscribe("Sheet1", "B3", "MSFT", "price")
key3 = rtd.subscribe("Sheet1", "B4", "TSLA", "change_pct")

assert key1 == "Sheet1!B2"
assert key2 == "Sheet1!B3"
assert key3 == "Sheet1!B4"
subs = rtd.list_subscriptions()
assert len(subs) == 3, f"Should have 3 subscriptions: {len(subs)}"
print(f"[OK] RTDPollingServer: {len(subs)} subscriptions registered")

# get_updates with mock values (pure computation — no API call)
mock_vals = {
    "Sheet1!B2": 185.50,
    "Sheet1!B3": 420.25,
    "Sheet1!B4": -1.23,
}
updates = rtd.get_updates(mock_values=mock_vals)
assert len(updates) == 3, f"Should return 3 updates: {len(updates)}"
assert updates["Sheet1!B2"]["value"] == 185.50
assert updates["Sheet1!B2"]["ticker"] == "AAPL"
assert updates["Sheet1!B3"]["value"] == 420.25
assert updates["Sheet1!B4"]["value"] == -1.23
assert updates["Sheet1!B4"]["field"] == "change_pct"
print(f"[OK] RTDPollingServer.get_updates: 3 cells with mock values verified")

# Unsubscribe one cell
removed = rtd.unsubscribe("Sheet1", "B4")
assert removed is True
assert len(rtd.list_subscriptions()) == 2
print(f"[OK] RTDPollingServer.unsubscribe: 2 subscriptions remain")

# ── Named Range Manager ──────────────────────────────────────────────────────
nr = NamedRangeManager()
# Built-in ranges should exist
all_ranges = nr.all_ranges()
assert "SENTINEL_UNIVERSE" in all_ranges, "SENTINEL_UNIVERSE should be built-in"
assert "SENTINEL_WATCHLIST" in all_ranges, "SENTINEL_WATCHLIST should be built-in"
assert "$E$" in all_ranges["SENTINEL_UNIVERSE"], "SENTINEL_UNIVERSE should reference col E"
print(f"[OK] NamedRangeManager built-ins: {list(all_ranges.keys())}")

# Define custom named ranges
nr.define("MY_PORTFOLIO", "Portfolio!$A$2:$A$10")
nr.define("MACRO_RANGE",  "Macro!$A$2:$E$12")
assert nr.get("MY_PORTFOLIO") == "Portfolio!$A$2:$A$10"
assert nr.get("my_portfolio") == "Portfolio!$A$2:$A$10"  # case-insensitive
assert nr.get("MACRO_RANGE")  == "Macro!$A$2:$E$12"
print(f"[OK] NamedRangeManager.define/get: custom ranges work, case-insensitive")

# Remove a range
removed_nr = nr.remove("MY_PORTFOLIO")
assert removed_nr is True
assert nr.get("MY_PORTFOLIO") is None
print(f"[OK] NamedRangeManager.remove: range deleted")

# to_dict includes all remaining ranges
d = nr.to_dict()
assert "SENTINEL_UNIVERSE" in d
assert "MACRO_RANGE" in d
assert "MY_PORTFOLIO" not in d
print(f"[OK] NamedRangeManager.to_dict: {len(d)} ranges returned")

# ── Valuation Workbook Template ──────────────────────────────────────────────
# Use a mock fetcher so no network calls are made
class MockFetcher:
    def get_financials(self, ticker, metric=None, period="ttm"):
        data = {"revenue_ttm": 400_000_000_000, "ebitda": 120_000_000_000,
                "fcf": 100_000_000_000, "market_cap": 2_800_000_000_000,
                "beta": 1.2, "pe_ratio": 29.5, "eps": 6.43}
        if metric:
            return {metric: data.get(metric)}
        return data
    def get_quote(self, ticker):
        return {"price": 185.50, "change_pct": 0.42, "market_cap": 2_800_000_000_000}

wb = create_valuation_workbook("AAPL", fetcher=MockFetcher())
assert wb["ticker"] == "AAPL", f"ticker should be AAPL: {wb['ticker']}"
assert "sheets" in wb, "workbook should have 'sheets' key"
assert "DCF" in wb["sheets"], "workbook should have DCF sheet"
assert "Assumptions" in wb["sheets"], "workbook should have Assumptions sheet"
assert "Comps" in wb["sheets"], "workbook should have Comps sheet"
assert "Summary" in wb["sheets"], "workbook should have Summary sheet"
print(f"[OK] create_valuation_workbook: 4 sheets returned (DCF, Assumptions, Comps, Summary)")

dcf = wb["sheets"]["DCF"]
assert dcf["ticker"] == "AAPL"
assert "assumptions" in dcf
assert len(dcf["projections"]) == 5, f"Should have 5 projection years: {len(dcf['projections'])}"
assert dcf["enterprise_value_m"] > 0, "Enterprise value should be positive"
# Assumptions sanity
assert 0.04 < dcf["assumptions"]["wacc"] < 0.20, f"WACC should be 4-20%: {dcf['assumptions']['wacc']}"
assert dcf["assumptions"]["terminal_growth"] > 0
print(f"[OK] DCF sheet: 5 projections, EV={dcf['enterprise_value_m']:,.0f}M, WACC={dcf['assumptions']['wacc']:.2%}")

assumptions = wb["sheets"]["Assumptions"]
param_names = [r["parameter"] for r in assumptions["rows"]]
assert "WACC" in param_names, "Assumptions should include WACC"
assert "Beta" in param_names, "Assumptions should include Beta"
print(f"[OK] Assumptions sheet: {len(assumptions['rows'])} parameters including WACC and Beta")

summary = wb["sheets"]["Summary"]
assert summary["ticker"] == "AAPL"
assert summary["current_price"] == 185.50
print(f"[OK] Summary sheet: ticker={summary['ticker']} price={summary['current_price']}")

# ── Pydantic models ──────────────────────────────────────────────────────────
holding = PortfolioHolding(ticker="NVDA", shares=100, cost_basis=450.0)
assert holding.ticker == "NVDA" and holding.shares == 100
print(f"[OK] PortfolioHolding: ticker={holding.ticker} shares={holding.shares}")

yc = YieldCurveResult(curve={"1M":5.25,"3M":5.30,"2Y":4.80,"10Y":4.50,"30Y":4.65},
                      spread_2s10s=-0.30, inverted=True, as_of="2024-01-15")
assert yc.inverted and yc.spread_2s10s < 0
print(f"[OK] YieldCurveResult: inverted={yc.inverted} spread={yc.spread_2s10s}")

req = WorkbookCreateRequest()
assert len(req.tickers) >= 1 and req.include_charts is True
print(f"[OK] WorkbookCreateRequest defaults OK")

print("\n[PASS] dim_093: Excel/Sheets plugin")
PYEOF
