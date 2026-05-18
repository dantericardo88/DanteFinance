#!/usr/bin/env bash
# dim_093: Excel/Sheets plugin — pure data formatting logic and constants
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
)

# Test constants
assert len(_DEFAULT_UNIVERSE) >= 10, f"Universe should have >= 10 tickers: {len(_DEFAULT_UNIVERSE)}"
assert "AAPL" in _DEFAULT_UNIVERSE, "AAPL should be in universe"
assert "MSFT" in _DEFAULT_UNIVERSE, "MSFT should be in universe"
print(f"[OK] _DEFAULT_UNIVERSE: {len(_DEFAULT_UNIVERSE)} tickers")

assert len(_DEFAULT_WATCHLIST) >= 3, f"Watchlist should have >= 3 tickers: {len(_DEFAULT_WATCHLIST)}"
assert "AAPL" in _DEFAULT_WATCHLIST, "AAPL should be in watchlist"
print(f"[OK] _DEFAULT_WATCHLIST: {len(_DEFAULT_WATCHLIST)} tickers")

assert len(_TREASURY_SERIES) >= 8, f"Expected >= 8 treasury series: {len(_TREASURY_SERIES)}"
assert "10Y" in _TREASURY_SERIES, "'10Y' should be in treasury series"
assert "2Y" in _TREASURY_SERIES, "'2Y' should be in treasury series"
assert _TREASURY_SERIES["10Y"] == "DGS10", f"10Y should map to DGS10: {_TREASURY_SERIES['10Y']}"
assert _TREASURY_SERIES["2Y"] == "DGS2", f"2Y should map to DGS2: {_TREASURY_SERIES['2Y']}"
print(f"[OK] _TREASURY_SERIES: {len(_TREASURY_SERIES)} tenors")

# Color constants should be valid hex
for name, color in [
    ("_SENTINEL_BLUE_DARK", _SENTINEL_BLUE_DARK),
    ("_SENTINEL_GREEN", _SENTINEL_GREEN),
    ("_SENTINEL_RED", _SENTINEL_RED),
]:
    assert len(color) == 6, f"{name} should be 6-char hex: '{color}'"
    assert all(c in "0123456789ABCDEFabcdef" for c in color), \
        f"{name} should be hex: '{color}'"
print(f"[OK] Color constants: BLUE={_SENTINEL_BLUE_DARK} GREEN={_SENTINEL_GREEN} RED={_SENTINEL_RED}")

# Timing constants
assert _RTD_INTERVAL_SECONDS > 0, "RTD interval must be positive"
assert _CACHE_TTL_PRICE > 0, "Price TTL must be positive"
assert _CACHE_TTL_FUNDAMENTAL > _CACHE_TTL_PRICE, \
    "Fundamental cache TTL should be longer than price TTL"
print(f"[OK] Cache TTLs: price={_CACHE_TTL_PRICE}s fundamental={_CACHE_TTL_FUNDAMENTAL}s")

# Test _clean_val
assert _clean_val(None) == "", f"None should clean to empty string: {_clean_val(None)}"
assert _clean_val(float('nan')) == "", f"NaN should clean to empty string: {_clean_val(float('nan'))}"
assert _clean_val(42) == 42, f"Int should pass through: {_clean_val(42)}"
assert _clean_val(3.14) == 3.14, f"Float should pass through: {_clean_val(3.14)}"
assert _clean_val("AAPL") == "AAPL", f"String should pass through: {_clean_val('AAPL')}"
d_val = _clean_val({"a": 1})
assert isinstance(d_val, str), f"Dict should become string: {d_val}"
l_val = _clean_val([1, 2, 3])
assert isinstance(l_val, str), f"List should become string: {l_val}"
print("[OK] _clean_val: None->'' NaN->'' int->int dict->str list->str")

# Test NaN detection (val != val is the idiom for NaN)
nan_val = float('nan')
assert nan_val != nan_val, "NaN != NaN should be True (IEEE 754)"
assert _clean_val(nan_val) == "", "NaN should be cleaned to ''"
assert _clean_val(float('inf')) == float('inf'), "Infinity is not NaN, should pass through"
print(f"[OK] _clean_val NaN edge cases correct")

# Test _df_to_sheet_values
df = pd.DataFrame({
    "ticker": ["AAPL", "MSFT", "GOOGL"],
    "price":  [185.50, 420.25, 175.80],
    "volume": [55000000, 32000000, 18000000],
})
sheet_vals = _df_to_sheet_values(df)
assert isinstance(sheet_vals, list), "Should return list"
assert len(sheet_vals) == 4, f"Should have 4 rows (1 header + 3 data): {len(sheet_vals)}"
assert sheet_vals[0] == ["ticker", "price", "volume"], \
    f"First row should be headers: {sheet_vals[0]}"
assert sheet_vals[1][0] == "AAPL", f"First data row ticker: {sheet_vals[1]}"
print(f"[OK] _df_to_sheet_values: {len(sheet_vals)} rows (header+data)")

# Test _df_to_sheet_values handles NaN values
df_with_nan = pd.DataFrame({
    "ticker": ["AAPL", "MSFT"],
    "pe_ratio": [28.5, float('nan')],
})
sheet_vals2 = _df_to_sheet_values(df_with_nan)
# NaN values should be converted to ""
assert sheet_vals2[2][1] == "", f"NaN should become '' in sheet values: {sheet_vals2[2]}"
print(f"[OK] _df_to_sheet_values NaN handling: NaN->''")

# Test _df_to_sheet_values with None values
df_with_none = pd.DataFrame({
    "ticker": ["AAPL"],
    "value":  [None],
})
sheet_vals3 = _df_to_sheet_values(df_with_none)
assert sheet_vals3[1][1] == "", f"None should become '' in sheet values: {sheet_vals3[1]}"
print(f"[OK] _df_to_sheet_values None handling: None->''")

# Test PortfolioHolding Pydantic model
holding = PortfolioHolding(
    ticker="NVDA",
    shares=100,
    cost_basis=450.0,
)
assert holding.ticker == "NVDA"
assert holding.shares == 100
assert holding.cost_basis == 450.0
print(f"[OK] PortfolioHolding: ticker={holding.ticker} shares={holding.shares}")

# Test YieldCurveResult Pydantic model
yc = YieldCurveResult(
    curve={"1M": 5.25, "3M": 5.30, "2Y": 4.80, "10Y": 4.50, "30Y": 4.65},
    spread_2s10s=-0.30,
    inverted=True,
    as_of="2024-01-15",
)
assert yc.inverted is True, "Yield curve should be inverted"
assert yc.spread_2s10s < 0, "2s10s spread should be negative (inverted)"
assert "10Y" in yc.curve, "Yield curve should have 10Y tenor"
assert len(yc.curve) >= 4, f"Yield curve should have >= 4 tenors: {len(yc.curve)}"
print(f"[OK] YieldCurveResult: inverted={yc.inverted} spread_2s10s={yc.spread_2s10s}")

# Test WorkbookCreateRequest defaults
req = WorkbookCreateRequest()
assert len(req.tickers) >= 1, "Default tickers should not be empty"
assert req.include_charts is True, "Default should include charts"
print(f"[OK] WorkbookCreateRequest defaults: tickers={req.tickers} include_charts={req.include_charts}")

# Test WorkbookCreateRequest with custom values
req2 = WorkbookCreateRequest(
    tickers=["AAPL", "MSFT", "GOOGL"],
    include_charts=False,
    rtd_enabled=True,
)
assert req2.tickers == ["AAPL", "MSFT", "GOOGL"]
assert req2.include_charts is False
assert req2.rtd_enabled is True
print(f"[OK] WorkbookCreateRequest custom: tickers={req2.tickers}")

print("\n[PASS] dim_093: Excel/Sheets plugin")
PYEOF
