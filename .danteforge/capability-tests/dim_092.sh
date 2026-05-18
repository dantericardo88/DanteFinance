#!/usr/bin/env bash
# dim_092: TradingView UDF — config constants, symbol resolver, cache logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import time

from sentinel.api.tradingview_enhanced import (
    UDF_CONFIG,
    SymbolResolver,
    _bar_cache_key,
    _bar_cache_get,
    _bar_cache_set,
    _BAR_CACHE,
    _BAR_CACHE_TTL,
)

# Test UDF_CONFIG structure
assert "supported_resolutions" in UDF_CONFIG, "UDF_CONFIG should have supported_resolutions"
assert "supports_marks" in UDF_CONFIG, "UDF_CONFIG should have supports_marks"
assert "supports_search" in UDF_CONFIG, "UDF_CONFIG should have supports_search"
assert "exchanges" in UDF_CONFIG, "UDF_CONFIG should have exchanges"
assert "symbols_types" in UDF_CONFIG, "UDF_CONFIG should have symbols_types"
print(f"[OK] UDF_CONFIG keys: {list(UDF_CONFIG.keys())}")

# Check supported resolutions include standard intervals
resolutions = UDF_CONFIG["supported_resolutions"]
for r in ["1", "5", "15", "60", "D", "W", "M"]:
    assert r in resolutions, f"Missing resolution '{r}': {resolutions}"
print(f"[OK] UDF_CONFIG supported_resolutions: {resolutions}")

# Check exchanges
exchanges = UDF_CONFIG["exchanges"]
exchange_values = [e["value"] for e in exchanges]
assert "NASDAQ" in exchange_values, "Should include NASDAQ exchange"
assert "NYSE" in exchange_values, "Should include NYSE exchange"
assert "CRYPTO" in exchange_values, "Should include CRYPTO exchange"
print(f"[OK] UDF_CONFIG exchanges: {exchange_values}")

# Check symbol types
sym_types = [s["value"] for s in UDF_CONFIG["symbols_types"]]
assert "stock" in sym_types, "Should include stock type"
assert "crypto" in sym_types, "Should include crypto type"
assert "etf" in sym_types, "Should include ETF type"
print(f"[OK] UDF_CONFIG symbols_types: {sym_types}")

# Check TTLs
assert _BAR_CACHE_TTL["intraday"] > 0, "Intraday TTL must be positive"
assert _BAR_CACHE_TTL["daily"] > 0, "Daily TTL must be positive"
assert _BAR_CACHE_TTL["daily"] > _BAR_CACHE_TTL["intraday"], \
    "Daily TTL should be larger than intraday TTL"
print(f"[OK] _BAR_CACHE_TTL: intraday={_BAR_CACHE_TTL['intraday']}s daily={_BAR_CACHE_TTL['daily']}s")

# Test SymbolResolver.resolve() — pure string parsing, no network
resolver = SymbolResolver()

# Plain ticker
r1 = resolver.resolve("AAPL")
assert r1["ticker"] == "AAPL", f"Plain ticker: {r1}"
assert r1["exchange"] == "", f"No exchange for plain ticker: {r1}"
print(f"[OK] resolve('AAPL'): {r1}")

# SENTINEL: prefix stripped
r2 = resolver.resolve("SENTINEL:MSFT")
assert r2["ticker"] == "MSFT", f"SENTINEL: prefix stripped: {r2}"
assert r2["exchange"] == "", f"Exchange empty after SENTINEL: strip: {r2}"
print(f"[OK] resolve('SENTINEL:MSFT'): {r2}")

# TICKER:EXCHANGE format
r3 = resolver.resolve("AAPL:NASDAQ")
assert r3["ticker"] == "AAPL", f"Ticker from TICKER:EXCHANGE: {r3}"
assert r3["exchange"] == "NASDAQ", f"Exchange from TICKER:EXCHANGE: {r3}"
print(f"[OK] resolve('AAPL:NASDAQ'): {r3}")

# Crypto detection by suffix
r4 = resolver.resolve("BTC-USD")
assert r4["ticker"] == "BTC-USD", f"Crypto ticker: {r4}"
assert r4["exchange"] == "CRYPTO", f"Crypto exchange: {r4}"
print(f"[OK] resolve('BTC-USD'): {r4}")

r5 = resolver.resolve("ETH-USDT")
assert r5["exchange"] == "CRYPTO", f"USDT suffix → CRYPTO: {r5}"
print(f"[OK] resolve('ETH-USDT'): exchange={r5['exchange']}")

# Case normalization (lowercase input)
r6 = resolver.resolve("aapl")
assert r6["ticker"] == "AAPL", f"Should uppercase ticker: {r6}"
print(f"[OK] resolve('aapl'): uppercased to {r6['ticker']}")

# Test TIMEZONE_MAP coverage
tz_map = resolver.TIMEZONE_MAP
assert "NYSE" in tz_map, "Should have NYSE timezone"
assert "NASDAQ" in tz_map, "Should have NASDAQ timezone"
assert "CCC" in tz_map, "Should have CCC (crypto) timezone"
assert tz_map["CCC"] == "UTC", f"Crypto timezone should be UTC: {tz_map['CCC']}"
assert tz_map["NYSE"] == "America/New_York", f"NYSE should be America/New_York: {tz_map['NYSE']}"
print(f"[OK] TIMEZONE_MAP: {len(tz_map)} entries, CCC={tz_map['CCC']} NYSE={tz_map['NYSE']}")

# Test SESSION_MAP
session_map = resolver.SESSION_MAP
assert "America/New_York" in session_map
assert session_map["America/New_York"] == "0930-1600", \
    f"NYSE session: {session_map['America/New_York']}"
assert session_map["UTC"] == "24x7", f"Crypto session: {session_map['UTC']}"
print(f"[OK] SESSION_MAP: NY='{session_map['America/New_York']}' UTC='{session_map['UTC']}'")

# Test PRICESCALE_MAP
scale_map = resolver.PRICESCALE_MAP
assert scale_map["USD"] == 100, "USD pricescale should be 100 (2 decimal places)"
assert scale_map["JPY"] == 1, "JPY pricescale should be 1 (no decimal places)"
assert scale_map["BTC"] > 1000, f"BTC pricescale should be large: {scale_map['BTC']}"
print(f"[OK] PRICESCALE_MAP: USD={scale_map['USD']} JPY={scale_map['JPY']} BTC={scale_map['BTC']}")

# Test bar cache: _bar_cache_key
key = _bar_cache_key("AAPL", "D", 1700000000, 1700100000)
assert isinstance(key, str), "Cache key should be a string"
assert "AAPL" in key, "Cache key should contain ticker"
assert "D" in key, "Cache key should contain resolution"
print(f"[OK] _bar_cache_key: '{key}'")

# Test _bar_cache_set and _bar_cache_get
test_data = {"t": [100, 200, 300], "c": [150.0, 152.0, 148.0], "s": "ok"}
_bar_cache_set(key, test_data)
retrieved = _bar_cache_get(key, is_intraday=False)
assert retrieved is not None, "Should retrieve data right after setting"
assert retrieved["s"] == "ok", f"Data mismatch: {retrieved}"
print(f"[OK] _bar_cache_set/_bar_cache_get: round-trip works")

# Daily TTL is 3600s — verify that same key is still valid immediately after set
retrieved2 = _bar_cache_get(key, is_intraday=False)
assert retrieved2 is not None, "Cache entry should still be valid"
print(f"[OK] Cache entry valid immediately after set (TTL not expired)")

# Test that a different key returns None
absent_key = _bar_cache_key("MSFT", "1", 1700000000, 1700100000)
assert _bar_cache_get(absent_key, is_intraday=True) is None, \
    "Non-existent cache key should return None"
print(f"[OK] Non-existent cache key returns None")

print("\n[PASS] dim_092: TradingView UDF")
PYEOF
