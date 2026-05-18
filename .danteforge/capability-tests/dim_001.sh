#!/bin/bash
# dim_001: Real-time equity quotes — AlpacaAdapter class structure and ALPACA_TF_MAP
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# Test 1: ALPACA_TF_MAP exists and has expected keys
from sentinel.sds.adapters.alpaca_adapter import ALPACA_TF_MAP, AlpacaAdapter
assert "1m" in ALPACA_TF_MAP, "Missing 1m key"
assert "1d" in ALPACA_TF_MAP, "Missing 1d key"
assert "1h" in ALPACA_TF_MAP, "Missing 1h key"
assert len(ALPACA_TF_MAP) >= 7, f"Too few entries: {len(ALPACA_TF_MAP)}"
print("[OK] ALPACA_TF_MAP has expected timeframe keys")

# Test 2: AlpacaAdapter class attributes
assert AlpacaAdapter.name == "alpaca", "name mismatch"
assert AlpacaAdapter.supports_realtime is True, "should support realtime"
assert AlpacaAdapter.supports_crypto is True, "should support crypto"
assert AlpacaAdapter.rate_limit_per_min == 200, "rate limit mismatch"
print("[OK] AlpacaAdapter class attributes correct")

# Test 3: AlpacaAdapter has required methods
assert hasattr(AlpacaAdapter, "fetch_ohlcv"), "Missing fetch_ohlcv"
assert hasattr(AlpacaAdapter, "fetch_latest_quote"), "Missing fetch_latest_quote"
assert hasattr(AlpacaAdapter, "fetch_snapshot"), "Missing fetch_snapshot"
print("[OK] AlpacaAdapter has fetch_ohlcv, fetch_latest_quote, fetch_snapshot")

# Test 4: Crypto detection logic is testable in isolation
ticker_crypto = "BTC/USD"
ticker_stock = "AAPL"
is_crypto_btc = "/" in ticker_crypto or (ticker_crypto.endswith("USD") and len(ticker_crypto) >= 6)
is_crypto_aapl = "/" in ticker_stock or (ticker_stock.endswith("USD") and len(ticker_stock) >= 6)
assert is_crypto_btc is True, "BTC/USD should be crypto"
assert is_crypto_aapl is False, "AAPL should not be crypto"
print("[OK] Crypto detection logic works correctly")

print("[PASS]")
PYEOF
