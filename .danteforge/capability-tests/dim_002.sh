#!/bin/bash
# dim_002: Historical OHLCV daily — YFinanceAdapter + HistoricalOHLCVDeep
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# Test 1: YFinanceAdapter INTERVAL_MAP
from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter, INTERVAL_MAP
assert "1d" in INTERVAL_MAP, "Missing 1d"
assert "1wk" in INTERVAL_MAP, "Missing 1wk"
assert "1mo" in INTERVAL_MAP, "Missing 1mo"
print("[OK] INTERVAL_MAP has daily/weekly/monthly intervals")

# Test 2: YFinanceAdapter class attributes
assert YFinanceAdapter.name == "yfinance"
assert YFinanceAdapter.supports_crypto is True
assert YFinanceAdapter.rate_limit_per_min == 60
print("[OK] YFinanceAdapter class attributes correct")

# Test 3: date chunking static method works with pure computation
from datetime import datetime
chunks = YFinanceAdapter._date_chunks(
    datetime(2010, 1, 1), datetime(2024, 1, 1), chunk_years=2
)
assert len(chunks) >= 6, f"Expected 6+ chunks for 14 year range, got {len(chunks)}"
assert all(start < end for start, end in chunks), "All chunks must have start < end"
assert chunks[0][0] == datetime(2010, 1, 1), "First chunk must start at start"
print(f"[OK] _date_chunks produces {len(chunks)} chunks for 14-year range")

# Test 4: historical_ohlcv_deep — EXCHANGE_MAP
from sentinel.sds.adapters.historical_ohlcv_deep import EXCHANGE_MAP
assert "US" in EXCHANGE_MAP, "Missing US"
assert "GB" in EXCHANGE_MAP, "Missing GB"
assert "JP" in EXCHANGE_MAP, "Missing JP"
assert len(EXCHANGE_MAP) >= 20, f"Too few exchanges: {len(EXCHANGE_MAP)}"
assert EXCHANGE_MAP["GB"] == (".L", "GBP"), f"GB mapping wrong: {EXCHANGE_MAP['GB']}"
print(f"[OK] EXCHANGE_MAP has {len(EXCHANGE_MAP)} markets including US, GB, JP")

print("[PASS]")
PYEOF
