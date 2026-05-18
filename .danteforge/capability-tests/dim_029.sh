#!/bin/bash
# dim_029: CongressAdapter — STOCK Act congressional trading tracker
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from decimal import Decimal
from datetime import date

from sentinel.sds.adapters.congress_adapter import (
    HOUSE_URL,
    SENATE_URL,
    AMOUNT_MIDPOINTS,
    _parse_amount,
    _parse_date,
    _tx_code,
    CongressAdapter,
)

# --- constants ---
assert "house-stock-watcher-data" in HOUSE_URL, f"HOUSE_URL unexpected: {HOUSE_URL}"
assert "senate-stock-watcher-data" in SENATE_URL, f"SENATE_URL unexpected: {SENATE_URL}"
assert isinstance(AMOUNT_MIDPOINTS, dict)
assert len(AMOUNT_MIDPOINTS) >= 5
print("[OK] HOUSE_URL, SENATE_URL, AMOUNT_MIDPOINTS constants present")

# --- _parse_amount ---
lo, hi = _parse_amount("$1,001 - $15,000")
assert lo == Decimal("1001"), f"lo={lo}"
assert hi == Decimal("15000"), f"hi={hi}"
lo2, hi2 = _parse_amount("")
assert lo2 is None and hi2 is None
print("[OK] _parse_amount: range parse and empty-string return None")

# --- _parse_date ---
d = _parse_date("2024-03-15")
assert d == date(2024, 3, 15)
d2 = _parse_date("03/15/2024")
assert d2 == date(2024, 3, 15)
assert _parse_date("") is None
assert _parse_date(None) is None
print("[OK] _parse_date: ISO, US-format, empty, None")

# --- _tx_code ---
assert _tx_code("Purchase") == "P"
assert _tx_code("Sale (Full)") == "S"
assert _tx_code("Exchange") == "X"
assert _tx_code("Other") == "O"
print("[OK] _tx_code: purchase/sale/exchange/other")

# --- CongressAdapter class structure ---
from sentinel.sds.base_adapter import BaseAdapter
assert issubclass(CongressAdapter, BaseAdapter)
assert CongressAdapter.name == "congress"
assert hasattr(CongressAdapter, "fetch_house_trades")
assert hasattr(CongressAdapter, "fetch_ohlcv")
print("[OK] CongressAdapter inherits BaseAdapter, has expected attrs")

print("\n[PASS] dim_029: CongressAdapter -- all checks passed")
PYEOF
