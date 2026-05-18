#!/bin/bash
# dim_005: Futures term structure — FUTURES_UNIVERSE, roll yield, HV computation
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

# Test 1: FUTURES_UNIVERSE has expected contracts
from sentinel.sds.adapters.futures_v3 import FUTURES_UNIVERSE, _UNIVERSE_MAP, FuturesContract
assert len(FUTURES_UNIVERSE) >= 40, f"Expected 40+ contracts, got {len(FUTURES_UNIVERSE)}"
symbols = [c.symbol for c in FUTURES_UNIVERSE]
assert "ES" in symbols, "Missing ES (E-mini S&P)"
assert "CL" in symbols, "Missing CL (WTI Crude)"
assert "GC" in symbols, "Missing GC (Gold)"
assert "NG" in symbols, "Missing NG (Natural Gas)"
print(f"[OK] FUTURES_UNIVERSE has {len(FUTURES_UNIVERSE)} contracts incl. ES, CL, GC, NG")

# Test 2: _UNIVERSE_MAP maps symbols correctly
es = _UNIVERSE_MAP["ES"]
assert es.exchange == "CME"
assert es.asset_class == "equity_index"
assert es.tick_size == 0.25
assert es.contract_size == 50.0
print(f"[OK] ES contract: exchange={es.exchange}, tick={es.tick_size}, size={es.contract_size}")

# Test 3: _annualise_roll_yield pure computation
from sentinel.sds.adapters.futures_v3 import _annualise_roll_yield
# 1% roll over 30 days → annualised ~12.2%
ry = _annualise_roll_yield(near=100.0, far=99.0, days_between=30)
expected = (100.0 - 99.0) / 100.0 * (365.0 / 30)
assert abs(ry - expected) < 1e-10, f"Roll yield mismatch: {ry} vs {expected}"
assert ry > 0, "Contango negative roll should be 0 in this form; near > far = positive"
# zero near price returns 0
assert _annualise_roll_yield(0.0, 99.0, 30) == 0.0, "near=0 should return 0"
print(f"[OK] _annualise_roll_yield: 1% over 30d = {ry:.4f} ({ry*100:.1f}% ann.)")

# Test 4: _hv (historical volatility) pure computation
from sentinel.sds.adapters.futures_v3 import _hv
prices = pd.Series([100, 101, 99, 102, 100, 103, 101, 104, 102, 105,
                    103, 106, 104, 107, 105, 108, 106, 109, 107, 110,
                    108, 111, 109], dtype=float)
hv = _hv(prices, window=21)
hv_clean = hv.dropna()
assert len(hv_clean) > 0, "HV series should have non-NaN values"
assert all(v > 0 for v in hv_clean), "All HV values should be positive"
print(f"[OK] _hv computed: latest HV = {hv_clean.iloc[-1]:.4f} (annualised)")

# Test 5: FuturesContract dataclass is usable directly
custom = FuturesContract(
    symbol="TEST", yf_symbol="TEST=F", name="Test Futures",
    exchange="CME", asset_class="equity_index",
    tick_size=0.5, contract_size=100.0
)
assert custom.symbol == "TEST"
assert custom.is_continuous is True  # default
print("[OK] FuturesContract dataclass instantiates correctly with defaults")

print("[PASS]")
PYEOF
