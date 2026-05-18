#!/bin/bash
# dim_003: Historical OHLCV intraday — IntradayDataManager, VolumeProfileAnalyzer
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

# Test 1: DEPTH_BY_SOURCE metadata
from sentinel.sds.adapters.intraday_deep import DEPTH_BY_SOURCE
assert "alpaca_iex" in DEPTH_BY_SOURCE, "Missing alpaca_iex source"
assert "yfinance" in DEPTH_BY_SOURCE, "Missing yfinance source"
assert "polygon_free" in DEPTH_BY_SOURCE, "Missing polygon_free source"
assert DEPTH_BY_SOURCE["yfinance"]["max_history_days"] == 7, "yfinance should be 7 day limit"
print(f"[OK] DEPTH_BY_SOURCE has {len(DEPTH_BY_SOURCE)} sources with correct metadata")

# Test 2: AlpacaIntradayAdapter class exists with ALPACA_TF_MAP
from sentinel.sds.adapters.intraday_deep import AlpacaIntradayAdapter
assert "1min" in AlpacaIntradayAdapter.ALPACA_TF_MAP
assert "1h" in AlpacaIntradayAdapter.ALPACA_TF_MAP
print("[OK] AlpacaIntradayAdapter.ALPACA_TF_MAP present")

# Test 3: VolumeProfileAnalyzer — VWAP computation (pure computation)
from sentinel.sds.adapters.intraday_deep import VolumeProfileAnalyzer
analyzer = VolumeProfileAnalyzer()

# Build a simple 5-bar intraday DataFrame
bars = pd.DataFrame({
    "time": pd.date_range("2024-01-02 09:30", periods=5, freq="1min", tz="UTC"),
    "open":  [100.0, 101.0, 102.0, 101.5, 102.5],
    "high":  [101.5, 102.5, 103.0, 102.0, 103.5],
    "low":   [99.5,  100.5, 101.0, 101.0, 102.0],
    "close": [101.0, 102.0, 101.5, 101.8, 103.0],
    "volume":[1000,  1500,  1200,  800,   1100],
})
vwap = analyzer.compute_vwap(bars)
assert len(vwap) == 5, f"Expected 5 VWAP values, got {len(vwap)}"
assert all(vwap > 0), "All VWAP values must be positive"
# VWAP should be within the bar price range
assert vwap.min() > 99.0 and vwap.max() < 104.0, f"VWAP out of range: {vwap.values}"
print(f"[OK] VWAP computed correctly, range [{vwap.min():.2f}, {vwap.max():.2f}]")

# Test 4: Volume profile computation
profile = analyzer.compute_volume_profile(bars, bins=10)
assert not profile.empty, "Volume profile should not be empty"
assert "volume" in profile.columns, "profile needs volume column"
assert "poc" in profile.columns, "profile needs poc column"
assert profile["poc"].sum() == 1, "Exactly one POC (Point of Control)"
print(f"[OK] Volume profile computed with {len(profile)} bins, POC identified")

print("[PASS]")
PYEOF
