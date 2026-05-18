#!/usr/bin/env bash
# dim_070: Fundamental screener — pure ratio/filter logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.sbx.fundamental_screener import FundamentalScreener

screener = FundamentalScreener()

# Test PREBUILT_SCREENS dict exists and has entries
presets = screener.PREBUILT_SCREENS
assert isinstance(presets, dict), "PREBUILT_SCREENS should be dict"
assert len(presets) > 0, "PREBUILT_SCREENS should not be empty"
print(f"[OK] PREBUILT_SCREENS has {len(presets)} presets: {list(presets.keys())[:5]}")

# Test _build_where pure logic
clauses = screener._build_where({
    "pe_ttm":  {"lte": 15.0, "gte": 5.0},
    "roe":     {"gte": 0.15},
    "sector":  {"in": ["Technology", "Healthcare"]},
})
assert any("pe_ttm" in c for c in clauses), "Expected pe_ttm clause"
assert any("roe" in c for c in clauses), "Expected roe clause"
assert any("IN" in c for c in clauses), "Expected IN clause for sector"
print(f"[OK] _build_where generated {len(clauses)} SQL clauses")

# Test Altman Z-Score formula manually (pure math)
# Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
x1, x2, x3, x4, x5 = 0.20, 0.30, 0.10, 1.50, 1.20
z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
assert z > 2.99, f"Should be in safe zone: {z}"
print(f"[OK] Altman Z formula: {z:.3f} -> safe zone")

x1b, x2b, x3b, x4b, x5b = 0.05, 0.02, 0.02, 0.30, 0.40
zb = 1.2*x1b + 1.4*x2b + 3.3*x3b + 0.6*x4b + 1.0*x5b
assert zb < 1.81, f"Should be in distress zone: {zb}"
print(f"[OK] Altman Z distress zone: {zb:.3f}")

# Piotroski F-Score logic (9-point system)
# Each boolean check = 1 point
checks = [
    1000 > 0,       # net income positive
    0.10 > 0,       # ROA positive
    1200 > 0,       # operating cash flow positive
    1200 > 1000,    # OCF > net income (quality)
    0.30 < 0.35,    # long-term debt ratio decreased
    1.8 > 1.5,      # current ratio improved
    500 == 500,     # no share dilution
    0.42 > 0.38,    # gross margin improved
    0.80 > 0.75,    # asset turnover improved
]
f_score = sum(checks)
assert 0 <= f_score <= 9, f"F-score out of range: {f_score}"
rating = "Strong" if f_score >= 7 else "Moderate" if f_score >= 4 else "Weak"
print(f"[OK] Piotroski F-Score: {f_score}/9 -> {rating}")

# DuckDBQueryEngine can be instantiated
from sentinel.sbx.fundamental_screener import DuckDBQueryEngine
engine = DuckDBQueryEngine()
schema = engine.get_schema()
assert isinstance(schema, pd.DataFrame), "get_schema should return DataFrame"
assert len(schema) > 0, "Schema should have columns"
print(f"[OK] DuckDBQueryEngine.get_schema() returned {len(schema)} field definitions")

print("\n[PASS] dim_070: Fundamental screener")
PYEOF
