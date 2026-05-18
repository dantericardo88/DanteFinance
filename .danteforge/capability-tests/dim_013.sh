#!/bin/bash
# dim_013: fundamental_data_layer_v3 — EdgarCikResolver, XbrlFactsFetcher, FundamentalDataLayerV3
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.fundamental_data_layer_v3 import (
    EdgarCikResolver,
    XbrlFactsFetcher,
    EdgarSubmissionMonitor,
    MetricCalculator,
    UniverseManager,
    FundamentalDataLayerV3,
    XBRL_MAP,
)

# Test 1: Key classes exist and are importable
assert EdgarCikResolver is not None
assert XbrlFactsFetcher is not None
assert MetricCalculator is not None
assert UniverseManager is not None
assert FundamentalDataLayerV3 is not None
print("[OK] All key classes importable")

# Test 2: XBRL_MAP contains income statement metrics
assert "revenue" in XBRL_MAP, "revenue must be in XBRL_MAP"
assert "gross_profit" in XBRL_MAP
assert "operating_income" in XBRL_MAP
assert "net_income" in XBRL_MAP
assert "eps_diluted" in XBRL_MAP
print(f"[OK] XBRL_MAP has {len(XBRL_MAP)} entries with income statement metrics")

# Test 3: XBRL_MAP entries have correct tuple structure (taxonomy, concept, unit)
rev_entry = XBRL_MAP["revenue"]
assert len(rev_entry) == 3, f"Expected 3-tuple, got {len(rev_entry)}"
taxonomy, concept, unit = rev_entry
assert taxonomy == "us-gaap"
assert concept == "Revenues"
assert unit == "USD"
print(f"[OK] revenue XBRL_MAP entry: ({taxonomy}, {concept}, {unit})")

# Test 4: Balance sheet metrics in XBRL_MAP
assert "total_assets" in XBRL_MAP
assert "total_equity" in XBRL_MAP
assert "cash" in XBRL_MAP
assert "long_term_debt" in XBRL_MAP
print("[OK] Balance sheet metrics in XBRL_MAP")

# Test 5: Cash flow metrics in XBRL_MAP
assert "capex" in XBRL_MAP
print("[OK] Cash flow metrics in XBRL_MAP")

# Test 6: EdgarCikResolver has resolve method
resolver = EdgarCikResolver.__new__(EdgarCikResolver)
assert hasattr(resolver, 'resolve') or hasattr(EdgarCikResolver, 'resolve')
print("[OK] EdgarCikResolver has resolve method")

# Test 7: FundamentalDataLayerV3 has load and get_metric methods
assert hasattr(FundamentalDataLayerV3, 'load')
assert hasattr(FundamentalDataLayerV3, 'get_metric')
assert hasattr(FundamentalDataLayerV3, 'batch_metrics')
assert hasattr(FundamentalDataLayerV3, 'refresh')
print("[OK] FundamentalDataLayerV3 has load, get_metric, batch_metrics, refresh methods")

print("\n[PASS] dim_013: fundamental_data_layer_v3 -- all checks passed")
PYEOF
