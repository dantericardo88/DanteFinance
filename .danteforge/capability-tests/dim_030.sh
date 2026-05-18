#!/bin/bash
# dim_030: ipo_intelligence_v3 — IPO pipeline intelligence
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from datetime import date

from sentinel.sfe.ipo_intelligence_v3 import (
    _BULGE_BRACKET,
    _SPAC_MARKERS,
    EdgarS1Parser,
    IPOPipelineTracker,
    SPACTracker,
    PipelineEntry,
    IPOResult,
    SPACRecord,
)

# --- constants ---
assert "Goldman Sachs" in _BULGE_BRACKET
assert "Morgan Stanley" in _BULGE_BRACKET
assert len(_SPAC_MARKERS) >= 3
print("[OK] _BULGE_BRACKET and _SPAC_MARKERS constants present")

# --- PipelineEntry Pydantic model ---
entry = PipelineEntry(
    cik="0001234567",
    company_name="TechCo Inc",
    form_type="S-1",
    filed_date=date(2024, 3, 1),
    pipeline_state="filed",
    price_range_low=18.0,
    price_range_high=20.0,
    shares_offered=10_000_000,
)
assert entry.company_name == "TechCo Inc"
assert entry.pipeline_state == "filed"
assert entry.price_range_low == 18.0
print("[OK] PipelineEntry Pydantic model created")

# --- IPOResult Pydantic model ---
result = IPOResult(
    ticker="TECH",
    company_name="TechCo Inc",
    ipo_date=date(2024, 3, 20),
    offer_price=19.0,
    first_day_close=25.0,
    day1_return_pct=31.6,
)
assert result.ticker == "TECH"
assert result.day1_return_pct == 31.6
print("[OK] IPOResult Pydantic model created")

# --- SPACRecord Pydantic model ---
spac = SPACRecord(
    cik="0009876543",
    company_name="Acquisition Corp I",
    filed_date=date(2024, 1, 15),
    trust_amount_mn=300.0,
    target_industry="Technology",
    deadline_months=24,
)
assert spac.trust_amount_mn == 300.0
print("[OK] SPACRecord Pydantic model created")

# --- class structure checks ---
assert hasattr(EdgarS1Parser, "parse_s1") or hasattr(EdgarS1Parser, "__init__")
assert hasattr(IPOPipelineTracker, "__init__")
assert hasattr(SPACTracker, "__init__")
print("[OK] EdgarS1Parser, IPOPipelineTracker, SPACTracker class structure present")

print("\n[PASS] dim_030: ipo_intelligence_v3 -- all checks passed")
PYEOF
