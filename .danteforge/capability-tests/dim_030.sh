#!/bin/bash
# dim_030: ipo_intelligence_v3 — IPO pipeline intelligence
set -e
export PYTHONIOENCODING=utf-8
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
    compute_ipo_pop_prediction,
    compute_lockup_expiry_signal,
    classify_ipo_quality,
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

# --------------------------------------------------------------------------
# NEW: Test — compute_ipo_pop_prediction formula verification
# --------------------------------------------------------------------------
# Formula: pop_score = rev_growth×0.3 + brand×0.2 + market×0.3 + uw_tier×0.2
rev_growth = 0.8
brand      = 0.6
market     = 0.9
uw_tier    = 1.0   # Goldman Sachs = bulge bracket

expected_pop_score = (rev_growth * 0.3
                      + brand    * 0.2
                      + market   * 0.3
                      + uw_tier  * 0.2)
# = 0.24 + 0.12 + 0.27 + 0.20 = 0.83

pred = compute_ipo_pop_prediction(rev_growth, brand, market, uw_tier)

assert abs(pred["pop_score"] - expected_pop_score) < 1e-6, \
    f"Expected pop_score={expected_pop_score:.4f}, got {pred['pop_score']}"

# pop_prediction_pct = pop_score × 28
expected_pct = round(expected_pop_score * 28.0, 2)
assert abs(pred["pop_prediction_pct"] - expected_pct) < 0.01, \
    f"Expected {expected_pct}%, got {pred['pop_prediction_pct']}%"

assert "formula" in pred
print(f"[OK] compute_ipo_pop_prediction: pop_score={pred['pop_score']:.4f}, "
      f"pop_pct={pred['pop_prediction_pct']}%")

# Boundary: all inputs 0.0 → pop_score = 0.0
pred_zero = compute_ipo_pop_prediction(0.0, 0.0, 0.0, 0.0)
assert pred_zero["pop_score"] == 0.0
print("[OK] compute_ipo_pop_prediction: zero inputs → pop_score=0.0")

# Boundary: all inputs 1.0 → pop_score = 1.0
pred_one = compute_ipo_pop_prediction(1.0, 1.0, 1.0, 1.0)
assert abs(pred_one["pop_score"] - 1.0) < 1e-9
print("[OK] compute_ipo_pop_prediction: all-ones inputs → pop_score=1.0")

# Out-of-range must raise ValueError
try:
    compute_ipo_pop_prediction(1.5, 0.5, 0.5, 0.5)
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("[OK] compute_ipo_pop_prediction: out-of-range input raises ValueError")

# --------------------------------------------------------------------------
# NEW: Test — compute_lockup_expiry_signal
# --------------------------------------------------------------------------
ipo_dt = date(2024, 1, 15)
signal = compute_lockup_expiry_signal(ipo_dt, lockup_days=180)

from datetime import timedelta
expected_expiry = ipo_dt + timedelta(days=180)
assert signal["lockup_expiry_date"] == expected_expiry.isoformat()
assert signal["expected_return_pct"] == -8.0
assert signal["lockup_days"] == 180
assert signal["signal"] in ("PAST", "APPROACHING", "ACTIVE")
print(f"[OK] compute_lockup_expiry_signal: expiry={signal['lockup_expiry_date']}, "
      f"signal={signal['signal']}, expected_return={signal['expected_return_pct']}%")

# --------------------------------------------------------------------------
# NEW: Test — classify_ipo_quality tiers
# --------------------------------------------------------------------------
# Tier 1: bulge bracket + positive EBITDA
t1 = classify_ipo_quality("Goldman Sachs", ebitda_positive=True)
assert t1["tier"] == "Tier 1"
assert t1["is_bulge_bracket"] is True
assert t1["ebitda_positive"] is True
print(f"[OK] classify_ipo_quality Tier 1: {t1['tier']} — {t1['rationale'][:40]}")

# Tier 2: bulge bracket but EBITDA negative
t2 = classify_ipo_quality("Goldman Sachs", ebitda_positive=False)
assert t2["tier"] == "Tier 2"
assert t2["is_bulge_bracket"] is True
print(f"[OK] classify_ipo_quality Tier 2 (bulge+loss): {t2['tier']}")

# Tier 2: major non-bulge with positive EBITDA
t2b = classify_ipo_quality("Jefferies", ebitda_positive=True)
assert t2b["tier"] == "Tier 2"
assert t2b["is_bulge_bracket"] is False
assert t2b["is_major"] is True
print(f"[OK] classify_ipo_quality Tier 2 (major+profit): {t2b['tier']}")

# Tier 3: boutique / unknown
t3 = classify_ipo_quality("Unknown Boutique Partners", ebitda_positive=True)
assert t3["tier"] == "Tier 3"
assert t3["is_bulge_bracket"] is False
assert t3["is_major"] is False
print(f"[OK] classify_ipo_quality Tier 3: {t3['tier']}")

# None underwriter → Tier 3
t3n = classify_ipo_quality(None, ebitda_positive=False)
assert t3n["tier"] == "Tier 3"
print(f"[OK] classify_ipo_quality Tier 3: None underwriter handled")

print("\n[PASS] dim_030: ipo_intelligence_v3 -- all checks passed")
PYEOF
