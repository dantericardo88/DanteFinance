#!/bin/bash
# dim_034: ria_adviser_v3 — Form ADV / RIA adviser intelligence
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.ria_adviser_v3 import (
    IAPD_BULK_ZIP,
    EDGAR_SEARCH,
    _safe_float,
    _safe_int,
    _millions,
    AUMBreakdown,
    AdviserScore,
    AdviserProfile,
    RIAIntelligenceEngine,
)

# --- constants ---
assert "adviserinfo.sec.gov" in IAPD_BULK_ZIP
assert "efts.sec.gov" in EDGAR_SEARCH
print("[OK] IAPD_BULK_ZIP and EDGAR_SEARCH constants present")

# --- _safe_float ---
assert _safe_float("1,234.56") == 1234.56
assert _safe_float("$5,000") == 5000.0
assert _safe_float("invalid") == 0.0
print("[OK] _safe_float handles currency and comma formats")

# --- _safe_int ---
assert _safe_int("100,000") == 100000
assert _safe_int("abc") == 0
print("[OK] _safe_int handles comma-formatted integers")

# --- _millions ---
assert _millions(5_000_000_000) == 5000.0
assert _millions(250.0) == 250.0
print("[OK] _millions converts raw dollar values to millions")

# --- AUMBreakdown dataclass ---
aum = AUMBreakdown(
    total_aum=1500.0,
    discretionary_aum=1200.0,
    non_discretionary_aum=300.0,
    us_aum=1425.0,
    foreign_aum=75.0,
)
assert aum.total_aum == 1500.0
assert abs(aum.discretionary_pct - 80.0) < 0.01
print("[OK] AUMBreakdown dataclass and discretionary_pct property correct")

# --- AdviserScore dataclass ---
score = AdviserScore(
    crd_number="123456",
    composite_score=82.5,
    aum_score=20.0,
    growth_score=15.0,
    client_diversity_score=25.0,
    disciplinary_penalty=0.0,
    strategy_breadth_score=12.5,
    institutional_focus_score=10.0,
    percentile=78.0,
)
assert score.composite_score == 82.5
assert score.percentile == 78.0
print("[OK] AdviserScore dataclass created")

# --- AdviserProfile dataclass ---
profile = AdviserProfile(
    crd_number="123456",
    firm_name="Acme Wealth Management LLC",
    hq_state="CA",
    state_registered=False,
    sec_registered=True,
    num_employees=45,
    num_clients=500,
    aum_breakdown=aum,
    score=score,
    strategies=["Equity", "Fixed Income", "Multi-Asset"],
    states_registered=["CA", "NY", "TX"],
    has_criminal_history=False,
    disciplinary_count=0,
)
assert profile.firm_name == "Acme Wealth Management LLC"
assert profile.num_clients == 500
assert "Equity" in profile.strategies
print("[OK] AdviserProfile dataclass created")

# --- RIAIntelligenceEngine class structure ---
assert hasattr(RIAIntelligenceEngine, "__init__")
assert hasattr(RIAIntelligenceEngine, "search_advisers") or hasattr(RIAIntelligenceEngine, "get_adviser_profile")
print("[OK] RIAIntelligenceEngine class structure present")

print("\n[PASS] dim_034: ria_adviser_v3 -- all checks passed")
PYEOF
