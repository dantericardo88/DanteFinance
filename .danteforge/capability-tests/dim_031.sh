#!/bin/bash
# dim_031: form_d_screener_v3 — Form D Regulation D screener
set -e
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.form_d_screener_v3 import (
    _EFTS_BASE,
    _EDGAR_ARCHIVE,
    _NS,
    FormDFiling,
    ExemptionAnalysis,
    OfferingTimeline,
    RegDExemptionAnalyzer,
    FormDIntelligenceEngine,
    compute_raise_velocity,
    detect_regulation_a_plus,
    compute_investor_count_signal,
)

# --- constants ---
assert "efts.sec.gov" in _EFTS_BASE
assert "sec.gov" in _EDGAR_ARCHIVE
assert "sec.gov" in _NS
print("[OK] EDGAR URL constants present")

# --- FormDFiling dataclass ---
filing = FormDFiling(
    company_name="StartupAI LLC",
    cik="0001234567",
    file_date="2024-03-01",
    accession_no="0001234567-24-000001",
    state="CA",
    city="San Francisco",
    total_offering_amount=5_000_000.0,
    amount_sold=2_500_000.0,
    offering_type="Equity",
    exemption_normalized="506b",
    num_accredited_investors=50,
    num_non_accredited_investors=0,
)
assert filing.company_name == "StartupAI LLC"
assert filing.exemption_normalized == "506b"
print("[OK] FormDFiling dataclass created")

# --- RegDExemptionAnalyzer: 506b no violation ---
analyzer = RegDExemptionAnalyzer()
analysis = analyzer.analyze_exemption(filing)
assert isinstance(analysis, ExemptionAnalysis)
violations = [f for f in analysis.flags if "VIOLATION" in f]
assert len(violations) == 0, f"Unexpected violations: {violations}"
print("[OK] 506b with 0 non-accredited — no violations")

# --- RegDExemptionAnalyzer: 506b violation ---
filing_bad = FormDFiling(
    company_name="BadCo LLC",
    cik="0009876543",
    file_date="2024-03-01",
    accession_no="0009876543-24-000001",
    exemption_normalized="506b",
    num_non_accredited_investors=40,
    num_accredited_investors=100,
)
analysis_bad = analyzer.analyze_exemption(filing_bad)
violations_bad = [f for f in analysis_bad.flags if "VIOLATION" in f]
assert len(violations_bad) > 0, "Expected violation for 40 non-accredited under 506b"
print("[OK] 506b with 40 non-accredited — violation detected")

# --- 504 cap violation ---
filing_504 = FormDFiling(
    company_name="SmallCo LLC",
    cik="0001111111",
    file_date="2024-01-01",
    accession_no="0001111111-24-000001",
    exemption_normalized="504",
    total_offering_amount=15_000_000.0,
)
analysis_504 = analyzer.analyze_exemption(filing_504)
violations_504 = [f for f in analysis_504.flags if "VIOLATION" in f]
assert len(violations_504) > 0, "Expected violation for 504 exceeding $10M"
print("[OK] 504 exceeding $10M — violation detected")

# --- OfferingTimeline dataclass ---
timeline = OfferingTimeline(
    company_name="StartupAI LLC",
    cik="0001234567",
    filings=[filing],
    first_file_date="2024-03-01",
    total_amendments=0,
    max_offering_amount=5_000_000.0,
)
assert timeline.max_offering_amount == 5_000_000.0
print("[OK] OfferingTimeline dataclass created")

# --- FormDIntelligenceEngine class structure ---
assert hasattr(FormDIntelligenceEngine, "get_market_overview")
assert hasattr(FormDIntelligenceEngine, "track_ipo_pipeline")
print("[OK] FormDIntelligenceEngine has expected methods")

# --------------------------------------------------------------------------
# NEW: Test — compute_raise_velocity formula verification
# --------------------------------------------------------------------------
# Formula: raise_velocity = (total_raised / days_since_formation) × 365
total_raised        = 12_000_000.0   # $12M
days_since_formation = 365            # exactly 1 year

expected_velocity = (total_raised / days_since_formation) * 365  # = 12_000_000
rv = compute_raise_velocity(total_raised, days_since_formation)

assert abs(rv["raise_velocity_annual"] - expected_velocity) < 0.01, \
    f"Expected {expected_velocity:.2f}, got {rv['raise_velocity_annual']}"
assert rv["total_raised"] == total_raised
assert rv["days_since_formation"] == days_since_formation
assert "formula" in rv
print(f"[OK] compute_raise_velocity: ${total_raised/1e6:.0f}M in {days_since_formation}d "
      f"→ ${rv['raise_velocity_annual']/1e6:.2f}M/yr")

# Non-round number test
rv2 = compute_raise_velocity(5_000_000.0, 500)
expected2 = (5_000_000.0 / 500) * 365   # 3_650_000
assert abs(rv2["raise_velocity_annual"] - expected2) < 0.01
print(f"[OK] compute_raise_velocity: $5M / 500d → ${rv2['raise_velocity_annual']/1e6:.2f}M/yr")

# Zero days must raise
try:
    compute_raise_velocity(1_000_000.0, 0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("[OK] compute_raise_velocity: days_since_formation=0 raises ValueError")

# Negative total_raised must raise
try:
    compute_raise_velocity(-1_000.0, 100)
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("[OK] compute_raise_velocity: negative total_raised raises ValueError")

# --------------------------------------------------------------------------
# NEW: Test — detect_regulation_a_plus
# --------------------------------------------------------------------------
# Tier 1 filing within cap
filing_rega1 = FormDFiling(
    company_name="MiniIPO Inc",
    cik="0002222222",
    file_date="2024-05-01",
    accession_no="0002222222-24-000001",
    exemption_normalized="rega_tier1",
    total_offering_amount=15_000_000.0,
)
r1 = detect_regulation_a_plus(filing_rega1)
assert r1["is_reg_a_plus"] is True
assert r1["tier"] == "Tier 1"
assert r1["max_amount"] == 20_000_000.0
assert r1["within_cap"] is True
print(f"[OK] detect_regulation_a_plus: Tier 1 $15M within $20M cap")

# Tier 2 exceeding cap
filing_rega2_over = FormDFiling(
    company_name="BigMiniIPO Inc",
    cik="0003333333",
    file_date="2024-05-01",
    accession_no="0003333333-24-000001",
    exemption_normalized="rega_tier2",
    total_offering_amount=80_000_000.0,   # > $75M cap
)
r2 = detect_regulation_a_plus(filing_rega2_over)
assert r2["is_reg_a_plus"] is True
assert r2["tier"] == "Tier 2"
assert r2["max_amount"] == 75_000_000.0
assert r2["within_cap"] is False
print(f"[OK] detect_regulation_a_plus: Tier 2 $80M exceeds $75M cap → within_cap=False")

# 506b → not Reg A+
filing_506b = FormDFiling(
    company_name="Normal VC Inc",
    cik="0004444444",
    file_date="2024-01-01",
    accession_no="0004444444-24-000001",
    exemption_normalized="506b",
    total_offering_amount=10_000_000.0,
)
r3 = detect_regulation_a_plus(filing_506b)
assert r3["is_reg_a_plus"] is False
assert r3["tier"] is None
assert r3["within_cap"] is None
print(f"[OK] detect_regulation_a_plus: 506b correctly identified as non-Reg-A+")

# --------------------------------------------------------------------------
# NEW: Test — compute_investor_count_signal
# --------------------------------------------------------------------------
# Exactly at threshold: 500 investors, $10M AUM → NOT triggered (> required)
at_threshold = compute_investor_count_signal(500, 10_000_000.0)
assert at_threshold["mandatory_registration_signal"] is False, \
    "Exactly 500 investors is NOT > 500, should not trigger"
print(f"[OK] compute_investor_count_signal: exactly 500 investors → no signal (needs >500)")

# Triggered: 501 investors, $10.1M AUM
triggered = compute_investor_count_signal(501, 10_100_000.0)
assert triggered["mandatory_registration_signal"] is True
assert triggered["investor_threshold"] == 500
assert triggered["aum_threshold"] == 10_000_000.0
print(f"[OK] compute_investor_count_signal: 501 investors + $10.1M AUM → signal=True")

# Not triggered: 501 investors but only $9M AUM (below AUM threshold)
not_triggered = compute_investor_count_signal(501, 9_000_000.0)
assert not_triggered["mandatory_registration_signal"] is False
print(f"[OK] compute_investor_count_signal: 501 investors but $9M AUM < threshold → signal=False")

# Not triggered: large AUM but only 10 investors
not_triggered2 = compute_investor_count_signal(10, 100_000_000.0)
assert not_triggered2["mandatory_registration_signal"] is False
print(f"[OK] compute_investor_count_signal: 10 investors + $100M AUM → signal=False")

print("\n[PASS] dim_031: form_d_screener_v3 -- all checks passed")
PYEOF
