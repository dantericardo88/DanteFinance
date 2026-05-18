#!/bin/bash
# dim_031: form_d_screener_v3 — Form D Regulation D screener
set -e
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

print("\n[PASS] dim_031: form_d_screener_v3 -- all checks passed")
PYEOF
