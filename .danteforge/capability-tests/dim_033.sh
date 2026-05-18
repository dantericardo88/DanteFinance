#!/bin/bash
# dim_033: edgar_search_v3 — EDGAR full-text search / NLP pipeline
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.edgar_search_v3 import (
    EFTS_BASE,
    EDGAR_DATA,
    EDGAR_ARCHIVES,
    _strip_html,
    _accession_to_path,
    _cik_10,
    EDGARSearchResult,
    RiskFactor,
    FilingAnalysis,
    EDGARSearchEngine,
)

# --- constants ---
assert "efts.sec.gov" in EFTS_BASE
assert "sec.gov" in EDGAR_DATA
assert "sec.gov" in EDGAR_ARCHIVES
print("[OK] EDGAR URL constants present")

# --- _strip_html ---
html = "<p>Hello <b>World</b></p>"
text = _strip_html(html)
assert "Hello" in text and "World" in text and "<p>" not in text
print("[OK] _strip_html removes HTML tags")

# --- _accession_to_path ---
path = _accession_to_path("0000320193-23-000077")
assert path == "000032019323000077", f"Got {path}"
print("[OK] _accession_to_path strips dashes")

# --- _cik_10 ---
assert _cik_10("320193") == "0000320193"
assert _cik_10("1") == "0000000001"
print("[OK] _cik_10 zero-pads to 10 digits")

# --- EDGARSearchResult dataclass ---
result = EDGARSearchResult(
    accession_no="0000320193-24-000001",
    filing_date="2024-01-15",
    form_type="10-K",
    entity_name="Apple Inc",
    cik="320193",
    period_of_report="2023-09-30",
    file_num="001-36491",
    category="form-type",
    display_names="Apple Inc (AAPL)",
)
assert result.entity_name == "Apple Inc"
assert result.form_type == "10-K"
print("[OK] EDGARSearchResult dataclass created")

# --- RiskFactor dataclass ---
rf = RiskFactor(
    text="We face intense competition in all our markets.",
    severity_score=0.15,
    category="COMPETITIVE",
    word_count=9,
    key_terms=["competition", "markets"],
)
assert rf.category == "COMPETITIVE"
assert rf.severity_score == 0.15
print("[OK] RiskFactor dataclass created")

# --- FilingAnalysis dataclass ---
fa = FilingAnalysis(
    ticker="AAPL",
    form_type="10-K",
    entity_name="Apple Inc",
    filing_date="2024-01-15",
    period="2023-09-30",
    risk_factors=[rf],
    forward_looking_statements=["We expect to invest further in AI."],
    material_weaknesses=[],
    quantitative_disclosures=[("Revenue", 394.3e9, "USD")],
    readability_score=42.5,
    fog_index=14.2,
    overall_sentiment=0.55,
    document_length=180000,
    section_lengths={"Risk Factors": 45000},
    summary="Apple reported strong fiscal 2023 results.",
)
assert fa.ticker == "AAPL"
assert fa.risk_factors[0].category == "COMPETITIVE"
print("[OK] FilingAnalysis dataclass created")

# --- EDGARSearchEngine class structure ---
engine = EDGARSearchEngine()
assert hasattr(engine, "search")
assert hasattr(engine, "analyze")
print("[OK] EDGARSearchEngine instantiates with search and analyze methods")

print("\n[PASS] dim_033: edgar_search_v3 -- all checks passed")
PYEOF
