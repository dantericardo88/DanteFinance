#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')

from sentinel.sfe.institutional_ownership_v3 import (
    _categorize_institution,
    _parse_int,
    _normalize_cik,
    _quarter_from_date,
    Holding13F,
    Institution,
    OwnershipDelta,
)
from datetime import date

# Test _categorize_institution
cat = _categorize_institution("VANGUARD GROUP INC")
assert cat == "etf_provider", f"Expected etf_provider, got {cat}"
print("[OK] VANGUARD categorized as etf_provider")

cat2 = _categorize_institution("FIDELITY MANAGEMENT LLC")
assert cat2 == "mutual_fund", f"Expected mutual_fund, got {cat2}"
print("[OK] FIDELITY categorized as mutual_fund")

cat3 = _categorize_institution("CALPERS PENSION FUND")
assert cat3 == "pension", f"Expected pension, got {cat3}"
print("[OK] CALPERS categorized as pension")

cat4 = _categorize_institution("JPMORGAN CHASE BANK")
assert cat4 == "bank", f"Expected bank, got {cat4}"
print("[OK] JPMORGAN categorized as bank")

# Test _parse_int
assert _parse_int("1,234,567") == 1234567
print("[OK] _parse_int handles commas")

assert _parse_int("abc", 99) == 99
print("[OK] _parse_int returns default on failure")

# Test _normalize_cik (strips leading zeros)
assert _normalize_cik("001234") == "1234"
assert _normalize_cik("0") == "0"
print("[OK] _normalize_cik strips leading zeros")

# Test _quarter_from_date
q = _quarter_from_date(date(2024, 5, 15))
assert q == "2024-Q2", f"Expected 2024-Q2, got {q}"
print("[OK] _quarter_from_date returns correct quarter")

# Test Holding13F dataclass creation
h = Holding13F(
    institution_cik="0000102909",
    institution_name="BLACKROCK INC",
    cusip="037833100",
    issuer_name="Apple Inc",
    ticker="AAPL",
    shares=1_000_000,
    value_usd=180_000_000,
    filing_date=date(2024, 3, 31),
    period_of_report=date(2024, 3, 31),
)
assert h.ticker == "AAPL"
assert h.shares == 1_000_000
print("[OK] Holding13F dataclass created successfully")

# Test Institution dataclass
inst = Institution(cik="0000102909", name="BLACKROCK INC", category="etf_provider")
assert inst.category == "etf_provider"
print("[OK] Institution dataclass created successfully")

# Test OwnershipDelta
delta = OwnershipDelta(ticker="AAPL", q1="2024Q1", q2="2024Q2", net_shares_change=500_000)
assert delta.net_shares_change == 500_000
print("[OK] OwnershipDelta dataclass created successfully")

print("[PASS]")
PYEOF
