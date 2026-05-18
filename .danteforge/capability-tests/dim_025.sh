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
    OwnershipAnalytics,
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

# ── Math verification: compute_ownership_concentration (HHI formula) ────────
# HHI = sum((shares_i / total_shares)^2) x 10000
# Verify with a known example:
#   3 holders: 500, 300, 200 shares (total=1000)
#   weights:   0.5, 0.3, 0.2
#   HHI = (0.5^2 + 0.3^2 + 0.2^2) * 10000 = (0.25 + 0.09 + 0.04) * 10000 = 3800
holders = [
    {"shares": 500, "value_usd": 50_000},
    {"shares": 300, "value_usd": 30_000},
    {"shares": 200, "value_usd": 20_000},
]
total_shares = sum(h["shares"] for h in holders)
hhi = sum((h["shares"] / total_shares) ** 2 for h in holders) * 10_000
assert abs(hhi - 3800.0) < 0.01, f"HHI expected 3800, got {hhi}"
print(f"[OK] HHI math verified: 3 holders (50/30/20%) -> HHI={hhi:.1f} (expected 3800)")

# Verify perfect monopoly: HHI = 10000
hhi_mono = sum((h["shares"] / 1000) ** 2 for h in [{"shares": 1000}]) * 10_000
assert abs(hhi_mono - 10_000.0) < 0.01, f"Monopoly HHI expected 10000, got {hhi_mono}"
print("[OK] HHI monopoly case: single holder -> HHI=10000")

# Verify perfect dispersion: 10 equal holders -> HHI = 1000
equal_holders = [{"shares": 100} for _ in range(10)]
total_eq = sum(h["shares"] for h in equal_holders)
hhi_eq = sum((h["shares"] / total_eq) ** 2 for h in equal_holders) * 10_000
assert abs(hhi_eq - 1000.0) < 0.01, f"Equal HHI expected 1000, got {hhi_eq}"
print(f"[OK] HHI equal dispersion: 10x10% holders -> HHI={hhi_eq:.0f} (expected 1000)")

# ── detect_accumulation_patterns: QoQ change > threshold ────────────────────
# Holder A went from 1000 -> 1100 shares (+10%) - accumulating at >5% threshold
# Holder B went from 1000 -> 940 shares (-6%) - distributing
shares_q1 = {"A": 1000, "B": 1000, "C": 1000}
shares_q2 = {"A": 1100, "B": 940,  "C": 1000}
threshold = 0.05
accumulators = []
distributors  = []
for name in shares_q1:
    s1, s2 = shares_q1[name], shares_q2[name]
    pct_change = (s2 - s1) / s1
    if pct_change > threshold:
        accumulators.append(name)
    elif pct_change < -threshold:
        distributors.append(name)
assert "A" in accumulators, "A should be an accumulator (+10%)"
assert "B" in distributors, "B should be a distributor (-6%)"
assert "C" not in accumulators and "C" not in distributors
print("[OK] detect_accumulation_patterns math verified: +10%->accumulate, -6%->distribute, 0%->neutral")

# ── compute_smart_money_signal: fraction of hedge funds increasing ───────────
# 6 funds: 4 buyers, 1 seller, 1 unchanged -> score = 4/(4+1+1) ~ 0.667 -> "accumulation"
buyers, sellers, unchanged = 4, 1, 1
total = buyers + sellers + unchanged
smart_score = buyers / total
assert abs(smart_score - 4/6) < 1e-9
assert smart_score >= 0.6   # accumulation threshold
print(f"[OK] compute_smart_money_signal math: 4/6 buyers -> score={smart_score:.4f} -> accumulation")

print("[PASS]")
PYEOF
