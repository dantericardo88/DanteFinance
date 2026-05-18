#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')

from sentinel.sfe.insider_v3 import (
    _classify_title,
    _is_director_kw,
    _is_ten_pct_kw,
    _safe_float,
    _safe_date,
    Form4Transaction,
    InsiderTransactionType,
)
from datetime import date

# Test _classify_title
tier = _classify_title("Chief Executive Officer")
assert tier == "CEO", f"Expected CEO, got {tier}"
print("[OK] CEO title classified correctly")

tier2 = _classify_title("Chief Financial Officer")
assert tier2 == "CFO", f"Expected CFO, got {tier2}"
print("[OK] CFO title classified correctly")

tier3 = _classify_title("Unknown VP level")
assert isinstance(tier3, str) and len(tier3) > 0
print("[OK] _classify_title returns a non-empty string for unknown titles")

# Test _is_director_kw
assert _is_director_kw("Board Director") is True
assert _is_director_kw("Chief Executive Officer") is False
print("[OK] _is_director_kw correctly identifies directors")

# Test _is_ten_pct_kw
assert _is_ten_pct_kw("10% Owner") is True
assert _is_ten_pct_kw("Director") is False
print("[OK] _is_ten_pct_kw correctly identifies 10% owners")

# Test _safe_float
assert _safe_float("123.45") == 123.45
assert _safe_float("invalid", 0.0) == 0.0
print("[OK] _safe_float handles valid and invalid input")

# Test _safe_date
d = _safe_date("2024-01-15")
assert d == date(2024, 1, 15)
print("[OK] _safe_date parses ISO format")

d2 = _safe_date("invalid")
assert d2 is None
print("[OK] _safe_date returns None for invalid input")

# Test InsiderTransactionType enum
assert InsiderTransactionType.OPEN_MARKET_BUY.value == "open_market_buy"
assert InsiderTransactionType.OPEN_MARKET_SELL.value == "open_market_sell"
print("[OK] InsiderTransactionType enum values correct")

# Test Form4Transaction dataclass
tx = Form4Transaction(
    accession_number="0001234567-24-000001",
    filing_cik="0000320193",
    issuer_name="Apple Inc",
    issuer_ticker="AAPL",
    owner_name="Tim Cook",
    owner_title="Chief Executive Officer",
    tx_date=date(2024, 1, 15),
    shares=50000.0,
    price_per_share=185.0,
    is_open_market=True,
    is_buy=True,
)
assert tx.issuer_ticker == "AAPL"
assert tx.shares == 50000.0
assert tx.is_buy is True
print("[OK] Form4Transaction dataclass created successfully")

print("[PASS]")
PYEOF
