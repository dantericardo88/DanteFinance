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
    BUY_CODES,
    OPEN_MARKET_CODES,
    _ROLE_WEIGHTS,
    compute_cluster_buy_signal,
    compute_insider_conviction_score,
    director_vs_officer_split,
)
from datetime import date, timedelta

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

# ── Math verification: compute_insider_conviction_score ─────────────────────
# Formula: score = (shares x price / (shares_outstanding x price)) x role_weight x 10000
# = (tx_value / market_cap) x role_weight x 10000
#
# Example: CEO buys 50000 shares @ $185 = $9.25M tx_value
#   shares_outstanding = 15,000,000,000, price = $185 -> mktcap = $2.775T
#   raw = 9_250_000 / 2_775_000_000_000 ~ 3.333e-6
#   role_weight = 3.0 (CEO)
#   raw_score = 3.333e-6 x 3.0 x 10000 ~ 0.1
tx_ceo = Form4Transaction(
    accession_number="test-001",
    filing_cik="0000320193",
    issuer_ticker="AAPL",
    owner_name="Tim Cook",
    role_tier="CEO",
    shares=50_000.0,
    price_per_share=185.0,
    is_buy=True,
)
shares_out = 15_000_000_000
price      = 185.0
score = compute_insider_conviction_score(tx_ceo, shares_out, price)
expected_raw   = (50_000 * 185.0) / (15_000_000_000 * 185.0)
expected_role  = _ROLE_WEIGHTS["CEO"]        # 3.0
expected_score = min(10.0, expected_raw * expected_role * 10_000)
assert abs(score - round(expected_score, 4)) < 1e-4, \
    f"Score {score} != expected {expected_score}"
print(f"[OK] compute_insider_conviction_score: CEO 50k shares -> score={score:.4f}")

# Edge case: zero market cap returns 0
score_zero = compute_insider_conviction_score(tx_ceo, 0, 185.0)
assert score_zero == 0.0
print("[OK] compute_insider_conviction_score: zero shares_out -> 0.0")

# Score must be in [0, 10]
tx_big = Form4Transaction(
    accession_number="test-002",
    filing_cik="XXXX",
    issuer_ticker="TEST",
    role_tier="CEO",
    shares=1_000_000_000.0,
    price_per_share=100.0,
    is_buy=True,
)
score_big = compute_insider_conviction_score(tx_big, 1_000_000, 100.0)
assert score_big == 10.0, f"Expected cap at 10.0, got {score_big}"
print("[OK] compute_insider_conviction_score: large buy capped at 10.0")

# ── Math verification: compute_cluster_buy_signal ───────────────────────────
# Build 3 transactions from different insiders on same ticker within 30 days
base_date = date(2024, 3, 1)
txns = []
for i, (name, title, days_offset) in enumerate([
    ("Tim Cook",   "CEO",      0),
    ("Luca Mae",   "CFO",      5),
    ("Jeff W",     "Director", 12),
]):
    txns.append(Form4Transaction(
        accession_number=f"acc-{i:04d}",
        filing_cik="0000320193",
        issuer_ticker="AAPL",
        owner_name=name,
        owner_title=title,
        tx_date=base_date + timedelta(days=days_offset),
        shares=10_000.0,
        price_per_share=180.0,
        tx_code="P",
        is_buy=True,
        is_open_market=True,
        estimated_value=1_800_000.0,
    ))

result = compute_cluster_buy_signal(txns, "AAPL", window_days=30)
assert result["cluster_detected"] is True, "Should detect cluster"
assert result["max_cluster_size"] >= 3, f"Cluster size {result['max_cluster_size']} < 3"
print(f"[OK] compute_cluster_buy_signal: 3 insiders in 30d -> cluster_detected=True, size={result['max_cluster_size']}")

# Only 1 insider -> no cluster
single = [txns[0]]
result_single = compute_cluster_buy_signal(single, "AAPL", window_days=30)
assert result_single["cluster_detected"] is False
print("[OK] compute_cluster_buy_signal: 1 insider -> cluster_detected=False")

# ── Math verification: director_vs_officer_split ────────────────────────────
from datetime import date as _date
today = _date.today()
d_txns = []
# Director buy
d_txns.append(Form4Transaction(
    accession_number="d-001",
    filing_cik="CIK",
    issuer_ticker="XYZ",
    owner_name="Dir A",
    role_tier="Director",
    is_director=True,
    is_officer=False,
    tx_date=today,
    tx_code="P",
    is_buy=True,
    is_sell=False,
    is_open_market=True,
    estimated_value=500_000.0,
    conviction_score=6.0,
))
# Officer buy
d_txns.append(Form4Transaction(
    accession_number="d-002",
    filing_cik="CIK",
    issuer_ticker="XYZ",
    owner_name="Off B",
    role_tier="CEO",
    is_director=False,
    is_officer=True,
    tx_date=today,
    tx_code="P",
    is_buy=True,
    is_sell=False,
    is_open_market=True,
    estimated_value=1_000_000.0,
    conviction_score=8.0,
))

split = director_vs_officer_split(d_txns, "XYZ", days=90)
assert split["directors"]["buy_count"] == 1
assert split["officers"]["buy_count"]  == 1
assert split["directors"]["net_value_usd"] > 0
assert split["officers"]["net_value_usd"]  > 0
assert split["agreement"] is True   # both buying
print(f"[OK] director_vs_officer_split: directors={split['directors']['direction']}, "
      f"officers={split['officers']['direction']}, agreement={split['agreement']}")

print("[PASS]")
PYEOF
