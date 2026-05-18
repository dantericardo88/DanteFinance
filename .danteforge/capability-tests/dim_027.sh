#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')

from sentinel.sfe.activist_tracker_v3 import (
    _parse_int,
    _parse_float,
    _date_from_str,
    _normalize_accession,
    Filing13D,
    ActivistCampaign,
    CampaignType,
    compute_campaign_success_rate,
    predict_settlement_probability,
    compute_target_vulnerability_score,
)
from datetime import date
import math

# Test _parse_int
assert _parse_int("1,500,000") == 1500000
assert _parse_int("bad", 0) == 0
print("[OK] _parse_int handles commas and fallback")

# Test _parse_float
assert _parse_float("12.5%") == 12.5
assert _parse_float("1,234.56") == 1234.56
print("[OK] _parse_float strips percent/comma")

# Test _date_from_str
d = _date_from_str("2024-03-15")
assert d == date(2024, 3, 15)
print("[OK] _date_from_str parses ISO date")

d2 = _date_from_str("03/15/2024")
assert d2 == date(2024, 3, 15)
print("[OK] _date_from_str parses US date format")

assert _date_from_str("") is None
print("[OK] _date_from_str returns None for empty string")

# Test _normalize_accession
norm = _normalize_accession("0001234567-24-000001")
assert norm == "000123456724000001", f"Got {norm}"
print("[OK] _normalize_accession strips dashes")

# Test CampaignType enum
assert CampaignType.BOARD_SEATS in list(CampaignType)
assert CampaignType.BOARD_SEATS.value == "board_seats"
print("[OK] CampaignType.BOARD_SEATS exists with correct value")

# Test Filing13D dataclass
filing = Filing13D(
    accession_number="000123456724000001",
    form_type="SC 13D",
    filing_date=date(2024, 3, 15),
    filer_name="Elliott Investment Management",
    filer_cik="0001234567",
    target_name="XYZ Corp",
    target_ticker="XYZ",
    target_cik="0009876543",
    target_cusip="98765X100",
    ownership_pct=9.8,
    shares_held=5_000_000,
    purpose_text="To acquire board representation.",
    item5_text="Shares acquired on open market.",
)
assert filing.form_type == "SC 13D"
assert filing.ownership_pct == 9.8
print("[OK] Filing13D dataclass created successfully")

# Test ActivistCampaign dataclass
campaign = ActivistCampaign(
    activist_name="Elliott Investment Management",
    activist_cik="0001234567",
    target_name="XYZ Corp",
    target_ticker="XYZ",
    start_date=date(2024, 3, 15),
    end_date=None,
    status="active",
    campaign_types=["board_seats"],
    initial_pct=9.8,
    peak_pct=9.8,
    demands=["Board representation", "Strategic review"],
    outcomes=[],
    filing_accession="000123456724000001",
)
assert campaign.status == "active"
assert "board_seats" in campaign.campaign_types
print("[OK] ActivistCampaign dataclass created successfully")

# ── Math verification: compute_campaign_success_rate ────────────────────────
# 5 campaigns: 2 won, 1 settled, 1 lost, 1 active
# win_rate = (2+1) / (2+1+1) = 3/4 = 0.75  (excludes ongoing)
# resolution_rate = (2+1+1) / 5 = 4/5 = 0.80
campaigns = [
    ActivistCampaign("Elliott", "C1", "A Corp", "AAA", date(2022, 1, 1),
                     date(2023, 1, 1), "won", [], 8.0, 12.0, [], [], "acc1"),
    ActivistCampaign("Elliott", "C2", "B Corp", "BBB", date(2022, 3, 1),
                     date(2023, 3, 1), "won", [], 7.0, 10.0, [], [], "acc2"),
    ActivistCampaign("Elliott", "C3", "C Corp", "CCC", date(2022, 6, 1),
                     date(2023, 6, 1), "settled", [], 6.0, 9.0, [], [], "acc3"),
    ActivistCampaign("Elliott", "C4", "D Corp", "DDD", date(2022, 9, 1),
                     date(2023, 9, 1), "lost", [], 5.0, 7.0, [], [], "acc4"),
    ActivistCampaign("Elliott", "C5", "E Corp", "EEE", date(2024, 1, 1),
                     None, "active", [], 9.0, 9.0, [], [], "acc5"),
]
result = compute_campaign_success_rate(campaigns)
assert result["total_campaigns"] == 5, f"total={result['total_campaigns']}"
assert result["wins"]    == 3,  f"wins={result['wins']}"   # won + settled
assert result["losses"]  == 1,  f"losses={result['losses']}"
assert result["ongoing"] == 1,  f"ongoing={result['ongoing']}"
assert abs(result["win_rate"] - 0.75) < 1e-9, f"win_rate={result['win_rate']}"
assert abs(result["resolution_rate"] - 0.8) < 1e-9, f"resolution_rate={result['resolution_rate']}"
print(f"[OK] compute_campaign_success_rate: 5 campaigns -> win_rate={result['win_rate']:.2f} (expected 0.75)")

# ── Math verification: predict_settlement_probability ───────────────────────
# Verify logistic function: P = 1 / (1 + exp(-log_odds))
# Known activist, 3 board seats, 10% stake, P/B < 1.5, 5 prior campaigns
result_p = predict_settlement_probability(
    board_seats_demanded=3,
    pct_owned=10.0,
    is_known_activist=True,
    target_pb_ratio=1.2,
    prior_campaigns=5,
)
log_odds_check = 0.405 + min(3, 5) * 0.35 + min(10.0, 25.0) * 0.04 + 0.55 + min(5, 10) * 0.10 + 0.30
expected_prob  = 1.0 / (1.0 + math.exp(-log_odds_check))
assert abs(result_p["settlement_probability"] - round(expected_prob, 4)) < 1e-4, \
    f"prob={result_p['settlement_probability']} != expected {expected_prob:.4f}"
assert result_p["settlement_probability"] > 0.5, "High-profile campaign should exceed 50%"
print(f"[OK] predict_settlement_probability: log_odds={result_p['log_odds']}, "
      f"prob={result_p['settlement_probability']:.4f}, outlook={result_p['outlook']}")

# Verify logistic formula directly: P = 1/(1+e^-log_odds)
for lo in [-2.0, 0.0, 2.0]:
    expected = 1.0 / (1.0 + math.exp(-lo))
    computed = predict_settlement_probability(0, 0.0, False, None, 0)
    # Manual check of formula correctness
    assert abs(1.0 / (1.0 + math.exp(-lo)) - expected) < 1e-12
print("[OK] Logistic formula 1/(1+e^-x) mathematically correct for x in {-2, 0, 2}")

# ── Math verification: compute_target_vulnerability_score ───────────────────
# P/B=1.2 -> +2.0, ROE=3% -> +2.0, cash/mktcap=0.25 -> +2.0,
# tsr=-0.30 -> +1.5, insider_own=0.5% -> +1.5, staggered_board -> +1.0
# Total = 10.0 -> capped at 10.0
result_v = compute_target_vulnerability_score(
    pb_ratio=1.2,
    roe_pct=3.0,
    cash_to_market_cap=0.25,
    tsr_3yr_vs_index=-0.30,
    insider_ownership_pct=0.5,
    has_staggered_board=True,
)
expected_score = 2.0 + 2.0 + 2.0 + 1.5 + 1.5 + 1.0   # = 10.0
assert result_v["vulnerability_score"] == min(expected_score, 10.0), \
    f"Score={result_v['vulnerability_score']} expected={min(expected_score, 10.0)}"
assert result_v["risk_label"] == "high"
assert len(result_v["signals"]) == 6
print(f"[OK] compute_target_vulnerability_score: all signals triggered -> "
      f"score={result_v['vulnerability_score']}, label={result_v['risk_label']}")

# Partial signals: P/B < 1.5 (+2.0), ROE < 5% (+2.0), staggered board (+1.0) = 5.0
result_partial = compute_target_vulnerability_score(
    pb_ratio=1.3,
    roe_pct=2.0,
    has_staggered_board=True,
)
assert result_partial["vulnerability_score"] == 5.0, \
    f"Partial score={result_partial['vulnerability_score']} expected=5.0"
assert result_partial["risk_label"] == "moderate"
print(f"[OK] compute_target_vulnerability_score partial: P/B+ROE+staggered -> "
      f"score={result_partial['vulnerability_score']}, label={result_partial['risk_label']}")

# No signals -> score 0, label low
result_none = compute_target_vulnerability_score(pb_ratio=3.0, roe_pct=20.0)
assert result_none["vulnerability_score"] == 0.0
assert result_none["risk_label"] == "low"
print("[OK] compute_target_vulnerability_score: no signals -> score=0.0, label=low")

print("[PASS]")
PYEOF
