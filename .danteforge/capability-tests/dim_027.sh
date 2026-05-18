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
)
from datetime import date

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

print("[PASS]")
PYEOF
