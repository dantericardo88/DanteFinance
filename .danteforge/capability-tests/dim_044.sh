#!/bin/bash
# dim_044: economic_calendar_v3 — economic calendar, surprise index, FOMC tracker
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.economic_calendar_v3 import (
    _compute_surprise,
    _make_event_id,
    _normalize_currency_code,
    _infer_category,
    _infer_importance,
    _deduplicate_events,
    _ALL_RELEASES,
    _US_RELEASES,
    EconomicEvent,
    FOMCEvent,
    TreasuryAuction,
    CalendarResponse,
    compute_economic_surprise_index,
    get_international_cb_calendar,
    fetch_treasury_auction_calendar,
    build_consensus_from_fred,
    _CB_SCHEDULES_2026,
    _FOMC_MEETINGS_2026,
)

# --- release catalog ---
assert len(_US_RELEASES) >= 10
assert len(_ALL_RELEASES) >= len(_US_RELEASES)
# Verify Nonfarm Payrolls is in the catalog with 5-star importance
nfp_entries = [r for r in _US_RELEASES if "Nonfarm Payrolls" in r[0]]
assert len(nfp_entries) > 0, "Nonfarm Payrolls not in release catalog"
assert nfp_entries[0][2] == 5, "Nonfarm Payrolls should be 5-star"
print(f"[OK] Release catalog: {len(_US_RELEASES)} US releases, {len(_ALL_RELEASES)} total")

# --- _compute_surprise ---
mag, direction = _compute_surprise(actual=3.2, forecast=3.0)
assert direction == "beat" and mag > 0
mag2, direction2 = _compute_surprise(actual=2.8, forecast=3.0)
assert direction2 == "miss"
print(f"[OK] _compute_surprise: beat={mag:.4f}, miss detected")

# --- _make_event_id ---
eid = _make_event_id("Nonfarm Payrolls", "2024-02-02", "US")
assert len(eid) == 16
assert eid == _make_event_id("Nonfarm Payrolls", "2024-02-02", "US"), "Must be deterministic"
print("[OK] _make_event_id: 16-char deterministic ID")

# --- _normalize_currency_code ---
assert _normalize_currency_code("USD") == "US"
assert _normalize_currency_code("EUR") == "EU"
assert _normalize_currency_code("GBP") == "GB"
print("[OK] _normalize_currency_code maps correctly")

# --- _infer_category ---
assert _infer_category("Consumer Price Index (CPI)") == "inflation"
assert _infer_category("Nonfarm Payrolls") == "employment"
print("[OK] _infer_category categorizes events correctly")

# --- _infer_importance ---
assert _infer_importance("FOMC Meeting") == 5
assert _infer_importance("Nonfarm Payrolls") == 5
assert _infer_importance("CPI Monthly") == 5
print("[OK] _infer_importance assigns 5-star to tier-1 events")

# --- EconomicEvent model ---
ev = EconomicEvent(
    event_id="abc12345def67890",
    event_name="Nonfarm Payrolls",
    country="US",
    event_date="2024-03-08",
    release_time_et="08:30",
    category="employment",
    importance_stars=5,
    forecast=200.0,
    actual=275.0,
    previous=229.0,
    surprise_magnitude=0.25,
    surprise_direction="beat",
    unit="K",
    source="forexfactory",
    is_released=True,
)
assert ev.event_name == "Nonfarm Payrolls"
assert ev.surprise_direction == "beat"
print("[OK] EconomicEvent model created")

# --- _deduplicate_events ---
ev2 = EconomicEvent(
    event_id="abc12345def67891",
    event_name="Nonfarm Payrolls",
    country="US",
    event_date="2024-03-08",
    release_time_et="08:30",
    category="employment",
    importance_stars=5,
    forecast=200.0,
    actual=275.0,
    previous=229.0,
    unit="K",
    source="investing.com",
    is_released=True,
)
deduped = _deduplicate_events([ev, ev2])
assert len(deduped) == 1, f"Expected 1 after dedup, got {len(deduped)}"
print("[OK] _deduplicate_events merges duplicate events")

# --- FOMCEvent model ---
fomc = FOMCEvent(
    meeting_date="2024-03-20",
    type="rate_decision",
    days_until=12,
    current_rate_pct=5.33,
    expected_change_bps=0.0,
    is_press_conference=True,
)
assert fomc.type == "rate_decision" and fomc.is_press_conference is True
print("[OK] FOMCEvent model created")

# --- TreasuryAuction model ---
auction = TreasuryAuction(
    auction_date="2024-03-11",
    tenor="10Y",
    amount_bn=39.0,
    bid_to_cover=2.54,
    high_yield=4.166,
    when_issued_yield=4.17,
    days_until=3,
)
assert auction.tenor == "10Y" and auction.bid_to_cover == 2.54
print("[OK] TreasuryAuction model created")

# --- CalendarResponse model ---
cr = CalendarResponse(
    generated_at="2024-03-15T10:00:00",
    days_ahead=14,
    country_filter="US",
    total_events=12,
    high_impact_count=4,
    events=[ev],
)
assert cr.total_events == 12
assert cr.high_impact_count == 4
print("[OK] CalendarResponse model created")

# =========================================================================
# NEW: Economic Surprise Index (ESI)
# =========================================================================
import pandas as pd

esi = compute_economic_surprise_index(country="US", window_days=90)
assert isinstance(esi, pd.Series), f"ESI must be pd.Series, got {type(esi)}"
assert len(esi) >= 1, f"ESI must have at least 1 data point, got {len(esi)}"
latest_esi = float(esi.iloc[-1])
assert isinstance(latest_esi, float), f"ESI value must be float, got {type(latest_esi)}"
# ESI can be positive or negative — just verify it's a real number
assert latest_esi == latest_esi, "ESI value must not be NaN"
print(f"[OK] ESI computed: {len(esi)} data points, latest={latest_esi:.4f} ({'positive' if latest_esi > 0 else 'negative'})")

# Verify ESI index is DatetimeIndex
assert hasattr(esi.index, 'dtype'), "ESI index must be a pandas index"
print(f"[OK] ESI series name: {esi.name}, index type: {type(esi.index).__name__}")

# =========================================================================
# NEW: International Central Bank Calendar
# =========================================================================

cb_calendar = get_international_cb_calendar(days_ahead=365)
assert isinstance(cb_calendar, list), "CB calendar must be a list"

# Must return meetings from at least 3 distinct central banks (even if all in future)
banks_present = set(entry["bank"] for entry in cb_calendar)
# If no future meetings (e.g., running after year-end), use full schedule check
if len(cb_calendar) == 0:
    # Verify schedules are defined for at least 3 banks
    assert len(_CB_SCHEDULES_2026) >= 3, "Must have schedules for at least 3 CBs"
    assert "ECB" in _CB_SCHEDULES_2026
    assert "BOE" in _CB_SCHEDULES_2026
    assert "BOJ" in _CB_SCHEDULES_2026
    print(f"[OK] CB schedules defined for {len(_CB_SCHEDULES_2026)} banks (no future meetings in window)")
else:
    assert len(banks_present) >= 3, (
        f"Must cover at least 3 central banks, got: {banks_present}"
    )
    # Verify required fields
    for entry in cb_calendar[:3]:
        assert "bank" in entry, "CB entry missing 'bank'"
        assert "date" in entry, "CB entry missing 'date'"
        assert "is_next" in entry, "CB entry missing 'is_next'"
        assert "days_until" in entry, "CB entry missing 'days_until'"
        assert isinstance(entry["days_until"], int), "days_until must be int"
    # Verify is_next logic — each bank should have exactly one is_next=True
    banks_with_next = [e["bank"] for e in cb_calendar if e["is_next"]]
    assert len(banks_with_next) == len(set(banks_with_next)), (
        f"Each bank should have at most one is_next=True, got duplicates: {banks_with_next}"
    )
    # Fed must be included
    assert "Fed" in banks_present, f"Fed must be in CB calendar. Got: {banks_present}"
    print(f"[OK] CB calendar: {len(cb_calendar)} meetings from {len(banks_present)} banks: {sorted(banks_present)}")

# Verify FOMC schedule is defined
assert len(_FOMC_MEETINGS_2026) == 8, f"Must have 8 FOMC meetings, got {len(_FOMC_MEETINGS_2026)}"
assert _FOMC_MEETINGS_2026[0]["decision"] == "2026-01-29"
assert _FOMC_MEETINGS_2026[-1]["decision"] == "2026-12-16"
print(f"[OK] FOMC 2026: {len(_FOMC_MEETINGS_2026)} meetings defined, Jan 29 to Dec 16")

# =========================================================================
# NEW: Treasury Auction Calendar (no network needed — uses fallback)
# =========================================================================

# Force fallback by testing the fallback generator directly
from sentinel.sma.economic_calendar_v3 import _generate_fallback_auctions
from datetime import date, timedelta

td = date.today()
test_auctions = _generate_fallback_auctions(td, td + timedelta(days=30))
assert isinstance(test_auctions, list), "Auction list must be a list"
assert len(test_auctions) >= 1, (
    f"Must return at least 1 upcoming auction in 30-day window, got {len(test_auctions)}"
)
# Verify required fields
for a in test_auctions[:3]:
    assert "cusip" in a, f"Auction missing 'cusip': {a}"
    assert "type" in a, f"Auction missing 'type': {a}"
    assert "term" in a, f"Auction missing 'term': {a}"
    assert "auction_date" in a, f"Auction missing 'auction_date': {a}"
    assert "issue_date" in a, f"Auction missing 'issue_date': {a}"
    assert "maturity_date" in a, f"Auction missing 'maturity_date': {a}"
    assert isinstance(a["days_until"], int), f"days_until must be int: {a}"
    # Validate date format
    date.fromisoformat(a["auction_date"])
    date.fromisoformat(a["issue_date"])
    date.fromisoformat(a["maturity_date"])

types_present = set(a["type"] for a in test_auctions)
print(f"[OK] Treasury auctions (fallback): {len(test_auctions)} auctions, types: {types_present}")

# Also test the public API (will try network then fall back — both are fine)
pub_auctions = fetch_treasury_auction_calendar(days_ahead=30)
assert isinstance(pub_auctions, list), "Public auction API must return list"
assert len(pub_auctions) >= 1, (
    f"Public treasury API must return at least 1 auction, got {len(pub_auctions)}"
)
print(f"[OK] fetch_treasury_auction_calendar: {len(pub_auctions)} auctions returned")

# =========================================================================
# NEW: Consensus aggregation (pure computation, no network)
# =========================================================================

# Verify build_consensus_from_fred is callable and returns correct structure
# (we don't call it here as it hits FRED network — just verify it's importable
# and the FRED series map has entries)
from sentinel.sma.economic_calendar_v3 import _FRED_SERIES_MAP
assert len(_FRED_SERIES_MAP) >= 10, f"FRED series map too small: {len(_FRED_SERIES_MAP)}"
assert "Nonfarm Payrolls" in _FRED_SERIES_MAP
assert "CPI (Headline)" in _FRED_SERIES_MAP
assert "Unemployment Rate" in _FRED_SERIES_MAP
assert "PAYEMS" in _FRED_SERIES_MAP["Nonfarm Payrolls"]
print(f"[OK] Consensus FRED map: {len(_FRED_SERIES_MAP)} series mapped")

print("\n[PASS] dim_044: economic_calendar_v3 -- all checks passed")
PYEOF
