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

print("\n[PASS] dim_044: economic_calendar_v3 -- all checks passed")
PYEOF
