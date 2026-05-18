#!/bin/bash
# dim_011: Extended hours — ExtendedBar, GapStatistics, GapAlert, session constants
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sds.adapters.extended_hours_v3 import (
    _PRE_START_MIN, _PRE_END_MIN, _AH_START_MIN, _AH_END_MIN,
    _GAP_ALERT_THRESHOLD,
    ExtendedBar, GapStatistics, GapAlert, SessionVolume, ExtendedQuote,
)
from datetime import datetime, date, timezone

# Test 1: Session boundary constants
assert _PRE_START_MIN == 240, f"pre_start should be 240 (4:00), got {_PRE_START_MIN}"
assert _PRE_END_MIN   == 570, f"pre_end should be 570 (9:30), got {_PRE_END_MIN}"
assert _AH_START_MIN  == 960, f"ah_start should be 960 (16:00), got {_AH_START_MIN}"
assert _AH_END_MIN    == 1200, f"ah_end should be 1200 (20:00), got {_AH_END_MIN}"
assert _GAP_ALERT_THRESHOLD == 0.02
print("[OK] Session boundary constants correct")

# Test 2: ExtendedBar pydantic model
bar = ExtendedBar(
    ticker="AAPL",
    time=datetime(2024, 3, 15, 8, 0, 0, tzinfo=timezone.utc),
    open=180.0,
    high=181.5,
    low=179.5,
    close=181.0,
    volume=50000,
    session="pre_market",
)
assert bar.ticker == "AAPL"
assert bar.open == 180.0
assert bar.high == 181.5
assert bar.session == "pre_market"
print(f"[OK] ExtendedBar: {bar.ticker} open={bar.open} session={bar.session}")

# Test 3: GapStatistics pydantic model — attributes
gs = GapStatistics(
    ticker="NVDA",
    days_analyzed=30,
    avg_gap_pct=1.5,
    max_gap_pct=5.2,
    gap_fill_rate=0.65,
    earnings_gap_count=2,
    significant_gap_count=4,
)
assert gs.ticker == "NVDA"
assert gs.days_analyzed == 30
assert gs.avg_gap_pct == 1.5
assert gs.gap_fill_rate == 0.65
assert gs.significant_gap_count == 4
print(f"[OK] GapStatistics: {gs.ticker} avg_gap={gs.avg_gap_pct}% fill_rate={gs.gap_fill_rate}")

# Test 4: GapAlert model — significant flag
ga = GapAlert(
    ticker="TSLA",
    session="pre_market",
    gap_date=date(2024, 3, 15),
    prev_close=200.0,
    current_price=208.0,
    gap_pct=4.0,
    is_significant=True,
    earnings_driven=True,
)
assert ga.is_significant is True
assert ga.earnings_driven is True
assert ga.gap_pct == 4.0
print(f"[OK] GapAlert: {ga.ticker} gap={ga.gap_pct}% significant={ga.is_significant}")

# Test 5: ExtendedQuote model
UTC = timezone.utc
eq = ExtendedQuote(
    ticker="MSFT",
    session="after_hours",
    as_of=datetime.now(UTC),
    price=420.50,
    bid=420.40,
    ask=420.60,
    volume=100000,
)
assert eq.ticker == "MSFT"
assert eq.session == "after_hours"
print(f"[OK] ExtendedQuote: {eq.ticker} session={eq.session} price={eq.price}")

print("\n[PASS] dim_011: extended_hours_v3 -- all checks passed")
PYEOF
