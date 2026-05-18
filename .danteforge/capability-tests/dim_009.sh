#!/bin/bash
# dim_009: Short interest — _compute_squeeze_score, ShortMetrics models
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import date

from sentinel.sds.adapters.short_interest_v3 import (
    ShortInterestRecord, DailyShortVolume, FTDRecord,
    ShortMetrics, SqueezeCandidateRecord, ShortChangeRecord,
    _compute_squeeze_score,
)

# Test 1: _compute_squeeze_score pure computation
score1 = _compute_squeeze_score(dtc=10.0, short_pct_float=0.50)
# score = 10 * 0.50 * 100 = 500
assert abs(score1 - 500.0) < 1e-6, f"Expected 500, got {score1}"

score2 = _compute_squeeze_score(dtc=1.0, short_pct_float=0.05)
# score = 1 * 0.05 * 100 = 5
assert abs(score2 - 5.0) < 1e-6, f"Expected 5, got {score2}"

# None inputs return None
assert _compute_squeeze_score(None, 0.5) is None
assert _compute_squeeze_score(5.0, None) is None
print(f"[OK] _compute_squeeze_score: GME-like={score1}, low={score2}, None handled")

# Test 2: ShortInterestRecord model
si = ShortInterestRecord(
    ticker="GME",
    settlement_date=date(2021, 1, 15),
    short_interest=71_200_000,
    avg_daily_volume=29_000_000,
    days_to_cover=round(71_200_000 / 29_000_000, 4),
    source="finra"
)
assert si.ticker == "GME"
assert si.days_to_cover is not None and si.days_to_cover > 2.0
print(f"[OK] ShortInterestRecord: GME DTC={si.days_to_cover:.2f}")

# Test 3: DailyShortVolume — field is short_pct not short_pct_volume
dv = DailyShortVolume(
    ticker="AMC",
    trade_date=date(2021, 5, 25),
    short_volume=12_000_000,
    total_volume=50_000_000,
    short_pct=round(12_000_000 / 50_000_000, 4),
    source="finra_daily"
)
assert abs(dv.short_pct - 0.24) < 1e-4, f"Short pct wrong: {dv.short_pct}"
print(f"[OK] DailyShortVolume: {dv.ticker} short pct={dv.short_pct:.2%}")

# Test 4: FTDRecord — uses quantity not fails_to_deliver
ftd = FTDRecord(
    ticker="BBBY",
    settlement_date=date(2022, 8, 10),
    quantity=5_000_000,
    price=6.50,
    source="sec_ftd"
)
assert ftd.quantity == 5_000_000
assert ftd.price == 6.50
print(f"[OK] FTDRecord: {ftd.ticker} qty={ftd.quantity:,}")

# Test 5: SqueezeCandidateRecord — uses as_of and rank
sq = SqueezeCandidateRecord(
    ticker="GME",
    as_of=date(2021, 1, 25),
    squeeze_score=score1,
    days_to_cover=10.0,
    short_pct_float=0.50,
    float_shares=50_000_000,
    recent_price_chg=-15.0,
    rank=1
)
assert sq.rank == 1
assert sq.squeeze_score == 500.0
print(f"[OK] SqueezeCandidateRecord: {sq.ticker} score={sq.squeeze_score}, rank={sq.rank}")

print("[PASS]")
PYEOF
