#!/bin/bash
# dim_008: Corporate actions — Pydantic models, adjustment factor, SplitRecord
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import date

from sentinel.sds.adapters.corporate_actions_v3 import (
    CorporateActionRecord, DividendRecord, SplitRecord,
    SpinoffRecord, MAActionRecord, AdjustmentFactor, UpcomingAction,
)

# Test 1: CorporateActionRecord instantiation
rec = CorporateActionRecord(
    ticker="AAPL",
    action_type="split",
    ex_date=date(2020, 8, 28),
    ratio="4:1",
    factor=0.25,
    source="sec_8937",
    notes="AAPL 4-for-1 split"
)
assert rec.ticker == "AAPL"
assert rec.action_type == "split"
assert rec.factor == 0.25
print("[OK] CorporateActionRecord instantiates correctly")

# Test 2: DividendRecord with pct_change and is_cut flag
div = DividendRecord(
    ticker="T",
    ex_date=date(2022, 4, 8),
    amount=0.2775,
    div_type="cut",
    source="yfinance",
    prev_amount=0.52,
    pct_change=(0.2775 - 0.52) / 0.52 * 100,
    is_cut=True
)
assert div.is_cut is True
assert div.pct_change is not None and div.pct_change < 0
print(f"[OK] DividendRecord: cut={div.is_cut}, pct_change={div.pct_change:.1f}%")

# Test 3: SplitRecord — forward vs reverse, factor computation
# 4-for-1 forward split: new=4, old=1, factor=0.25 (adjust backwards)
split_fwd = SplitRecord(
    ticker="TSLA", ex_date=date(2022, 8, 25),
    ratio_new=3, ratio_old=1, factor=round(1/3, 6),
    split_type="forward", source="yfinance"
)
assert split_fwd.split_type == "forward"
assert abs(split_fwd.factor - 1/3) < 1e-5
# 1-for-20 reverse split: new=1, old=20, factor=20.0
split_rev = SplitRecord(
    ticker="GME", ex_date=date(2023, 7, 20),
    ratio_new=1, ratio_old=20, factor=20.0,
    split_type="reverse", source="yfinance"
)
assert split_rev.split_type == "reverse"
assert split_rev.factor == 20.0
print(f"[OK] SplitRecord: fwd factor={split_fwd.factor:.4f}, rev factor={split_rev.factor}")

# Test 4: AdjustmentFactor validation logic
# Cumulative adjustment: product of factors
factors = [0.25, 0.5]  # two splits
cumulative = 1.0
for f in factors:
    cumulative *= f
assert abs(cumulative - 0.125) < 1e-10
af = AdjustmentFactor(
    ticker="AAPL", as_of_date=date(2020, 8, 28),
    factor=cumulative, source="sec_8937", action_type="split"
)
assert af.factor == 0.125
print(f"[OK] AdjustmentFactor: cumulative={af.factor}")

# Test 5: MAActionRecord
ma = MAActionRecord(
    ticker="ATVI", action_type="merger_completion",
    announcement_date=date(2022, 1, 18),
    expiry_date=date(2023, 10, 13),
    consideration="$95 per share cash",
    acquirer="Microsoft",
    source="edgar"
)
assert ma.acquirer == "Microsoft"
assert ma.action_type == "merger_completion"
print(f"[OK] MAActionRecord: {ma.ticker} acquired by {ma.acquirer}")

print("[PASS]")
PYEOF
