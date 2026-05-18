#!/bin/bash
# dim_022: PIT integrity — STATUTORY_DEADLINES, FilingRecord, LookAheadFlag, ScanResult
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import date, timedelta

from sentinel.sfe.pit_integrity_v3 import (
    STATUTORY_DEADLINES, EARNINGS_FORMS, NT_FORMS,
    FilingRecord, LookAheadFlag, ScanResult, SP500Member,
    LagModelResult,
)

# Test 1: STATUTORY_DEADLINES — regulatory deadlines
assert "large_accelerated" in STATUTORY_DEADLINES
assert "non_accelerated" in STATUTORY_DEADLINES
assert STATUTORY_DEADLINES["large_accelerated"]["10-K"] == 60
assert STATUTORY_DEADLINES["large_accelerated"]["10-Q"] == 40
assert STATUTORY_DEADLINES["non_accelerated"]["10-K"] == 90
assert STATUTORY_DEADLINES["non_accelerated"]["10-Q"] == 45
print(f"[OK] STATUTORY_DEADLINES: large_accel 10-K={STATUTORY_DEADLINES['large_accelerated']['10-K']}d, non_accel 10-K={STATUTORY_DEADLINES['non_accelerated']['10-K']}d")

# Test 2: EARNINGS_FORMS and NT_FORMS
assert "10-Q" in EARNINGS_FORMS and "10-K" in EARNINGS_FORMS and "20-F" in EARNINGS_FORMS
assert "NT 10-Q" in NT_FORMS and "NT 10-K" in NT_FORMS
print(f"[OK] EARNINGS_FORMS={len(EARNINGS_FORMS)}, NT_FORMS={len(NT_FORMS)}")

# Test 3: FilingRecord model
fr = FilingRecord(
    ticker="AAPL",
    cik="0000320193",
    form="10-K",
    period_end="2023-09-30",
    filed_date="2023-11-03",
    report_date="2023-09-30",
    lag_days=34,
    filer_category="large_accelerated",
    statutory_deadline_days=60,
    is_late=False
)
# Key PIT check: data from period_end 2023-09-30 wasn't available until 2023-11-03
assert fr.lag_days == 34
assert fr.is_late is False  # 34 days < 60 day deadline
print(f"[OK] FilingRecord: {fr.ticker} {fr.form} lag={fr.lag_days}d, late={fr.is_late}")

# Test 4: LookAheadBiasScanner logic (pure simulation)
# Scenario: backtest uses Q3 earnings data dated 2023-09-30, as_of 2023-10-15
# Filing date = 2023-11-03 -> LOOK-AHEAD! Data wasn't available yet.
data_date = date(2023, 9, 30)   # when the data refers to
as_of_date = date(2023, 10, 15)  # backtest checkpoint
filed_date = date(2023, 11, 3)   # actual filing
look_ahead = filed_date > as_of_date  # used data before it was filed
lag_days_behind = (filed_date - as_of_date).days if look_ahead else 0

laf = LookAheadFlag(
    row_index=42,
    ticker="AAPL",
    metric="revenue",
    data_date=data_date.isoformat(),
    as_of_date=as_of_date.isoformat(),
    filed_date=filed_date.isoformat(),
    lag_days_behind=lag_days_behind,
    look_ahead_pct=1.0  # 100% of the row is look-ahead
)
assert laf.lag_days_behind == 19, f"Expected 19 days look-ahead, got {laf.lag_days_behind}"
print(f"[OK] LookAheadFlag: AAPL Q3 data used {laf.lag_days_behind}d before filing, row={laf.row_index}")

# Test 5: ScanResult
sr = ScanResult(
    total_rows=100,
    flagged_rows=15,
    clean_rows=85,
    look_ahead_pct=15.0,
    flags=[laf],
    summary="15 of 100 rows used data before it was publicly filed"
)
assert sr.look_ahead_pct == 15.0
assert sr.clean_rows == sr.total_rows - sr.flagged_rows
print(f"[OK] ScanResult: {sr.look_ahead_pct}% look-ahead, {sr.clean_rows} clean rows")

# Test 6: SP500Member model
member = SP500Member(
    ticker="NVDA",
    name="NVIDIA Corporation",
    added_date="1999-11-01",
    is_current=True,
    sector="Information Technology",
    sub_industry="Semiconductors"
)
assert member.is_current is True
assert member.removed_date is None
print(f"[OK] SP500Member: {member.ticker} in S&P500, sector={member.sector}")

print("[PASS]")
PYEOF
