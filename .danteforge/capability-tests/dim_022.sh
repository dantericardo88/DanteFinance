#!/bin/bash
# dim_022: PIT integrity — STATUTORY_DEADLINES, FilingRecord, LookAheadFlag, ScanResult
set -e
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import date, timedelta

from sentinel.sfe.pit_integrity_v3 import (
    STATUTORY_DEADLINES, EARNINGS_FORMS, NT_FORMS,
    FilingRecord, LookAheadFlag, ScanResult, SP500Member,
    LagModelResult,
    compute_stale_data_penalty,
    validate_fiscal_year_consistency,
    flag_restatement_risk,
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

# --------------------------------------------------------------------------
# NEW: Test 7 — compute_stale_data_penalty math
# --------------------------------------------------------------------------
# Case: exactly 90 days → still within grace period → penalty = 0
r0 = compute_stale_data_penalty(90)
assert r0["penalty"] == 0.0, f"Expected 0.0 at 90 days, got {r0['penalty']}"
assert r0["is_stale"] is False
assert r0["blocks_past_grace"] == 0
print(f"[OK] compute_stale_data_penalty(90): penalty={r0['penalty']}, is_stale={r0['is_stale']}")

# Case: 91 days → 1 day past grace, first 30-day block NOT complete → 0 blocks → penalty = 0
r1 = compute_stale_data_penalty(91)
assert r1["is_stale"] is True
assert r1["blocks_past_grace"] == 0, f"Expected 0 complete blocks at 91d, got {r1['blocks_past_grace']}"
assert r1["penalty"] == 0.0
print(f"[OK] compute_stale_data_penalty(91): penalty={r1['penalty']} (first block not yet complete)")

# Case: 90 + 30 = 120 days → exactly 1 complete block → penalty = 0.1
r2 = compute_stale_data_penalty(120)
assert r2["blocks_past_grace"] == 1, f"Expected 1 block at 120d, got {r2['blocks_past_grace']}"
assert abs(r2["penalty"] - 0.1) < 1e-9, f"Expected 0.1, got {r2['penalty']}"
assert r2["is_stale"] is True
print(f"[OK] compute_stale_data_penalty(120): penalty={r2['penalty']:.1f} (1 block × 0.1)")

# Case: 90 + 60 = 150 days → 2 complete blocks → penalty = 0.2
r3 = compute_stale_data_penalty(150)
assert r3["blocks_past_grace"] == 2
assert abs(r3["penalty"] - 0.2) < 1e-9, f"Expected 0.2, got {r3['penalty']}"
print(f"[OK] compute_stale_data_penalty(150): penalty={r3['penalty']:.1f} (2 blocks × 0.1)")

# Case: 0 days → fresh
r4 = compute_stale_data_penalty(0)
assert r4["penalty"] == 0.0
assert r4["is_stale"] is False
print(f"[OK] compute_stale_data_penalty(0): fresh data, penalty=0.0")

# Negative days must raise
try:
    compute_stale_data_penalty(-1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("[OK] compute_stale_data_penalty(-1): raises ValueError")

# --------------------------------------------------------------------------
# NEW: Test 8 — validate_fiscal_year_consistency
# --------------------------------------------------------------------------
filings_consistent = [
    {"ticker": "AAPL", "period_end": "2022-09-30"},
    {"ticker": "AAPL", "period_end": "2023-09-30"},
    {"ticker": "AAPL", "period_end": "2024-09-30"},
]
vc = validate_fiscal_year_consistency(filings_consistent)
assert vc["is_consistent"] is True
assert vc["fiscal_month_end"] == 9   # September
assert vc["inconsistent_periods"] == []
print(f"[OK] validate_fiscal_year_consistency: AAPL consistent, month={vc['fiscal_month_end']}")

filings_inconsistent = [
    {"ticker": "ODD", "period_end": "2022-09-30"},
    {"ticker": "ODD", "period_end": "2023-09-30"},
    {"ticker": "ODD", "period_end": "2024-03-31"},  # different month!
]
vi = validate_fiscal_year_consistency(filings_inconsistent)
assert vi["is_consistent"] is False
assert "2024-03-31" in vi["inconsistent_periods"]
print(f"[OK] validate_fiscal_year_consistency: inconsistency correctly flagged")

# Empty list → consistent (no data = no conflict)
ve = validate_fiscal_year_consistency([])
assert ve["is_consistent"] is True
assert ve["fiscal_month_end"] is None
print(f"[OK] validate_fiscal_year_consistency: empty list handled")

# --------------------------------------------------------------------------
# NEW: Test 9 — flag_restatement_risk
# --------------------------------------------------------------------------
from datetime import datetime
today_str = datetime.utcnow().strftime("%Y-%m-%d")

# 10-K/A filed today → within 2-year window → risk = True
filings_with_amendment = [
    {"form": "10-K",   "filed_date": "2023-11-03"},
    {"form": "10-K/A", "filed_date": today_str},   # amendment today
]
rr = flag_restatement_risk(filings_with_amendment, lookback_years=2)
assert rr["risk_flag"] is True, "10-K/A in window should set risk_flag=True"
assert len(rr["amendment_filings"]) == 1
print(f"[OK] flag_restatement_risk: 10-K/A within 2yr → risk_flag=True")

# No amendment → risk = False
filings_clean = [
    {"form": "10-K", "filed_date": "2023-11-03"},
    {"form": "10-Q", "filed_date": "2024-02-01"},
]
rr2 = flag_restatement_risk(filings_clean, lookback_years=2)
assert rr2["risk_flag"] is False
assert rr2["amendment_filings"] == []
print(f"[OK] flag_restatement_risk: no amendment → risk_flag=False")

# Old 10-K/A (>2 years ago) → risk = False
filings_old = [
    {"form": "10-K/A", "filed_date": "2020-01-15"},  # >2 years ago
]
rr3 = flag_restatement_risk(filings_old, lookback_years=2)
assert rr3["risk_flag"] is False, "Old amendment outside window should not flag"
print(f"[OK] flag_restatement_risk: old 10-K/A outside window → risk_flag=False")

print("[PASS]")
PYEOF
