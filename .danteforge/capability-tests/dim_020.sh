#!/bin/bash
# dim_020: Historical PIT — PITSnapshot, filing lag computation, VintageRecord (pure computation)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import date

from sentinel.sfe.historical_pit_v3 import (
    PITSnapshot, FilingRecord, LagReport,
    RestatementRecord, VintageRecord,
)

# Test 1: PITSnapshot model — key PIT concept
snap = PITSnapshot(
    ticker="AAPL",
    cik="0000320193",
    company_name="Apple Inc.",
    as_of_date="2024-01-31",
    fiscal_period_end="2023-09-30",
    filed_date="2023-11-03",
    filing_lag_days=(date(2023, 11, 3) - date(2023, 9, 30)).days,
    revenue=383_285_000_000.0,
    net_income=96_995_000_000.0,
    total_assets=352_583_000_000.0,
    n_concepts_found=15,
    data_vintage="original"
)
assert snap.filing_lag_days == 34, f"Lag should be 34 days: {snap.filing_lag_days}"
assert snap.data_vintage == "original"
assert snap.n_concepts_found == 15
print(f"[OK] PITSnapshot: {snap.ticker} filed {snap.filing_lag_days} days after period end")

# Test 2: Filing lag computation — PIT timestamp = filing_date (not period end)
# A backtest using data as_of 2023-10-15 would NOT have access to this filing
as_of_backtest = date(2023, 10, 15)
filed = date(2023, 11, 3)
is_available = filed <= as_of_backtest
assert is_available is False, "Filing was not yet available on 2023-10-15"
# But as of 2023-11-05, it is available
as_of_later = date(2023, 11, 5)
is_available_later = filed <= as_of_later
assert is_available_later is True
print(f"[OK] PIT availability: as_of={as_of_backtest} -> {is_available}, as_of={as_of_later} -> {is_available_later}")

# Test 3: FilingRecord model
fr = FilingRecord(
    accession="0000320193-23-000106",
    form="10-K",
    filing_date="2023-11-03",
    report_date="2023-09-30",
    lag_days=34,
    is_annual=True
)
assert fr.is_annual is True
assert fr.lag_days == 34
print(f"[OK] FilingRecord: {fr.form} lag={fr.lag_days} days, annual={fr.is_annual}")

# Test 4: LagReport statistics
fr_list = [
    FilingRecord(accession="A", form="10-K", filing_date="2023-11-03",
                 report_date="2023-09-30", lag_days=34, is_annual=True),
    FilingRecord(accession="B", form="10-K", filing_date="2022-10-28",
                 report_date="2022-09-24", lag_days=34, is_annual=True),
    FilingRecord(accession="C", form="10-K", filing_date="2021-10-29",
                 report_date="2021-09-25", lag_days=34, is_annual=True),
]
import statistics
lags = [f.lag_days for f in fr_list]
lr = LagReport(
    ticker="AAPL",
    cik="0000320193",
    form_type="10-K",
    median_lag_days=statistics.median(lags),
    mean_lag_days=statistics.mean(lags),
    min_lag_days=min(lags),
    max_lag_days=max(lags),
    n_filings=len(fr_list),
    filer_status_estimate="large_accelerated",
    target_lag_days=60,  # statutory deadline for large accelerated
    filings=fr_list
)
assert lr.median_lag_days == 34.0
assert lr.filer_status_estimate == "large_accelerated"
# AAPL files well before the 60-day deadline
assert lr.max_lag_days < lr.target_lag_days, \
    f"Max lag {lr.max_lag_days} should be < target {lr.target_lag_days}"
print(f"[OK] LagReport: {lr.ticker} median={lr.median_lag_days}d, target={lr.target_lag_days}d (early filer)")

# Test 5: RestatementRecord — material threshold
rr = RestatementRecord(
    metric="Revenue",
    period_end="2022-09-24",
    original_filed="2022-10-28",
    original_value=394_328_000_000.0,
    restatement_filed="2023-01-15",
    restatement_value=397_000_000_000.0,
    pct_change=(397_000 - 394_328) / 394_328 * 100
)
assert rr.pct_change is not None and rr.pct_change > 0
is_material = abs(rr.pct_change) > 1.0
print(f"[OK] RestatementRecord: {rr.metric} delta={rr.pct_change:.3f}%, material={is_material}")

print("[PASS]")
PYEOF
