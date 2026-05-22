#!/usr/bin/env bash
# dim_126: Executive Compensation Benchmarking
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import numpy as np

from sentinel.spm.exec_compensation_v3 import (
    ExecComp, PeerGroup, CompBenchmark, CompensationBenchmarker,
    EquityDilutionAnalyzer,
    ceo_pay_ratio, pay_percentile, pay_performance_correlation, say_on_pay_risk,
)

FAILS = []

def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  [FAIL] " + msg)
    else:
        print("  [OK]   " + msg)

# ---------------------------------------------------------------------------
# CEO comp: base=1M, bonus=2M, stock=8M, options=4M  -> total=15M
# ---------------------------------------------------------------------------
print("\n--- ExecComp ---")
ceo = ExecComp(
    name="Jane Smith",
    role="CEO",
    year=2024,
    base_salary=1_000_000,
    annual_bonus=2_000_000,
    stock_awards=8_000_000,
    option_awards=4_000_000,
    other=0.0,
)
print(f"  total = ${ceo.total:,.0f}")
check(abs(ceo.total - 15_000_000) < 1,
      f"CEO total == 15M (got {ceo.total:,.0f})")
check(0 < ceo.equity_fraction < 1,
      f"equity_fraction in (0,1) (got {ceo.equity_fraction:.3f})")
check(0 < ceo.at_risk_fraction < 1,
      f"at_risk_fraction in (0,1) (got {ceo.at_risk_fraction:.3f})")

# ---------------------------------------------------------------------------
# Peer group: pays=[10M, 12M, 14M, 15M, 16M, 18M, 20M], median=14.5M?
# We pass the company's own pay to pay_percentile separately.
# ---------------------------------------------------------------------------
print("\n--- Peer benchmarking ---")
peer_pays = [10e6, 12e6, 14e6, 15e6, 16e6, 18e6, 20e6]
peers = PeerGroup(
    companies=["Co1","Co2","Co3","Co4","Co5","Co6","Co7"],
    ceo_pays=peer_pays,
    tsr_5yr=[5.0, 8.0, -2.0, 12.0, 6.0, 4.0, 9.0],
    ebitda_growth=[3.0, 5.0, -1.0, 8.0, 4.0, 2.0, 6.0],
    median_employee_pay=50_000.0,
)

benchmarker = CompensationBenchmarker(ceo, peers)

pctile = benchmarker.percentile()
print(f"  pay_percentile = {pctile:.1f}")
check(50 < pctile < 75,
      f"percentile in (50, 75) (got {pctile:.1f}): company at $15M, peer median ~$15M")

premium = benchmarker.pay_premium()
print(f"  pay_premium = {premium:.1f}%")
check(-10 < premium < 50,
      f"pay_premium in (-10, 50) (got {premium:.1f}%)")

# ---------------------------------------------------------------------------
# Pay-for-performance
# TSR series=[5,8,-2,12,6]%, CEO pay=[10,11,9,14,15]M
# ---------------------------------------------------------------------------
print("\n--- Pay-for-performance ---")
pay_series = np.array([10e6, 11e6, 9e6, 14e6, 15e6])
tsr_series = np.array([5.0, 8.0, -2.0, 12.0, 6.0])
r = pay_performance_correlation(pay_series, tsr_series)
print(f"  pay_performance_correlation r = {r:.4f}")
check(-1 <= r <= 1, f"pay_performance_correlation in [-1, 1] (got {r:.4f})")

pf = benchmarker.pay_for_performance(company_tsr_5yr=6.0)
print(f"  pay_for_performance = {pf}")
check(-1 <= pf['r_squared'] <= 1,
      f"r_squared in [-1, 1] (got {pf['r_squared']:.4f})")
check(isinstance(pf['excess_pay'], float),
      f"excess_pay is float (got {type(pf['excess_pay']).__name__})")
check(isinstance(pf['aligned'], bool),
      "aligned is bool")

# ---------------------------------------------------------------------------
# CEO pay ratio: 15M / 50000 = 300
# ---------------------------------------------------------------------------
print("\n--- CEO pay ratio ---")
ratio = ceo_pay_ratio(15_000_000, 50_000)
print(f"  CEO pay ratio = {ratio:.1f}")
check(ratio >= 200,
      f"CEO pay ratio >= 200 (got {ratio:.1f}): 15M / 50k = 300")

# ---------------------------------------------------------------------------
# EquityDilutionAnalyzer
# options=5M, restricted=2M, shares=100M -> dilution < 0.1
# ---------------------------------------------------------------------------
print("\n--- Equity dilution ---")
eda = EquityDilutionAnalyzer()
dilution = eda.dilution(options=5e6, restricted=2e6, shares_outstanding=100e6)
print(f"  dilution = {dilution:.4f}")
check(dilution < 0.1,
      f"dilution < 0.1 (got {dilution:.4f}): (5+2)/(100+5+2) = 7/107 ~= 0.065")
check(dilution > 0,
      f"dilution > 0 (got {dilution:.4f})")

overhang = eda.overhang(unvested=7e6, diluted_shares=107e6)
print(f"  overhang = {overhang:.4f}")
check(0 < overhang < 1, f"overhang in (0,1) (got {overhang:.4f})")

burn = eda.burn_rate(new_grants=1e6, shares_outstanding=100e6)
print(f"  burn_rate = {burn:.4f}")
check(0 < burn < 1, f"burn_rate in (0,1) (got {burn:.4f})")

# ---------------------------------------------------------------------------
# say_on_pay_risk module-level function
# ---------------------------------------------------------------------------
print("\n--- say_on_pay_risk (module function) ---")
sop = say_on_pay_risk(excess_pay_z=0.5, tsr_z=-0.3, pay_ratio_z=1.2)
print(f"  say_on_pay_risk = {sop:.4f}")
check(0 <= sop <= 1, f"say_on_pay_risk in [0,1] (got {sop:.4f})")

sop_low  = say_on_pay_risk(excess_pay_z=-2.0, tsr_z=-1.0, pay_ratio_z=-1.5)
sop_high = say_on_pay_risk(excess_pay_z=3.0,  tsr_z=2.0,  pay_ratio_z=2.5)
check(sop_low < sop_high,
      f"higher z-scores -> higher SOP risk ({sop_low:.4f} < {sop_high:.4f})")

# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------
print("\n--- CompensationBenchmarker.full_benchmark ---")
bench = benchmarker.full_benchmark(company_tsr_5yr=6.0)
check(isinstance(bench, CompBenchmark), "full_benchmark returns CompBenchmark")
check(0 <= bench.pay_percentile <= 100,
      f"pay_percentile in [0,100] (got {bench.pay_percentile:.1f})")
check(0 <= bench.say_on_pay_risk <= 1,
      f"say_on_pay_risk in [0,1] (got {bench.say_on_pay_risk:.4f})")
check(bench.sop_recommendation in ("FOR", "AGAINST"),
      f"sop_recommendation is FOR or AGAINST (got '{bench.sop_recommendation}')")
check(bench.ceo_pay_ratio >= 200,
      f"bench.ceo_pay_ratio >= 200 (got {bench.ceo_pay_ratio:.1f})")

# pay_percentile convenience
pctile2 = pay_percentile(15e6, peer_pays)
check(50 < pctile2 < 75,
      f"pay_percentile convenience in (50, 75) (got {pctile2:.1f})")

# Final result
print()
if FAILS:
    print("[FAIL] dim_126: %d check(s) failed:" % len(FAILS))
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
else:
    print("[PASS] dim_126: Executive compensation benchmarking")
PYEOF
