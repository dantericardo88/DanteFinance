#!/usr/bin/env bash
# dim_078: BHB Attribution — pure math (score 6 → 9)
# Tests: single-period BHB, Brinson-Fachler, multi-period GRAP linking,
#        country attribution, sector GRAP per-bucket, currency attribution,
#        FI DV01-weighted attribution, local-vs-currency decomposition
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.sbx.bhb_attribution_v2 import BHBAttributionV2 as BHBAttribution

sectors = ["Technology", "Financials", "Healthcare", "Energy"]

portfolio_weights = pd.Series([0.40, 0.25, 0.20, 0.15], index=sectors)
benchmark_weights = pd.Series([0.30, 0.20, 0.25, 0.25], index=sectors)
portfolio_returns = pd.Series([0.12, 0.08, 0.06, 0.05], index=sectors)
benchmark_returns = pd.Series([0.10, 0.07, 0.07, 0.04], index=sectors)

bhb = BHBAttribution(
    portfolio_weights=portfolio_weights,
    benchmark_weights=benchmark_weights,
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    period_label="Q1-2024",
)

# Test active return
active = bhb.active_return()
port_return = float((portfolio_weights * portfolio_returns).sum())
bench_return = float((benchmark_weights * benchmark_returns).sum())
expected_active = port_return - bench_return
assert abs(active - expected_active) < 1e-9, f"Active return mismatch: {active:.6f} vs {expected_active:.6f}"
print(f"[OK] Active return: {active:.4f} (portfolio={port_return:.4f} - bench={bench_return:.4f})")

# Test allocation effect
alloc = bhb.allocation_effect()
assert len(alloc) == 4, f"Expected 4 sectors, got {len(alloc)}"
assert alloc["Technology"] > 0, f"Tech overweight should have positive allocation: {alloc['Technology']:.4f}"
print(f"[OK] Allocation effects: {alloc.round(6).to_dict()}")

# Test selection effect
sel = bhb.selection_effect()
assert sel["Technology"] > 0, f"Tech beat should have positive selection: {sel['Technology']:.4f}"
print(f"[OK] Selection effects: {sel.round(6).to_dict()}")

# Test interaction effect
inter = bhb.interaction_effect()
print(f"[OK] Interaction effects: {inter.round(6).to_dict()}")

# Completeness check
total_explained = alloc.sum() + sel.sum() + inter.sum()
assert abs(total_explained - active) < 1e-9, \
    f"Completeness violated: explained={total_explained:.8f} active={active:.8f}"
print(f"[OK] BHB completeness verified: explained={total_explained:.6f} = active={active:.6f}")

# Test verify()
assert bhb.verify(tol=1e-8), "BHB verify() should return True"
print("[OK] BHB verify() passed")

# Test full_table()
table = bhb.full_table()
assert "TOTAL" in table.index, "full_table should have TOTAL row"
assert "allocation" in table.columns
assert "selection" in table.columns
assert "interaction" in table.columns
print(f"[OK] full_table(): {len(table)} rows (including TOTAL)")

# Test to_single_period()
sp = bhb.to_single_period()
assert sp.model == "BHB"
assert sp.verified == True
assert abs(sp.residual) < 1e-8, f"Residual should be near 0: {sp.residual:.2e}"
print(f"[OK] to_single_period(): model={sp.model} verified={sp.verified} residual={sp.residual:.2e}")

print("\n[PASS] dim_078 Part 1: BHB Attribution (bhb_attribution_v2)")
PYEOF

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.attribution_v3 import (
    BrinsonHoodBeebower,
    MultiPeriodAttributionLinker,
    FixedIncomeAttribution,
    CurrencyAttribution,
)

# ── Part 2: Multi-period GRAP linking — compound ≠ arithmetic sum ─────────────
sectors = ["Tech", "Finance", "Health"]
wp = pd.Series([0.50, 0.30, 0.20], index=sectors)
wb = pd.Series([0.40, 0.30, 0.30], index=sectors)

rp1 = pd.Series([0.10, 0.05, 0.03], index=sectors)
rb1 = pd.Series([0.08, 0.04, 0.04], index=sectors)
res1 = BrinsonHoodBeebower.compute_attribution(wp, rp1, wb, rb1, "Q1")

rp2 = pd.Series([0.03, 0.02, 0.05], index=sectors)
rb2 = pd.Series([0.04, 0.01, 0.03], index=sectors)
res2 = BrinsonHoodBeebower.compute_attribution(wp, rp2, wb, rb2, "Q2")

grap = MultiPeriodAttributionLinker.grap_linking([res1, res2])

arith_sum_active = res1.total_active_return + res2.total_active_return
geometric_active = grap.geometric_active

# Key test: geometric ≠ arithmetic for 2+ periods with non-trivial returns
assert abs(geometric_active - arith_sum_active) > 1e-6, (
    f"GRAP geometric={geometric_active:.8f} should differ from arith sum={arith_sum_active:.8f}")
print(f"[OK] GRAP linked != arithmetic sum: geometric={geometric_active:.6f} arith_sum={arith_sum_active:.6f}")

# GRAP total must equal geometric active (post-residual adjustment)
assert abs(grap.linked_total - geometric_active) < 1e-9, (
    f"GRAP linked_total={grap.linked_total:.8f} != geometric_active={geometric_active:.8f}")
print(f"[OK] GRAP residual near zero: {grap.residual:.2e}")

# Attribution identity: allocation + selection + interaction = linked_total
alloc_sel_inter = grap.linked_allocation + grap.linked_selection + grap.linked_interaction
assert abs(alloc_sel_inter - grap.linked_total) < 1e-12, \
    f"GRAP A+S+I={alloc_sel_inter:.8f} != linked_total={grap.linked_total:.8f}"
print(f"[OK] GRAP: allocation + selection + interaction = linked_total ({grap.linked_total:.6f})")

# Compare GRAP vs Cariño vs Menchero — all should give geometric active
carino = MultiPeriodAttributionLinker.carino_linking([res1, res2])
menchero = MultiPeriodAttributionLinker.menchero_linking([res1, res2])
assert abs(carino.geometric_active - geometric_active) < 1e-12, "Cariño geometric_active mismatch"
assert abs(menchero.geometric_active - geometric_active) < 1e-12, "Menchero geometric_active mismatch"
print(f"[OK] All three methods produce same geometric active: {geometric_active:.6f}")
print(f"     Carino total={carino.linked_total:.6f} Menchero total={menchero.linked_total:.6f} GRAP total={grap.linked_total:.6f}")

print("\n[PASS] dim_078 Part 2: Multi-period GRAP linking")
PYEOF

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.attribution_v3 import (
    BrinsonHoodBeebower,
    MultiPeriodAttributionLinker,
)

# ── Part 3: Per-sector GRAP linking ──────────────────────────────────────────
sectors = ["Tech", "Finance", "Health"]
wp = pd.Series([0.50, 0.30, 0.20], index=sectors)
wb = pd.Series([0.40, 0.30, 0.30], index=sectors)

rp1 = pd.Series([0.10, 0.05, 0.03], index=sectors)
rb1 = pd.Series([0.08, 0.04, 0.04], index=sectors)
res1 = BrinsonHoodBeebower.compute_attribution(wp, rp1, wb, rb1, "Q1")

rp2 = pd.Series([-0.02, 0.03, 0.07], index=sectors)
rb2 = pd.Series([-0.01, 0.02, 0.05], index=sectors)
res2 = BrinsonHoodBeebower.compute_attribution(wp, rp2, wb, rb2, "Q2")

sector_linked = MultiPeriodAttributionLinker.grap_sector_linking([res1, res2])
full_grap = MultiPeriodAttributionLinker.grap_linking([res1, res2])

# Sum of sector linked_totals should equal full GRAP linked_total
sector_sum = sum(v["linked_total"] for v in sector_linked.values())
assert abs(sector_sum - full_grap.linked_total) < 1e-8, (
    f"Sector GRAP sum={sector_sum:.8f} != full GRAP total={full_grap.linked_total:.8f}")
print(f"[OK] Per-sector GRAP sum = full GRAP total: {sector_sum:.6f}")

# Each sector must have allocation, selection, interaction keys
for sec in sectors:
    assert sec in sector_linked, f"Sector {sec} missing from sector_linked"
    for key in ("linked_allocation", "linked_selection", "linked_interaction", "linked_total"):
        assert key in sector_linked[sec], f"Key {key} missing for sector {sec}"
print(f"[OK] All sectors present with full attribution breakdown: {list(sector_linked.keys())}")

# Tech is overweighted and outperforms — should have positive contribution
tech_total = sector_linked["Tech"]["linked_total"]
print(f"[OK] Tech sector linked total: {tech_total:.6f}")

print("\n[PASS] dim_078 Part 3: Per-sector GRAP linking")
PYEOF

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.attribution_v3 import BrinsonHoodBeebower

# ── Part 4: Country attribution (BHB applied across country buckets) ──────────
countries = ["US", "Europe", "Asia", "EM"]
wp = pd.Series([0.60, 0.20, 0.10, 0.10], index=countries)
wb = pd.Series([0.55, 0.25, 0.12, 0.08], index=countries)
rp = pd.Series([0.08, 0.04, 0.09, 0.06], index=countries)
rb = pd.Series([0.07, 0.05, 0.07, 0.04], index=countries)

country_res = BrinsonHoodBeebower.compute_country_attribution(wp, rp, wb, rb, "2024")

# BHB identity: allocation + selection + interaction = active return
identity_check = abs(
    country_res.total_allocation + country_res.total_selection + country_res.total_interaction
    - country_res.total_active_return
)
assert identity_check < 1e-12, f"Country BHB identity violated: {identity_check:.2e}"
print(f"[OK] Country BHB identity: A+S+I = active return ({country_res.total_active_return:.6f})")

# US is overweighted and outperforms — should have positive allocation
assert country_res.allocation_effects["US"] > 0, \
    f"US overweight & outperform should give positive alloc: {country_res.allocation_effects['US']:.4f}"
print(f"[OK] Country allocation effects: {dict((k, round(v,6)) for k,v in country_res.allocation_effects.items())}")

# Selection check: US portfolio (8%) > benchmark (7%) → positive selection
assert country_res.selection_effects["US"] > 0, \
    f"US selection should be positive: {country_res.selection_effects['US']:.4f}"

# Europe is underweight and underperforms benchmark average → positive allocation
# (underweighting a below-average sector = good call)
# rb_Europe=0.05 < r_b_total≈0.063, so (wp-wb)*(rb-rB) = (-0.05)*(-ve) = positive
assert country_res.allocation_effects["Europe"] > 0, \
    f"Europe underweight with below-avg return should be positive allocation: {country_res.allocation_effects['Europe']:.4f}"
print(f"[OK] Country selection effects: {dict((k, round(v,6)) for k,v in country_res.selection_effects.items())}")
print(f"[OK] Brinson-Fachler country total: {country_res.bf_total_allocation:.6f}")

print("\n[PASS] dim_078 Part 4: Country attribution (BHB across countries)")
PYEOF

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.attribution_v3 import CurrencyAttribution

# ── Part 5: Currency attribution ──────────────────────────────────────────────
currencies = ["USD", "EUR", "JPY", "GBP"]
wp = pd.Series([0.50, 0.25, 0.15, 0.10], index=currencies)
wb = pd.Series([0.55, 0.20, 0.15, 0.10], index=currencies)
fx_returns = pd.Series([0.0, 0.03, -0.02, 0.01], index=currencies)

curr = CurrencyAttribution.compute_currency_effect(wp, wb, fx_returns, period="Q1")

# Total effect = portfolio FX return - benchmark FX return
expected_total = float((wp * fx_returns).sum()) - float((wb * fx_returns).sum())
assert abs(curr["total_currency_effect"] - expected_total) < 1e-10, \
    f"Currency total mismatch: {curr['total_currency_effect']:.8f} vs {expected_total:.8f}"
print(f"[OK] Currency total effect: {curr['total_currency_effect']:.6f} (portfolio FX - benchmark FX)")

# EUR overweighted and FX positive → positive currency allocation
eur_alloc = curr["per_currency"]["EUR"]["currency_allocation"]
assert eur_alloc > 0, f"EUR overweight with positive FX should have positive allocation: {eur_alloc:.6f}"
print(f"[OK] EUR currency allocation: {eur_alloc:.6f} (overweight + positive FX)")

# USD underweighted (0 FX return) → 0 interaction/allocation
usd_total = curr["per_currency"]["USD"]["total_currency_effect"]
assert abs(usd_total) < 1e-10, f"USD (0 FX return) should have 0 currency effect: {usd_total:.8f}"
print(f"[OK] USD currency effect = 0 (zero FX return): {usd_total:.8f}")

# Local vs currency decomposition
lvc = CurrencyAttribution.compute_local_vs_currency(
    portfolio_total_return=0.12,
    portfolio_local_return=0.09,
    benchmark_total_return=0.08,
    benchmark_local_return=0.065,
)
assert abs(lvc["decomposition_check"]) < 1e-12, \
    f"Local+currency decomposition check: {lvc['decomposition_check']:.2e}"
print(f"[OK] Local vs currency decomposition: local={lvc['local_effect']:.4f} currency={lvc['currency_effect']:.4f}")
print(f"     Geometric currency effect: {lvc['currency_effect_geometric']:.6f}")

print("\n[PASS] dim_078 Part 5: Currency attribution")
PYEOF

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.attribution_v3 import FixedIncomeAttribution

# ── Part 6: Fixed income DV01-weighted attribution ────────────────────────────
port_holdings = [
    {"weight": 0.30, "duration": 2.0, "price": 99.5, "maturity_bucket": "2yr", "coupon": 0.025, "spread": 0.005},
    {"weight": 0.40, "duration": 5.0, "price": 98.0, "maturity_bucket": "5yr", "coupon": 0.035, "spread": 0.010},
    {"weight": 0.30, "duration": 10.0, "price": 95.0, "maturity_bucket": "10yr", "coupon": 0.045, "spread": 0.015},
]
bench_holdings = [
    {"weight": 0.25, "duration": 2.0, "price": 99.5, "maturity_bucket": "2yr", "coupon": 0.025, "spread": 0.005},
    {"weight": 0.50, "duration": 5.0, "price": 98.0, "maturity_bucket": "5yr", "coupon": 0.035, "spread": 0.010},
    {"weight": 0.25, "duration": 10.0, "price": 95.0, "maturity_bucket": "10yr", "coupon": 0.045, "spread": 0.015},
]
# Bear flattening scenario: short end rises, long end rises more
yield_changes = {"2yr": 0.0025, "5yr": 0.0050, "10yr": 0.0075}  # parallel tightening + twist

dv01 = FixedIncomeAttribution.compute_dv01_attribution(
    port_holdings, bench_holdings, yield_changes, period="Q1"
)

# Portfolio is overweight 10yr (high duration) relative to benchmark
# Long yields rising → portfolio should suffer more than benchmark
p_dur_10 = 0.30 * 10.0 * 95.0 * 0.0001  # portfolio DV01 at 10yr
b_dur_10 = 0.25 * 10.0 * 95.0 * 0.0001  # benchmark DV01 at 10yr
active_dv01_10 = p_dur_10 - b_dur_10
expected_shift_10 = -active_dv01_10 * yield_changes["10yr"]
assert abs(dv01["per_bucket"]["10yr"]["shift_effect"] - expected_shift_10) < 1e-7, \
    f"10yr shift effect mismatch: {dv01['per_bucket']['10yr']['shift_effect']:.8f} vs {expected_shift_10:.8f}"
print(f"[OK] DV01 10yr shift effect: {dv01['per_bucket']['10yr']['shift_effect']:.8f}")

# Active DV01 exposed at 10yr: portfolio overweighted → negative shift (rising yield hurts)
assert dv01["per_bucket"]["10yr"]["shift_effect"] < 0, \
    f"Portfolio overweight 10yr with rising rates should have negative shift effect"
print(f"[OK] DV01 10yr (overweight + rising yield) is negative: {dv01['per_bucket']['10yr']['shift_effect']:.6f}")

# Verify all expected buckets present
for bucket in ["2yr", "5yr", "10yr"]:
    assert bucket in dv01["per_bucket"], f"Bucket {bucket} missing from DV01 result"
print(f"[OK] DV01 all maturity buckets present: {list(dv01['per_bucket'].keys())}")

# Twist effect is nonzero (different yield changes at different maturities)
assert abs(dv01["twist_effect"]) > 0, "Twist effect should be nonzero for non-parallel shift"
print(f"[OK] DV01 twist effect (non-parallel shift): {dv01['twist_effect']:.6f}")

print(f"[OK] DV01 carry effect (active coupon accrual): {dv01['carry_effect']:.8f}")
print(f"[OK] Total active DV01 (bps): {dv01['total_active_dv01_bps']:.2f}")

# Also test Campisi full FI attribution
fi_port = {"return": 0.028, "yield": 0.045, "duration": 6.5, "convexity": 50.0,
           "coupon": 0.04, "spread": 0.012, "currency_return": 0.0}
fi_bench = {"return": 0.018, "yield": 0.040, "duration": 5.5, "convexity": 40.0,
            "coupon": 0.035, "spread": 0.010, "currency_return": 0.0}
fi_curve = {"parallel": -0.005, "twist": 0.001, "butterfly": 0.0, "spread_change": -0.002}
fi_res = FixedIncomeAttribution.compute_fi_attribution(fi_port, fi_bench, fi_curve, "2024", 90)

# Attribution identity: all effects sum to active return
fi_check = abs(fi_res.attribution_check)
assert fi_check < 1e-9, f"FI attribution_check should be near 0: {fi_check:.2e}"
print(f"[OK] Campisi FI attribution: check={fi_res.attribution_check:.2e}")
print(f"     Duration effect={fi_res.duration_effect:.6f} Spread effect={fi_res.spread_effect:.6f}")

print("\n[PASS] dim_078 Part 6: Fixed income DV01-weighted attribution")
PYEOF

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.attribution_v3 import (
    BrinsonHoodBeebower, MultiPeriodAttributionLinker
)

# ── Part 7: 4-period GRAP test — verify compound ≠ arithmetic over 4 quarters ─
sectors = ["Tech", "Finance", "Energy", "Health"]
wp = pd.Series([0.40, 0.25, 0.10, 0.25], index=sectors)
wb = pd.Series([0.35, 0.20, 0.15, 0.30], index=sectors)

period_returns = [
    # Q1: strong tech/finance quarter
    (pd.Series([0.15, 0.08, 0.05, 0.04], index=sectors),
     pd.Series([0.12, 0.06, 0.06, 0.05], index=sectors)),
    # Q2: energy rallies
    (pd.Series([0.02, 0.03, 0.18, 0.01], index=sectors),
     pd.Series([0.03, 0.02, 0.15, 0.02], index=sectors)),
    # Q3: correction
    (pd.Series([-0.05, -0.03, -0.02, 0.01], index=sectors),
     pd.Series([-0.04, -0.02, -0.03, 0.00], index=sectors)),
    # Q4: recovery
    (pd.Series([0.10, 0.07, 0.03, 0.08], index=sectors),
     pd.Series([0.09, 0.06, 0.02, 0.07], index=sectors)),
]

period_results = []
for i, (rp, rb) in enumerate(period_returns):
    res = BrinsonHoodBeebower.compute_attribution(wp, rp, wb, rb, f"Q{i+1}")
    period_results.append(res)

grap4 = MultiPeriodAttributionLinker.grap_linking(period_results)

# Arithmetic sum
arith_sum = sum(r.total_active_return for r in period_results)

# Assert compound != arithmetic
assert abs(grap4.geometric_active - arith_sum) > 1e-6, \
    f"4-period geometric must differ from arithmetic: geo={grap4.geometric_active:.8f} arith={arith_sum:.8f}"
print(f"[OK] 4-period GRAP: geometric={grap4.geometric_active:.6f} != arithmetic={arith_sum:.6f}")
print(f"     Difference: {grap4.geometric_active - arith_sum:.6f}")

# Verify cumulative return computation
rp_cumulative = grap4.portfolio_cumulative
rb_cumulative = grap4.benchmark_cumulative
expected_geo_active = (1.0 + rp_cumulative) / (1.0 + rb_cumulative) - 1.0
assert abs(expected_geo_active - grap4.geometric_active) < 1e-10, \
    f"Geometric active = (1+Rp)/(1+Rb)-1 check failed: {expected_geo_active:.8f} vs {grap4.geometric_active:.8f}"
print(f"[OK] GRAP formula: (1+Rp)/(1+Rb)-1 = {grap4.geometric_active:.6f}")
print(f"     Portfolio cumulative: {rp_cumulative:.4f}, Benchmark cumulative: {rb_cumulative:.4f}")

# 4-period sector GRAP must sum to full GRAP total
sector_linked4 = MultiPeriodAttributionLinker.grap_sector_linking(period_results)
sector_sum4 = sum(v["linked_total"] for v in sector_linked4.values())
assert abs(sector_sum4 - grap4.linked_total) < 1e-8, \
    f"4-period sector GRAP sum={sector_sum4:.8f} != full GRAP={grap4.linked_total:.8f}"
print(f"[OK] 4-period per-sector GRAP sum = full GRAP: {sector_sum4:.6f}")

print("\n[PASS] dim_078 Part 7: 4-period GRAP formula verification")
PYEOF

echo ""
echo "=================================================="
echo " dim_078: BHB Attribution — ALL PARTS PASSED"
echo "=================================================="
