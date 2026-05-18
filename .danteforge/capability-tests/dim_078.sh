#!/usr/bin/env bash
# dim_078: BHB Attribution — pure math
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

print("\n[PASS] dim_078: BHB Attribution")
PYEOF
