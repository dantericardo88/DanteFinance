#!/usr/bin/env bash
# dim_149: Autonomous portfolio rebalancing orchestrator (drift + tax-loss harvest)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from sentinel.sai.agentic_portfolio_v3 import (
    RebalancingRule,
    RebalancingResult,
    DriftMonitor,
    TaxLossHarvester,
    PortfolioRebalancer,
    check_rebalance_needed,
    harvest_tax_losses,
)
print("[OK] All imports successful")

# ---------------------------------------------------------------------------
# 2. DriftMonitor
# ---------------------------------------------------------------------------
monitor = DriftMonitor()

current = np.array([0.40, 0.30, 0.30])
target  = np.array([0.33, 0.33, 0.34])

drift_arr = monitor.drift(current, target)
print(f"[OK] drift array: {drift_arr.round(4)}")
assert drift_arr.shape == (3,), f"FAIL: drift array shape wrong: {drift_arr.shape}"
# Expected drift: [0.07, 0.03, 0.04]
expected_drift = np.abs(current - target)
assert np.allclose(drift_arr, expected_drift, atol=1e-10), (
    f"FAIL: drift values wrong: {drift_arr} vs {expected_drift}"
)

max_d = monitor.max_drift(current, target)
print(f"[OK] max_drift = {max_d:.4f}")
assert abs(max_d - 0.07) < 1e-10, f"FAIL: max_drift should be 0.07, got {max_d:.6f}"
print(f"[OK] max_drift == 0.07 as expected")

# requires_rebalance with 5% threshold --> True (max drift is 7%)
needs = monitor.requires_rebalance(current, target, threshold=0.05)
assert needs is True, (
    f"FAIL: requires_rebalance(threshold=0.05) should be True when drift=0.07"
)
print(f"[OK] requires_rebalance(threshold=0.05) = True (drift=7% > 5%)")

# requires_rebalance with 10% threshold --> False
needs_10 = monitor.requires_rebalance(current, target, threshold=0.10)
assert needs_10 is False, (
    f"FAIL: requires_rebalance(threshold=0.10) should be False when drift=0.07"
)
print(f"[OK] requires_rebalance(threshold=0.10) = False (drift=7% < 10%)")

# ---------------------------------------------------------------------------
# 3. TaxLossHarvester
# ---------------------------------------------------------------------------
positions = {
    'AAPL': {'cost_basis': 150.0, 'current_price': 130.0, 'shares': 100},
    'MSFT': {'cost_basis': 300.0, 'current_price': 320.0, 'shares': 50},
}

harvester = TaxLossHarvester()
candidates = harvester.harvest_candidates(positions)

print(f"[OK] harvest_candidates returned {len(candidates)} candidate(s)")
# AAPL has loss: (130-150)*100 = -2000
# MSFT has gain: (320-300)*50 = +1000 --> NOT a candidate
candidate_tickers = [c['ticker'] for c in candidates]
assert 'AAPL' in candidate_tickers, (
    f"FAIL: AAPL (loss position) should be in candidates: {candidate_tickers}"
)
assert 'MSFT' not in candidate_tickers, (
    f"FAIL: MSFT (gain position) should NOT be in candidates: {candidate_tickers}"
)
print(f"[OK] AAPL in candidates, MSFT not in candidates — correct")

# Validate candidate structure
aapl_cand = next(c for c in candidates if c['ticker'] == 'AAPL')
assert aapl_cand['unrealized_loss'] > 0, (
    f"FAIL: AAPL unrealized_loss should be positive: {aapl_cand['unrealized_loss']}"
)
assert abs(aapl_cand['unrealized_loss'] - 2000.0) < 1e-6, (
    f"FAIL: AAPL unrealized_loss should be 2000, got {aapl_cand['unrealized_loss']}"
)
assert aapl_cand['tax_saving'] > 0, "FAIL: tax_saving should be positive"
print(f"[OK] AAPL candidate: unrealized_loss={aapl_cand['unrealized_loss']:.2f}, "
      f"tax_saving={aapl_cand['tax_saving']:.2f}")

# harvest --> sell orders
harvest_orders = harvester.harvest(candidates, wash_sale_days=30)
assert isinstance(harvest_orders, list), "FAIL: harvest must return a list"
assert len(harvest_orders) >= 1, "FAIL: should have at least 1 harvest order (AAPL)"
aapl_order = next((o for o in harvest_orders if o['ticker'] == 'AAPL'), None)
assert aapl_order is not None, "FAIL: AAPL harvest order missing"
assert aapl_order['side'] == 'SELL', f"FAIL: harvest order should be SELL, got {aapl_order['side']}"
print(f"[OK] harvest orders: {[(o['ticker'], o['side']) for o in harvest_orders]}")

# ---------------------------------------------------------------------------
# 4. PortfolioRebalancer
# ---------------------------------------------------------------------------
rules = [RebalancingRule(trigger_type='drift', threshold=0.05, description='5% drift rule')]
rebalancer = PortfolioRebalancer(target_weights=target, rules=rules)

# should_rebalance
should, reason = rebalancer.should_rebalance(current)
assert should is True, f"FAIL: should_rebalance should be True (drift=7%): {should}, {reason}"
assert 'drift' in reason.lower(), f"FAIL: reason should mention drift: {reason}"
print(f"[OK] should_rebalance = True, reason = '{reason}'")

# rebalance
portfolio_value = 500_000.0
result = rebalancer.rebalance(
    current_weights=current,
    positions=positions,
    portfolio_value=portfolio_value,
)
assert isinstance(result, RebalancingResult), (
    f"FAIL: rebalance must return RebalancingResult, got {type(result)}"
)
print(f"[OK] rebalance returned RebalancingResult")

assert result.triggered_by == 'drift', (
    f"FAIL: triggered_by should be 'drift', got '{result.triggered_by}'"
)
print(f"[OK] triggered_by = '{result.triggered_by}'")

assert isinstance(result.orders, list), "FAIL: orders must be a list"
assert len(result.orders) > 0, "FAIL: orders must be non-empty for 7% drift"
print(f"[OK] orders non-empty: {len(result.orders)} rebalancing trade(s)")

# Validate RebalancingResult fields
assert isinstance(result.estimated_tax_impact, float), "FAIL: estimated_tax_impact must be float"
assert isinstance(result.estimated_cost_bps, float), "FAIL: estimated_cost_bps must be float"
assert isinstance(result.net_benefit, float), "FAIL: net_benefit must be float"
print(f"[OK] RebalancingResult: triggered_by={result.triggered_by}, "
      f"tax_impact={result.estimated_tax_impact:.2f}, "
      f"cost_bps={result.estimated_cost_bps:.1f}, "
      f"net_benefit={result.net_benefit:.2f}")

# ---------------------------------------------------------------------------
# 5. Standalone functions
# ---------------------------------------------------------------------------
# check_rebalance_needed
needs_check = check_rebalance_needed(current, target, threshold=0.05)
assert needs_check is True, (
    f"FAIL: check_rebalance_needed should be True for 7% drift: {needs_check}"
)
print(f"[OK] check_rebalance_needed(threshold=0.05) = True")

needs_check_tight = check_rebalance_needed(current, target, threshold=0.10)
assert needs_check_tight is False, (
    f"FAIL: check_rebalance_needed should be False for 10% threshold with 7% drift"
)
print(f"[OK] check_rebalance_needed(threshold=0.10) = False")

# harvest_tax_losses
loss_candidates = harvest_tax_losses(positions)
assert isinstance(loss_candidates, list), "FAIL: harvest_tax_losses must return list"
assert any(c['ticker'] == 'AAPL' for c in loss_candidates), (
    "FAIL: AAPL should be in harvest_tax_losses results"
)
print(f"[OK] harvest_tax_losses: {[c['ticker'] for c in loss_candidates]}")

# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------
# No drift --> no rebalance needed
close_weights = np.array([0.34, 0.33, 0.33])
no_drift = monitor.requires_rebalance(close_weights, target, threshold=0.05)
assert not no_drift, (
    f"FAIL: weights close to target should not trigger rebalance: drift={monitor.max_drift(close_weights, target):.4f}"
)
print(f"[OK] Near-target weights (drift={monitor.max_drift(close_weights, target):.4f}) --> no rebalance needed")

# Empty positions --> no harvest candidates
empty_candidates = harvest_tax_losses({})
assert empty_candidates == [], f"FAIL: empty positions should yield empty candidates"
print("[OK] Empty positions --> empty harvest candidates")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("\n--- Rebalancing Orchestrator Summary ---")
print(f"  Current weights : {current.tolist()}")
print(f"  Target weights  : {target.tolist()}")
print(f"  Max drift       : {max_d:.4f}")
print(f"  Triggered       : {should} ({reason})")
print(f"  Rebalance orders: {len(result.orders)}")
print(f"  Tax candidates  : {candidate_tickers}")
print(f"  Tax saving      : ${aapl_cand['tax_saving']:.2f}")

print("\n[PASS] dim_149: Autonomous rebalancing orchestrator")
PYEOF
