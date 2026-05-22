#!/usr/bin/env bash
# dim_150: Portfolio copilot (NL query-driven what-if + trade ideas)
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
    QueryParser,
    WhatIfEngine,
    PortfolioCopilot,
    portfolio_query,
)
print("[OK] All imports successful from sentinel.sai.agentic_portfolio_v3")

# ---------------------------------------------------------------------------
# 2. Setup
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)
weights = np.array([0.4, 0.35, 0.25])
asset_names = ['AAPL', 'MSFT', 'GOOGL']
n = len(weights)

# Generate 252-day synthetic returns (3 assets, realistic covariance)
vols = np.array([0.20, 0.18, 0.22])
corr = np.array([
    [1.00, 0.65, 0.58],
    [0.65, 1.00, 0.55],
    [0.58, 0.55, 1.00],
])
L = np.linalg.cholesky(corr)
z = rng.standard_normal((252, n))
daily_vol = vols / np.sqrt(252)
returns = (z @ L.T) * daily_vol   # (252, 3)

# Covariance for WhatIfEngine
cov_matrix = np.cov(returns.T) * 252

print(f"[OK] Setup: weights={weights.tolist()}, assets={asset_names}")
print(f"[OK] Generated {returns.shape} returns array")

# ---------------------------------------------------------------------------
# 3. QueryParser
# ---------------------------------------------------------------------------
parser = QueryParser()

# trade_idea intent
parsed_buy = parser.parse("buy more AAPL")
assert parsed_buy['intent'] == 'trade_idea', (
    f"FAIL: 'buy more AAPL' should be 'trade_idea', got '{parsed_buy['intent']}'"
)
print(f"[OK] QueryParser: 'buy more AAPL' --> intent='{parsed_buy['intent']}'")

# risk intent
parsed_risk = parser.parse("what is my portfolio risk")
assert parsed_risk['intent'] == 'risk', (
    f"FAIL: 'what is my portfolio risk' should be 'risk', got '{parsed_risk['intent']}'"
)
print(f"[OK] QueryParser: 'what is my portfolio risk' --> intent='{parsed_risk['intent']}'")

# what_if intent
parsed_whatif = parser.parse("what would happen if I increased AAPL to 50%")
assert parsed_whatif['intent'] == 'what_if', (
    f"FAIL: what-if query should map to 'what_if', got '{parsed_whatif['intent']}'"
)
print(f"[OK] QueryParser: what-if query --> intent='{parsed_whatif['intent']}'")

# rebalance intent
parsed_reb = parser.parse("should I rebalance my portfolio?")
assert parsed_reb['intent'] == 'rebalance', (
    f"FAIL: rebalance query should map to 'rebalance', got '{parsed_reb['intent']}'"
)
print(f"[OK] QueryParser: rebalance query --> intent='{parsed_reb['intent']}'")

# performance intent
parsed_perf = parser.parse("how are my returns?")
assert parsed_perf['intent'] == 'performance', (
    f"FAIL: performance query should map to 'performance', got '{parsed_perf['intent']}'"
)
print(f"[OK] QueryParser: performance query --> intent='{parsed_perf['intent']}'")

# parse returns required keys
for q, expected in [
    ("buy AAPL", "trade_idea"),
    ("explain my risk exposure", "risk"),
]:
    p = parser.parse(q)
    assert 'intent' in p, f"FAIL: parse result missing 'intent'"
    assert 'entities' in p, f"FAIL: parse result missing 'entities'"
    assert 'parameters' in p, f"FAIL: parse result missing 'parameters'"
print(f"[OK] parse always returns intent, entities, parameters")

# ---------------------------------------------------------------------------
# 4. WhatIfEngine
# ---------------------------------------------------------------------------
what_if_engine = WhatIfEngine()

# weight_change: increase AAPL (idx=0) to 50%
result_wi = what_if_engine.weight_change(
    current_weights=weights,
    ticker_idx=0,
    new_weight=0.50,
    cov_matrix=cov_matrix,
)
assert 'new_vol' in result_wi, f"FAIL: weight_change must return 'new_vol': {result_wi.keys()}"
assert 'old_vol' in result_wi, f"FAIL: weight_change must return 'old_vol': {result_wi.keys()}"
assert 'vol_change_bps' in result_wi, "FAIL: weight_change must return 'vol_change_bps'"
assert 'marginal_contribution' in result_wi, "FAIL: weight_change must return 'marginal_contribution'"
assert result_wi['new_vol'] > 0, f"FAIL: new_vol must be positive: {result_wi['new_vol']}"
print(f"[OK] WhatIfEngine.weight_change: new_vol={result_wi['new_vol']:.4f}, "
      f"old_vol={result_wi['old_vol']:.4f}, "
      f"vol_change={result_wi['vol_change_bps']:+.2f}bps")

# market_shock
shock_returns = np.array([-0.10, -0.05, -0.15])
shock_result = what_if_engine.market_shock(weights, shock_returns)
assert 'portfolio_return' in shock_result, "FAIL: market_shock missing 'portfolio_return'"
assert 'winner' in shock_result, "FAIL: market_shock missing 'winner'"
assert 'loser' in shock_result, "FAIL: market_shock missing 'loser'"
expected_port_ret = float(weights @ shock_returns)
assert abs(shock_result['portfolio_return'] - expected_port_ret) < 1e-10, (
    f"FAIL: portfolio_return {shock_result['portfolio_return']:.6f} vs {expected_port_ret:.6f}"
)
# loser should be asset 2 (GOOGL at -15%)
assert shock_result['loser'] == 2, (
    f"FAIL: loser should be asset 2 (GOOGL at -15%), got {shock_result['loser']}"
)
print(f"[OK] WhatIfEngine.market_shock: port_return={shock_result['portfolio_return']:.4f}, "
      f"loser=asset{shock_result['loser']}")

# add_position
correlations = np.array([0.3, 0.25, 0.2])  # new asset correlations to existing 3
add_result = what_if_engine.add_position(
    current_weights=weights,
    new_weight=0.10,
    new_ticker_vol=0.25,
    correlations=correlations,
    cov_matrix=cov_matrix,
)
assert 'new_vol' in add_result, "FAIL: add_position missing 'new_vol'"
assert 'diversification_benefit' in add_result, "FAIL: add_position missing 'diversification_benefit'"
assert add_result['new_vol'] > 0, "FAIL: new_vol must be positive"
print(f"[OK] WhatIfEngine.add_position: new_vol={add_result['new_vol']:.4f}, "
      f"benefit={add_result['diversification_benefit']:.4f}")

# ---------------------------------------------------------------------------
# 5. PortfolioCopilot
# ---------------------------------------------------------------------------
copilot = PortfolioCopilot(weights=weights, returns=returns, asset_names=asset_names)

# what_if query via NL
response_whatif = copilot.query("what would happen if I increased AAPL to 50%")
assert isinstance(response_whatif, str) and len(response_whatif) > 0, (
    "FAIL: what-if query must return non-empty string"
)
print(f"[OK] copilot.query(what-if): '{response_whatif[:80]}...'")

# risk query
response_risk = copilot.query("what is my portfolio risk")
assert isinstance(response_risk, str) and len(response_risk) > 0, (
    "FAIL: risk query must return non-empty string"
)
# must contain 'vol' or 'risk'
lower_risk = response_risk.lower()
assert 'vol' in lower_risk or 'risk' in lower_risk, (
    f"FAIL: risk response should mention 'vol' or 'risk': '{response_risk[:100]}'"
)
print(f"[OK] copilot.query(risk): '{response_risk[:80]}...' (contains 'vol'/'risk')")

# trade ideas
response_trade = copilot.trade_ideas(momentum_window=63)
assert isinstance(response_trade, str) and len(response_trade) > 0, (
    "FAIL: trade_ideas must return non-empty string"
)
print(f"[OK] copilot.trade_ideas(): '{response_trade[:80]}...'")

# rebalancing check
target = np.array([0.33, 0.33, 0.34])
response_reb = copilot.rebalancing_check(target_weights=target)
assert isinstance(response_reb, str) and len(response_reb) > 0, (
    "FAIL: rebalancing_check must return non-empty string"
)
print(f"[OK] copilot.rebalancing_check(): '{response_reb[:80]}...'")

# explain_risk
response_explain = copilot.explain_risk()
assert isinstance(response_explain, str) and len(response_explain) > 0, (
    "FAIL: explain_risk must return non-empty string"
)
lower_explain = response_explain.lower()
assert 'vol' in lower_explain or 'risk' in lower_explain, (
    f"FAIL: explain_risk should mention vol/risk: '{response_explain[:100]}'"
)
print(f"[OK] copilot.explain_risk(): '{response_explain[:80]}...'")

# what_if method directly
response_whatif_direct = copilot.what_if('MSFT', new_weight=0.50)
assert isinstance(response_whatif_direct, str) and len(response_whatif_direct) > 0, (
    "FAIL: what_if method must return non-empty string"
)
print(f"[OK] copilot.what_if('MSFT', 0.50): '{response_whatif_direct[:80]}...'")

# ---------------------------------------------------------------------------
# 6. Standalone portfolio_query function
# ---------------------------------------------------------------------------
pq_result = portfolio_query("explain my risk", weights, returns, asset_names)
assert isinstance(pq_result, str) and len(pq_result) > 0, (
    "FAIL: portfolio_query must return non-empty string"
)
print(f"[OK] portfolio_query('explain my risk'): '{pq_result[:80]}...'")

pq_trade = portfolio_query("buy more AAPL", weights, returns, asset_names)
assert isinstance(pq_trade, str) and len(pq_trade) > 0, (
    "FAIL: portfolio_query trade idea must return non-empty string"
)
print(f"[OK] portfolio_query('buy more AAPL'): '{pq_trade[:80]}...'")

# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------
# Unknown ticker in what-if
resp_unknown = copilot.what_if('TSLA', new_weight=0.20)
assert isinstance(resp_unknown, str), "FAIL: what_if with unknown ticker must return string"
print(f"[OK] copilot.what_if('TSLA') gracefully handles unknown ticker")

# Invalid query still returns something
resp_garbage = copilot.query("asdfghjkl xyz")
assert isinstance(resp_garbage, str) and len(resp_garbage) > 0, (
    "FAIL: garbage query must still return non-empty string"
)
print(f"[OK] copilot.query(garbage) graceful fallback: '{resp_garbage[:60]}...'")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n--- Portfolio Copilot Summary ---")
print(f"  Assets          : {asset_names}")
print(f"  Weights         : {weights.tolist()}")
print(f"  Returns shape   : {returns.shape}")
print(f"  Query: what-if  : {response_whatif[:60]}...")
print(f"  Query: risk     : {response_risk[:60]}...")
print(f"  Trade ideas     : {response_trade[:60]}...")
print(f"  Rebalance check : {response_reb[:60]}...")

print("\n[PASS] dim_150: Portfolio copilot NL query")
PYEOF
