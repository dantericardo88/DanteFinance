#!/usr/bin/env bash
# dim_121: Almgren-Chriss optimal execution / market impact modeling
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import numpy as np

# Imports
from sentinel.sma.market_impact_v3 import (
    ACParams, ExecutionPlan, AlmgrenChriss,
    ImplementationShortfall, TCAAnalytics,
    optimal_execution, twap_execution, compute_is, kyle_lambda,
)

FAILS = []

def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  [FAIL] " + msg)
    else:
        print("  [OK]   " + msg)

# Parameters
params = ACParams(sigma=2.0, gamma=1e-7, eta=2.5e-6, epsilon=0.0625, tau=1.0)
X = 1_000_000.0   # shares to liquidate
T = 5.0           # days
n = 5             # intervals
lam = 1e-6        # risk aversion

# --- 1. Optimal trajectory ---
print("\n--- Optimal trajectory ---")
plan = optimal_execution(X, T, n, params, lam)

check(isinstance(plan, ExecutionPlan), "optimal_execution returns ExecutionPlan")
check(len(plan.holdings) == n + 1, "holdings length == n+1 (%d)" % (n+1))
check(len(plan.trades)   == n,     "trades length == n (%d)" % n)

check(abs(plan.holdings[0] - X) < 1.0,
      "holdings[0] == X (got %.1f)" % plan.holdings[0])
check(abs(plan.holdings[-1]) < 1.0,
      "holdings[-1] ~= 0 (got %.6f)" % plan.holdings[-1])

monotone = all(plan.holdings[i] >= plan.holdings[i+1] - 1e-6
               for i in range(len(plan.holdings)-1))
check(monotone, "holdings monotonically decrease (selling)")
check(plan.expected_cost > 0, "expected_cost > 0 (got %.2f)" % plan.expected_cost)
check(plan.cost_variance  > 0, "cost_variance > 0")

# --- 2. TWAP trajectory ---
print("\n--- TWAP trajectory ---")
twap = twap_execution(X, T, n, params)

check(isinstance(twap, ExecutionPlan), "twap_execution returns ExecutionPlan")
check(abs(twap.holdings[0] - X) < 1.0, "TWAP holdings[0] == X")
check(abs(twap.holdings[-1]) < 1.0,    "TWAP holdings[-1] ~= 0")

per_step = X / n
for i, v in enumerate(twap.trades * (T/n)):
    check(abs(v - per_step) / per_step < 0.10,
          "TWAP trade[%d] ~= X/n +/- 10%% (got %.0f, expected %.0f)" % (i, v, per_step))

check(twap.expected_cost > 0, "TWAP expected_cost > 0")

# --- 3. Efficient frontier ---
print("\n--- Efficient frontier ---")
ac = AlmgrenChriss(params)
lambdas = np.logspace(-8, -3, 10)
costs, variances = ac.efficient_frontier(X, T, n, lambdas)

check(len(costs) == 10 and len(variances) == 10,
      "efficient_frontier returns arrays of length 10")
check(variances[-1] < variances[0],
      "higher lambda -> lower variance  (%.3e < %.3e)" % (variances[-1], variances[0]))
check(costs[-1] > costs[0],
      "higher lambda -> higher expected cost  (%.2f > %.2f)" % (costs[-1], costs[0]))

var_decreasing  = bool(np.all(np.diff(variances) <= 1e-6 * variances[:-1]))
cost_increasing = bool(np.all(np.diff(costs) >= -1e-6 * np.abs(costs[:-1])))
check(var_decreasing,  "variance monotonically non-increasing across frontier")
check(cost_increasing, "expected cost monotonically non-decreasing across frontier")

# --- 4. Implementation Shortfall ---
print("\n--- Implementation Shortfall ---")
fill_prices = np.array([100.15, 100.12, 100.08])
fill_sizes  = np.array([333.0,  334.0,  333.0])
is_result = compute_is(
    decision_price=100.0,
    arrival_price=100.10,
    fill_prices=fill_prices,
    fill_sizes=fill_sizes,
    target=1000.0,
)

check(isinstance(is_result, dict), "compute_is returns dict")
required_keys = {"IS_bps", "paper_profit", "execution_cost", "opportunity_cost"}
check(required_keys.issubset(is_result.keys()),
      "IS dict contains required keys")

# Fills (100.08-100.15) average above arrival (100.10)
# -> execution cost is negative (paid more), IS_bps > 0
check(is_result["IS_bps"] > 0,
      "IS_bps > 0 -- paid more than arrival (got %.4f bps)" % is_result["IS_bps"])

# arrival (100.10) > decision (100.0) -> paper profit is positive
check(is_result["paper_profit"] > 0,
      "paper_profit > 0 -- arrival > decision (got %.4f)" % is_result["paper_profit"])

print("  IS breakdown: IS_bps=%.4f, exec_cost=%.4f, opp_cost=%.4f, avg_fill=%.4f" % (
    is_result["IS_bps"], is_result["execution_cost"],
    is_result["opportunity_cost"], is_result["avg_fill_price"]))

# --- 5. Kyle lambda ---
print("\n--- Kyle lambda ---")
rng = np.random.default_rng(42)
order_flow   = rng.normal(0, 1e4, 200)
true_lambda  = 5e-5
price_changes = true_lambda * order_flow + rng.normal(0, 0.01, 200)

lam_est = kyle_lambda(price_changes, order_flow)
check(lam_est > 0,
      "kyle_lambda > 0 (got %.6e)" % lam_est)
check(abs(lam_est - true_lambda) / true_lambda < 0.50,
      "kyle_lambda within 50%% of true value (%.4e vs %.4e)" % (lam_est, true_lambda))

# --- 6. TCA analytics ---
print("\n--- TCA Analytics ---")
tca = TCAAnalytics()

part_cost = tca.participation_cost(adv_fraction=0.05, sigma=0.01, participation_rate=0.05)
check(part_cost > 0, "participation_cost > 0 (got %.6f)" % part_cost)

report = tca.market_impact_report(plan, params)
check("expected_cost_bps"   in report, "market_impact_report has expected_cost_bps")
check("cost_std_bps"        in report, "market_impact_report has cost_std_bps")
check("sharpe_of_cost"      in report, "market_impact_report has sharpe_of_cost")
check("twap_vs_optimal_bps" in report, "market_impact_report has twap_vs_optimal_bps")
check(report["expected_cost_bps"] > 0, "report expected_cost_bps > 0")

# --- 7. VWAP benchmark ---
print("\n--- VWAP benchmark ---")
is_obj = ImplementationShortfall()
vwap_slippage = is_obj.vwap_benchmark(
    fill_prices=np.array([100.15, 100.12, 100.08]),
    fill_sizes=np.array([333.0, 334.0, 333.0]),
    vwap=100.0,
)
check(isinstance(vwap_slippage, float), "vwap_benchmark returns float")
check(vwap_slippage > 0,
      "vwap_benchmark > 0 when fills > vwap (got %.4f bps)" % vwap_slippage)

# Final result
print()
if FAILS:
    print("[FAIL] dim_121: %d check(s) failed:" % len(FAILS))
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
else:
    print("[PASS] dim_121: Almgren-Chriss market impact")

PYEOF
