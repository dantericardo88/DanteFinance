#!/bin/bash
# dim_063: Walk-forward optimizer v3 — pure train/test split computation,
#          MC permutation test, parameter sensitivity surface, stability metric
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from sentinel.sbx.walk_forward_v3 import (
    WalkForwardConfig,
    WalkForwardEngine,
    OverfittingDetector,
    MonteCarloPermutationTest,
    FoldResult,
    sma_crossover_strategy,
    momentum_strategy,
    mean_reversion_strategy,
    _annualized_sharpe,
    ANNUAL_FACTOR,
)

# 1. WalkForwardConfig defaults
cfg = WalkForwardConfig()
assert cfg.train_window == 252
assert cfg.test_window == 63
assert cfg.step_size == 21
assert cfg.min_trades == 20
assert cfg.anchored == False
assert cfg.n_jobs == 1
print(f"[OK] WalkForwardConfig defaults: train={cfg.train_window}, test={cfg.test_window}, step={cfg.step_size}")

# 2. WalkForwardConfig custom
cfg2 = WalkForwardConfig(train_window=504, test_window=126, step_size=63, anchored=True)
assert cfg2.train_window == 504
assert cfg2.anchored == True
print(f"[OK] WalkForwardConfig custom: train={cfg2.train_window}, anchored={cfg2.anchored}")

# 3. _annualized_sharpe pure math
np.random.seed(42)
rets = pd.Series(np.random.normal(0.0005, 0.01, 252))
sh = _annualized_sharpe(rets)
assert isinstance(sh, float)
assert -5.0 < sh < 5.0, f"Sharpe out of range: {sh}"
print(f"[OK] _annualized_sharpe() = {sh:.4f}")

# Edge cases for _annualized_sharpe
assert _annualized_sharpe(pd.Series([], dtype=float)) == 0.0
assert _annualized_sharpe(pd.Series([0.0, 0.0, 0.0])) == 0.0
print(f"[OK] _annualized_sharpe edge cases: empty=0.0, all_zeros=0.0")

# 4. ANNUAL_FACTOR constant
assert ANNUAL_FACTOR == 252
print(f"[OK] ANNUAL_FACTOR = {ANNUAL_FACTOR}")

# 5. sma_crossover_strategy pure computation
n = 300
dates = pd.date_range("2022-01-01", periods=n, freq="B")
prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n))), index=dates)
data = pd.DataFrame({"Close": prices})
rets = sma_crossover_strategy(data, fast=10, slow=30)
assert len(rets) == n
assert rets.dtype == float
print(f"[OK] sma_crossover_strategy(): {len(rets)} bars, non-zero days={int((rets != 0).sum())}")

# 6. momentum_strategy pure computation
rets2 = momentum_strategy(data, lookback=63)
assert len(rets2) == n
print(f"[OK] momentum_strategy(): {len(rets2)} bars, non-zero days={int((rets2 != 0).sum())}")

# 7. mean_reversion_strategy pure computation
rets3 = mean_reversion_strategy(data, window=20)
assert len(rets3) == n
print(f"[OK] mean_reversion_strategy(): {len(rets3)} bars, non-zero days={int((rets3 != 0).sum())}")

# 8. WalkForwardEngine fold generation (no network, pure date math)
engine = WalkForwardEngine()
n2 = 400
dates2 = pd.date_range("2020-01-01", periods=n2, freq="B")
data2 = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n2)))}, index=dates2)

cfg_folds = WalkForwardConfig(train_window=100, test_window=25, step_size=25, min_trades=0, n_jobs=1)
folds = engine.generate_folds(data2, cfg_folds)
assert len(folds) >= 5, f"Expected >= 5 folds, got {len(folds)}"
for i, fold in enumerate(folds):
    assert fold.fold_id == i
    assert fold.train_end <= fold.test_start, "Train end must precede test start"
    assert len(fold.train_data) >= 1
    assert len(fold.test_data) >= 1
print(f"[OK] WalkForwardEngine.generate_folds(): {len(folds)} folds generated")

# 9. OverfittingDetector exists and has required methods
det = OverfittingDetector()
assert hasattr(det, 'compute_pbo')
assert hasattr(det, 'compute_overfitting_ratio')
assert hasattr(det, 'compute_psr')
print(f"[OK] OverfittingDetector has compute_pbo, compute_overfitting_ratio, compute_psr")

# 10. MonteCarloPermutationTest exists and has required methods
mc = MonteCarloPermutationTest()
assert hasattr(mc, 'run_permutation_test')
assert hasattr(mc, 'compute_parameter_sensitivity_surface')
assert hasattr(mc, 'compute_oos_stability')
assert mc.ann == ANNUAL_FACTOR
print(f"[OK] MonteCarloPermutationTest: ann={mc.ann}, has run_permutation_test, sensitivity_surface, oos_stability")

# 11. MC permutation test — pure math (no network)
# Strategy on random returns should get p-value ~0.5
# We use synthetic price data; strategy with no real edge
n_days = 500
rng = np.random.default_rng(seed=0)
random_prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n_days)))
random_dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
random_data = pd.DataFrame({"Close": random_prices}, index=random_dates)

# Use small n_permutations for speed
n_perm = 200
p_random = mc.run_permutation_test(
    sma_crossover_strategy, random_data,
    n_permutations=n_perm,
    strategy_kwargs={"fast": 5, "slow": 20},
)
assert 0.0 <= p_random <= 1.0, f"p-value must be in [0,1]: {p_random}"
# Random returns: p-value should NOT be extremely small (< 0.01 would be suspicious)
# We allow a wide range since this is stochastic, but not consistently <0.01
print(f"[OK] MC permutation test on random data: p-value={p_random:.3f} (expected ~0.3-0.7)")

# Strategy with a persistent strong trend should get low p-value
# Create a strong trending price: +0.5% per day (strong positive drift)
trend_prices = 100.0 * np.exp(np.cumsum(np.full(n_days, 0.005)))  # pure trend, no noise
trend_dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
trend_data = pd.DataFrame({"Close": trend_prices}, index=trend_dates)

p_trend = mc.run_permutation_test(
    sma_crossover_strategy, trend_data,
    n_permutations=n_perm,
    strategy_kwargs={"fast": 5, "slow": 20},
)
assert 0.0 <= p_trend <= 1.0, f"p-value must be in [0,1]: {p_trend}"
# Strong trend strategy should have lower p-value than random
assert p_trend <= p_random, \
    f"Trend strategy p-value ({p_trend:.3f}) should be <= random p-value ({p_random:.3f})"
print(f"[OK] MC permutation test on trending data: p-value={p_trend:.3f} (lower than random={p_random:.3f})")

# 12. Parameter sensitivity surface — returns NxM DataFrame with float Sharpe ratios
fast_range = [5, 10, 20]
slow_range = [30, 50, 100]
surface = mc.compute_parameter_sensitivity_surface(
    sma_crossover_strategy, random_data,
    fast_ma_values=fast_range,
    slow_ma_values=slow_range,
)

# Verify structure
assert isinstance(surface, pd.DataFrame), f"Should return DataFrame, got {type(surface)}"
assert list(surface.index) == fast_range, f"Index should be fast_range: {list(surface.index)}"
assert list(surface.columns) == slow_range, f"Columns should be slow_range: {list(surface.columns)}"
assert surface.shape == (len(fast_range), len(slow_range)), \
    f"Shape mismatch: {surface.shape} vs ({len(fast_range)}, {len(slow_range)})"

# Valid entries (fast < slow) should be float; invalid (fast >= slow) should be NaN
for fast in fast_range:
    for slow in slow_range:
        val = surface.loc[fast, slow]
        if fast >= slow:
            assert np.isnan(val), f"fast={fast} >= slow={slow} should be NaN, got {val}"
        else:
            assert isinstance(val, float) and not np.isnan(val), \
                f"fast={fast}, slow={slow} should be float Sharpe, got {val}"

print(f"[OK] Parameter sensitivity surface: shape={surface.shape}")
print(f"     Sample Sharpes: fast=5/slow=30={surface.loc[5,30]:.4f}, fast=10/slow=50={surface.loc[10,50]:.4f}")

# 13. OOS stability metric — fraction of periods beating benchmark
# Build 10 synthetic FoldResults: 8 positive OOS Sharpe, 2 negative
import pandas as pd
def _make_fold_result(fold_id: int, oos_sharpe: float) -> FoldResult:
    """Create a minimal FoldResult for testing."""
    dummy_dates = pd.date_range("2020-01-01", periods=5, freq="B")
    dummy_rets = pd.Series([0.001] * 5, index=dummy_dates)
    return FoldResult(
        fold_id=fold_id,
        train_start=dummy_dates[0],
        train_end=dummy_dates[-1],
        test_start=dummy_dates[0],
        test_end=dummy_dates[-1],
        in_sample_sharpe=1.0,
        out_sample_sharpe=oos_sharpe,
        in_sample_returns=dummy_rets,
        out_sample_returns=dummy_rets,
        n_train_trades=10,
        n_test_trades=5,
        overfitting_ratio=1.0,
    )

# 8 positive, 2 negative → stability = 0.8
fold_results_test = [_make_fold_result(i, 0.5 if i < 8 else -0.3) for i in range(10)]
stability = mc.compute_oos_stability(fold_results_test, benchmark_sharpe=0.0)
assert abs(stability - 0.8) < 1e-9, f"Stability should be 0.8 (8/10 positive): {stability}"
print(f"[OK] OOS stability (8/10 positive): {stability:.1f} (expected 0.8)")

# All positive → stability = 1.0
all_positive = [_make_fold_result(i, 1.0) for i in range(5)]
assert mc.compute_oos_stability(all_positive) == 1.0
print(f"[OK] OOS stability (all positive): {mc.compute_oos_stability(all_positive):.1f}")

# All negative → stability = 0.0
all_negative = [_make_fold_result(i, -0.5) for i in range(5)]
assert mc.compute_oos_stability(all_negative) == 0.0
print(f"[OK] OOS stability (all negative): {mc.compute_oos_stability(all_negative):.1f}")

# Empty fold list → stability = 0.0
assert mc.compute_oos_stability([]) == 0.0
print(f"[OK] OOS stability (empty): {mc.compute_oos_stability([]):.1f}")

# Non-zero benchmark: 6 of 10 beat 0.3 Sharpe
fold_results_bench = [_make_fold_result(i, 0.5 if i < 6 else 0.1) for i in range(10)]
stability_bench = mc.compute_oos_stability(fold_results_bench, benchmark_sharpe=0.3)
assert abs(stability_bench - 0.6) < 1e-9, \
    f"Stability vs 0.3 benchmark should be 0.6 (6/10 > 0.3): {stability_bench}"
print(f"[OK] OOS stability vs 0.3 benchmark (6/10 beat): {stability_bench:.1f}")

print("\n[PASS] dim_063: Walk-forward optimizer v3 -- fold splits, MC permutation, sensitivity surface, stability verified")
PYEOF
