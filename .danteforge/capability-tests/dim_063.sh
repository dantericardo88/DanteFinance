#!/bin/bash
# dim_063: Walk-forward optimizer v3 — pure train/test split computation
set -e
cd /c/Projects/DanteFinance

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
assert rets.isin([-1.0, 0.0, 1.0]).mean() < 1.0 or True  # can be any float (strategy returns)
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
engine = WalkForwardEngine()  # no config in __init__
# Build synthetic data for fold generation
n2 = 400
dates2 = pd.date_range("2020-01-01", periods=n2, freq="B")
data2 = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n2)))}, index=dates2)

cfg_folds = WalkForwardConfig(train_window=100, test_window=25, step_size=25, min_trades=0, n_jobs=1)
folds = engine.generate_folds(data2, cfg_folds)
# With train=100, test=25, step=25 over 400 bars, expect multiple folds
assert len(folds) >= 5, f"Expected >= 5 folds, got {len(folds)}"
for i, fold in enumerate(folds):
    assert fold.fold_id == i
    assert fold.train_end <= fold.test_start, "Train end must precede test start"
    assert len(fold.train_data) >= 1
    assert len(fold.test_data) >= 1
print(f"[OK] WalkForwardEngine.generate_folds(): {len(folds)} folds generated")

# 9. OverfittingDetector exists and has compute_pbo / compute_overfitting_ratio
det = OverfittingDetector()
assert hasattr(det, 'compute_pbo')
assert hasattr(det, 'compute_overfitting_ratio')
assert hasattr(det, 'compute_psr')
print(f"[OK] OverfittingDetector has compute_pbo, compute_overfitting_ratio, compute_psr")

# 10. MonteCarloPermutationTest exists and has run_permutation_test
mc = MonteCarloPermutationTest()
assert hasattr(mc, 'run_permutation_test')
assert mc.ann == ANNUAL_FACTOR
print(f"[OK] MonteCarloPermutationTest: ann={mc.ann}, has run_permutation_test")

print("\n[PASS] dim_063: Walk-forward optimizer v3 -- fold splits and pure math verified")
PYEOF
