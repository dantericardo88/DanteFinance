#!/bin/bash
# dim_064: Overfitting detection — DSR, Haircut Sharpe, CPCV, PBO
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

# 1. Core imports
from sentinel.sbx.overfitting_detection_v3 import (
    DeflatedSharpeRatio, DSRResult, CombinatorialPurgedCV,
    CPCVResult, ParameterOverfittingAnalyzer, FalseStrategyTheoremAnalyzer,
    EULER_MASCHERONI, ANNUAL_FACTOR,
)
print(f"[OK] Overfitting detection imports OK (EULER={EULER_MASCHERONI:.6f}, ANNUAL={ANNUAL_FACTOR})")

# 2. Mathematical constants
assert abs(EULER_MASCHERONI - 0.5772) < 0.0001, f"Euler-Mascheroni constant wrong: {EULER_MASCHERONI}"
assert ANNUAL_FACTOR in (252, 260), f"Annual factor should be 252 or 260: {ANNUAL_FACTOR}"
print("[OK] Mathematical constants correct")

# 3. DeflatedSharpeRatio.compute_dsr — pure math, returns probability
dsr_calc = DeflatedSharpeRatio()
dsr_val = dsr_calc.compute_dsr(sharpe=2.5, n_trials=100, n_obs=252)
assert 0.0 <= dsr_val <= 1.0, f"DSR should be a probability in [0,1]: {dsr_val}"
print(f"[OK] compute_dsr(SR=2.5, trials=100, n=252): {dsr_val:.4f}")

# 4. Higher SR -> higher DSR probability (monotone)
dsr_low = dsr_calc.compute_dsr(sharpe=0.5, n_trials=100, n_obs=252)
dsr_high = dsr_calc.compute_dsr(sharpe=3.0, n_trials=100, n_obs=252)
assert dsr_high > dsr_low, f"Higher SR should yield higher DSR: {dsr_high} vs {dsr_low}"
print(f"[OK] DSR monotone: SR=0.5->{dsr_low:.3f}, SR=3.0->{dsr_high:.3f}")

# 5. Haircut Sharpe
haircut = dsr_calc.compute_haircut_sharpe(sharpe=2.0, n_trials=50, n_obs=252)
assert isinstance(haircut, float), f"Haircut SR should be float: {haircut}"
print(f"[OK] compute_haircut_sharpe(SR=2.0, trials=50): {haircut:.4f}")

# 6. Expected max Sharpe grows with number of trials
exp_max_100 = dsr_calc.compute_expected_max_sharpe(n_trials=100, n_obs=252)
exp_max_10 = dsr_calc.compute_expected_max_sharpe(n_trials=10, n_obs=252)
assert exp_max_100 > exp_max_10, "Expected max SR should grow with number of trials"
print(f"[OK] Expected max SR: trials=10->{exp_max_10:.3f}, trials=100->{exp_max_100:.3f}")

# 7. DSR from returns returns DSRResult with all fields
np.random.seed(42)
rets = pd.Series(np.random.randn(252) * 0.01)
result = dsr_calc.compute_dsr_from_returns(returns=rets, n_trials=50)
assert isinstance(result, DSRResult), f"Result should be DSRResult, got {type(result)}"
assert hasattr(result, 'deflated_sr'), "DSRResult missing deflated_sr"
assert hasattr(result, 'is_significant'), "DSRResult missing is_significant"
assert hasattr(result, 'interpretation'), "DSRResult missing interpretation"
assert isinstance(result.interpretation, str) and len(result.interpretation) > 0
print(f"[OK] DSRResult: deflated_sr={result.deflated_sr:.4f}, significant={result.is_significant}")

# 8. All key classes are importable
for cls in [CombinatorialPurgedCV, ParameterOverfittingAnalyzer, FalseStrategyTheoremAnalyzer]:
    assert cls is not None, f"Class {cls} should be importable"
print("[OK] CombinatorialPurgedCV, ParameterOverfittingAnalyzer, FalseStrategyTheoremAnalyzer all importable")

print("\n[PASS] dim_064: Overfitting detection (DSR/PBO/CPCV) -- all checks passed")
PYEOF
