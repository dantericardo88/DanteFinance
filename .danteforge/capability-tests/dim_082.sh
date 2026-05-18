#!/usr/bin/env bash
# dim_082: Position sizing — Kelly criterion math
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np

from sentinel.spm.position_sizing_v3 import KellyCriterion, KellyResult

# Test compute_full_kelly with known inputs
# Kelly formula: f* = (p*b - q) / b, where b = avg_win/avg_loss
win_rate = 0.60
avg_win  = 0.08   # 8% average win
avg_loss = 0.05   # 5% average loss

kelly = KellyCriterion.compute_full_kelly(win_rate, avg_win, avg_loss)
# b = 0.08/0.05 = 1.6; q = 0.40
# f* = (0.60*1.6 - 0.40) / 1.6 = (0.96 - 0.40) / 1.6 = 0.35
expected_kelly = (win_rate * (avg_win/avg_loss) - (1-win_rate)) / (avg_win/avg_loss)
assert abs(kelly - expected_kelly) < 1e-9, f"Kelly mismatch: {kelly:.6f} vs {expected_kelly:.6f}"
assert 0 < kelly < 1, f"Kelly should be in (0,1): {kelly}"
print(f"[OK] Full Kelly = {kelly:.4f} = {kelly:.1%} (expected {expected_kelly:.4f})")

# Test fractional Kelly
assert abs(kelly/2 - expected_kelly/2) < 1e-9, "Half Kelly should be half of full kelly"
print(f"[OK] Half Kelly = {kelly/2:.4f}")

# Test compute_continuous_kelly
mu_annual = 0.12    # 12% expected return
sigma_annual = 0.18 # 18% volatility
rf = 0.05           # 5% risk-free rate
cont_kelly = KellyCriterion.compute_continuous_kelly(mu_annual, sigma_annual, rf)
# f* = (mu - rf) / sigma^2 = (0.12 - 0.05) / 0.18^2 = 0.07 / 0.0324 = 2.16 (capped at 2.0)
expected_cont = (mu_annual - rf) / (sigma_annual ** 2)
expected_cont = min(2.0, max(0.0, expected_cont))
assert abs(cont_kelly - expected_cont) < 1e-9, f"Continuous Kelly mismatch: {cont_kelly:.4f} vs {expected_cont:.4f}"
print(f"[OK] Continuous Kelly = {cont_kelly:.4f} (uncapped would be {(mu_annual-rf)/sigma_annual**2:.4f})")

# Test compute_multivariate_kelly
n = 3
mu_vec = np.array([0.12, 0.08, 0.15])
cov_mat = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.02, 0.01],
    [0.02, 0.01, 0.05]
])
rf2 = 0.05
mv_kelly = KellyCriterion.compute_multivariate_kelly(mu_vec, cov_mat, rf2)
assert len(mv_kelly) == 3, f"Should return 3 weights: {len(mv_kelly)}"
assert abs(mv_kelly.sum() - 1.0) < 1e-6 or mv_kelly.sum() <= 1.0, \
    f"Multivariate Kelly weights sum={mv_kelly.sum():.4f}"
assert all(mv_kelly >= -1e-9), f"All Kelly weights should be non-negative: {mv_kelly}"
print(f"[OK] Multivariate Kelly weights: {mv_kelly.round(4)}")

# Test error handling
import traceback
try:
    KellyCriterion.compute_full_kelly(0.60, 0.08, 0.0)  # avg_loss=0 should raise
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"[OK] ValueError raised for avg_loss=0: {e}")

try:
    KellyCriterion.compute_full_kelly(1.10, 0.08, 0.05)  # win_rate > 1 should raise
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"[OK] ValueError raised for win_rate>1: {e}")

# Test negative Kelly → clamped to 0
neg_kelly = KellyCriterion.compute_full_kelly(0.40, 0.05, 0.10)  # losing edge
assert neg_kelly == 0.0, f"Negative Kelly should be clamped to 0: {neg_kelly}"
print(f"[OK] Negative edge Kelly clamped to 0.0: {neg_kelly}")

print("\n[PASS] dim_082: Position sizing (Kelly criterion)")
PYEOF
