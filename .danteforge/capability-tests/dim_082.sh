#!/usr/bin/env bash
# dim_082: Position sizing — Kelly criterion math (multivariate, shrinkage, options)
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

# Test compute_multivariate_kelly — 2-asset case with known covariance gives correct weights
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

# Verify multivariate Kelly: 2-asset case analytical check
# f* = Sigma^{-1} * (mu - rf) / gamma (gamma=1 in unnormalized version, then normalize)
# 2x2 case: cov=[[0.04, 0.0], [0.0, 0.01]], mu=[0.10, 0.08], rf=0.04
cov_2 = np.array([[0.04, 0.0], [0.0, 0.01]])
mu_2 = np.array([0.10, 0.08])
rf_2 = 0.04
mv2 = KellyCriterion.compute_multivariate_kelly(mu_2, cov_2, rf_2)
# Exact: Sigma^{-1} = diag(25, 100); excess = [0.06, 0.04]
# unnorm_f = [25*0.06, 100*0.04] = [1.5, 4.0]; sum=5.5 → weights = [1.5/5.5, 4.0/5.5]
expected_w = np.array([1.5, 4.0]) / 5.5
# Higher Sharpe asset (asset2: 0.04/0.10=0.4 SR) vs asset1 (0.06/0.20=0.3 SR) gets more weight
assert mv2[1] > mv2[0], f"Lower-vol higher-excess-return asset should get more weight: {mv2}"
assert abs(mv2[0] - expected_w[0]) < 1e-9 and abs(mv2[1] - expected_w[1]) < 1e-9, \
    f"2-asset multivariate Kelly mismatch: {mv2} vs {expected_w}"
print(f"[OK] 2-asset multivariate Kelly analytical check: {mv2.round(4)} (expected {expected_w.round(4)})")

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

# Edge case: zero win rate → Kelly fraction = 0
zero_win_kelly = KellyCriterion.compute_full_kelly(0.01, 0.10, 0.05)  # near-zero win rate
# f* = (0.01 * 2 - 0.99) / 2 = (0.02 - 0.99)/2 = -0.485 → clamped to 0
assert zero_win_kelly == 0.0, f"Near-zero win rate should give Kelly=0: {zero_win_kelly}"
print(f"[OK] Edge case near-zero win rate: Kelly={zero_win_kelly}")

# Test Kelly with James-Stein shrinkage
# Properties to verify:
# 1. f_shrunk <= full_kelly always
# 2. More observations → less shrinkage (f_shrunk closer to full_kelly)
# 3. Zero full_kelly → zero shrunk Kelly
full_k = 0.35  # 35% Kelly fraction
f_shrunk_small = KellyCriterion.compute_kelly_with_shrinkage(full_k, n_obs=10)
f_shrunk_large = KellyCriterion.compute_kelly_with_shrinkage(full_k, n_obs=1000)

assert f_shrunk_small <= full_k, f"Shrunk Kelly must be <= full Kelly: {f_shrunk_small} <= {full_k}"
assert f_shrunk_large <= full_k, f"Shrunk Kelly must be <= full Kelly: {f_shrunk_large} <= {full_k}"
assert f_shrunk_large > f_shrunk_small, \
    f"More data => less shrinkage: {f_shrunk_large:.4f} > {f_shrunk_small:.4f}"
assert KellyCriterion.compute_kelly_with_shrinkage(0.0, n_obs=100) == 0.0, \
    "Zero full Kelly => zero shrunk Kelly"

# Verify shrinkage formula: lambda = n_params / (n_obs * f^2 + n_params)
n_params = 1
n_obs = 10
lam = n_params / (n_obs * full_k**2 + n_params)
expected_shrunk = (1 - lam) * full_k
assert abs(f_shrunk_small - expected_shrunk) < 1e-9, \
    f"Shrinkage formula mismatch: {f_shrunk_small:.6f} vs {expected_shrunk:.6f}"

print(f"[OK] Kelly with shrinkage: full={full_k:.4f}, shrunk(n=10)={f_shrunk_small:.4f}, shrunk(n=1000)={f_shrunk_large:.4f}")
print(f"     lambda(n=10)={lam:.4f}, expected={expected_shrunk:.4f} OK")

# Test Kelly for options — leverage and payoff asymmetry
# Option: delta=0.5, option_price=5.0, underlying=100.0
# Leverage = 0.5 * 100 / 5 = 10x
# win_rate=0.55, avg_win=2.0 (200% of premium), avg_loss=1.0 (100% of premium)
# b = 2.0; f_base = (0.55*2 - 0.45)/2 = (1.10-0.45)/2 = 0.325
# f_options = 0.325 / 10 = 0.0325
delta = 0.5
option_price = 5.0
underlying = 100.0
win_r = 0.55
avg_win_opt = 2.0   # 200% return on premium when winning
avg_loss_opt = 1.0  # 100% loss on premium when losing
f_opt = KellyCriterion.compute_kelly_for_options(
    delta=delta, option_price=option_price, underlying_price=underlying,
    win_rate=win_r, avg_win_pct=avg_win_opt, avg_loss_pct=avg_loss_opt
)

# Verify
b_opt = avg_win_opt / avg_loss_opt
q_opt = 1 - win_r
f_base_opt = (win_r * b_opt - q_opt) / b_opt
leverage = delta * underlying / option_price  # = 10x
expected_f_opt = f_base_opt / leverage
expected_f_opt = max(0.0, min(1.0, expected_f_opt))

assert abs(f_opt - expected_f_opt) < 1e-9, \
    f"Options Kelly mismatch: {f_opt:.6f} vs {expected_f_opt:.6f}"
assert f_opt < KellyCriterion.compute_full_kelly(win_r, avg_win_opt, avg_loss_opt), \
    "Options Kelly should be smaller than plain Kelly (leverage reduces fraction)"
assert 0 < f_opt < 0.10, f"Options Kelly should be small due to leverage: {f_opt:.4f}"
print(f"[OK] Kelly for options: delta={delta}, leverage={leverage:.1f}x, f_opt={f_opt:.4f} (base={f_base_opt:.4f})")

# Options Kelly with losing edge (expect 0)
f_opt_losing = KellyCriterion.compute_kelly_for_options(
    delta=0.5, option_price=5.0, underlying_price=100.0,
    win_rate=0.30, avg_win_pct=1.0, avg_loss_pct=1.0
)
assert f_opt_losing == 0.0, f"Losing-edge options should give Kelly=0: {f_opt_losing}"
print(f"[OK] Options Kelly with losing edge = {f_opt_losing}")

# Options Kelly with explicit leverage override
f_opt_lever = KellyCriterion.compute_kelly_for_options(
    delta=0.5, option_price=5.0, underlying_price=100.0,
    win_rate=win_r, avg_win_pct=avg_win_opt, avg_loss_pct=avg_loss_opt,
    leverage_multiple=5.0,
)
expected_lever = f_base_opt / 5.0
assert abs(f_opt_lever - expected_lever) < 1e-9, \
    f"Leverage override mismatch: {f_opt_lever:.6f} vs {expected_lever:.6f}"
print(f"[OK] Options Kelly with 5x leverage override: {f_opt_lever:.4f}")

print("\n[PASS] dim_082: Position sizing (Kelly criterion, shrinkage, options)")
PYEOF
