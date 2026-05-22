#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
np.random.seed(42)
# Two-regime data: 300 low-vol, 200 high-vol
r1 = np.random.normal(0.001, 0.01, 300)
r2 = np.random.normal(-0.002, 0.03, 200)
returns = np.concatenate([r1, r2])

from sentinel.sma.vol_regime_v3 import GaussianHMM, RegimeAnalytics, fit_hmm, detect_regimes, regime_vol_ratio

hmm = GaussianHMM(n_states=2, n_iter=200, tol=1e-5)
hmm.fit(returns)
p = hmm.params
assert p.converged, "HMM must converge"
assert p.n_states == 2
stds = sorted(p.stds)
assert stds[0] < 0.02, f"Low-vol state std should be < 0.02, got {stds[0]:.4f}"
assert stds[1] > 0.015, f"High-vol state std should be > 0.015, got {stds[1]:.4f}"
ratio = stds[1] / stds[0]
assert ratio >= 1.5, f"Vol ratio should be >= 1.5, got {ratio:.2f}"
states = hmm.predict(returns)
assert 0 in states and 1 in states, "Both states must appear in Viterbi path"
proba = hmm.predict_proba(returns)
assert proba.shape == (500, 2), f"proba shape {proba.shape}"
assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "Probabilities must sum to 1"
assert np.allclose(p.trans_matrix.sum(axis=1), 1.0, atol=1e-6), "Trans rows sum to 1"
ll = hmm.score(returns)
assert np.isfinite(ll), f"Log-likelihood must be finite, got {ll}"
# Note: for continuous Gaussian distributions with small sigma, total log-likelihood
# can be positive (PDF > 1 at peak); here we verify it is a large finite number
assert abs(ll) > 100, f"Log-likelihood magnitude should be > 100, got {ll:.2f}"
ra = RegimeAnalytics(hmm, returns)
vol_r = ra.regime_vol_ratio()
assert vol_r >= 1.5, f"Regime vol ratio >= 1.5"
periods = ra.regime_periods()
assert len(periods) >= 2, "Must detect at least 2 regime periods"
print(f"Low-vol std: {stds[0]:.4f}, High-vol std: {stds[1]:.4f}, Ratio: {ratio:.2f}")
print(f"Log-likelihood: {ll:.2f}, Regimes: {len(periods)}")
print("[PASS] dim_116: Vol regime detection (HMM)")
PYEOF
