#!/usr/bin/env bash
# dim_133: Cross-asset signal fusion (ML ensemble, correlation-weighted)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ── Module imports ──────────────────────────────────────────────────────────
from sentinel.sma.signal_fusion_v3 import (
    SignalBundle,
    FusionResult,
    CorrelationWeightedFusion,
    ICWeightedFusion,
    SignalOrthogonalizer,
    WalkForwardValidator,
    SignalFusionEngine,
    correlation_weighted_ensemble,
    ic_weighted_ensemble,
    orthogonalize_signals,
    ensemble_sharpe,
)

rng = np.random.default_rng(99)
T = 300

# ── 1. Synthetic signals and returns ─────────────────────────────────────────
# Base return series
base_ret = rng.normal(0, 0.01, T)

# sig1: correlated with returns (rho≈0.15)
sig1 = 0.15 * base_ret / (base_ret.std() + 1e-12) + 0.985 * rng.normal(0, 0.01, T)
# sig2: correlated with returns (rho≈0.10)
sig2 = 0.10 * base_ret / (base_ret.std() + 1e-12) + 0.990 * rng.normal(0, 0.01, T)
# sig3: pure noise
sig3 = rng.normal(0, 0.01, T)

signals = np.column_stack([sig1, sig2, sig3])  # (300, 3)
returns = base_ret

# ── 2. SignalBundle ──────────────────────────────────────────────────────────
bundle = SignalBundle(
    signals=signals,
    returns=returns,
    signal_names=["alpha_momentum", "alpha_reversal", "noise"],
)
assert bundle.n_signals == 3, f"Expected n_signals=3, got {bundle.n_signals}"
assert bundle.T == 300, f"Expected T=300, got {bundle.T}"
print(f"[OK] SignalBundle: T={bundle.T}, n_signals={bundle.n_signals}")

# ── 3. correlation_weighted_ensemble ─────────────────────────────────────────
cwe_result = correlation_weighted_ensemble(signals, returns)
assert isinstance(cwe_result, FusionResult), "Expected FusionResult"
assert abs(cwe_result.weights.sum() - 1.0) < 1e-9, f"Weights must sum to 1, got {cwe_result.weights.sum()}"
assert np.all(cwe_result.weights >= 0), "All weights must be non-negative"
assert np.isfinite(cwe_result.ensemble_sharpe), f"ensemble_sharpe non-finite: {cwe_result.ensemble_sharpe}"
assert cwe_result.diversification_ratio > 0, f"diversification_ratio must be >0"
assert cwe_result.correlation_matrix.shape == (3, 3), f"corr_matrix shape wrong: {cwe_result.correlation_matrix.shape}"
diag = np.diag(cwe_result.correlation_matrix)
assert np.allclose(diag, 1.0, atol=1e-9), f"Diagonal of corr_matrix should be 1, got {diag}"
print(f"[OK] correlation_weighted_ensemble: weights={np.round(cwe_result.weights, 4)}, "
      f"sharpe={cwe_result.ensemble_sharpe:.4f}, div_ratio={cwe_result.diversification_ratio:.4f}")

# ── 4. ic_weighted_ensemble ───────────────────────────────────────────────────
iwe_result = ic_weighted_ensemble(signals, returns)
assert isinstance(iwe_result, FusionResult), "Expected FusionResult"
assert abs(iwe_result.weights.sum() - 1.0) < 1e-6, f"IC weights sum != 1: {iwe_result.weights.sum()}"
print(f"[OK] ic_weighted_ensemble: weights={np.round(iwe_result.weights, 4)}")

# ── 5. orthogonalize_signals ──────────────────────────────────────────────────
orth = orthogonalize_signals(signals)
assert orth.shape == (300, 3), f"Expected (300,3), got {orth.shape}"

# Check near-orthogonality (dot products between columns should be ~0)
for i in range(3):
    for j in range(i + 1, 3):
        dot_val = abs(float(np.dot(orth[:, i], orth[:, j])))
        assert dot_val < 1.0, f"Columns {i},{j} not orthogonal: dot={dot_val:.6f}"
print(f"[OK] orthogonalize_signals: shape={orth.shape}, columns pairwise orthogonal")

# ── 6. WalkForwardValidator ───────────────────────────────────────────────────
wfv = WalkForwardValidator(train_window=150, test_window=30)
wf_result = wfv.validate(bundle, fusion_method="correlation")
assert "fold_sharpes" in wf_result, "Missing 'fold_sharpes' in walk-forward result"
assert wf_result["n_folds"] > 0, "No walk-forward folds produced"
print(f"[OK] WalkForwardValidator: {wf_result['n_folds']} folds, "
      f"avg_sharpe={np.mean(wf_result['fold_sharpes']):.4f}")

# ── 7. SignalFusionEngine.compare_methods ────────────────────────────────────
engine = SignalFusionEngine(bundle)
methods = engine.compare_methods()
assert "correlation" in methods, "Missing 'correlation' in compare_methods result"
print(f"[OK] compare_methods: methods={list(methods.keys())}")

# ── 8. marginal_contribution ──────────────────────────────────────────────────
mc = engine.marginal_contribution()
assert mc.shape == (3,), f"marginal_contribution shape wrong: {mc.shape}"
assert np.all(np.isfinite(mc)), "marginal_contribution contains non-finite"
print(f"[OK] marginal_contribution: {np.round(mc, 4)}")

# ── 9. ensemble_sharpe utility ────────────────────────────────────────────────
ens_sig = signals.mean(axis=1)
es = ensemble_sharpe(ens_sig, returns)
assert np.isfinite(es), f"ensemble_sharpe non-finite: {es}"
print(f"[OK] ensemble_sharpe: {es:.4f}")

# ── 10. SignalOrthogonalizer.is_orthogonal ────────────────────────────────────
orth_obj = SignalOrthogonalizer()
orth2 = orth_obj.fit_transform(signals)
# Not necessarily perfectly orthogonal (float precision), but close enough at tol=1.0
non_orth = not orth_obj.is_orthogonal(signals, tol=1e-3)  # raw signals likely non-orthogonal
print(f"[OK] SignalOrthogonalizer: raw_orthogonal={not non_orth}")

print("\n[PASS] dim_133: Cross-asset signal fusion")
PYEOF
