#!/usr/bin/env bash
# dim_129: ML alpha signals (gradient boosting from scratch)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ── Module imports ──────────────────────────────────────────────────────────
from sentinel.sma.ml_alpha_v3 import (
    TreeNode,
    DecisionTree,
    GradientBoostingRegressor,
    AlphaFeatureEngine,
    SignalEvaluator,
    MLAlphaPipeline,
    gradient_boosting_alpha,
    compute_ic,
    compute_ir,
)

rng = np.random.default_rng(42)

# ── 1. Synthetic price series ────────────────────────────────────────────────
T = 500
trend = np.linspace(100, 120, T)
noise = rng.normal(0, 1, T).cumsum() * 0.3
prices = trend + noise
prices = np.clip(prices, 1.0, None)
volumes = rng.uniform(1e6, 2e6, T)

# ── 2. AlphaFeatureEngine ────────────────────────────────────────────────────
engine = AlphaFeatureEngine(prices, volumes)
X, feature_names = engine.build_feature_matrix()
assert X.shape[1] >= 5, f"Expected >=5 features, got {X.shape[1]}"
assert not np.any(np.isnan(X)), "NaN found in feature matrix valid rows"
print(f"[OK] AlphaFeatureEngine: {X.shape[0]} valid rows, {X.shape[1]} features: {feature_names}")

# ── 3. GradientBoostingRegressor ─────────────────────────────────────────────
y = rng.normal(0, 0.01, len(X))
gb = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3,
                                min_samples_split=5, seed=7)
gb.fit(X, y)
preds = gb.predict(X)
assert np.all(np.isfinite(preds)), "GBR predictions contain non-finite values"
imp = gb.feature_importance()
assert abs(imp.sum() - 1.0) < 1e-9 or imp.sum() == 0.0, f"Feature importance should sum to 1, got {imp.sum()}"
print(f"[OK] GradientBoostingRegressor: predictions finite, importance sum={imp.sum():.6f}")

# ── 4. DecisionTree simple smoke test ────────────────────────────────────────
X_simple = np.array([[1.0], [2.0], [3.0]])
y_simple = np.array([1.0, 2.0, 3.0])
dt = DecisionTree(max_depth=2, min_samples_split=1)
dt.fit(X_simple, y_simple)
pred_2_5 = float(dt.predict(np.array([[2.5]]))[0])
assert 1.5 <= pred_2_5 <= 3.0, f"Expected predict([2.5]) between 1.5 and 3.0, got {pred_2_5}"
print(f"[OK] DecisionTree: predict([2.5]) = {pred_2_5:.4f} (expected 2.0-3.0)")

# ── 5. gradient_boosting_alpha convenience function ───────────────────────────
result = gradient_boosting_alpha(prices, volumes, n_estimators=10)
assert "sharpe" in result, "Missing 'sharpe' key"
assert "ir" in result, "Missing 'ir' key"
assert np.isfinite(result["sharpe"]), f"sharpe non-finite: {result['sharpe']}"
assert np.isfinite(result["ir"]), f"ir non-finite: {result['ir']}"
print(f"[OK] gradient_boosting_alpha: sharpe={result['sharpe']:.4f}, ir={result['ir']:.4f}")

# ── 6. SignalEvaluator ───────────────────────────────────────────────────────
ev = SignalEvaluator()

# IC on correlated series
sig = rng.normal(0, 1, 200)
fwd = sig * 0.3 + rng.normal(0, 1, 200) * 0.7
ic_val = compute_ic(sig, fwd)
assert abs(ic_val) < 1.0, f"|IC| must be < 1, got {ic_val}"
assert np.isfinite(ic_val), f"IC non-finite: {ic_val}"
print(f"[OK] compute_ic: IC = {ic_val:.4f}")

# hit_rate
hr = ev.hit_rate(sig, fwd)
assert 0 < hr < 1, f"Expected hit_rate in (0,1), got {hr}"
print(f"[OK] SignalEvaluator.hit_rate: {hr:.4f}")

# IR
ic_series = ev.information_coefficient(sig, fwd, window=40)
ir_val = compute_ir(ic_series)
assert np.isfinite(ir_val), f"IR non-finite: {ir_val}"
print(f"[OK] compute_ir: IR = {ir_val:.4f}")

# turnover
to = ev.turnover(sig)
assert np.isfinite(to) and to >= 0, f"turnover invalid: {to}"
print(f"[OK] SignalEvaluator.turnover: {to:.4f}")

# ── 7. MLAlphaPipeline ───────────────────────────────────────────────────────
pipeline = MLAlphaPipeline(prices, volumes)
pipeline.fit(n_estimators=15)
bt = pipeline.backtest()
assert "sharpe" in bt, "Missing 'sharpe' in backtest result"
assert np.isfinite(bt["sharpe"]), f"sharpe non-finite: {bt['sharpe']}"
print(f"[OK] MLAlphaPipeline.backtest: sharpe={bt['sharpe']:.4f}, hit_rate={bt['hit_rate']:.4f}")

# ── staged_predict ────────────────────────────────────────────────────────────
X_small = X[:50]
staged = gb.staged_predict(X_small)
assert len(staged) == 10, f"Expected 10 staged predictions, got {len(staged)}"
print(f"[OK] staged_predict: {len(staged)} stages")

print("\n[PASS] dim_129: ML alpha signals (gradient boosting)")
PYEOF
