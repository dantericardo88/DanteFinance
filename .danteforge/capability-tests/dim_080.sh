#!/usr/bin/env bash
# dim_080: Correlation monitor — pure correlation matrix math
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.correlation_monitor_v3 import CorrelationComputer

np.random.seed(42)
dates = pd.date_range("2023-01-02", periods=252, freq="B")

# Build correlated return series
# SPY and QQQ: high correlation (~0.90)
# SPY and TLT: negative correlation (~-0.30)
mkt = np.random.normal(0.0003, 0.01, 252)
spy = mkt + np.random.normal(0, 0.002, 252)
qqq = 1.2 * mkt + np.random.normal(0, 0.003, 252)
tlt = -0.3 * mkt + np.random.normal(0, 0.004, 252)
gld = 0.1 * mkt + np.random.normal(0, 0.005, 252)

returns = pd.DataFrame({
    "SPY": spy, "QQQ": qqq, "TLT": tlt, "GLD": gld
}, index=dates)

# Test compute_pearson
corr = CorrelationComputer.compute_pearson(returns)
assert corr.shape == (4, 4), f"Correlation matrix should be 4x4: {corr.shape}"
assert all(abs(corr.loc[t, t] - 1.0) < 1e-9 for t in corr.columns), "Diagonal should be 1.0"
spy_qqq = corr.loc["SPY", "QQQ"]
spy_tlt = corr.loc["SPY", "TLT"]
assert spy_qqq > 0.7, f"SPY-QQQ correlation should be high: {spy_qqq:.3f}"
assert spy_tlt < 0.0, f"SPY-TLT correlation should be negative: {spy_tlt:.3f}"
print(f"[OK] Pearson correlation: SPY-QQQ={spy_qqq:.3f} SPY-TLT={spy_tlt:.3f}")

# Test windowed correlation
corr_60d = CorrelationComputer.compute_pearson(returns, window=60)
assert corr_60d.shape == (4, 4), "Windowed correlation should be 4x4"
print(f"[OK] Windowed Pearson (60d): SPY-QQQ={corr_60d.loc['SPY','QQQ']:.3f}")

# Test Spearman correlation
spear = CorrelationComputer.compute_spearman(returns)
assert spear.shape == (4, 4), "Spearman matrix should be 4x4"
assert all(abs(spear.loc[t, t] - 1.0) < 0.01 for t in spear.columns), "Spearman diagonal ~1"
spy_qqq_spear = spear.loc["SPY", "QQQ"]
assert spy_qqq_spear > 0.6, f"Spearman SPY-QQQ should be high: {spy_qqq_spear:.3f}"
print(f"[OK] Spearman correlation: SPY-QQQ={spy_qqq_spear:.3f}")

# Test rolling correlation
roll_corr = CorrelationComputer.compute_rolling_correlation(
    returns["SPY"], returns["QQQ"], window=30
)
valid = roll_corr.dropna()
assert len(valid) > 100, f"Rolling correlation should have >100 valid points: {len(valid)}"
assert valid.between(-1, 1).all(), "Rolling correlation should be in [-1, 1]"
print(f"[OK] Rolling correlation (30d): mean={valid.mean():.3f} std={valid.std():.3f}")

# Test EWM correlation
ewm_corr = CorrelationComputer.compute_ewm_correlation(returns, span=60)
assert ewm_corr.shape == (4, 4), "EWM correlation should be 4x4"
assert all(abs(ewm_corr.loc[t, t] - 1.0) < 0.01 for t in ewm_corr.columns), "EWM diagonal ~1"
print(f"[OK] EWM correlation (span=60): SPY-QQQ={ewm_corr.loc['SPY','QQQ']:.3f}")

# Test correlation matrix symmetry
for corr_mat, name in [(corr, "Pearson"), (spear, "Spearman"), (ewm_corr, "EWM")]:
    is_sym = np.allclose(corr_mat.values, corr_mat.values.T, atol=1e-9)
    assert is_sym, f"{name} correlation matrix should be symmetric"
print("[OK] All correlation matrices are symmetric")

print("\n[PASS] dim_080: Correlation monitor")
PYEOF
