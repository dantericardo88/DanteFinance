#!/usr/bin/env bash
# dim_079: Factor risk model — pure OLS regression
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.factor_risk_v3 import (
    FactorExposures, FactorModelFitter, FrenchDataLoader, _ols, FACTOR_NAMES
)

# Test _ols pure computation
np.random.seed(42)
n = 200
X = np.random.normal(0, 1, (n, 3))
true_betas = np.array([0.5, 1.2, -0.3])
y = 0.002 + X @ true_betas + np.random.normal(0, 0.01, n)

result = _ols(y, X, add_const=True)
assert "alpha" in result, "Should have alpha key"
assert "betas" in result, "Should have betas key"
assert "r_squared" in result, "Should have r_squared key"
assert result["r_squared"] > 0.9, f"R-squared should be high: {result['r_squared']:.4f}"
betas = result["betas"]
for i, (est, true) in enumerate(zip(betas, true_betas)):
    assert abs(est - true) < 0.05, f"Beta[{i}] estimate {est:.4f} far from true {true:.4f}"
print(f"[OK] OLS regression: R²={result['r_squared']:.4f} betas={[f'{b:.3f}' for b in betas]}")

# Test FactorRiskModel.fit with synthetic factor data
# Build synthetic FF5+MOM factors and a stock's excess returns
dates = pd.date_range("2023-01-02", periods=252, freq="B")
factor_data = pd.DataFrame({
    "Mkt-RF": np.random.normal(0.0003, 0.01, 252),
    "SMB":    np.random.normal(0.0001, 0.005, 252),
    "HML":    np.random.normal(0.0001, 0.005, 252),
    "RMW":    np.random.normal(0.0001, 0.004, 252),
    "CMA":    np.random.normal(0.0001, 0.004, 252),
    "MOM":    np.random.normal(0.0002, 0.006, 252),
    "RF":     np.full(252, 0.00019),   # ~5% annual
}, index=dates)

# Synthetic stock: beta_mkt=1.1, beta_smb=0.3, alpha~0
stock_returns = (
    0.0005
    + 1.1 * factor_data["Mkt-RF"]
    + 0.3 * factor_data["SMB"]
    - 0.2 * factor_data["HML"]
    + np.random.normal(0, 0.005, 252)
)
stock_returns = pd.Series(stock_returns, index=dates, name="TEST")

model = FactorModelFitter(factor_data=factor_data)
exposures = model.fit(stock_returns, factors=factor_data, factor_names=FACTOR_NAMES)

assert isinstance(exposures, FactorExposures), "Should return FactorExposures"
assert exposures.n_obs == 252, f"n_obs should be 252: {exposures.n_obs}"
assert exposures.r_squared > 0.5, f"R² should be > 0.5: {exposures.r_squared:.4f}"
assert abs(exposures.betas.get("Mkt-RF", 0) - 1.1) < 0.15, \
    f"Mkt-RF beta should be ~1.1: {exposures.betas.get('Mkt-RF', 0):.4f}"
print(f"[OK] FactorRiskModel.fit: R²={exposures.r_squared:.4f} alpha={exposures.alpha:.6f}")
print(f"[OK] Factor betas: {exposures.betas}")
assert exposures.residual_vol > 0, f"Residual vol should be positive: {exposures.residual_vol}"
print(f"[OK] Residual vol (annualized): {exposures.residual_vol:.4f}")

# Test summary string
summary = exposures.summary()
assert "Factor Exposures" in summary, "summary should contain title"
assert "Mkt-RF" in summary, "summary should contain factor names"
print("[OK] FactorExposures.summary() generated successfully")

# Test FACTOR_NAMES constant
assert "Mkt-RF" in FACTOR_NAMES
assert "SMB" in FACTOR_NAMES
assert "HML" in FACTOR_NAMES
assert "MOM" in FACTOR_NAMES
assert len(FACTOR_NAMES) == 6, f"Expected 6 factors, got {len(FACTOR_NAMES)}"
print(f"[OK] FACTOR_NAMES: {FACTOR_NAMES}")

print("\n[PASS] dim_079: Factor risk model")
PYEOF
