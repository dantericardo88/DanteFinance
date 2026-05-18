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
    FactorExposures, FactorModelFitter, FrenchDataLoader, PortfolioFactorAnalyzer,
    _ols, FACTOR_NAMES
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

# ---- New math: PortfolioFactorAnalyzer additions ----
pa = PortfolioFactorAnalyzer()

# -- compute_active_factor_exposure --
portfolio_betas = {"Mkt-RF": 1.2, "SMB": 0.4, "HML": -0.1, "RMW": 0.2, "CMA": 0.0, "MOM": 0.3}
benchmark_betas = {"Mkt-RF": 1.0, "SMB": 0.0, "HML":  0.0, "RMW": 0.0, "CMA": 0.0, "MOM": 0.0}
port_exp = FactorExposures(
    ticker="Portfolio", alpha=0.0002, betas=portfolio_betas, t_stats={},
    r_squared=0.85, adj_r_squared=0.84, residual_vol=0.05, residual_vol_daily=0.003,
    n_obs=252, factor_names=FACTOR_NAMES,
)
bench_exp = FactorExposures(
    ticker="Benchmark", alpha=0.0, betas=benchmark_betas, t_stats={},
    r_squared=1.0, adj_r_squared=1.0, residual_vol=0.0, residual_vol_daily=0.0,
    n_obs=252, factor_names=FACTOR_NAMES,
)
active = pa.compute_active_factor_exposure(port_exp, bench_exp)
assert abs(active["Mkt-RF"] - 0.2) < 1e-9, f"Active Mkt-RF: expected 0.2, got {active['Mkt-RF']}"
assert abs(active["SMB"]    - 0.4) < 1e-9, f"Active SMB: expected 0.4, got {active['SMB']}"
assert abs(active["HML"]   - (-0.1)) < 1e-9, f"Active HML: expected -0.1, got {active['HML']}"
assert abs(active["MOM"]    - 0.3) < 1e-9, f"Active MOM: expected 0.3, got {active['MOM']}"
print(f"[OK] compute_active_factor_exposure: Mkt-RF={active['Mkt-RF']:.4f}  SMB={active['SMB']:.4f}  HML={active['HML']:.4f}")

# -- compute_factor_timing_score --
np.random.seed(0)
n_days = 500
factor_rets    = pd.Series(np.random.normal(0.0001, 0.01, n_days))
# When factor is UP (positive), our exposure is higher -> positive timing
# Build exposures that are perfectly timed
high_exp = 1.5   # exposure when factor is positive
low_exp  = 0.5   # exposure when factor is negative
port_exp_series = pd.Series([
    high_exp if r > 0 else low_exp for r in factor_rets
])
timing = pa.compute_factor_timing_score(factor_rets, port_exp_series, factor_name="HML")
assert "timing_score" in timing
assert timing["timing_score"] > 0.0, f"Good timing should give positive score: {timing['timing_score']}"
expected_timing = timing["mean_exposure_up"] - timing["mean_exposure_down"]
assert abs(timing["timing_score"] - expected_timing) < 1e-9, \
    f"Timing score = up_exp - down_exp: expected {expected_timing:.6f} got {timing['timing_score']:.6f}"
print(f"[OK] compute_factor_timing_score = {timing['timing_score']:.6f}  (expected {expected_timing:.6f})")
print(f"[OK]   mean_exp_up={timing['mean_exposure_up']:.4f}  mean_exp_down={timing['mean_exposure_down']:.4f}")

# -- compute_residual_alpha --
# Build fully synthetic residual alpha test
np.random.seed(42)
n = 252
dates2 = pd.date_range("2023-01-02", periods=n, freq="B")
rf_series   = pd.Series(np.full(n, 0.00019), index=dates2)  # constant RF
true_alpha  = 0.0003   # daily alpha
true_betas2 = {"Mkt-RF": 1.1, "SMB": 0.3, "HML": -0.2, "RMW": 0.1, "CMA": 0.0, "MOM": 0.0}

fac2 = pd.DataFrame({
    "Mkt-RF": np.random.normal(0.0003, 0.01, n),
    "SMB":    np.random.normal(0.0001, 0.005, n),
    "HML":    np.random.normal(0.0001, 0.005, n),
    "RMW":    np.random.normal(0.0001, 0.004, n),
    "CMA":    np.random.normal(0.0001, 0.004, n),
    "MOM":    np.random.normal(0.0002, 0.006, n),
}, index=dates2)

# portfolio return = alpha + rf + sum(beta_i * F_i) + noise
port_ret2 = (
    true_alpha
    + rf_series.values
    + sum(true_betas2[f] * fac2[f].values for f in true_betas2)
    + np.random.normal(0, 0.002, n)
)
port_series2 = pd.Series(port_ret2, index=dates2)

exp2 = FactorExposures(
    ticker="SyntheticPortfolio", alpha=true_alpha, betas=true_betas2, t_stats={},
    r_squared=0.90, adj_r_squared=0.89, residual_vol=0.03, residual_vol_daily=0.002,
    n_obs=n, factor_names=FACTOR_NAMES,
)
alpha_result = pa.compute_residual_alpha(port_series2, rf_series, fac2, exp2)
assert "alpha_annualized" in alpha_result
assert "alpha_t_stat" in alpha_result
assert "residual_series" in alpha_result
# Annualized alpha should be close to true_alpha * 252 (within noise)
expected_alpha_ann = true_alpha * 252
assert abs(alpha_result["alpha_annualized"] - expected_alpha_ann) < 0.05, \
    f"Residual alpha: expected ~{expected_alpha_ann:.4f}, got {alpha_result['alpha_annualized']:.4f}"
print(f"[OK] compute_residual_alpha: alpha_ann={alpha_result['alpha_annualized']:.4f}  (expected ~{expected_alpha_ann:.4f})")
print(f"[OK]   t_stat={alpha_result['alpha_t_stat']:.4f}  n_obs={alpha_result['n_obs']}")

# Verify the formula: alpha = r_p - r_f - sum(beta_i * F_i)
# Compute manually for first observation
day0 = dates2[0]
manual_alpha_d0 = (
    port_series2.iloc[0]
    - rf_series.iloc[0]
    - sum(true_betas2[f] * fac2[f].iloc[0] for f in true_betas2)
)
resid_series = alpha_result["residual_series"]
assert abs(resid_series.iloc[0] - manual_alpha_d0) < 1e-9, \
    f"Formula mismatch day 0: {resid_series.iloc[0]:.8f} vs {manual_alpha_d0:.8f}"
print(f"[OK] Residual alpha formula verified day 0: {resid_series.iloc[0]:.8f} == {manual_alpha_d0:.8f}")

print("\n[PASS] dim_079: Factor risk model + active_exposure/timing/residual_alpha verified")
PYEOF
