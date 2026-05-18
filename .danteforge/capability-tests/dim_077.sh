#!/usr/bin/env bash
# dim_077: Portfolio VaR/CVaR — pure math
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.portfolio_risk_v3 import (
    HistoricalSimulationVaR, GARCHModel, GARCHParams, _norm_ppf
)

np.random.seed(42)

# Test _norm_ppf approximation
ppf_99 = _norm_ppf(0.99)
ppf_95 = _norm_ppf(0.95)
assert 2.2 < ppf_99 < 2.4, f"norm_ppf(0.99) = {ppf_99}"
assert 1.6 < ppf_95 < 1.7, f"norm_ppf(0.95) = {ppf_95}"
print(f"[OK] _norm_ppf: ppf(0.99)={ppf_99:.4f} ppf(0.95)={ppf_95:.4f}")

# Test HistoricalSimulationVaR
hsvar = HistoricalSimulationVaR()
returns = pd.Series(np.random.normal(0.0005, 0.01, 252))

var_99 = hsvar.compute_var(returns, confidence=0.99, horizon=1)
var_95 = hsvar.compute_var(returns, confidence=0.95, horizon=1)

assert var_99 > 0, f"VaR 99% should be positive: {var_99}"
assert var_95 > 0, f"VaR 95% should be positive: {var_95}"
assert var_99 > var_95, f"VaR 99% should exceed VaR 95%: {var_99:.4f} vs {var_95:.4f}"
print(f"[OK] HistoricalSimulationVaR: 99%={var_99:.4f} 95%={var_95:.4f}")

# Test CVaR (conditional VaR / ES)
cvar_99 = hsvar.compute_cvar(returns, confidence=0.99)
assert cvar_99 >= var_99, f"CVaR should be >= VaR: {cvar_99:.4f} vs {var_99:.4f}"
print(f"[OK] CVaR 99% = {cvar_99:.4f} >= VaR 99% = {var_99:.4f}")

# Test horizon scaling (square root of time rule)
var_10d = hsvar.compute_var(returns, confidence=0.99, horizon=10)
ratio = var_10d / var_99
assert 2.5 < ratio < 4.0, f"10-day VaR should be ~sqrt(10) x 1-day: {ratio:.2f}"
print(f"[OK] Horizon scaling: 10d/1d VaR ratio = {ratio:.3f} (expected ~{10**0.5:.2f})")

# Test GARCHModel fit (pure numpy, no external data)
garch = GARCHModel(model_type="GARCH")
returns_arr = np.random.normal(0.0, 0.01, 300)
garch_params = garch.fit(returns_arr)
assert isinstance(garch_params, GARCHParams), "Should return GARCHParams"
assert garch_params.alpha > 0, f"GARCH alpha should be positive: {garch_params.alpha}"
assert garch_params.beta > 0, f"GARCH beta should be positive: {garch_params.beta}"
assert garch_params.alpha + garch_params.beta < 1.0, \
    f"GARCH stationarity: alpha+beta={garch_params.alpha + garch_params.beta:.4f}"
print(f"[OK] GARCH(1,1) fit: alpha={garch_params.alpha:.4f} beta={garch_params.beta:.4f}")

# Test GARCH variance forecast
if garch._sigma2 is not None:
    forecasts = garch.forecast_variance(h=10)
    assert len(forecasts) == 10, f"Expected 10 forecasts, got {len(forecasts)}"
    assert all(f > 0 for f in forecasts), "All variance forecasts should be positive"
    print(f"[OK] GARCH 10-day variance forecast: {[f'{v:.2e}' for v in forecasts[:3]]} ...")

# Test portfolio VaR via weighted returns (pure numpy)
n_assets = 4
n_obs = 252
asset_returns = np.random.normal(0.0005, 0.015, (n_obs, n_assets))
weights = np.array([0.4, 0.3, 0.2, 0.1])
port_returns = asset_returns @ weights
port_returns_series = pd.Series(port_returns)

port_var99 = hsvar.compute_var(port_returns_series, confidence=0.99)
assert port_var99 > 0, f"Portfolio VaR should be positive: {port_var99}"
print(f"[OK] Portfolio VaR 99% (4 assets, weighted) = {port_var99:.4f}")

print("\n[PASS] dim_077: Portfolio VaR/CVaR")
PYEOF
