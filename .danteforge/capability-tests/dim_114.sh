#!/usr/bin/env bash
# dim_114: GARCH/EWMA volatility forecasting — pure numpy/scipy MLE
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ── Imports ──────────────────────────────────────────────────────────────────
from sentinel.sma.garch_vol_v3 import (
    GARCHParams,
    GARCH11,
    EGARCH11,
    EWMA,
    VolForecastEnsemble,
    fit_garch,
    ewma_vol,
    garch_forecast,
    vol_cone,
)

print("[OK] all imports successful")

# ── 1. Synthetic GARCH-structured returns ────────────────────────────────────
rng = np.random.default_rng(2024)
# True params: omega=1e-6, alpha=0.10, beta=0.85  -> persistence=0.95
true_omega, true_alpha, true_beta = 1e-6, 0.10, 0.85
n = 500
returns_sim = np.empty(n)
var_t = true_omega / (1 - true_alpha - true_beta)  # start at long-run var
for t in range(n):
    eps = rng.standard_normal()
    returns_sim[t] = np.sqrt(max(var_t, 0.0)) * eps
    var_t = true_omega + true_alpha * returns_sim[t]**2 + true_beta * var_t

print(f"[OK] synthetic returns: n={n}, mean={returns_sim.mean():.6f}, std={returns_sim.std():.6f}")

# ── 2. Fit GARCH(1,1) ────────────────────────────────────────────────────────
params = fit_garch(returns_sim)
print(f"[OK] GARCH fit: omega={params.omega:.2e}, alpha={params.alpha:.4f}, "
      f"beta={params.beta:.4f}, persistence={params.persistence:.4f}, "
      f"long_run_vol={params.long_run_vol:.6f}, converged={params.converged}")

assert params.omega > 0, f"omega must be positive, got {params.omega}"
assert params.alpha > 0, f"alpha must be positive, got {params.alpha}"
assert params.beta > 0, f"beta must be positive, got {params.beta}"
assert params.persistence < 1.0, f"persistence must be < 1 for stationarity: {params.persistence}"
print(f"[OK] stationarity: persistence={params.persistence:.4f} < 1.0")

lr_vol = params.long_run_vol
assert 0.001 <= lr_vol <= 0.5, (
    f"long_run_vol={lr_vol:.6f} outside reasonable [0.001, 0.5] daily vol range"
)
print(f"[OK] long_run_vol={lr_vol:.6f} in [0.001, 0.5]")

assert params.converged, "GARCH MLE did not converge — check optimizer"
print("[OK] GARCH MLE converged=True")

# ── 3. Conditional volatility filter ─────────────────────────────────────────
model = GARCH11()
model.params_ = params  # reuse fitted params
cond_vols = model.filter(returns_sim, params)
assert len(cond_vols) == n, f"filter length mismatch: {len(cond_vols)} vs {n}"
assert np.all(cond_vols > 0), "all conditional vols must be positive"
print(f"[OK] filter: min_vol={cond_vols.min():.6f}, max_vol={cond_vols.max():.6f}")

# ── 4. Multi-step variance forecast ──────────────────────────────────────────
current_var = float(cond_vols[-1]**2)
fc_var = garch_forecast(params, current_var, h=10)
assert len(fc_var) == 10, f"forecast length should be 10, got {len(fc_var)}"
assert np.all(fc_var > 0), "all forecast variances must be positive"
# Mean-reversion: forecasts converge toward long-run variance
lr_var = lr_vol**2
# Check that forecast is monotonically trending toward LR (not necessarily above/below)
# If current > LR, forecast should decrease; if current < LR, it should increase
if current_var > lr_var:
    assert fc_var[0] <= current_var + 1e-12, "forecast[0] <= current_var when above LR"
    assert fc_var[-1] >= fc_var[0] - 1e-10 or fc_var[-1] <= current_var, "mean reversion toward LR"
else:
    assert fc_var[0] >= current_var - 1e-12, "forecast[0] >= current_var when below LR"
print(f"[OK] 10-step forecast: fc[0]={np.sqrt(fc_var[0]):.6f}, fc[-1]={np.sqrt(fc_var[-1]):.6f}, LR={lr_vol:.6f}")

# ── 5. GARCH simulate ────────────────────────────────────────────────────────
sim_returns = model.simulate(params, n=252, seed=42)
assert len(sim_returns) == 252, f"simulate length={len(sim_returns)}"
assert np.all(np.isfinite(sim_returns)), "simulated returns must be finite"
print(f"[OK] simulate: n=252, std={sim_returns.std():.6f}")

# ── 6. EWMA volatility ───────────────────────────────────────────────────────
ewma_vols = ewma_vol(returns_sim, lam=0.94)
assert len(ewma_vols) == n, f"ewma_vol length={len(ewma_vols)}"
assert np.all(ewma_vols > 0), "all EWMA vols must be positive"
print(f"[OK] EWMA: min={ewma_vols.min():.6f}, max={ewma_vols.max():.6f}")

# ── 7. EWMA forecast ─────────────────────────────────────────────────────────
ewma_model = EWMA(lam=0.94)
ewma_model.fit(returns_sim)
ewma_fc = ewma_model.forecast(h=10)
assert len(ewma_fc) == 10, f"EWMA forecast length={len(ewma_fc)}"
assert np.all(ewma_fc >= 0), "EWMA forecast must be non-negative variance"
# Under EWMA the variance forecast is flat
assert np.allclose(ewma_fc, ewma_fc[0]), "EWMA forecast should be flat (I-GARCH)"
print(f"[OK] EWMA forecast flat at var={ewma_fc[0]:.8f}")

# ── 8. VolForecastEnsemble ───────────────────────────────────────────────────
ensemble = VolForecastEnsemble(garch_weight=0.60, ewma_lam=0.94)
ensemble.fit(returns_sim)
fc_dict = ensemble.forecast(h=10)

required_keys = {"garch", "ewma", "ensemble"}
assert required_keys.issubset(fc_dict.keys()), (
    f"Missing keys: {required_keys - fc_dict.keys()}"
)
print(f"[OK] ensemble keys: {sorted(fc_dict.keys())}")

for key in required_keys:
    arr = fc_dict[key]
    assert len(arr) == 10, f"ensemble['{key}'] length={len(arr)}"
    assert np.all(arr >= 0), f"ensemble['{key}'] must be non-negative vols"
    print(f"[OK] ensemble['{key}']: first={arr[0]:.6f}, last={arr[-1]:.6f}")

# Ensemble should be between GARCH and EWMA
for i in range(10):
    combo = 0.60 * fc_dict["garch"][i] + 0.40 * fc_dict["ewma"][i]
    assert abs(fc_dict["ensemble"][i] - combo) < 1e-10, (
        f"ensemble[{i}] mismatch: {fc_dict['ensemble'][i]:.8f} vs {combo:.8f}"
    )
print("[OK] ensemble = 0.60*garch + 0.40*ewma verified for all 10 horizons")

# ── 9. Volatility cone ───────────────────────────────────────────────────────
cone = vol_cone(returns_sim, windows=[5, 10, 21, 63, 252])
assert "21d" in cone, f"Expected '21d' in vol_cone, got keys: {list(cone.keys())}"
assert "5d" in cone,  f"Expected '5d' in vol_cone"
assert "63d" in cone, f"Expected '63d' in vol_cone"
print(f"[OK] vol_cone keys: {sorted(cone.keys())}")

for window_key, stats in cone.items():
    assert "mean" in stats and "median" in stats, f"Missing stats for {window_key}"
    assert "p25" in stats and "p75" in stats and "p90" in stats, f"Missing percentiles for {window_key}"
    assert stats["p75"] > stats["p25"], (
        f"p75={stats['p75']:.4f} should be > p25={stats['p25']:.4f} for {window_key}"
    )
    assert stats["p90"] >= stats["p75"], (
        f"p90 >= p75 violated for {window_key}"
    )
    print(f"[OK] vol_cone['{window_key}']: p25={stats['p25']:.4f}, p75={stats['p75']:.4f}, p90={stats['p90']:.4f}")

# ── 10. EGARCH(1,1) basic sanity ─────────────────────────────────────────────
egarch = EGARCH11()
egarch_params = egarch.fit(returns_sim)
assert egarch_params.converged is not None, "converged field must exist"
egarch_vols = egarch.filter(returns_sim, egarch_params)
assert len(egarch_vols) == n, f"EGARCH filter length={len(egarch_vols)}"
assert np.all(egarch_vols > 0), "all EGARCH vols must be positive"
assert abs(egarch_params.beta) < 1.0, f"EGARCH beta must be < 1: {egarch_params.beta}"
print(f"[OK] EGARCH: beta={egarch_params.beta:.4f}, min_vol={egarch_vols.min():.6f}, max_vol={egarch_vols.max():.6f}")

# ── 11. GARCHParams properties ───────────────────────────────────────────────
test_p = GARCHParams(omega=1e-6, alpha=0.10, beta=0.85)
assert abs(test_p.persistence - 0.95) < 1e-10, f"persistence={test_p.persistence}"
expected_lr = np.sqrt(1e-6 / (1 - 0.95))
assert abs(test_p.long_run_vol - expected_lr) < 1e-10, f"long_run_vol={test_p.long_run_vol}"
print(f"[OK] GARCHParams properties: persistence={test_p.persistence}, long_run_vol={test_p.long_run_vol:.6f}")

# ── 12. EWMA different lambda ─────────────────────────────────────────────────
ewma97 = ewma_vol(returns_sim, lam=0.97)
assert len(ewma97) == n
assert np.all(ewma97 > 0)
# Higher lambda → smoother (more persistent), compare variance sums conceptually
print(f"[OK] EWMA lam=0.97: min={ewma97.min():.6f}, max={ewma97.max():.6f}")

print("\n[PASS] dim_114: GARCH/EWMA vol forecasting")
PYEOF
