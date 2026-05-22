#!/usr/bin/env bash
# dim_117: Cross-asset vol correlation — DCC-GARCH, contagion, spillover
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ── Imports ──────────────────────────────────────────────────────────────────
from sentinel.sma.cross_asset_vol_v3 import (
    GARCHResult,
    DCCResult,
    SpilloverTable,
    DCCGARCHModel,
    ContagionAnalyzer,
    VolatilitySpillover,
    CorrelationMetrics,
    garch11_vol,
    rolling_correlation,
    contagion_test,
    correlation_regime,
)
print("[OK] all imports successful")

# ── 1. Generate 3-asset correlated returns (T=300) ───────────────────────────
rng = np.random.default_rng(42)
T, n = 300, 3
true_corr = np.array([[1.0, 0.7, 0.3],
                       [0.7, 1.0, 0.5],
                       [0.3, 0.5, 1.0]])
L = np.linalg.cholesky(true_corr)
z = rng.standard_normal((T, n))
returns = (z @ L.T) * 0.01   # scale to ~1% daily vol
print(f"[OK] synthetic 3-asset returns: shape={returns.shape}, "
      f"means={returns.mean(axis=0).round(5)}")

# ── 2. Rolling correlation: shape (240, 3, 3), diagonals = 1.0 ───────────────
rc = rolling_correlation(returns, window=60)
assert rc.shape == (240, 3, 3), f"rolling_correlation shape mismatch: {rc.shape}"
for t in range(0, 240, 40):
    diag = np.diag(rc[t])
    assert np.allclose(diag, 1.0, atol=1e-10), f"diagonal != 1.0 at t={t}: {diag}"
print(f"[OK] rolling_correlation shape={rc.shape}, diagonals=1.0")

# ── 3. GARCH on asset 1: conditional vol > 0 everywhere ─────────────────────
vol1 = garch11_vol(returns[:, 0])
assert len(vol1) == T, f"garch11_vol length mismatch: {len(vol1)} vs {T}"
assert np.all(vol1 > 0), f"conditional vol not > 0 everywhere: min={vol1.min()}"
print(f"[OK] GARCH asset1: len={len(vol1)}, min_vol={vol1.min():.6f}, max_vol={vol1.max():.6f}")

# ── 4. DCCGARCHModel.fit_garch ───────────────────────────────────────────────
model = DCCGARCHModel()
gr = model.fit_garch(returns[:, 0])
assert isinstance(gr, GARCHResult), f"Expected GARCHResult, got {type(gr)}"
assert gr.omega > 0, f"omega must be > 0: {gr.omega}"
assert gr.alpha > 0, f"alpha must be > 0: {gr.alpha}"
assert gr.beta > 0, f"beta must be > 0: {gr.beta}"
assert gr.alpha + gr.beta < 1.0, f"alpha+beta must be < 1: {gr.alpha+gr.beta}"
assert np.all(gr.conditional_vols > 0), "conditional_vols must be > 0"
print(f"[OK] GARCHResult: omega={gr.omega:.2e}, alpha={gr.alpha:.4f}, beta={gr.beta:.4f}, a+b={gr.alpha+gr.beta:.4f}")

# ── 5. DCC fit: a+b < 1, shape (300,3,3), diagonals = 1.0 ───────────────────
dcc = model.fit_dcc(returns)
assert isinstance(dcc, DCCResult), f"Expected DCCResult, got {type(dcc)}"
assert dcc.a + dcc.b < 1.0, f"a+b must be < 1: a={dcc.a}, b={dcc.b}, sum={dcc.a+dcc.b}"
assert dcc.conditional_correlations.shape == (T, n, n), \
    f"conditional_correlations shape mismatch: {dcc.conditional_correlations.shape}"
for t in [0, 100, 200, 299]:
    diag = np.diag(dcc.conditional_correlations[t])
    assert np.allclose(diag, 1.0, atol=1e-8), f"DCC diagonal != 1.0 at t={t}: {diag}"
print(f"[OK] DCCResult: a={dcc.a:.4f}, b={dcc.b:.4f}, a+b={dcc.a+dcc.b:.4f}, "
      f"corr shape={dcc.conditional_correlations.shape}")

# ── 6. Average correlation time series: len=300, values in [-1, 1] ───────────
avg_corr = dcc.average_correlation()
assert len(avg_corr) == T, f"average_correlation length mismatch: {len(avg_corr)}"
assert np.all(avg_corr >= -1.0) and np.all(avg_corr <= 1.0), \
    f"average_correlation out of [-1,1]: min={avg_corr.min()}, max={avg_corr.max()}"
print(f"[OK] average_correlation: len={len(avg_corr)}, min={avg_corr.min():.4f}, max={avg_corr.max():.4f}")

# ── 7. correlation_at ────────────────────────────────────────────────────────
Rt = dcc.correlation_at(150)
assert Rt.shape == (n, n), f"correlation_at shape mismatch: {Rt.shape}"
assert np.allclose(np.diag(Rt), 1.0, atol=1e-8), "correlation_at diagonal != 1.0"
print(f"[OK] correlation_at(150): shape={Rt.shape}")

# ── 8. CorrelationMetrics: average_pairwise > 0 ──────────────────────────────
cm = CorrelationMetrics()
sample_corr = np.corrcoef(returns.T)
np.fill_diagonal(sample_corr, 1.0)

avg_pw = cm.average_pairwise(sample_corr)
assert avg_pw > 0, f"average_pairwise should be > 0 for positively correlated assets: {avg_pw}"
print(f"[OK] average_pairwise={avg_pw:.4f} > 0")

# ── 9. eigenvalue_dispersion in (0, 1] ───────────────────────────────────────
ed = cm.eigenvalue_dispersion(sample_corr)
assert 0.0 < ed <= 1.0, f"eigenvalue_dispersion must be in (0,1]: {ed}"
print(f"[OK] eigenvalue_dispersion={ed:.4f} in (0,1]")

# ── 10. effective_n_factors ──────────────────────────────────────────────────
en = cm.effective_n_factors(sample_corr)
assert 1.0 <= en <= n + 0.01, f"effective_n_factors={en} should be in [1, n={n}]"
print(f"[OK] effective_n_factors={en:.4f}")

# ── 11. market_correlation ───────────────────────────────────────────────────
mc = cm.market_correlation(returns)
assert len(mc) == n, f"market_correlation length={len(mc)}"
assert np.all(mc >= -1.0) and np.all(mc <= 1.0), f"market_correlation out of [-1,1]: {mc}"
print(f"[OK] market_correlation={mc.round(4)}")

# ── 12. Contagion test: crisis higher variance, p_value finite ───────────────
crisis_returns = rng.standard_normal((80, 2)) * 0.03   # higher vol
tranquil_returns = rng.standard_normal((100, 2)) * 0.01
analyzer = ContagionAnalyzer()
ct = analyzer.forbes_rigobon_test(crisis_returns, tranquil_returns)
required_keys = {"rho_crisis", "rho_tranquil", "rho_crisis_adj", "contagion_detected", "p_value"}
assert required_keys.issubset(ct.keys()), f"Missing keys: {required_keys - ct.keys()}"
assert np.isfinite(ct["p_value"]), f"p_value must be finite: {ct['p_value']}"
assert 0.0 <= ct["p_value"] <= 1.0, f"p_value must be in [0,1]: {ct['p_value']}"
assert np.isfinite(ct["rho_crisis_adj"]), "rho_crisis_adj must be finite"
print(f"[OK] contagion_test: rho_crisis={ct['rho_crisis']:.4f}, "
      f"rho_tranquil={ct['rho_tranquil']:.4f}, "
      f"rho_crisis_adj={ct['rho_crisis_adj']:.4f}, "
      f"p_value={ct['p_value']:.4f}, contagion={ct['contagion_detected']}")

# ── 13. contagion_test convenience function ──────────────────────────────────
ct2 = contagion_test(returns[:, 0], returns[:, 1], split=150)
assert np.isfinite(ct2["p_value"]), f"contagion_test p_value must be finite"
print(f"[OK] contagion_test convenience: p_value={ct2['p_value']:.4f}")

# ── 14. Spillover: realized vol spillover table shape (3,3), index in [0,100] ─
spillover = VolatilitySpillover()
realized_vol = np.abs(returns)   # (300, 3) proxy for realized vol
tbl = spillover.realized_vol_spillover(realized_vol, asset_names=["A", "B", "C"])
assert isinstance(tbl, SpilloverTable), f"Expected SpilloverTable"
assert tbl.from_to.shape == (n, n), f"from_to shape mismatch: {tbl.from_to.shape}"
assert 0.0 <= tbl.spillover_index <= 100.0, \
    f"spillover_index must be in [0,100]: {tbl.spillover_index}"
assert len(tbl.net_spillovers) == n, f"net_spillovers length={len(tbl.net_spillovers)}"
print(f"[OK] realized_vol_spillover: shape={tbl.from_to.shape}, "
      f"spillover_index={tbl.spillover_index:.2f}%, "
      f"transmitters={tbl.transmitters()}, receivers={tbl.receivers()}")

# ── 15. Diebold-Yilmaz VAR-based spillover ───────────────────────────────────
dyz = spillover.diebold_yilmaz(returns, lag=2, horizon=5,
                                asset_names=["A", "B", "C"])
assert isinstance(dyz, SpilloverTable), f"Expected SpilloverTable"
assert dyz.from_to.shape == (n, n), f"Diebold-Yilmaz from_to shape: {dyz.from_to.shape}"
assert 0.0 <= dyz.spillover_index <= 100.0, \
    f"Diebold-Yilmaz spillover_index: {dyz.spillover_index}"
print(f"[OK] diebold_yilmaz: shape={dyz.from_to.shape}, "
      f"spillover_index={dyz.spillover_index:.2f}%")

# ── 16. correlation_regime ───────────────────────────────────────────────────
regimes = correlation_regime(returns, n_regimes=2, window=40)
assert len(regimes) == T, f"correlation_regime length={len(regimes)}"
assert set(regimes).issubset({0, 1}), f"regimes must be 0/1: {set(regimes)}"
print(f"[OK] correlation_regime: len={len(regimes)}, "
      f"high_corr_pct={regimes.mean()*100:.1f}%")

# ── 17. correlation_breakdown ────────────────────────────────────────────────
bd = analyzer.correlation_breakdown(returns, crisis_start=100, crisis_end=200)
assert "pre" in bd and "crisis" in bd and "post" in bd
assert bd["pre"].shape == (n, n)
assert bd["crisis"].shape == (n, n)
print(f"[OK] correlation_breakdown: pre avg={bd['pre'][0,1]:.3f}, "
      f"crisis avg={bd['crisis'][0,1]:.3f}")

# ── 18. DCCGARCHModel rolling_correlation ────────────────────────────────────
rc2 = model.rolling_correlation(returns, window=60)
assert rc2.shape == (240, n, n), f"rolling_correlation shape: {rc2.shape}"
print(f"[OK] DCCGARCHModel.rolling_correlation: shape={rc2.shape}")

print("\n[PASS] dim_117: Cross-asset vol correlation")
PYEOF
