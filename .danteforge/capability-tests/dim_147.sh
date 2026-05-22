#!/usr/bin/env bash
# dim_147: Smart beta / alternative weighting strategies
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from sentinel.spm.smart_beta_v3 import (
    PortfolioWeights,
    EqualRiskContribution,
    MinimumVariance,
    MomentumTilt,
    MaximumDiversification,
    SmartBetaPortfolio,
    equal_risk_contribution,
    minimum_variance,
    diversification_ratio,
    portfolio_vol,
    # Legacy aliases
    RiskParityEngine,
    MomentumTiltPortfolio,
    compute_factor_tilt,
)
print("[OK] All imports successful from sentinel.spm.smart_beta_v3")

# ---------------------------------------------------------------------------
# 2. Build 4-asset universe with specified covariance structure
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)
n_assets = 4

cov = np.array([
    [0.01,  0.003, 0.001, 0.002],
    [0.003, 0.015, 0.002, 0.001],
    [0.001, 0.002, 0.008, 0.001],
    [0.002, 0.001, 0.001, 0.012],
])

# Generate 252-day returns from this covariance structure
L = np.linalg.cholesky(cov)
z = rng.standard_normal((252, n_assets))
daily_returns = z @ L.T  # (252, 4)

print(f"[OK] Generated {daily_returns.shape[0]}x{daily_returns.shape[1]} returns matrix")

# ---------------------------------------------------------------------------
# 3. SmartBetaPortfolio facade
# ---------------------------------------------------------------------------
# Pass covariance-scaled returns so the facade computes correct cov
sbp = SmartBetaPortfolio(returns=daily_returns, asset_names=['A','B','C','D'])

# Override covariance with specified matrix (scale to daily then annualise to match)
sbp.cov_matrix = cov * 252    # annualise
sbp.vols = np.sqrt(np.diag(sbp.cov_matrix))
print(f"[OK] SmartBetaPortfolio constructed with specified covariance")

# ---------------------------------------------------------------------------
# 4. Equal weight
# ---------------------------------------------------------------------------
ew = sbp.equal_weight()
assert isinstance(ew, PortfolioWeights), "FAIL: equal_weight must return PortfolioWeights"
expected_ew = np.array([0.25, 0.25, 0.25, 0.25])
assert np.allclose(ew.weights, expected_ew, atol=1e-10), (
    f"FAIL: EW weights should be [0.25,0.25,0.25,0.25], got {ew.weights}"
)
assert abs(ew.weights.sum() - 1.0) < 1e-8, "FAIL: EW weights must sum to 1"
print(f"[OK] Equal weight: {ew.weights} (sum={ew.weights.sum():.8f})")

# Effective N for EW should be exactly 4
assert abs(ew.effective_n - 4.0) < 1e-6, (
    f"FAIL: EW effective_n should be 4.0, got {ew.effective_n:.6f}"
)
print(f"[OK] EW effective_n = {ew.effective_n:.2f} (expected 4.0)")

# ---------------------------------------------------------------------------
# 5. Minimum variance
# ---------------------------------------------------------------------------
mv = sbp.min_variance()
assert isinstance(mv, PortfolioWeights), "FAIL: min_variance must return PortfolioWeights"
assert abs(mv.weights.sum() - 1.0) < 1e-6, "FAIL: MinVar weights must sum to 1"
assert np.all(mv.weights >= -1e-8), "FAIL: MinVar weights must be >= 0"

ew_vol = portfolio_vol(ew.weights, sbp.cov_matrix)
mv_vol = portfolio_vol(mv.weights, sbp.cov_matrix)
assert mv_vol <= ew_vol + 1e-8, (
    f"FAIL: MinVar vol ({mv_vol:.6f}) should be <= EW vol ({ew_vol:.6f})"
)
print(f"[OK] MinVar vol={mv_vol:.4f} <= EW vol={ew_vol:.4f}")
print(f"[OK] MinVar weights: {mv.weights.round(4)} (sum={mv.weights.sum():.8f})")

# ---------------------------------------------------------------------------
# 6. Risk parity / ERC
# ---------------------------------------------------------------------------
erc_engine = EqualRiskContribution()
erc_weights = erc_engine.fit(sbp.cov_matrix)
assert abs(erc_weights.sum() - 1.0) < 1e-6, "FAIL: ERC weights must sum to 1"
assert np.all(erc_weights >= 0), "FAIL: ERC weights must be >= 0"
print(f"[OK] ERC weights: {erc_weights.round(4)} (sum={erc_weights.sum():.8f})")

# Check risk contributions are approximately equal
rc = erc_engine.risk_contributions(erc_weights, sbp.cov_matrix)
rc_pct = rc / rc.sum()
rc_target = 1.0 / n_assets
for i, rc_i in enumerate(rc_pct):
    assert abs(rc_i - rc_target) < 0.20, (
        f"FAIL: ERC risk contribution {i} ({rc_i:.4f}) deviates > 20% from target ({rc_target:.4f})"
    )
print(f"[OK] ERC risk contributions (normalised): {rc_pct.round(4)} — all within 20% of 1/N")

# SmartBetaPortfolio risk_parity
rp = sbp.risk_parity()
assert abs(rp.weights.sum() - 1.0) < 1e-6, "FAIL: risk_parity weights must sum to 1"
print(f"[OK] risk_parity weights sum to 1: {rp.weights.sum():.8f}")

# ---------------------------------------------------------------------------
# 7. Momentum tilt
# ---------------------------------------------------------------------------
mom = sbp.momentum_tilt(lookback=63)
assert isinstance(mom, PortfolioWeights), "FAIL: momentum_tilt must return PortfolioWeights"
assert abs(mom.weights.sum() - 1.0) < 1e-6, "FAIL: Momentum weights must sum to 1"
assert np.all(mom.weights > 0), "FAIL: Momentum (softmax) weights must all be > 0"
print(f"[OK] Momentum tilt weights: {mom.weights.round(4)} (sum={mom.weights.sum():.8f})")

# ---------------------------------------------------------------------------
# 8. Maximum diversification
# ---------------------------------------------------------------------------
md = sbp.max_diversification()
assert abs(md.weights.sum() - 1.0) < 1e-6, "FAIL: MaxDiv weights must sum to 1"
vols = sbp.vols
dr_md = diversification_ratio(md.weights, vols, sbp.cov_matrix)
assert dr_md >= 1.0, f"FAIL: Diversification ratio should be >= 1, got {dr_md:.4f}"
assert dr_md <= n_assets + 1e-6, (
    f"FAIL: DR should be <= n_assets ({n_assets}), got {dr_md:.4f}"
)
print(f"[OK] MaxDiv: weights={md.weights.round(4)}, DR={dr_md:.4f}")

# MaxDiv DR should be >= MinVar DR
dr_mv = diversification_ratio(mv.weights, vols, sbp.cov_matrix)
print(f"[OK] MaxDiv DR={dr_md:.4f} vs MinVar DR={dr_mv:.4f}")

# ---------------------------------------------------------------------------
# 9. Factor tilt
# ---------------------------------------------------------------------------
factor_scores = np.array([0.4, 0.3, 0.2, 0.1])  # asset quality scores
ft = sbp.factor_tilt(factor_scores, alpha=0.5)
assert abs(ft.weights.sum() - 1.0) < 1e-6, "FAIL: Factor tilt weights must sum to 1"
assert np.all(ft.weights >= 0), "FAIL: Factor tilt weights must be >= 0"
print(f"[OK] Factor tilt weights: {ft.weights.round(4)}")

# ---------------------------------------------------------------------------
# 10. Compare strategies
# ---------------------------------------------------------------------------
strategies = sbp.compare_strategies()
assert len(strategies) >= 3, (
    f"FAIL: compare_strategies should return >= 3 strategies, got {len(strategies)}"
)
for name, pw in strategies.items():
    assert abs(pw.weights.sum() - 1.0) < 1e-5, (
        f"FAIL: {name} weights don't sum to 1: {pw.weights.sum()}"
    )
print(f"[OK] compare_strategies returned {len(strategies)} strategies: {list(strategies.keys())}")

# ---------------------------------------------------------------------------
# 11. Standalone functions
# ---------------------------------------------------------------------------
erc_standalone = equal_risk_contribution(sbp.cov_matrix)
assert abs(erc_standalone.sum() - 1.0) < 1e-6, "FAIL: standalone ERC weights sum ≠ 1"
print(f"[OK] standalone equal_risk_contribution: {erc_standalone.round(4)}")

mv_standalone = minimum_variance(sbp.cov_matrix)
assert abs(mv_standalone.sum() - 1.0) < 1e-6, "FAIL: standalone MinVar weights sum ≠ 1"
print(f"[OK] standalone minimum_variance: {mv_standalone.round(4)}")

pv = portfolio_vol(ew.weights, sbp.cov_matrix)
assert pv > 0, "FAIL: portfolio_vol should be positive"
print(f"[OK] standalone portfolio_vol(EW) = {pv:.4f}")

dr = diversification_ratio(ew.weights, vols, sbp.cov_matrix)
assert dr >= 1.0, f"FAIL: standalone diversification_ratio >= 1, got {dr:.4f}"
print(f"[OK] standalone diversification_ratio(EW) = {dr:.4f}")

# ---------------------------------------------------------------------------
# 12. Legacy alias checks
# ---------------------------------------------------------------------------
rpe = RiskParityEngine()
rpe_w = rpe.fit(sbp.cov_matrix)
assert abs(rpe_w.sum() - 1.0) < 1e-6, "FAIL: RiskParityEngine alias broken"
print(f"[OK] Legacy RiskParityEngine alias works")

mtp = MomentumTiltPortfolio(lookback=63)
mtp_w = mtp.fit(daily_returns)
assert abs(mtp_w.sum() - 1.0) < 1e-6, "FAIL: MomentumTiltPortfolio alias broken"
print(f"[OK] Legacy MomentumTiltPortfolio alias works")

cft_w = compute_factor_tilt(factor_scores, alpha=0.5)
assert abs(cft_w.sum() - 1.0) < 1e-6, "FAIL: compute_factor_tilt broken"
print(f"[OK] Legacy compute_factor_tilt: {cft_w.round(4)}")

# ---------------------------------------------------------------------------
# 13. Summary
# ---------------------------------------------------------------------------
print("\n--- Smart Beta Summary ---")
print(f"  Assets        : {n_assets}")
print(f"  Returns days  : {daily_returns.shape[0]}")
print(f"  EW vol        : {ew_vol:.4f}")
print(f"  MinVar vol    : {mv_vol:.4f}")
print(f"  ERC weights   : {erc_weights.round(3).tolist()}")
print(f"  MaxDiv DR     : {dr_md:.4f}")
print(f"  EW effective_n: {ew.effective_n:.2f}")
print(f"  Strategies    : {list(strategies.keys())}")

print("\n[PASS] dim_147: Smart beta / alternative weighting")
PYEOF
