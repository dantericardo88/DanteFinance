#!/usr/bin/env bash
# dim_081: Portfolio optimizer — mean-variance pure math
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.spm.portfolio_optimizer_v3 import (
    MeanVarianceOptimizer, CovarianceEstimator, MeanEstimator, OptResult
)

np.random.seed(42)

# Build synthetic return data for 4 assets
n_obs = 252
n_assets = 4
asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Generate correlated returns: tech-heavy with positive correlation
base = np.random.normal(0.0004, 0.012, n_obs)
returns_data = np.column_stack([
    0.8 * base + np.random.normal(0.0003, 0.008, n_obs),
    0.9 * base + np.random.normal(0.0002, 0.007, n_obs),
    0.7 * base + np.random.normal(0.0002, 0.009, n_obs),
    0.6 * base + np.random.normal(0.0001, 0.010, n_obs),
])
returns_df = pd.DataFrame(returns_data, columns=asset_names)

# Test CovarianceEstimator.sample_covariance
cov = CovarianceEstimator.sample_covariance(returns_df)
assert cov.shape == (4, 4), f"Cov matrix should be 4x4: {cov.shape}"
assert np.all(np.linalg.eigvalsh(cov) > -1e-9), "Covariance matrix should be PSD"
print(f"[OK] Sample covariance: shape={cov.shape} min_eigenvalue={np.linalg.eigvalsh(cov).min():.2e}")

# Test Ledoit-Wolf shrinkage
lw_cov = CovarianceEstimator.ledoit_wolf_shrinkage(returns_df)
assert lw_cov.shape == (4, 4), "LW cov should be 4x4"
print(f"[OK] Ledoit-Wolf covariance: min_eigval={np.linalg.eigvalsh(lw_cov).min():.2e}")

# Test EWMA covariance
ewm_cov = CovarianceEstimator.exponential_weighted_covariance(returns_df, lambda_=0.94)
assert ewm_cov.shape == (4, 4), "EWM cov should be 4x4"
print(f"[OK] EWMA covariance (lambda=0.94): diag={np.diag(ewm_cov).round(6)}")

# Test MeanEstimator.sample_mean
mu = MeanEstimator.sample_mean(returns_df)
assert len(mu) == 4, f"Expected 4 mean estimates: {len(mu)}"
assert len(mu) == 4, "All 4 mean returns should exist"
print(f"[OK] Sample mean returns (annualized): {mu.round(4)}")

# Test MeanVarianceOptimizer.max_sharpe
optimizer = MeanVarianceOptimizer(allow_short=False)
result = optimizer.max_sharpe(mu=mu, Sigma=cov, rf=0.05/252, asset_names=asset_names)
assert isinstance(result, OptResult), "Should return OptResult"
assert len(result.weights) == 4, "Should have 4 weights"
assert abs(result.weights.sum() - 1.0) < 1e-6, f"Weights should sum to 1: {result.weights.sum()}"
assert all(result.weights >= -1e-9), f"All weights should be non-negative: {result.weights}"
assert result.sharpe_ratio > 0, f"Sharpe ratio should be positive: {result.sharpe_ratio}"
print(f"[OK] max_sharpe: weights={result.weights.round(4)} sharpe={result.sharpe_ratio:.4f}")

# Test minimum variance
from sentinel.spm.portfolio_optimizer_v3 import MeanVarianceOptimizer
result_mv = optimizer.min_variance(Sigma=cov, asset_names=asset_names)
assert abs(result_mv.weights.sum() - 1.0) < 1e-6, "Min-var weights should sum to 1"
assert result_mv.expected_volatility <= result.expected_volatility + 1e-6, \
    f"Min-var vol should be <= max-sharpe vol: {result_mv.expected_volatility:.4f} vs {result.expected_volatility:.4f}"
print(f"[OK] min_variance: vol={result_mv.expected_volatility:.4f} <= max_sharpe vol={result.expected_volatility:.4f}")

# Test to_dict
d = result.to_dict()
assert "method" in d, "OptResult.to_dict() should have 'method'"
assert "weights" in d, "OptResult.to_dict() should have 'weights'"
assert "sharpe_ratio" in d, "OptResult.to_dict() should have 'sharpe_ratio'"
print(f"[OK] OptResult.to_dict(): {list(d.keys())}")

print("\n[PASS] dim_081: Portfolio optimizer")
PYEOF
