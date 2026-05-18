#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import math

from sentinel.sfe.credit_spread_v3 import (
    _norm_cdf,
    _norm_pdf,
    MertonModel,
    MertonResult,
    KMVResult,
    IssuerCreditProfile,
)
from datetime import date

# Test _norm_cdf boundary values
assert abs(_norm_cdf(0.0) - 0.5) < 1e-6, "CDF(0) should be 0.5"
assert _norm_cdf(5.0) > 0.999, "CDF(5) should be ~1"
assert _norm_cdf(-5.0) < 0.001, "CDF(-5) should be ~0"
print("[OK] _norm_cdf boundary values correct")

# Test _norm_pdf: peaks at 0
pdf0 = _norm_pdf(0.0)
assert abs(pdf0 - 1.0 / math.sqrt(2 * math.pi)) < 1e-9
print(f"[OK] _norm_pdf(0) = {pdf0:.6f} ~= 0.398942")

# Test MertonModel.solve_firm_value_vol with a simple case
# E = 500 (market cap $500M), sigma_E = 0.30 (30% equity vol)
# D = 300 (debt $300M), T = 1 year, r = 0.05
model = MertonModel()
V, sigma_V = model.solve_firm_value_vol(
    equity_value=500.0,
    equity_vol=0.30,
    debt_face=300.0,
    T=1.0,
    r=0.05,
)
# Firm value V should be in range [E, E+D] ~ [500, 800]
assert 450.0 < V < 900.0, f"Firm value out of range: {V}"
# Asset vol sigma_V should be less than equity vol (leverage reduces vol)
assert 0.01 < sigma_V < 0.30, f"Asset vol out of range: {sigma_V}"
print(f"[OK] MertonModel solved: V={V:.1f}M, sigma_V={sigma_V:.4f}")

# Test MertonResult dataclass
mr = MertonResult(
    ticker="AAPL",
    equity_value=3000.0,
    equity_vol=0.25,
    firm_value=3200.0,
    asset_vol=0.20,
    debt_face=200.0,
    time_horizon=1.0,
    risk_free_rate=0.05,
    d1=5.5,
    d2=5.3,
    distance_to_default=5.3,
    risk_neutral_pd=0.0001,
    physical_pd=0.0001,
    credit_spread_bps=5.0,
    credit_quality="AAA",
    leverage_ratio=0.0625,
    as_of=date(2024, 3, 15),
)
assert mr.ticker == "AAPL"
assert mr.credit_quality == "AAA"
assert mr.credit_spread_bps == 5.0
print("[OK] MertonResult dataclass created successfully")

# Test KMVResult dataclass
kmv = KMVResult(
    ticker="TSLA",
    equity_value=800.0,
    equity_vol=0.50,
    debt_short=40.0,
    debt_long=10.0,
    default_point=45.0,
    firm_value=850.0,
    asset_vol=0.45,
    kmv_dd=5.8,
    edf=0.02,
    credit_quality="BB",
    as_of=date(2024, 3, 15),
)
assert kmv.edf == 0.02
assert kmv.credit_quality == "BB"
print("[OK] KMVResult dataclass created successfully")

# Verify Merton model _merton_call produces sensible call value
E_call, d1, d2 = model._merton_call(V=800.0, D=300.0, T=1.0, r=0.05, sigma_V=0.20)
assert E_call > 0, "Call value should be positive"
assert d1 > d2, "d1 should be > d2"
print(f"[OK] _merton_call: E={E_call:.2f}, d1={d1:.4f}, d2={d2:.4f}")

print("[PASS]")
PYEOF
