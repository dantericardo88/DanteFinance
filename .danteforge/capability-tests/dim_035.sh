#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import math

from sentinel.sfe.treasury_yield_v3 import (
    NelsonSiegelSvensson,
    NSSParams,
    YieldCurve,
    ForwardCurve,
)

# Test NSS _nss_yield at zero maturity returns beta0 + beta1
params = [4.0, -2.0, 1.5, 0.5, 1.5, 3.0]  # beta0, beta1, beta2, beta3, lam1, lam2
y0 = NelsonSiegelSvensson._nss_yield(0.0, params)
assert abs(y0 - (4.0 + (-2.0))) < 1e-9, f"Expected {4.0 + -2.0}, got {y0}"
print(f"[OK] NSS yield at tau=0 = beta0+beta1 = {y0:.4f}")

# Test NSS _nss_yield at long maturity approaches beta0
y30 = NelsonSiegelSvensson._nss_yield(30.0, params)
assert abs(y30 - 4.0) < 0.5, f"Expected ~4.0, got {y30}"
print(f"[OK] NSS yield at tau=30yr approaches beta0=4.0 (got {y30:.4f})")

# Test _nss_yield_vec returns a list of the same length as tenors
tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
yields = NelsonSiegelSvensson._nss_yield_vec(tenors, params)
assert len(yields) == len(tenors)
print("[OK] _nss_yield_vec returns correct length list")

# Test NSSParams dataclass
nss = NSSParams(
    beta0=4.0,
    beta1=-2.0,
    beta2=1.5,
    beta3=0.5,
    lambda1=1.5,
    lambda2=3.0,
    rmse=0.05,
    n_tenors=8,
)
assert nss.beta0 == 4.0
assert nss.n_tenors == 8
print("[OK] NSSParams dataclass created successfully")

# Test YieldCurve dataclass
yc = YieldCurve(
    as_of="2024-03-15",
    nominal={"3M": 5.25, "2Y": 4.80, "10Y": 4.20, "30Y": 4.35},
    real_tips={"5Y": 2.0, "10Y": 1.8},
    breakeven={"5Y": 2.3, "10Y": 2.4},
    tenors_years=[0.25, 2.0, 10.0, 30.0],
    yields_pct=[5.25, 4.80, 4.20, 4.35],
    nss_params=nss,
    regime="normal",
    slope_2s10s=-60.0,
    slope_3m10y=-105.0,
)
assert yc.slope_2s10s == -60.0
assert yc.nominal["10Y"] == 4.20
print("[OK] YieldCurve dataclass created successfully")

# Test ForwardCurve dataclass
fc = ForwardCurve(
    as_of="2024-03-15",
    forward_rates={"1Y1Y": 4.5, "2Y3Y": 4.2, "5Y5Y": 3.9},
    policy_path={"3M": 5.25, "6M": 4.75, "12M": 4.25},
    five_year_five_year=2.35,
)
assert fc.five_year_five_year == 2.35
assert fc.forward_rates["5Y5Y"] == 3.9
print("[OK] ForwardCurve dataclass created successfully")

print("[PASS]")
PYEOF
