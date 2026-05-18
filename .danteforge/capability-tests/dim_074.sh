#!/usr/bin/env bash
# dim_074: Fixed income screener — pure bond math & filter logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math

from sentinel.sfe.fixed_income_screener import (
    _TREASURY_CURVE, _IG_SPREADS, _HY_SPREADS, _MUNI_TEY_FACTOR,
    _interp_treasury_yield, _bond_price,
)

# Test Treasury curve interpolation (pure math, no network)
y_3m = _interp_treasury_yield(0.25)
y_10y = _interp_treasury_yield(10.0)
y_5y = _interp_treasury_yield(5.0)
assert 4.0 < y_3m < 6.0, f"3M Treasury yield out of range: {y_3m}"
assert 4.0 < y_10y < 6.0, f"10Y Treasury yield out of range: {y_10y}"
print(f"[OK] Treasury curve: 3M={y_3m:.2f}% 5Y={y_5y:.2f}% 10Y={y_10y:.2f}%")

# Test interpolation between known points
y_interp = _interp_treasury_yield(7.5)  # between 7Y and 10Y
y_7 = _TREASURY_CURVE[7.0]
y_10 = _TREASURY_CURVE[10.0]
assert min(y_7, y_10) <= y_interp <= max(y_7, y_10) + 0.01, f"Interpolation out of bounds: {y_interp}"
print(f"[OK] Interpolated 7.5Y yield = {y_interp:.3f}% (between {y_7}% and {y_10}%)")

# Test bond price calculation (pure math)
# Par bond: coupon = YTM → price = 100
par_price = _bond_price(ytm_pct=4.5, coupon_rate_pct=4.5, maturity_years=5.0)
assert abs(par_price - 1000.0) < 1.0, f"Par bond price should be ~1000, got {par_price:.2f}"
print(f"[OK] Par bond price: {par_price:.2f} (should be ~1000)")

# Premium bond: coupon > YTM → price > par
premium_price = _bond_price(ytm_pct=3.0, coupon_rate_pct=5.0, maturity_years=5.0)
assert premium_price > 1000.0, f"Premium bond should price above par: {premium_price:.2f}"
print(f"[OK] Premium bond price: {premium_price:.2f} (should be >1000)")

# Discount bond: coupon < YTM → price < par
discount_price = _bond_price(ytm_pct=6.0, coupon_rate_pct=4.0, maturity_years=5.0)
assert discount_price < 1000.0, f"Discount bond should price below par: {discount_price:.2f}"
print(f"[OK] Discount bond price: {discount_price:.2f} (should be <1000)")

# Test IG spread lookup
assert _IG_SPREADS["AAA"] < _IG_SPREADS["BBB"], "AAA should have lower spread than BBB"
assert _IG_SPREADS["BBB-"] > 150, f"BBB- spread too low: {_IG_SPREADS['BBB-']}"
print(f"[OK] IG spreads: AAA={_IG_SPREADS['AAA']}bps BBB={_IG_SPREADS['BBB']}bps BBB-={_IG_SPREADS['BBB-']}bps")

# Test HY spread lookup
assert _HY_SPREADS["BB+"] < _HY_SPREADS["CCC"], "BB+ should have lower spread than CCC"
print(f"[OK] HY spreads: BB+={_HY_SPREADS['BB+']}bps CCC={_HY_SPREADS['CCC']}bps")

# Test muni tax-equivalent yield factor
# TEY = muni_yield / (1 - tax_rate); at 40% tax rate, factor = 1/0.6 = 1.667
assert abs(_MUNI_TEY_FACTOR - 1.6667) < 0.01, f"Muni TEY factor wrong: {_MUNI_TEY_FACTOR}"
muni_yield = 3.0  # 3% tax-free
tey = muni_yield * _MUNI_TEY_FACTOR
assert 4.5 < tey < 5.5, f"TEY out of range: {tey:.2f}%"
print(f"[OK] Muni TEY: {muni_yield}% tax-free = {tey:.2f}% taxable equivalent")

# Modified duration formula (pure math)
# For a par bond: modified_duration ≈ (1 - (1+y)^(-n)) / y ≈ (n for zero-coupon)
def modified_duration_approx(ytm_pct, maturity_years, freq=2):
    ytm = ytm_pct / 100.0 / freq
    n = maturity_years * freq
    if ytm == 0:
        return maturity_years
    # Approx for par bond
    dur = (1 - (1 + ytm)**(-n)) / ytm / freq
    return dur

md = modified_duration_approx(4.5, 10)
assert 6.0 < md < 9.0, f"Modified duration unexpected: {md:.2f}"
print(f"[OK] Approx modified duration (10Y, 4.5% par) = {md:.2f} years")

print("\n[PASS] dim_074: Fixed income screener")
PYEOF
