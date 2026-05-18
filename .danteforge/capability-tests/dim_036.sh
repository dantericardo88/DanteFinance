#!/bin/bash
# dim_036: trace_bond_pricer — FINRA TRACE bond pricing engine
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sbx.trace_bond_pricer import (
    FINRA_FIXED,
    FRED_CSV,
    SWAP_RATES,
    _bond_price,
    compute_ytm,
    compute_duration,
    BondPricingResult,
    TRACEBondPricer,
)
from datetime import date

# --- constants ---
assert "finra.org" in FINRA_FIXED
assert "fred.stlouisfed.org" in FRED_CSV
assert "10Y" in SWAP_RATES and SWAP_RATES["10Y"] == "DGS10"
assert "2Y" in SWAP_RATES and "30Y" in SWAP_RATES
print("[OK] FINRA_FIXED, FRED_CSV, SWAP_RATES constants present")

# --- _bond_price: par bond (coupon == ytm) => price ~ 100 ---
# coupon and ytm in percent (e.g. 5.0 = 5%)
price_par = _bond_price(coupon=5.0, ytm=5.0, maturity_years=10.0)
assert abs(price_par - 100.0) < 0.01, f"Par bond price should be ~100, got {price_par}"
print(f"[OK] _bond_price: par bond = {price_par:.4f}")

# premium bond: coupon > ytm => price > 100
price_prem = _bond_price(coupon=6.0, ytm=5.0, maturity_years=10.0)
assert price_prem > 100.0, f"Premium bond should price > 100, got {price_prem}"
print(f"[OK] _bond_price: premium bond = {price_prem:.4f} > 100")

# discount bond: coupon < ytm => price < 100
price_disc = _bond_price(coupon=4.0, ytm=5.0, maturity_years=10.0)
assert price_disc < 100.0, f"Discount bond should price < 100, got {price_disc}"
print(f"[OK] _bond_price: discount bond = {price_disc:.4f} < 100")

# --- compute_ytm round-trip ---
ytm_rt = compute_ytm(coupon=5.0, price=price_par, maturity_years=10.0)
assert abs(ytm_rt - 5.0) < 0.01, f"YTM round-trip failed: {ytm_rt} vs 5.0"
print(f"[OK] compute_ytm round-trip: {ytm_rt:.4f}% ~= 5.0%")

# premium bond round-trip
ytm_prem = compute_ytm(coupon=6.0, price=price_prem, maturity_years=10.0)
assert abs(ytm_prem - 5.0) < 0.01, f"Premium YTM round-trip failed: {ytm_prem}"
print(f"[OK] compute_ytm premium round-trip: {ytm_prem:.4f}% ~= 5.0%")

# --- compute_duration returns (macaulay, modified, convexity) ---
mac, mod, conv = compute_duration(coupon=5.0, ytm=5.0, maturity_years=10.0)
assert 5.0 < mac < 10.0, f"Macaulay duration out of range: {mac}"
assert 5.0 < mod < 10.0, f"Modified duration out of range: {mod}"
assert conv > 0, f"Convexity should be positive: {conv}"
print(f"[OK] compute_duration: macaulay={mac:.4f}, modified={mod:.4f}, convexity={conv:.4f}")

# --- BondPricingResult Pydantic model ---
br = BondPricingResult(
    cusip="037833AJ9",
    issuer_name="APPLE INC",
    coupon=3.0,
    maturity_date=date(2027, 2, 9),
    last_trade_price=96.5,
    last_trade_yield=4.2,
    ytm=4.18,
    duration_modified=2.8,
    dv01=270.0,
    z_spread=85.0,
)
assert br.cusip == "037833AJ9"
assert br.issuer_name == "APPLE INC"
assert br.ytm == 4.18
assert br.z_spread == 85.0
print("[OK] BondPricingResult Pydantic model created")

# --- TRACEBondPricer class structure ---
assert hasattr(TRACEBondPricer, "__init__")
assert hasattr(TRACEBondPricer, "price_bond") or hasattr(TRACEBondPricer, "screen_bonds")
print("[OK] TRACEBondPricer class structure present")

print("\n[PASS] dim_036: trace_bond_pricer -- all checks passed")
PYEOF
