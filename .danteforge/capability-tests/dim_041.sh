#!/bin/bash
# dim_041: High-yield bond and leveraged loan analytics
# Exit 0 = dimension verified  |  Exit 1 = not verified
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

import numpy as np

from sentinel.sfe.high_yield_v3 import (
    HYBond, LeveragedLoan, HYAnalytics, DefaultAnalysis, LoanAnalytics, HYIndex,
    bond_ytm, z_spread, merton_pd, expected_loss, distress_score,
)

# ---- Test 1: YTM > coupon when price < par --------------------------------
bond = HYBond(
    issuer="TestCorp",
    coupon=0.09,
    face_value=1000.0,
    maturity_years=5.0,
    price=920.0,
)
ytm_val = bond_ytm(bond.price, bond.coupon, bond.face_value, bond.maturity_years)
assert ytm_val > 0.09, f"YTM {ytm_val:.4f} should be > coupon 0.09 when price < par"
print(f"[OK] YTM={ytm_val:.4f} > coupon 0.09 (bond at discount)")

# ---- Test 2: Z-spread positive against flat 5% curve ----------------------
spot_curve = np.full(5, 0.05)
maturities  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
zs = z_spread(bond.price, bond.coupon, bond.face_value, bond.maturity_years,
              spot_curve, maturities)
assert zs > 0, f"Z-spread {zs*10000:.1f} bps should be > 0"
print(f"[OK] Z-spread={zs*10000:.1f} bps (above risk-free)")

# ---- Test 3: Duration 3 < dur < 5 ----------------------------------------
analytics = HYAnalytics()
dur = analytics.duration(bond)
assert 3.0 < dur < 5.0, f"Duration {dur:.2f} should be between 3 and 5"
print(f"[OK] Duration={dur:.2f} years")

# ---- Test 4: Distress score < 7 for bond at price=920 --------------------
ds = distress_score(bond.price, ytm_val)
assert ds < 7.0, f"Distress score {ds:.2f} should be < 7 (not deeply distressed)"
print(f"[OK] Distress score={ds:.2f} < 7 (not deeply distressed)")

# ---- Test 5: is_distressed = True when price < 70 -------------------------
distressed_bond = HYBond(
    issuer="DistressedCorp",
    coupon=0.12,
    face_value=1000.0,
    maturity_years=3.0,
    price=65.0,
)
assert analytics.is_distressed(distressed_bond), "Price=65 bond should be distressed"
print(f"[OK] is_distressed=True for price=65 bond")

# ---- Test 6: Merton PD in (0, 0.5) ----------------------------------------
da = DefaultAnalysis()
pd_val = da.merton_pd(
    asset_value=1_000_000_000,
    debt=800_000_000,
    asset_vol=0.30,
    risk_free=0.05,
    T=1.0,
)
assert 0 < pd_val < 0.5, f"Merton PD {pd_val:.4f} should be in (0, 0.5)"
print(f"[OK] Merton PD={pd_val:.4f} in (0, 0.5)")

# ---- Test 7: Expected loss = PD * LGD ------------------------------------
el = da.expected_loss(pd=0.05, lgd=0.55)
expected = 0.05 * 0.55
assert abs(el - expected) < 1e-8, f"EL {el} != {expected}"
print(f"[OK] EL={el:.4f} = PD*LGD = 0.05*0.55 = {expected:.4f}")

# ---- Test 8: LGD for senior unsecured = 0.55 -----------------------------
lgd = da.loss_given_default("senior_unsecured")
assert abs(lgd - 0.55) < 1e-9, f"LGD for senior_unsecured={lgd} != 0.55"
print(f"[OK] LGD('senior_unsecured')={lgd:.2f}")

# ---- Test 9: Break-even spread > 200 bps for PD=5%, LGD=55% -------------
bes = da.break_even_spread(pd=0.05, lgd=0.55)
assert bes > 200, f"Break-even spread {bes:.1f} bps should be > 200 bps"
print(f"[OK] Break-even spread={bes:.1f} bps > 200 bps")

# ---- Test 10: Leveraged loan effective rate > SOFR -----------------------
loan = LeveragedLoan(
    issuer="LoanCo",
    spread_bps=500.0,
    sofr_floor=0.005,
    face_value=1_000_000,
    maturity_years=7.0,
    price=99.0,
    first_lien_leverage=4.0,
    total_leverage=5.5,
    interest_coverage=3.0,
    is_cov_lite=True,
)
la = LoanAnalytics()
eff_rate = la.effective_rate(loan, sofr=0.05)
assert eff_rate > 0.05, f"Effective rate {eff_rate:.4f} should be > SOFR 5%"
print(f"[OK] Effective rate={eff_rate:.4f} (SOFR 5% + 500 bps spread)")

# ---- Test 11: HYIndex OAS and duration > 0 --------------------------------
bonds = [
    HYBond("A", coupon=0.08, face_value=1000, maturity_years=3.0, price=950, rating="BB"),
    HYBond("B", coupon=0.09, face_value=1000, maturity_years=5.0, price=920, rating="B"),
    HYBond("C", coupon=0.10, face_value=1000, maturity_years=7.0, price=880, rating="B"),
    HYBond("D", coupon=0.07, face_value=1000, maturity_years=4.0, price=970, rating="BB"),
    HYBond("E", coupon=0.11, face_value=1000, maturity_years=6.0, price=860, rating="CCC"),
]
idx = HYIndex(bonds)
idx_oas = idx.oas(risk_free=0.05)
idx_dur = idx.duration()
assert idx_oas > 0, f"Index OAS {idx_oas:.1f} bps should be > 0"
assert idx_dur > 0, f"Index duration {idx_dur:.2f} should be > 0"
print(f"[OK] HYIndex OAS={idx_oas:.1f} bps, duration={idx_dur:.2f} yrs")

print("[PASS] dim_041: High yield bond analytics")
PYEOF
