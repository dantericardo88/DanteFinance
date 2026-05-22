#!/usr/bin/env bash
# Capability test for dim_040: MBS/ABS/CLO structured products analytics
# Exit 0 = dimension verified  |  Exit 1 = not verified
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np

# ---------- Imports ----------
try:
    from sentinel.sfe.structured_products_v3 import (
        MBSPool, MBSCashFlow, CLOTranche, CLOWaterfall,
        MBSPricer, CLOAnalytics, ABSPricer,
        psa_cpr, smm_from_cpr,
        mbs_cash_flows, mbs_price, mbs_wal, clo_equity_irr,
    )
except ImportError as e:
    print("[FAIL] dim_040: import error - " + str(e))
    sys.exit(1)

errors = []

# 1. MBSPool construction
pool = MBSPool(balance=100_000_000, wac=0.06, wam=360, psa_speed=1.0)

# 2. mbs_cash_flows: 360 cash flows, month-1 CF > 0, fully amortised
cfs = mbs_cash_flows(pool)
if len(cfs) != 360:
    errors.append("Expected 360 cash flows, got " + str(len(cfs)))
if cfs[0].total_cashflow <= 0:
    errors.append("Month-1 cashflow must be > 0, got " + str(cfs[0].total_cashflow))
if cfs[-1].remaining_balance > 1000:
    errors.append(
        "Month-360 remaining balance should be ~0, got " +
        "{:.2f}".format(cfs[-1].remaining_balance)
    )

# 3. WAL: 5 < wal < 15 for 30yr mortgage at 100 PSA
pricer = MBSPricer()
wal = pricer.wal(pool)
if not (5 < wal < 15):
    errors.append("WAL should be in (5, 15) years, got " + "{:.4f}".format(wal))

# 4. Price at discount=WAC ~= par (100)
price_at_par = pricer.price(pool, discount_rate=0.06)
if not (98.0 <= price_at_par <= 102.0):
    errors.append("Price at 6% discount should be ~100, got " + "{:.4f}".format(price_at_par))

# 5. Price at discount=5% -> premium (> 100)
price_premium = pricer.price(pool, discount_rate=0.05)
if price_premium <= 100.0:
    errors.append("Price at 5% discount should be > 100, got " + "{:.4f}".format(price_premium))

# 6. Price at discount=7% -> discount (< 100)
price_discount = pricer.price(pool, discount_rate=0.07)
if price_discount >= 100.0:
    errors.append("Price at 7% discount should be < 100, got " + "{:.4f}".format(price_discount))

# 7. PSA sensitivity: WAL(200 PSA) < WAL(100 PSA)
psa_speeds = np.array([1.0, 2.0])
wals = pricer.psa_sensitivity(pool, discount_rate=0.06, psa_speeds=psa_speeds)
if not (wals[1] < wals[0]):
    errors.append(
        "WAL at 200 PSA ({:.4f}) should be < WAL at 100 PSA ({:.4f})".format(wals[1], wals[0])
    )

# 8. CLOWaterfall construction
tranches = [
    CLOTranche(name="AAA", rating="AAA", par_amount=70_000_000, coupon=0.015, oc_trigger=1.20),
    CLOTranche(name="AA",  rating="AA",  par_amount=10_000_000, coupon=0.020),
    CLOTranche(name="A",   rating="A",   par_amount= 7_000_000, coupon=0.030),
    CLOTranche(name="BBB", rating="BBB", par_amount= 5_000_000, coupon=0.050),
    CLOTranche(name="Equity", rating="Equity", par_amount=8_000_000, coupon=0.0),
]
wfall = CLOWaterfall(
    tranches=tranches,
    collateral_balance=100_000_000,
    collateral_coupon=0.06,
    default_rate=0.02,
    recovery_rate=0.40,
)
analytics = CLOAnalytics(wfall)

# 9. OC ratio: AAA OC = 100/70 ~= 1.43 > 1.2
oc = analytics.oc_ratio(period=0)
if "AAA" not in oc:
    errors.append("OC ratio dict missing 'AAA' key")
else:
    aaa_oc = oc["AAA"]
    if not (1.35 <= aaa_oc <= 1.55):
        errors.append("AAA OC ratio should be ~1.43, got " + "{:.4f}".format(aaa_oc))
    if aaa_oc <= 1.2:
        errors.append("AAA OC ({:.4f}) should pass the 1.2 trigger".format(aaa_oc))

# 10. equity_irr: returns finite float
irr = analytics.equity_irr(equity_investment=8_000_000, n_periods=5)
if not np.isfinite(irr):
    errors.append("equity_irr should be a finite float, got " + str(irr))

# module-level helper alias
irr2 = clo_equity_irr(wfall, equity_investment=8_000_000)
if not np.isfinite(irr2):
    errors.append("clo_equity_irr (module fn) should be finite, got " + str(irr2))

# 11. ABS: 50M, 5%, 60 months, WAL < 5yr
abs_pricer = ABSPricer(balance=50_000_000, coupon=0.05, term_months=60, cpr=0.06)
abs_wal = abs_pricer.wal()
if abs_wal >= 5.0:
    errors.append("ABS WAL should be < 5 years, got " + "{:.4f}".format(abs_wal))

# 12. ABS yield_to_maturity at par -> ~= 5%
abs_price_at_par = abs_pricer.price(discount_rate=0.05)
abs_ytm = abs_pricer.yield_to_maturity(abs_price_at_par)
if not (0.048 <= abs_ytm <= 0.052):
    errors.append(
        "ABS YTM at par price should be ~5%, got {:.4f}%".format(abs_ytm * 100)
    )

# ---------- Report ----------
if errors:
    for e in errors:
        print("  FAIL: " + e)
    print("[FAIL] dim_040: MBS/ABS/CLO structured products")
    sys.exit(1)

print("  WAL (100 PSA 30yr): {:.2f} years".format(wal))
print("  Price at 6%: {:.4f}  (~= par)".format(price_at_par))
print("  Price at 5%: {:.4f}  (premium)".format(price_premium))
print("  Price at 7%: {:.4f}  (discount)".format(price_discount))
print("  WAL 100 PSA: {:.2f} yr  |  WAL 200 PSA: {:.2f} yr".format(wals[0], wals[1]))
print("  AAA OC ratio: {:.4f}  (trigger 1.20 -> PASS)".format(oc["AAA"]))
print("  CLO equity IRR: {:.2f}%".format(irr * 100))
print("  ABS WAL: {:.2f} yr  (< 5.0 yr OK)".format(abs_wal))
print("  ABS YTM at par: {:.4f}%".format(abs_ytm * 100))
print("[PASS] dim_040: MBS/ABS/CLO structured products")
sys.exit(0)
PYEOF
