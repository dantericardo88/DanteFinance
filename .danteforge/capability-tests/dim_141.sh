#!/usr/bin/env bash
# dim_141: Commercial real estate (CRE) transaction comps & cap rates
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.cre_analytics_v3 import (
        CRETransaction, CRECompsEngine, TransactionComps, CREValuation,
        CapRateAnalytics, compute_cre_cap_rate, price_per_sf, dscr, dcf_value,
    )
    assert CRETransaction is not None, "Missing CRETransaction"
    assert CRECompsEngine is not None, "Missing CRECompsEngine"
    assert compute_cre_cap_rate is not None, "Missing compute_cre_cap_rate"
    print("[OK] CRETransaction present")
    print("[OK] CRECompsEngine present")
    print("[OK] compute_cre_cap_rate present")

    import numpy as np

    # -------------------------------------------------------------------------
    # Build 10 office transactions: prices 10M-50M, SF 10k-50k, NOI = price*0.06
    # -------------------------------------------------------------------------
    txns = []
    for i in range(10):
        price = 10e6 + i * (40e6 / 9)
        sf = 10_000 + i * (40_000 / 9)
        noi = price * 0.06
        txns.append(CRETransaction(
            property_id=f"OFF-{i:02d}",
            property_type="office",
            location="CBD",
            sale_price=price,
            square_feet=sf,
            noi=noi,
            year=2024,
            vacancy_rate=0.05,
        ))

    # --- CRETransaction properties ---
    t0 = txns[0]
    assert abs(t0.cap_rate - 0.06) < 1e-9, f"cap_rate wrong: {t0.cap_rate}"
    print(f"[OK] CRETransaction.cap_rate = {t0.cap_rate:.4f}  (expected 0.06)")
    assert t0.price_per_sf > 0, "price_per_sf must be positive"
    print(f"[OK] CRETransaction.price_per_sf = {t0.price_per_sf:.2f}")

    # --- Market metrics ---
    engine = CRECompsEngine(txns)
    mm = engine.market_metrics(property_type="office", treasury_yield=0.042)
    assert mm.n_transactions == 10, f"Expected 10 transactions, got {mm.n_transactions}"
    assert abs(mm.avg_cap_rate - 0.06) < 1e-6, f"avg_cap_rate wrong: {mm.avg_cap_rate}"
    assert mm.cap_rate_spread > 0, f"cap_rate_spread should be positive"
    print(f"[OK] market_metrics: avg_cap_rate={mm.avg_cap_rate:.4f}, n={mm.n_transactions}")
    print(f"[OK] cap_rate_spread = {mm.cap_rate_spread:.4f}")

    # --- Comparable value ---
    comp_val = engine.comparable_value(subject_noi=1_500_000, subject_sf=25_000,
                                        property_type="office")
    assert "avg_value" in comp_val, "comparable_value missing 'avg_value' key"
    assert comp_val["avg_value"] > 0, f"avg_value must be positive: {comp_val['avg_value']}"
    print(f"[OK] comparable_value avg_value = ${comp_val['avg_value']/1e6:.2f}M")

    # --- Cap rate percentile: 0.07 should be above most 6% caps ---
    pct = engine.percentile(cap_rate=0.07, property_type="office")
    assert pct > 50, f"Percentile of 0.07 among 0.06 comps should be >50, got {pct:.1f}"
    print(f"[OK] percentile(0.07 among 0.06 comps) = {pct:.1f}%  (expected >50)")

    # --- DSCR ---
    val = CREValuation()
    dscr_val = val.dscr(noi=1_000_000, annual_debt_service=700_000)
    assert dscr_val > 1.25, f"DSCR should be >1.25, got {dscr_val:.4f}"
    print(f"[OK] DSCR = {dscr_val:.4f}  (expected ~1.43)")

    # --- Max loan ---
    max_loan_val = val.max_loan(noi=1_000_000, interest_rate=0.055,
                                 amortization_years=25, dscr_min=1.25)
    assert max_loan_val > 0, f"max_loan must be positive, got {max_loan_val}"
    print(f"[OK] max_loan = ${max_loan_val:,.0f}")

    # --- Breakeven occupancy: opex=400k, rent=50psf, SF=20k → 40% ---
    beo = val.breakeven_occupancy(operating_expenses=400_000,
                                   asking_rent_psf=50.0, sf=20_000)
    assert 0.3 < beo < 0.5, f"Breakeven occupancy out of range: {beo:.4f}"
    print(f"[OK] breakeven_occupancy = {beo:.4f}  (expected ~0.40)")

    # --- CapRateAnalytics ---
    cra = CapRateAnalytics()

    spread = cra.cap_rate_spread(cap_rate_val=0.065, treasury_10yr=0.042)
    assert spread > 0, f"cap_rate_spread must be positive, got {spread}"
    assert abs(spread - 0.023) < 1e-9, f"spread wrong: {spread}"
    print(f"[OK] cap_rate_spread = {spread:.4f}  (expected 0.023)")

    # --- DCF value ---
    dcf_v = val.dcf_value(noi=1_000_000, growth_rate=0.03, discount_rate=0.07,
                           terminal_cap=0.055, years=10)
    assert dcf_v > 10_000_000, f"DCF value should be >10M, got {dcf_v:,.0f}"
    print(f"[OK] DCF value = ${dcf_v/1e6:.2f}M  (expected >$10M)")

    # --- Implied growth ---
    g = cra.implied_growth(cap_rate_val=0.06, discount_rate=0.08)
    assert abs(g - 0.02) < 1e-9, f"implied growth wrong: {g}"
    print(f"[OK] implied_growth = {g:.4f}  (expected 0.02)")

    # --- Standalone functions ---
    assert abs(compute_cre_cap_rate(60_000, 1_000_000) - 0.06) < 1e-9
    assert abs(price_per_sf(1_000_000, 20_000) - 50.0) < 1e-9
    assert abs(dscr(1_000_000, 700_000) - 1_000_000/700_000) < 1e-9
    dcf_sa = dcf_value(1_000_000, g=0.03, r=0.07, terminal_cap=0.055, years=10)
    assert dcf_sa > 10_000_000
    print("[OK] Standalone functions: compute_cre_cap_rate, price_per_sf, dscr, dcf_value")

    print("\n[PASS] dim_141: CRE transaction comps & cap rates -- all checks passed")

except ImportError as e:
    print(f"[NOT-BUILT] dim_141: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.cre_analytics_v3")
    sys.exit(0)
PYEOF
