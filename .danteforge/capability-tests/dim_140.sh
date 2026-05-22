#!/usr/bin/env bash
# dim_140: REIT fundamental analysis (FFO / AFFO / NAV / cap-rate)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.sfe.reit_analytics_v3 import (
        REITFundamentals, REITFinancials, REITAnalyzer, PropertyValuation,
        compute_ffo, compute_affo, compute_nav, compute_cap_rate, cap_rate,
        debt_to_ebitda,
    )
    assert REITFundamentals is not None, "Missing REITFundamentals"
    assert compute_ffo is not None, "Missing compute_ffo"
    assert compute_affo is not None, "Missing compute_affo"
    assert compute_nav is not None, "Missing compute_nav"
    assert compute_cap_rate is not None, "Missing compute_cap_rate"
    print("[OK] REITFundamentals present")
    print("[OK] compute_ffo present")
    print("[OK] compute_affo present")
    print("[OK] compute_nav present")
    print("[OK] compute_cap_rate present")

    import numpy as np

    # -------------------------------------------------------------------------
    # Test fixture
    # revenue=500M, opex=200M, D&A=100M, interest=50M, net_income=80M
    # gains=20M, recurring_capex=30M, total_debt=800M, cash=50M
    # shares=200M, price=25, dividend=1.50, sector_cap_rate=0.055
    # -------------------------------------------------------------------------
    fin = REITFinancials(
        name="TestREIT",
        revenue=500e6,
        operating_expenses=200e6,
        depreciation=100e6,
        interest_expense=50e6,
        net_income=80e6,
        gains_on_sales=20e6,
        straight_line_rent_adj=0.0,
        stock_comp=0.0,
        recurring_capex=30e6,
        total_assets=1_500e6,
        total_debt=800e6,
        cash=50e6,
        shares_outstanding=200e6,
        share_price=25.0,
        dividend_per_share=1.50,
        sector_cap_rate=0.055,
    )
    analyzer = REITAnalyzer(fin)

    # NOI = 500M - 200M = 300M
    noi_val = analyzer.noi()
    assert 295e6 < noi_val < 305e6, f"NOI out of range: {noi_val:.1f}"
    print(f"[OK] NOI = {noi_val/1e6:.1f}M  (expected ~300M)")

    # FFO = 80 + 100 - 20 = 160M
    ffo_val = analyzer.ffo()
    assert 155e6 < ffo_val < 165e6, f"FFO out of range: {ffo_val:.1f}"
    print(f"[OK] FFO = {ffo_val/1e6:.1f}M  (expected ~160M)")

    # AFFO = 160 - 30 = 130M
    affo_val = analyzer.affo()
    assert 125e6 < affo_val < 135e6, f"AFFO out of range: {affo_val:.1f}"
    print(f"[OK] AFFO = {affo_val/1e6:.1f}M  (expected ~130M)")

    # FFO per share = 160M / 200M = 0.80
    metrics = analyzer.compute_all()
    ffo_ps = metrics.ffo_per_share
    assert abs(ffo_ps - 0.80) < 0.05, f"FFO/share out of range: {ffo_ps:.4f}"
    print(f"[OK] FFO/share = {ffo_ps:.4f}  (expected ~0.80)")

    # NAV = NOI/cap_rate + cash - debt = 300M/0.055 + 50M - 800M ≈ 4705M
    nav_val = metrics.nav
    assert nav_val > 0, f"NAV should be positive, got {nav_val:.1f}"
    print(f"[OK] NAV = {nav_val/1e6:.1f}M  (expected ~4705M)")

    # Premium/discount: market_cap=5000M vs nav≈4705M → small premium
    import math
    assert math.isfinite(metrics.premium_discount_pct), "premium_discount_pct is not finite"
    print(f"[OK] Premium/Discount = {metrics.premium_discount_pct:.2f}%")

    # Implied cap rate = NOI / (market_cap + net_debt)
    implied_cr = metrics.implied_cap_rate
    assert implied_cr > 0, f"Implied cap rate must be positive, got {implied_cr}"
    assert implied_cr < 0.20, f"Implied cap rate unreasonably large: {implied_cr}"
    print(f"[OK] Implied cap rate = {implied_cr:.4f}")

    # Debt / EBITDA
    d_ebitda = metrics.debt_to_ebitda
    assert d_ebitda > 0, f"Debt/EBITDA must be positive, got {d_ebitda}"
    print(f"[OK] Debt/EBITDA = {d_ebitda:.2f}x")

    # PropertyValuation.dcf_value
    pv = PropertyValuation()
    dcf_val = pv.dcf_value(noi=300e6, cap_rate_market=0.055,
                            growth_rate=0.02, discount_rate=0.07,
                            terminal_cap=0.055, years=10)
    assert dcf_val > 0, f"DCF value must be positive, got {dcf_val}"
    print(f"[OK] PropertyValuation.dcf_value = {dcf_val/1e6:.1f}M")

    # cap_rate_sensitivity: 5 cap rates → 5 values
    cap_rates_arr = np.array([0.04, 0.05, 0.055, 0.06, 0.07])
    values = pv.cap_rate_sensitivity(300e6, cap_rates_arr)
    assert len(values) == 5, f"Expected 5 sensitivity values, got {len(values)}"
    assert all(v > 0 for v in values), "All sensitivity values must be positive"
    print(f"[OK] cap_rate_sensitivity returned {len(values)} values")

    # Standalone functions
    assert abs(compute_ffo(80e6, 100e6, 20e6) - 160e6) < 1e4
    assert abs(compute_affo(160e6, 30e6) - 130e6) < 1e4
    nav_sa = compute_nav(300e6, 0.055, other_assets=50e6, total_liabilities=800e6)
    assert nav_sa > 0
    cr = compute_cap_rate(300e6, 5000e6)
    assert abs(cr - 0.06) < 1e-6

    print("\n[PASS] dim_140: REIT fundamentals (FFO/AFFO/NAV) -- all checks passed")

except ImportError as e:
    print(f"[NOT-BUILT] dim_140: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sfe.reit_analytics_v3")
    sys.exit(0)
PYEOF
