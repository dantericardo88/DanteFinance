#!/usr/bin/env bash
# dim_074: Fixed income screener v3 -- expanded universe, TIPS, cheapness, liquidity
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math

from sentinel.sfe.fixed_income_screener_v3 import (
    _TREASURY_FALLBACK,
    _IG_SPREADS,
    _HY_SPREADS,
    _IG_CORPS,
    _HY_CORPS,
    _HY_EXPANDED,
    AGENCY_BONDS,
    TIPS_OTR,
    TREASURY_OTR,
    _interp_treasury,
    _ytm_newton,
    _modified_duration,
    _macaulay_duration,
    _convexity,
    _dv01_per_mm,
    compute_tips_real_yield,
    tips_analytics,
    compute_cheapness_score,
    screen_relative_value,
    compute_liquidity_score,
    get_expanded_universe_count,
)

# 1. Expanded universe count > 100 bonds
total_count = get_expanded_universe_count()
assert total_count > 100, f"Expanded universe too small: {total_count}"
print(f"[OK] Expanded universe: {total_count} bonds (> 100 threshold)")
print(f"     IG={len(_IG_CORPS)}, HY_core={len(_HY_CORPS)}, HY_expanded={len(_HY_EXPANDED)}, "
      f"TIPS={len(TIPS_OTR)}, Agency={len(AGENCY_BONDS)}, Treasury={len(TREASURY_OTR)}")

# 2. Treasury curve interpolation (pure math)
y_3m = _interp_treasury(0.25, _TREASURY_FALLBACK)
y_10y = _interp_treasury(10.0, _TREASURY_FALLBACK)
y_5y = _interp_treasury(5.0, _TREASURY_FALLBACK)
assert 3.0 < y_3m < 7.0, f"3M Treasury yield out of range: {y_3m}"
assert 3.0 < y_10y < 7.0, f"10Y Treasury yield out of range: {y_10y}"
print(f"[OK] Treasury curve: 3M={y_3m:.2f}% 5Y={y_5y:.2f}% 10Y={y_10y:.2f}%")

# 3. Bond YTM (pure math Newton-Raphson)
par_price = _ytm_newton(coupon_rate=4.5, years=5.0, price=100.0)
assert abs(par_price - 4.5) < 0.05, f"Par bond YTM should be ~4.5%, got {par_price:.4f}"
print(f"[OK] Par bond YTM: coupon=YTM -> {par_price:.4f}% (should be ~4.5%)")

# 4. TIPS real yield: real = nominal - breakeven
# Given nominal yield 4.40% and breakeven 2.30% -> real yield = 2.10%
nominal = 4.40
breakeven = 2.30
real = compute_tips_real_yield(nominal, breakeven)
assert abs(real - (nominal - breakeven)) < 0.001, f"TIPS real yield: expected {nominal - breakeven}, got {real}"
assert abs(real - 2.10) < 0.001, f"Real yield should be 2.10%, got {real}"
print(f"[OK] TIPS real yield: {nominal}% nominal - {breakeven}% breakeven = {real}% real")

# Test with different values
real2 = compute_tips_real_yield(5.0, 2.50)
assert abs(real2 - 2.50) < 0.001, f"TIPS real 5%-2.5% should be 2.5%, got {real2}"
print(f"[OK] TIPS real yield: 5.0% - 2.5% = {real2}%")

# TIPS analytics
ta = tips_analytics(real_yield_pct=2.10, breakeven_inflation_pct=2.30, tenor_years=10.0)
assert ta["real_yield_pct"] == 2.10
assert ta["nominal_equivalent_yield_pct"] == 4.40
assert ta["tenor_years"] == 10.0
assert ta["approx_modified_duration"] > 0
print(f"[OK] TIPS analytics: real={ta['real_yield_pct']}%, nominal_equiv={ta['nominal_equivalent_yield_pct']}%, "
      f"dur~{ta['approx_modified_duration']:.2f}")

# 5. Cheapness score: Z-spread 50bps above sector median -> "cheap"
rv = compute_cheapness_score(bond_z_spread_bps=200.0, sector_median_z_spread_bps=150.0)
assert rv["label"] == "cheap", f"Expected 'cheap', got {rv['label']}"
assert rv["cheapness_bps"] == 50.0
print(f"[OK] Cheapness score: 200bps bond vs 150bps sector median -> '{rv['label']}' (+{rv['cheapness_bps']}bps)")

# Z-spread below sector median -> "rich"
rv_rich = compute_cheapness_score(bond_z_spread_bps=100.0, sector_median_z_spread_bps=160.0)
assert rv_rich["label"] == "rich", f"Expected 'rich', got {rv_rich['label']}"
assert rv_rich["cheapness_bps"] == -60.0
print(f"[OK] Cheapness score: 100bps bond vs 160bps sector median -> '{rv_rich['label']}' ({rv_rich['cheapness_bps']}bps)")

# Within 25bps -> "fair"
rv_fair = compute_cheapness_score(bond_z_spread_bps=155.0, sector_median_z_spread_bps=150.0)
assert rv_fair["label"] == "fair", f"Expected 'fair', got {rv_fair['label']}"
print(f"[OK] Cheapness score: 155bps vs 150bps sector -> '{rv_fair['label']}'")

# 6. Relative value screener on a sample universe
sample_bonds = [
    {"bond_id": "A1", "sector": "CORP_IG", "oas_bps": 200.0, "issuer": "Alpha Corp"},
    {"bond_id": "A2", "sector": "CORP_IG", "oas_bps": 100.0, "issuer": "Beta Corp"},
    {"bond_id": "A3", "sector": "CORP_IG", "oas_bps": 150.0, "issuer": "Gamma Corp"},
    {"bond_id": "B1", "sector": "CORP_HY", "oas_bps": 400.0, "issuer": "Delta Corp"},
    {"bond_id": "B2", "sector": "CORP_HY", "oas_bps": 600.0, "issuer": "Epsilon Corp"},
]
rv_results = screen_relative_value(sample_bonds)
assert len(rv_results) == 5
# A1 (200bps) vs IG median (150bps) -> cheap (+50bps)
a1 = next(r for r in rv_results if r["bond_id"] == "A1")
assert a1["cheapness_label"] == "cheap", f"A1 should be cheap: {a1}"
assert a1["cheapness_bps"] == 50.0, f"A1 cheapness: expected 50, got {a1['cheapness_bps']}"
# A2 (100bps) vs IG median (150bps) -> rich (-50bps)
a2 = next(r for r in rv_results if r["bond_id"] == "A2")
assert a2["cheapness_label"] == "rich", f"A2 should be rich: {a2}"
# Results sorted cheapest first
assert rv_results[0]["cheapness_bps"] >= rv_results[-1]["cheapness_bps"]
print(f"[OK] Relative value screener: A1=cheap(+50bps), A2=rich(-50bps), sorted by cheapness")

# 7. Liquidity score: $1B issue > $100M issue
score_1b = compute_liquidity_score(1000.0)
score_100m = compute_liquidity_score(100.0)
assert score_1b > score_100m, f"$1B should score higher than $100M: {score_1b} vs {score_100m}"
print(f"[OK] Liquidity score: $1B={score_1b:.1f} > $100M={score_100m:.1f}")

# Very large (benchmark) -> highest score
score_benchmark = compute_liquidity_score(5000.0)
assert score_benchmark == 100.0, f"$5B should be 100 (benchmark): {score_benchmark}"
print(f"[OK] Liquidity score: $5B benchmark -> {score_benchmark:.0f}/100")

# Small issue -> low score
score_small = compute_liquidity_score(50.0)
assert score_small < 20.0, f"$50M should be illiquid (<20): {score_small}"
print(f"[OK] Liquidity score: $50M illiquid -> {score_small:.1f}/100")

# Monotone increasing with size
sizes = [50, 100, 250, 500, 1000, 2000, 5000]
scores = [compute_liquidity_score(s) for s in sizes]
assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1)), (
    f"Liquidity scores not monotone: {list(zip(sizes, scores))}"
)
print(f"[OK] Liquidity monotone: {[f'{s}->{sc:.0f}' for s, sc in zip(sizes, scores)]}")

# 8. IG spread ordering: AAA < BBB
assert _IG_SPREADS["AAA"] < _IG_SPREADS["BBB"]
assert _IG_SPREADS["BBB-"] > 150
print(f"[OK] IG spreads: AAA={_IG_SPREADS['AAA']}bps < BBB={_IG_SPREADS['BBB']}bps < BBB-={_IG_SPREADS['BBB-']}bps")

# 9. HY spreads ordering
assert _HY_SPREADS["BB+"] < _HY_SPREADS["CCC"]
print(f"[OK] HY spreads: BB+={_HY_SPREADS['BB+']}bps < CCC={_HY_SPREADS['CCC']}bps")

# 10. Modified duration (pure math)
mod_dur = _modified_duration(coupon_rate=4.5, years=10.0, ytm_pct=4.5)
assert 6.0 < mod_dur < 9.0, f"10Y par bond mod duration unexpected: {mod_dur:.3f}"
print(f"[OK] Modified duration: 10Y 4.5% par bond = {mod_dur:.3f} years")

# 11. DV01: larger DV01 for longer duration bonds
dv01_10y = _dv01_per_mm(mod_dur)
mod_dur_2y = _modified_duration(coupon_rate=4.5, years=2.0, ytm_pct=4.5)
dv01_2y = _dv01_per_mm(mod_dur_2y)
assert dv01_10y > dv01_2y, f"10Y DV01 ({dv01_10y:.2f}) should be > 2Y DV01 ({dv01_2y:.2f})"
print(f"[OK] DV01: 10Y=${dv01_10y:.2f}/MM > 2Y=${dv01_2y:.2f}/MM (correct)")

print(f"\n[PASS] dim_074: Fixed income screener v3 -- {total_count} bonds, TIPS, cheapness, liquidity all verified")
PYEOF
