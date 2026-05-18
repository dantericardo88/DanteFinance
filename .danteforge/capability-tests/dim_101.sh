#!/bin/bash
# dim_101: LBO / Merger Model Templates — capability verification
# Tests pure arithmetic: IRR solver, LBO model, merger PPA, PE fund economics
set -e

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# 1. Import all public classes
from sentinel.sfe.lbo_model_v3 import (
    LBOModel,
    MergerModel,
    DCFMergerValuation,
    LBOCandidateScreener,
    PEFundEconomics,
    ModelTemplates,
    LBOResult,
    LBOScore,
    MergerResult,
    _solve_irr,
)
print("[OK] All LBO/Merger classes imported")

# 2. IRR solver — pure Newton-Raphson (no network)
# invest 100, get 150 after 5 years -> ~8.45% IRR
cfs = [-100.0, 0, 0, 0, 0, 150.0]
irr = _solve_irr(cfs)
assert 0.08 < irr < 0.09, f"IRR out of range: {irr:.4f}"
print(f"[OK] IRR solver: {irr*100:.2f}%")

# 3. LBO model — pure computation (no ticker = no network)
model = LBOModel(
    target_name="TestCo",
    entry_ebitda=100_000_000,
    entry_multiple=10.0,
    hold_period=5,
)
model.set_capital_structure(leverage_pct=0.60)
model.set_operating_assumptions(
    entry_revenue=500_000_000,
    ebitda_margin=0.20,
    revenue_growth=0.07,
    capex_pct_revenue=0.04,
    nwc_change_pct=0.05,
)
result = model.run_full_model(exit_multiple=11.0)
assert result.irr > 0, f"IRR must be positive, got {result.irr}"
assert result.moic > 1.0, f"MOIC must exceed 1x, got {result.moic}"
assert result.equity_invested > 0
print(f"[OK] LBO model: IRR={result.irr*100:.1f}%, MOIC={result.moic:.2f}x")

# 4. Sensitivity table (pure math)
sens = model.sensitivity_table(base_exit_multiple=11.0)
assert not sens.empty, "Sensitivity table must not be empty"
assert sens.shape == (5, 5), f"Expected 5x5, got {sens.shape}"
print(f"[OK] Sensitivity table: {sens.shape[0]}x{sens.shape[1]}")

# 5. LBOResult structure
assert hasattr(result, 'irr'), "LBOResult missing irr"
assert hasattr(result, 'moic'), "LBOResult missing moic"
assert hasattr(result, 'entry_ev'), "LBOResult missing entry_ev"
assert hasattr(result, 'equity_invested'), "LBOResult missing equity_invested"
assert hasattr(result, 'financials_df'), "LBOResult missing financials_df"
assert hasattr(result, 'debt_schedule_df'), "LBOResult missing debt_schedule_df"
assert not result.financials_df.empty, "Financials DF must not be empty"
assert not result.debt_schedule_df.empty, "Debt schedule DF must not be empty"
print(f"[OK] LBOResult: EV=${result.entry_ev/1e9:.1f}B, {len(result.financials_df)} year projections")

# 6. Merger model PPA (pure math — set_deal_structure + compute_goodwill)
merger = MergerModel(acquirer_name="BigCo", target_name="SmallCo")
merger.set_deal_structure(
    deal_value=1_500_000_000,
    cash_pct=0.50,
    stock_pct=0.50,
    premium_pct=0.30,
    acquirer_stock_price=50.0,
    acquirer_shares_outstanding=200_000_000,
)
ppa = merger.compute_goodwill_and_ppa(
    target_book_value=400_000_000,
    identifiable_intangibles=200_000_000,
    customer_relationships=100_000_000,
)
assert ppa["goodwill"] > 0, f"Goodwill must be positive, got {ppa['goodwill']}"
print(f"[OK] Merger PPA: goodwill=${ppa['goodwill']/1e6:.0f}M")

# 7. PE fund economics
pe = PEFundEconomics(
    fund_size=500_000_000,
    mgmt_fee_pct=0.02,
    carry_pct=0.20,
    hurdle_rate=0.08,
    fund_life=10,
    investment_period=5,
)
fund_result = pe.project_fund(exit_irr=0.22, hold_period=5)
assert "fund_irr" in fund_result, f"Missing fund_irr, got keys: {list(fund_result.keys())}"
assert fund_result["fund_irr"] > 0, "Fund IRR must be positive"
assert "gp_carry" in fund_result, "Missing gp_carry"
assert fund_result["gp_carry"] > 0, "GP carry must be positive"
assert "dpi" in fund_result, "Missing dpi"
assert "tvpi" in fund_result, "Missing tvpi"
print(f"[OK] PE fund: IRR={fund_result['fund_irr_pct']:.1f}%, DPI={fund_result['dpi']:.2f}x, TVPI={fund_result['tvpi']:.2f}x")

# 8. Model templates — verify key methods exist
templates = ModelTemplates()
assert hasattr(templates, 'simple_lbo_template'), "Missing simple_lbo_template"
assert hasattr(templates, 'carve_out_lbo_template'), "Missing carve_out_lbo_template"
assert hasattr(templates, 'pe_fund_return_template'), "Missing pe_fund_return_template"
print("[OK] Model templates class verified")

# 9. Simple LBO template (pure computation)
simple = templates.simple_lbo_template(
    target_name="Demo Target",
    entry_ebitda=50_000_000,
    entry_multiple=8.0,
    exit_multiple=9.0,
    hold_period=5,
)
assert simple["irr"] > 0, "Simple LBO template IRR must be positive"
assert simple["moic"] > 1.0, "Simple LBO template MOIC must exceed 1x"
print(f"[OK] Simple LBO template: IRR={simple['irr']*100:.1f}%, MOIC={simple['moic']:.2f}x")

# 10. LBOCandidateScreener interface
screener = LBOCandidateScreener()
assert hasattr(screener, 'score_lbo_attractiveness'), "Missing score_lbo_attractiveness method"
print("[OK] LBOCandidateScreener interface verified")

print("\n[PASS] dim_101: LBO / Merger Model Templates — all checks passed")
PYEOF
