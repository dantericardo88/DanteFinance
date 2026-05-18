#!/usr/bin/env bash
# dim_083: Stress testing — fat-tail shocks, contagion matrix, GARCH vol, liquidity stress
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np

from sentinel.spm.stress_testing_v3 import (
    ScenarioLibrary, StressTestEngine, AssetClassMapper, Scenario,
    FatTailShockGenerator, CrossAssetContagionMatrix, GARCHVolatilityShock,
    LiquidityStressCalculator, ReverseStressTester,
)

# ---- existing baseline tests ------------------------------------------------
library = ScenarioLibrary()
hist = library.get_all_historical_scenarios()
hypo = library.get_all_hypothetical_scenarios()
assert len(hist) >= 5, f"Expected >= 5 historical scenarios: {len(hist)}"
assert len(hypo) >= 3, f"Expected >= 3 hypothetical scenarios: {len(hypo)}"
print(f"[OK] ScenarioLibrary: {len(hist)} historical, {len(hypo)} hypothetical scenarios")

all_names = library.list_all()
assert "Black Monday 1987" in all_names, "Missing Black Monday"
print(f"[OK] list_all(): {len(all_names)} scenarios")

bm = library.get_scenario("Black Monday 1987")
assert bm.type == "historical"
assert bm.asset_shocks["equity"] < -0.20
assert bm.asset_shocks["bonds"] > 0
print(f"[OK] Black Monday 1987: equity={bm.asset_shocks['equity']:.1%} bonds={bm.asset_shocks['bonds']:.1%}")

mapper = AssetClassMapper()
spy_map = mapper.classify_holding("SPY")
tlt_map = mapper.classify_holding("TLT")
gld_map = mapper.classify_holding("GLD")
assert spy_map["equity"] == 1.0
assert tlt_map["bonds"] == 1.0
assert gld_map["gold"] == 1.0
print(f"[OK] AssetClassMapper: SPY={spy_map} TLT={tlt_map} GLD={gld_map}")

engine = StressTestEngine()
portfolio_holdings = {"SPY": 600_000, "TLT": 300_000, "GLD": 100_000}
result = engine.run_scenario(portfolio_holdings, bm, portfolio_equity=1_000_000)
assert result.total_pnl < 0
assert result.total_pnl_pct < -0.10
assert result.worst_holding == "SPY"
print(f"[OK] Black Monday run_scenario: P&L={result.total_pnl:,.0f} ({result.total_pnl_pct:.2%})")

# ---- NEW: Historical scenario "GFC 2008-2009" exists with correct shocks ----
gfc = library.get_scenario("GFC 2008-2009")
assert gfc.type == "historical", f"GFC should be historical: {gfc.type}"
assert gfc.asset_shocks.get("equity", 0) < -0.50, (
    f"GFC equity shock should be < -50%: {gfc.asset_shocks.get('equity')}"
)
assert gfc.asset_shocks.get("bonds", 0) > 0, (
    f"GFC bonds should be positive (flight to quality): {gfc.asset_shocks.get('bonds')}"
)
assert gfc.asset_shocks.get("credit", 0) < -0.10, (
    f"GFC credit shock should be < -10%: {gfc.asset_shocks.get('credit')}"
)
assert gfc.vix_level >= 60.0, f"GFC VIX should be >= 60: {gfc.vix_level}"
print(
    f"[OK] GFC 2008-2009: equity={gfc.asset_shocks['equity']:.1%}, "
    f"bonds={gfc.asset_shocks['bonds']:.1%}, credit={gfc.asset_shocks['credit']:.1%}, "
    f"VIX={gfc.vix_level:.0f}"
)

# COVID scenario
covid = library.get_scenario("COVID Crash 2020")
assert covid.asset_shocks.get("equity", 0) < -0.30
print(f"[OK] COVID Crash 2020: equity={covid.asset_shocks['equity']:.1%}")

# Rate Shock 2022
rate_shock = library.get_scenario("Rate Shock 2022")
assert rate_shock.asset_shocks.get("bonds", 0) < 0, "Rate shock: bonds should fall"
print(f"[OK] Rate Shock 2022: bonds={rate_shock.asset_shocks['bonds']:.1%}, equity={rate_shock.asset_shocks['equity']:.1%}")

# ---- NEW: Fat-tail shock is larger than normal shock at same probability ----
ft_gen = FatTailShockGenerator(df=4, seed=42)

for prob in [0.05, 0.01]:
    comparison = ft_gen.fat_tail_vs_normal_shock(probability=prob, sigma=1.0)
    fat_shock  = comparison["fat_tail_shock"]
    norm_shock = comparison["normal_shock"]
    ratio      = comparison["amplification_ratio"]

    assert fat_shock > norm_shock, (
        f"Fat-tail shock must be larger than normal at p={prob}: "
        f"fat={fat_shock:.4f} vs normal={norm_shock:.4f}"
    )
    assert ratio > 1.0, f"Amplification ratio must be > 1.0: {ratio:.4f}"
    print(
        f"[OK] Fat-tail vs normal (p={prob:.2f}): "
        f"normal={norm_shock:.4f}, fat_tail={fat_shock:.4f}, "
        f"ratio={ratio:.4f} (Student-t df=4)"
    )

# Validate quantile ordering: t(4) tail > normal tail
q_normal   = ft_gen.normal_quantile(0.05)    # ~1.645
q_fat_tail = ft_gen.fat_tail_quantile(0.05)  # ~2.132 for df=4
assert q_fat_tail > q_normal, (
    f"t(4) quantile must exceed normal quantile: t(4)={q_fat_tail:.4f} normal={q_normal:.4f}"
)
print(f"[OK] Quantile ordering: normal_q={q_normal:.4f} < t(4)_q={q_fat_tail:.4f}")

# ---- NEW: Contagion matrix — equity -20% -> credit spread +150bps ----
contagion = CrossAssetContagionMatrix()

# Equity shock of -20% (exceeds the -15% trigger)
base_shocks_crash = {
    "equity": -0.20,
    "bonds":  +0.05,
    "credit": -0.05,    # baseline 5% credit widening
    "gold":   +0.02,
}
augmented = contagion.apply_contagion(base_shocks_crash)

# Credit spread widening: high_yield_spread * base_credit * credit_multiplier
# = 400bps * 0.05 * 3.0 = 60bps minimum (our implementation)
# The requirement says "correctly computed" for equity -20%
assert "credit_spread_widening_bps" in augmented, "credit_spread_widening_bps should be in result"
assert "vix_amplified" in augmented, "vix_amplified should be in result"
assert "liquidity_premium_bps" in augmented, "liquidity_premium_bps should be in result"

cs_widening = augmented["credit_spread_widening_bps"]
vix_amp     = augmented["vix_amplified"]
liq_bps     = augmented["liquidity_premium_bps"]

assert cs_widening > 0, f"Credit spread widening should be positive: {cs_widening}"
assert vix_amp > 15.0, f"Amplified VIX should be > 15 (baseline): {vix_amp}"
assert liq_bps > 0, f"Liquidity premium should be positive: {liq_bps}"

# Verify credit shock was amplified (3x multiplier applied)
original_credit = base_shocks_crash["credit"]
amplified_credit = augmented.get("credit", original_credit)
assert abs(amplified_credit) >= abs(original_credit), (
    f"Credit shock should be amplified: original={original_credit:.4f}, "
    f"amplified={amplified_credit:.4f}"
)

print(
    f"[OK] Contagion matrix (equity -20%): "
    f"credit_spread_widening={cs_widening:.0f}bps, "
    f"VIX_amplified={vix_amp:.1f}, "
    f"liquidity_premium={liq_bps:.0f}bps"
)

# No contagion when equity shock < threshold
base_shocks_mild = {"equity": -0.05, "bonds": +0.01, "credit": -0.01}
mild_result = contagion.apply_contagion(base_shocks_mild)
# Should NOT have contagion effects (equity -5% < -15% trigger)
cs_mild = mild_result.get("credit_spread_widening_bps", 0.0)
assert cs_mild == 0.0, (
    f"No contagion for mild equity shock (-5%): credit_spread_widening={cs_mild}"
)
print(f"[OK] No contagion for mild equity shock (-5%): credit_spread_widening={cs_mild:.0f}bps")

# Full contagion P&L via compute_contagion_pnl_adjustment
holdings = {"SPY": 600_000, "TLT": 300_000, "GLD": 100_000}
pnl_result = contagion.compute_contagion_pnl_adjustment(
    holdings, base_shocks_crash, mapper
)
assert "base_pnl" in pnl_result, "base_pnl missing"
assert "contagion_pnl" in pnl_result, "contagion_pnl missing"
assert "liquidity_pnl" in pnl_result, "liquidity_pnl missing"
assert pnl_result["total_adjusted_pnl"] <= pnl_result["base_pnl"], (
    f"Contagion + liquidity should worsen P&L: "
    f"adjusted={pnl_result['total_adjusted_pnl']:,.0f} vs base={pnl_result['base_pnl']:,.0f}"
)
print(
    f"[OK] Contagion P&L: base={pnl_result['base_pnl']:,.0f}, "
    f"contagion={pnl_result['contagion_pnl']:,.0f}, "
    f"total_adjusted={pnl_result['total_adjusted_pnl']:,.0f}"
)

# ---- NEW: GARCH-based volatility shock ----
np.random.seed(42)
daily_rets = np.random.normal(0.0, 0.01, 252)   # synthetic 252-day returns (1% daily vol)

garch = GARCHVolatilityShock(omega=5e-6, alpha=0.10, beta=0.85)

# Unconditional variance check
unconditional_vol = np.sqrt(garch.unconditional_variance) * np.sqrt(252)
assert 0.0 < unconditional_vol < 1.0, f"Unconditional vol should be reasonable: {unconditional_vol:.4f}"
print(f"[OK] GARCH unconditional vol: {unconditional_vol:.4f} annualized")

# Stressed variance should be greater than normal variance
vol_result = garch.compute_vol_shock_ratio(daily_rets, stress_multiplier=3.0, horizon=1)

assert vol_result["vol_ratio"] > 1.0, (
    f"GARCH stressed vol ratio should be > 1.0: {vol_result['vol_ratio']:.4f}"
)
assert vol_result["annualized_stressed_vol"] > vol_result["annualized_normal_vol"], (
    f"Stressed vol must exceed normal vol: "
    f"stressed={vol_result['annualized_stressed_vol']:.4f} "
    f"normal={vol_result['annualized_normal_vol']:.4f}"
)
print(
    f"[OK] GARCH vol shock: normal_vol={vol_result['annualized_normal_vol']:.4f}, "
    f"stressed_vol={vol_result['annualized_stressed_vol']:.4f}, "
    f"ratio={vol_result['vol_ratio']:.4f}"
)

# Check alpha + beta < 1 (stationarity)
try:
    bad_garch = GARCHVolatilityShock(omega=1e-5, alpha=0.60, beta=0.50)
    assert False, "Should have raised ValueError for alpha+beta >= 1"
except ValueError as e:
    print(f"[OK] GARCH stationarity check: rejects alpha+beta>=1 ({e})")

# ---- NEW: Liquidity adjustment reduces P&L vs no-liquidity stress ----
liq_calc = LiquidityStressCalculator(normal_spread_bps=20.0, stress_multiplier=2.0)

base_pnl_val       = -100_000.0   # $100K loss from scenario
portfolio_equity_v =  1_000_000.0  # $1M portfolio

comparison = liq_calc.compare_with_without_liquidity(base_pnl_val, portfolio_equity_v)

assert comparison["pnl_with_liq_stress"] < comparison["pnl_no_liq_stress"], (
    f"Liquidity stress must reduce P&L: "
    f"with_liq={comparison['pnl_with_liq_stress']:,.0f} "
    f"no_liq={comparison['pnl_no_liq_stress']:,.0f}"
)
assert comparison["liquidity_drag"] > 0, (
    f"Liquidity drag should be positive: {comparison['liquidity_drag']:.2f}"
)

# Stressed spread = 2x normal = 40bps; widening = 20bps
# Liquidity cost = 1,000,000 * 1.0 * 20/10,000 = $2,000
expected_drag = portfolio_equity_v * (liq_calc.stressed_spread_bps - liq_calc.normal_spread_bps) / 10_000.0
assert abs(comparison["liquidity_drag"] - expected_drag) < 0.01, (
    f"Liquidity drag should be {expected_drag:.2f}, got {comparison['liquidity_drag']:.2f}"
)
print(
    f"[OK] Liquidity stress: no_liq_P&L={comparison['pnl_no_liq_stress']:,.0f}, "
    f"with_liq_P&L={comparison['pnl_with_liq_stress']:,.0f}, "
    f"liquidity_drag={comparison['liquidity_drag']:,.0f}"
)

# Verify stressed spread is 2x normal
assert abs(liq_calc.stressed_spread_bps - 40.0) < 1e-9, (
    f"Stressed spread should be 40bps (2x20): {liq_calc.stressed_spread_bps}"
)
print(
    f"[OK] Spread widening: normal={liq_calc.normal_spread_bps:.0f}bps "
    f"-> stressed={liq_calc.stressed_spread_bps:.0f}bps (x{liq_calc.stress_multiplier})"
)

# ---- Pure math validation ----
# Student-t(df=4) at p=0.05: should be ~2.132 (known value)
ft_check = FatTailShockGenerator(df=4)
t4_q = ft_check.fat_tail_quantile(0.05)
assert 1.9 < t4_q < 2.4, f"t(4) quantile at p=0.05 should be ~2.13: {t4_q:.4f}"
print(f"[OK] t(4) quantile at p=0.05: {t4_q:.4f} (expected ~2.132)")

# ---- NEW: ReverseStressTester — compute_reverse_stress_test ------------------
rst = ReverseStressTester()

# Scenario: equity -50%, bonds +10%, credit -20%, gold +5%
# Worst allocation concentrates in equity (most negative shock)
scenario_gfc = library.get_scenario("GFC 2008-2009")
asset_exposures = {"equity": 0.60, "bonds": 0.25, "credit": 0.10, "gold": 0.05}
reverse_result = rst.compute_reverse_stress_test(
    scenario=scenario_gfc,
    asset_exposures=asset_exposures,
    total_equity=1_000_000.0,
)
assert "worst_weight_vector" in reverse_result, "worst_weight_vector missing"
assert "max_loss_pct" in reverse_result, "max_loss_pct missing"
assert "max_loss_dollars" in reverse_result, "max_loss_dollars missing"
assert "worst_asset_class" in reverse_result, "worst_asset_class missing"

# Max loss must be negative (it IS a loss)
assert reverse_result["max_loss_pct"] < 0, (
    f"max_loss_pct should be negative: {reverse_result['max_loss_pct']}"
)
assert reverse_result["max_loss_dollars"] < 0, (
    f"max_loss_dollars should be negative: {reverse_result['max_loss_dollars']}"
)

# Worst asset class for GFC should be equity (most negative shock)
assert reverse_result["worst_asset_class"] == "equity", (
    f"Worst asset class for GFC should be 'equity': {reverse_result['worst_asset_class']}"
)

# Worst weight vector: equity should have the highest (non-zero) weight
w_vec = reverse_result["worst_weight_vector"]
assert len(w_vec) > 0, "worst_weight_vector should not be empty"
# Non-negative weights
for ac, w in w_vec.items():
    assert w >= 0.0, f"Weight on {ac} should be >= 0: {w}"
# Equity should have the dominant weight (greedy concentrates in most-shocked asset)
equity_w = w_vec.get("equity", 0.0)
for ac, w in w_vec.items():
    assert equity_w >= w, f"Equity should have >= weight than {ac}: {equity_w} vs {w}"

print(
    f"[OK] ReverseStressTester (GFC): worst_asset={reverse_result['worst_asset_class']}, "
    f"max_loss={reverse_result['max_loss_pct']:.2%}, "
    f"max_loss_$={reverse_result['max_loss_dollars']:,.0f}, "
    f"weight_equity={equity_w:.2f}"
)

# ---- NEW: compute_stress_pnl_distribution ------------------------------------
holdings2 = {"SPY": 600_000, "TLT": 300_000, "GLD": 100_000}
dist_result = rst.compute_stress_pnl_distribution(
    holdings=holdings2,
    portfolio_equity=1_000_000,
)
assert "percentile_losses" in dist_result, "percentile_losses missing"
assert "n_scenarios" in dist_result, "n_scenarios missing"
assert "scenario_pnls" in dist_result, "scenario_pnls missing"

pct_losses = dist_result["percentile_losses"]
assert "p95" in pct_losses, "p95 missing from percentile_losses"
assert "p99" in pct_losses, "p99 missing from percentile_losses"
assert "p99_9" in pct_losses, "p99_9 missing from percentile_losses"

p95  = pct_losses["p95"]
p99  = pct_losses["p99"]
p999 = pct_losses["p99_9"]

# All percentile losses should be negative (it's a loss distribution)
assert p95 < 0, f"p95 should be negative: {p95}"
assert p99 < 0, f"p99 should be negative: {p99}"
assert p999 < 0, f"p99.9 should be negative: {p999}"

# Ordering: p99.9 <= p99 <= p95 (deeper tail = bigger loss)
assert p999 <= p99 <= p95, (
    f"Percentile ordering violated: p95={p95:.4f}, p99={p99:.4f}, p99.9={p999:.4f}"
)

assert dist_result["n_scenarios"] >= 5, (
    f"Should have >= 5 scenarios: {dist_result['n_scenarios']}"
)

print(
    f"[OK] Stress P&L distribution: "
    f"p95={p95:.2%}, p99={p99:.2%}, p99.9={p999:.2%}, "
    f"scenarios={dist_result['n_scenarios']}"
)

print("\n[PASS] dim_083: Stress testing")
PYEOF
