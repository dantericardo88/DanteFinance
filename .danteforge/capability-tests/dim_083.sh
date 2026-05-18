#!/usr/bin/env bash
# dim_083: Stress testing — scenario shock math
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.spm.stress_testing_v3 import (
    ScenarioLibrary, StressTestEngine, AssetClassMapper, Scenario
)

# Test ScenarioLibrary
library = ScenarioLibrary()

hist = library.get_all_historical_scenarios()
hypo = library.get_all_hypothetical_scenarios()

assert len(hist) >= 5, f"Expected >= 5 historical scenarios: {len(hist)}"
assert len(hypo) >= 3, f"Expected >= 3 hypothetical scenarios: {len(hypo)}"
print(f"[OK] ScenarioLibrary: {len(hist)} historical, {len(hypo)} hypothetical scenarios")

# Test list_all
all_names = library.list_all()
assert "Black Monday 1987" in all_names, "Missing Black Monday"
assert "Hard Landing" in all_names or any("Landing" in n for n in all_names), "Missing Landing scenario"
print(f"[OK] list_all(): {len(all_names)} scenarios")

# Test get_scenario
bm = library.get_scenario("Black Monday 1987")
assert bm.type == "historical"
assert bm.asset_shocks["equity"] < -0.20, f"Black Monday equity shock too small: {bm.asset_shocks['equity']}"
assert bm.asset_shocks["bonds"] > 0, f"Black Monday bonds should be positive: {bm.asset_shocks['bonds']}"
print(f"[OK] Black Monday 1987: equity={bm.asset_shocks['equity']:.1%} bonds={bm.asset_shocks['bonds']:.1%}")

# Test display_shocks
shocks_str = bm.display_shocks()
assert "equity" in shocks_str, f"Should mention equity: {shocks_str}"
print(f"[OK] display_shocks: {shocks_str[:80]}")

# Test AssetClassMapper
mapper = AssetClassMapper()
spy_map = mapper.classify_holding("SPY")
tlt_map = mapper.classify_holding("TLT")
gld_map = mapper.classify_holding("GLD")
assert spy_map["equity"] == 1.0, f"SPY should be 100% equity: {spy_map}"
assert tlt_map["bonds"] == 1.0, f"TLT should be 100% bonds: {tlt_map}"
assert gld_map["gold"] == 1.0, f"GLD should be 100% gold: {gld_map}"
print(f"[OK] AssetClassMapper: SPY={spy_map} TLT={tlt_map} GLD={gld_map}")

# Test compute_portfolio_asset_class_weights
holdings = {"SPY": 0.50, "TLT": 0.30, "GLD": 0.20}
ac_weights = mapper.compute_portfolio_asset_class_weights(holdings)
assert abs(ac_weights["equity"] - 0.50) < 1e-9, f"Equity weight should be 0.50: {ac_weights['equity']}"
assert abs(ac_weights["bonds"] - 0.30) < 1e-9, f"Bonds weight should be 0.30: {ac_weights['bonds']}"
assert abs(ac_weights["gold"] - 0.20) < 1e-9, f"Gold weight should be 0.20: {ac_weights['gold']}"
print(f"[OK] Portfolio asset class weights: equity={ac_weights['equity']:.2f} bonds={ac_weights['bonds']:.2f} gold={ac_weights['gold']:.2f}")

# Test StressTestEngine.run_scenario
engine = StressTestEngine()
portfolio_holdings = {
    "SPY":  600_000,   # $600K equities
    "TLT":  300_000,   # $300K bonds
    "GLD":  100_000,   # $100K gold
}
total_equity = 1_000_000

result = engine.run_scenario(
    holdings=portfolio_holdings,
    scenario=bm,
    portfolio_equity=total_equity,
)

assert result.total_pnl < 0, f"Black Monday should cause losses: {result.total_pnl:,.0f}"
assert result.total_pnl_pct < -0.10, f"Portfolio loss should be > 10%: {result.total_pnl_pct:.2%}"
assert result.worst_holding == "SPY", f"SPY should be worst holding: {result.worst_holding}"
assert result.verdict in ("WARNING", "SEVERE"), f"Verdict should be WARNING or SEVERE: {result.verdict}"
print(f"[OK] Black Monday run_scenario: P&L={result.total_pnl:,.0f} ({result.total_pnl_pct:.2%}) verdict={result.verdict}")

# Test summary string
summary = result.summary()
assert "Black Monday" in summary, "Summary should mention scenario name"
assert "P&L" in summary or "Portfolio" in summary
print(f"[OK] ScenarioResult.summary(): {summary.split(chr(10))[0]}")

print("\n[PASS] dim_083: Stress testing")
PYEOF
