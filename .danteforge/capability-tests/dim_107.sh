#!/bin/bash
# dim_107: DeFi Protocol Analytics — capability verification
# Tests pure computation logic (no network calls required)
set -e

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

# 1. Import all public classes
from sentinel.sfe.defi_analytics_v3 import (
    DefiLlamaAdvancedClient,
    ProtocolQualityScorer,
    LendingProtocolAnalyzer,
    YieldOptimizerEngine,
    BridgeFlowAnalyzer,
    GovernanceTokenAnalyzer,
    DeFiMarketMonitor,
    ProtocolScore,
    YieldAllocation,
    BridgeFlow,
    ImpermanentLossCalculator,
    KNOWN_HACKS,
    KNOWN_AUDITS,
    PROTOCOL_COINGECKO_IDS,
    TOKEN_EMISSION_SCHEDULES,
)
print("[OK] All DeFi analytics classes imported (including ImpermanentLossCalculator)")

# 2. LendingProtocolAnalyzer — pure math (no network)
lender = LendingProtocolAnalyzer()

# Utilization rate
util = lender.compute_utilization_rate(total_borrows=800_000_000, total_supply=1_000_000_000)
assert abs(util - 0.80) < 1e-9, f"Util should be 0.80, got {util}"
print(f"[OK] Utilization rate: {util:.2%}")

# Zero supply edge case
util_zero = lender.compute_utilization_rate(0, 0)
assert util_zero == 0.0
print("[OK] Zero supply edge case handled")

# Liquidation risk
hf_dist = [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
risk = lender.compute_liquidation_risk(hf_dist)
assert risk["cascade_risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
assert risk["liquidatable_pct"] >= 0
assert risk["at_risk_pct"] >= 0
print(f"[OK] Liquidation risk: {risk['cascade_risk_level']}, liq={risk['liquidatable_pct']:.1f}%")

# Critical cascade with all positions near liquidation
critical_hf = [0.95] * 20 + [1.05] * 30 + [1.10] * 50
critical_risk = lender.compute_liquidation_risk(critical_hf)
assert critical_risk["cascade_risk_level"] == "CRITICAL"
print("[OK] Critical cascade detection works")

# 3. ProtocolQualityScorer — security score (pure lookup, no network)
scorer = ProtocolQualityScorer()
sec_aave = scorer.compute_security_score("aave")
sec_hack = scorer.compute_security_score("ronin")  # heavily hacked protocol
assert sec_aave > sec_hack, "Aave should score higher security than Ronin"
assert 0 <= sec_aave <= 100
assert 0 <= sec_hack <= 100
print(f"[OK] Security scores: aave={sec_aave:.0f}, ronin={sec_hack:.0f}")

# Governance score (pure lookup, no network)
gov_aave = scorer.compute_governance_score("aave")
gov_unknown = scorer.compute_governance_score("unknown-protocol-xyz")
assert gov_aave > gov_unknown, "Aave should score higher governance"
assert 0 <= gov_aave <= 100
print(f"[OK] Governance scores: aave={gov_aave:.0f}, unknown={gov_unknown:.0f}")

# 4. GovernanceTokenAnalyzer — emission pressure (pure logic)
gov_analyzer = GovernanceTokenAnalyzer()
emission = gov_analyzer.detect_emission_pressure("pancakeswap")  # high inflation
assert emission["emission_pressure"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
assert emission["annual_emission_rate_pct"] >= 0
print(f"[OK] Emission pressure: {emission['emission_pressure']} for pancakeswap")

low_emission = gov_analyzer.detect_emission_pressure("makerdao")  # low inflation
assert low_emission["emission_pressure"] == "LOW"
print(f"[OK] Low emission confirmed for makerdao")

# 5. ProtocolScore dataclass (pure Python)
score = ProtocolScore(
    protocol="test",
    tvl_quality=80.0,
    revenue_quality=75.0,
    security_score=85.0,
    governance_score=70.0,
)
assert score.composite > 0
assert score.tier in ("S", "A", "B", "C", "D")
print(f"[OK] ProtocolScore: composite={score.composite:.1f}, tier={score.tier}")

# S tier (score >= 85)
s_score = ProtocolScore(
    protocol="top",
    tvl_quality=95.0,
    revenue_quality=90.0,
    security_score=92.0,
    governance_score=88.0,
)
assert s_score.tier == "S", f"Expected S tier, got {s_score.tier}"
print("[OK] S-tier protocol scoring works")

# 6. Known data integrity checks
assert "aave" in KNOWN_AUDITS, "Aave must be in audit database"
assert KNOWN_AUDITS["aave"] >= 8, "Aave should have 8+ audits"
assert "ronin" in KNOWN_HACKS, "Ronin hack must be in database"
assert KNOWN_HACKS["ronin"] >= 600_000_000, "Ronin hack >= $600M"
assert len(PROTOCOL_COINGECKO_IDS) >= 10, "Must have 10+ protocol CoinGecko IDs"
assert len(TOKEN_EMISSION_SCHEDULES) >= 8, "Must have 8+ emission schedules"
print(f"[OK] Known data: {len(KNOWN_AUDITS)} audits, {len(KNOWN_HACKS)} hacks tracked")

# 7. Class interface verification
assert hasattr(DefiLlamaAdvancedClient, 'get_protocols')
assert hasattr(DefiLlamaAdvancedClient, 'get_yield_pools')
assert hasattr(DefiLlamaAdvancedClient, 'get_bridge_flows')
assert hasattr(DefiLlamaAdvancedClient, 'get_stablecoin_breakdown')
assert hasattr(YieldOptimizerEngine, 'get_risk_adjusted_yields')
assert hasattr(YieldOptimizerEngine, 'compute_optimal_allocation')
assert hasattr(YieldOptimizerEngine, 'detect_yield_arb')
assert hasattr(YieldOptimizerEngine, 'estimate_gas_cost_impact')
assert hasattr(BridgeFlowAnalyzer, 'detect_capital_rotation')
assert hasattr(BridgeFlowAnalyzer, 'detect_bridge_stress')
assert hasattr(DeFiMarketMonitor, 'get_defi_market_dashboard')
assert hasattr(DeFiMarketMonitor, 'compute_defi_health_index')
assert hasattr(DeFiMarketMonitor, 'generate_weekly_report')
print("[OK] All class interfaces verified (14 methods)")

# 8. Gas cost estimation (pure math)
yield_engine = YieldOptimizerEngine()
gas = yield_engine.estimate_gas_cost_impact(apy=5.0, capital=100_000, chain="ethereum")
assert gas["annual_yield_gross"] == 5000.0
assert gas["gas_cost_usd"] == 150.0
assert gas["net_apy"] < 5.0  # gas reduces yield
print(f"[OK] Gas impact: ETH 5% APY $100k -> net {gas['net_apy']:.2f}%")

l2_gas = yield_engine.estimate_gas_cost_impact(apy=5.0, capital=10_000, chain="base")
assert l2_gas["gas_cost_usd"] < 5.0  # Base L2 very cheap
print(f"[OK] L2 gas (Base): ${l2_gas['gas_cost_usd']} per round trip")

# ===========================================================================
# 9. ImpermanentLossCalculator — IL formula (pure math)
# ===========================================================================

# IL when price doubles: ratio = 2
# Expected: IL = 2*sqrt(2)/(1+2) - 1 = 2*1.41421/3 - 1 = 0.94281 - 1 = -0.05719
price_ratio = 2.0
il = ImpermanentLossCalculator.compute_il(price_ratio)
expected_il = 2 * math.sqrt(2) / (1 + 2) - 1
assert abs(il - expected_il) < 1e-9, f"IL mismatch: {il} vs {expected_il}"
assert abs(il - (-0.05719)) < 1e-4, f"IL should be ~-5.72%, got {il*100:.4f}%"
print(f"[OK] IL formula (price doubles): {il*100:.4f}% (expected ~-5.72%)")

# IL is always <= 0
for ratio in [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]:
    il_r = ImpermanentLossCalculator.compute_il(ratio)
    assert il_r <= 0.0, f"IL must be <= 0, got {il_r} at ratio={ratio}"
print("[OK] IL is always <= 0 across all price ratios")

# IL = 0 when price unchanged (ratio = 1.0)
il_no_change = ImpermanentLossCalculator.compute_il(1.0)
assert abs(il_no_change) < 1e-12, f"IL should be 0 when no price change, got {il_no_change}"
print(f"[OK] IL = 0 when price unchanged (ratio=1.0)")

# Symmetric: IL(ratio=r) == IL(ratio=1/r)
il_up   = ImpermanentLossCalculator.compute_il(2.0)
il_down = ImpermanentLossCalculator.compute_il(0.5)
assert abs(il_up - il_down) < 1e-9, f"IL should be symmetric: up={il_up:.6f} down={il_down:.6f}"
print(f"[OK] IL is symmetric: price x2 and price /2 give same IL = {il_up*100:.4f}%")

# 10. IL in dollar terms
initial_value = 100_000.0
il_dollar = ImpermanentLossCalculator.compute_il_dollar(initial_value, price_ratio=2.0)
assert il_dollar < 0, "IL dollar must be negative"
assert abs(il_dollar - initial_value * expected_il) < 1e-6
print(f"[OK] IL dollar: $100k position, price doubles -> ${il_dollar:.2f} loss")

# 11. LP P&L — can be positive if fees > IL
fees_earned = 5_000.0  # $5k fees
pnl_positive = ImpermanentLossCalculator.compute_lp_pnl(
    initial_value=100_000.0,
    price_ratio=2.0,
    fees_earned=fees_earned,
    opportunity_cost=0.0,
)
# IL dollar ~ -$5,720, fees = $5,000 -> pnl ~ -$720 (still negative)
# But with higher fees it becomes positive:
pnl_profitable = ImpermanentLossCalculator.compute_lp_pnl(
    initial_value=100_000.0,
    price_ratio=2.0,
    fees_earned=10_000.0,
    opportunity_cost=0.0,
)
assert pnl_profitable > il_dollar, "Higher fees improve LP P&L"
print(f"[OK] LP P&L: fees=${fees_earned:.0f} -> pnl={pnl_positive:.2f}; fees=$10k -> pnl={pnl_profitable:.2f}")

# LP P&L sign: fees > |IL| -> positive P&L
big_fees = abs(il_dollar) + 1000
pnl_net_positive = ImpermanentLossCalculator.compute_lp_pnl(100_000.0, 2.0, big_fees)
assert pnl_net_positive > 0, f"Fees exceeding IL should yield positive P&L: {pnl_net_positive}"
print(f"[OK] LP P&L positive when fees > IL magnitude: {pnl_net_positive:.2f}")

# 12. IL fee APY breakeven
breakeven_apy = ImpermanentLossCalculator.compute_fee_apy_breakeven(price_ratio=2.0, holding_period_years=1.0)
assert breakeven_apy > 0, "Breakeven fee APY should be positive"
assert abs(breakeven_apy - abs(expected_il)) < 1e-9, f"Breakeven mismatch: {breakeven_apy}"
print(f"[OK] Breakeven fee APY for price doubling: {breakeven_apy*100:.4f}%/yr")

# 13. Liquidation cascade simulation
positions = [
    {"collateral_value": 100_000, "debt_value": 75_000, "liquidation_threshold": 0.80},
    {"collateral_value": 80_000,  "debt_value": 65_000, "liquidation_threshold": 0.80},
    {"collateral_value": 50_000,  "debt_value": 38_000, "liquidation_threshold": 0.80},
]
result = ImpermanentLossCalculator.simulate_liquidation_cascade(
    positions=positions,
    initial_price_drop_pct=20.0,
    cascade_multiplier=0.10,
)

# With 20% price drop: effective collateral = 80% of original
# Position 1: 100k*0.8=80k collateral vs 75k/0.8=93.75k threshold -> LIQUIDATED
# Position 2: 80k*0.8=64k collateral vs 65k/0.8=81.25k threshold -> LIQUIDATED
# Position 3: 50k*0.8=40k collateral vs 38k/0.8=47.5k threshold -> LIQUIDATED
assert result["positions_liquidated"] >= 1, f"At least 1 position should liquidate: {result}"
assert result["total_liquidated_value"] > 0, "Liquidated value must be > 0"
print(f"[OK] Cascade: {result['positions_liquidated']} positions liquidated, "
      f"${result['total_liquidated_value']:,.0f} total, "
      f"{result['liquidation_rounds']} rounds")

# Empty positions -> zero result
empty_result = ImpermanentLossCalculator.simulate_liquidation_cascade([], 20.0)
assert empty_result["total_liquidated_value"] == 0.0
print("[OK] Empty cascade returns zero liquidated value")

# 14. ImpermanentLossCalculator interface
assert hasattr(ImpermanentLossCalculator, 'compute_il')
assert hasattr(ImpermanentLossCalculator, 'compute_il_dollar')
assert hasattr(ImpermanentLossCalculator, 'compute_lp_pnl')
assert hasattr(ImpermanentLossCalculator, 'compute_fee_apy_breakeven')
assert hasattr(ImpermanentLossCalculator, 'simulate_liquidation_cascade')
print("[OK] ImpermanentLossCalculator: all 5 methods present")

print("\n[PASS] dim_107: DeFi Protocol Analytics — all checks passed (including IL + cascade)")
PYEOF
