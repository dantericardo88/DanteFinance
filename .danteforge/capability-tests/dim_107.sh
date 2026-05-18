#!/bin/bash
# dim_107: DeFi Protocol Analytics — capability verification
# Tests pure computation logic (no network calls required)
set -e

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
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
    KNOWN_HACKS,
    KNOWN_AUDITS,
    PROTOCOL_COINGECKO_IDS,
    TOKEN_EMISSION_SCHEDULES,
)
print("[OK] All DeFi analytics classes imported")

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

print("\n[PASS] dim_107: DeFi Protocol Analytics — all checks passed")
PYEOF
