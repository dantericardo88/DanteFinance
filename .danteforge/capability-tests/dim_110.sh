#!/usr/bin/env bash
# dim_110: DEX/AMM Analytics v3 — rugpull detection + IL modeling (pure math)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

from sentinel.sfe.dex_analytics_v3 import (
    DEXAnalyticsEngine,
    AMMPriceEngine,
    LiquidityFlowAnalyzer,
    YieldFarmingAnalyzer,
    DEXVolumeTracker,
    DEXScreener,
    ILCalculator,
    PoolData,
    SwapData,
    LiquidityTick,
    ProtocolHealth,
    YieldOpportunity,
    _TTL,
)
print("[OK] All DEX analytics classes imported (including ILCalculator)")

# -----------------------------------------------------------------------
# Existing structural tests (preserved)
# -----------------------------------------------------------------------
assert isinstance(_TTL, dict) and len(_TTL) >= 5
for key, val in _TTL.items():
    assert isinstance(val, int) and val > 0
print(f"[OK] _TTL: {len(_TTL)} entries, all positive integers")

# AMMPriceEngine.sqrt_price_x96_to_price — pure math
dec0, dec1 = 18, 6
target_price_raw = 2000.0
decimal_factor = 10 ** (dec0 - dec1)
sqrt_val = math.sqrt(target_price_raw / decimal_factor)
sqrt_price_x96 = int(sqrt_val * (2 ** 96))
price = AMMPriceEngine.sqrt_price_x96_to_price(sqrt_price_x96, token0_decimals=dec0, token1_decimals=dec1)
assert abs(price - target_price_raw) < 1.0
print(f"[OK] sqrt_price_x96_to_price: ETH/USDC = {price:.2f} USDC")

# PoolData properties
pool = PoolData(
    id="0xtest001",
    token0_symbol="ETH",
    token1_symbol="USDC",
    token0_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    token1_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    fee_tier=3000,
    liquidity=1_000_000.0,
    tvl_usd=50_000_000.0,
    volume_usd_24h=25_000_000.0,
    fees_usd_24h=75_000.0,
    volume_usd_total=10_000_000_000.0,
    token0_price=2000.0,
    token1_price=0.0005,
)
assert pool.fee_pct == 0.003
assert pool.vol_tvl_ratio == 0.5
print(f"[OK] PoolData: fee_pct={pool.fee_pct} vol_tvl={pool.vol_tvl_ratio}")

# -----------------------------------------------------------------------
# 1. Rugpull score — high TVL drop + no audit -> score > 70 (HIGH RISK)
# -----------------------------------------------------------------------
analyzer = LiquidityFlowAnalyzer.__new__(LiquidityFlowAnalyzer)

result_high = analyzer.compute_rugpull_score_from_signals(
    tvl_change_1d=-0.60,   # -60% TVL in 1 day
    has_audit=False,
    sell_tax_pct=0.0,
    ownership_renounced=True,
    price_impact_small_trade_pct=0.0,
)
assert result_high["rug_score"] >= 70, (
    f"60% TVL drop + no audit should score >= 70, got {result_high['rug_score']}"
)
assert result_high["risk_level"] == "HIGH RISK"
assert result_high["signals"]["liquidity_removal"]["points"] == 40
assert result_high["signals"]["no_audit"]["points"] == 30
print(f"[OK] Rugpull HIGH RISK: score={result_high['rug_score']} (liq=40pts + no_audit=30pts)")

# -----------------------------------------------------------------------
# 2. Safe protocol — low score
# -----------------------------------------------------------------------
result_safe = analyzer.compute_rugpull_score_from_signals(
    tvl_change_1d=0.02,
    has_audit=True,
    sell_tax_pct=0.0,
    ownership_renounced=True,
    price_impact_small_trade_pct=0.1,
)
assert result_safe["rug_score"] < 40
assert result_safe["risk_level"] == "LOW RISK"
print(f"[OK] Safe protocol: score={result_safe['rug_score']} (LOW RISK)")

# -----------------------------------------------------------------------
# 3. Honeypot — sell_tax > 10% adds points
# -----------------------------------------------------------------------
result_honeypot = analyzer.compute_rugpull_score_from_signals(
    tvl_change_1d=0.0, has_audit=True, sell_tax_pct=25.0,
    ownership_renounced=False, price_impact_small_trade_pct=0.0,
)
assert result_honeypot["signals"]["honeypot"]["points"] > 0
print(f"[OK] Honeypot: sell_tax=25% -> {result_honeypot['signals']['honeypot']['points']} pts")

# -----------------------------------------------------------------------
# 4. Price impact anomaly — >5% impact flagged
# -----------------------------------------------------------------------
result_thin = analyzer.compute_rugpull_score_from_signals(
    tvl_change_1d=0.0, has_audit=True, sell_tax_pct=0.0,
    ownership_renounced=True, price_impact_small_trade_pct=8.0,
)
assert result_thin["signals"]["price_impact_anomaly"]["points"] == 15
print(f"[OK] Price impact anomaly: 8% impact -> 15 pts")

result_normal = analyzer.compute_rugpull_score_from_signals(
    tvl_change_1d=0.0, has_audit=True, sell_tax_pct=0.0,
    ownership_renounced=True, price_impact_small_trade_pct=0.5,
)
assert result_normal["signals"]["price_impact_anomaly"]["points"] == 0
print("[OK] Price impact OK when < 5%: 0 pts")

# Score capped at 100
result_max = analyzer.compute_rugpull_score_from_signals(
    tvl_change_1d=-0.90, has_audit=False, sell_tax_pct=50.0,
    ownership_renounced=False, price_impact_small_trade_pct=20.0,
)
assert result_max["rug_score"] <= 100.0
print(f"[OK] Rug score capped at 100: got {result_max['rug_score']}")

# -----------------------------------------------------------------------
# 5. ILCalculator — formula matches dim_107
# -----------------------------------------------------------------------
# IL when price doubles: IL = 2*sqrt(2)/(1+2) - 1 ~ -0.05719
price_ratio = 2.0
il = ILCalculator.compute_il(price_ratio)
expected = 2 * math.sqrt(2) / (1 + 2) - 1
assert abs(il - expected) < 1e-9, f"IL formula mismatch: {il} vs {expected}"
assert abs(il - (-0.05719)) < 1e-4
print(f"[OK] ILCalculator.compute_il(2.0) = {il*100:.4f}% (matches dim_107)")

# Always <= 0
for ratio in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    assert ILCalculator.compute_il(ratio) <= 0.0 + 1e-12
print("[OK] ILCalculator: IL always <= 0 across 9 ratios")

# IL = 0 at ratio=1
assert abs(ILCalculator.compute_il(1.0)) < 1e-12
print("[OK] ILCalculator: IL = 0 when price unchanged")

# Symmetric: IL(r) == IL(1/r)
assert abs(ILCalculator.compute_il(2.0) - ILCalculator.compute_il(0.5)) < 1e-9
print("[OK] ILCalculator: symmetric (price x2 == price /2)")

# -----------------------------------------------------------------------
# 6. ILCalculator.compute_pool_il — ETH doubles from 2000->4000
# -----------------------------------------------------------------------
current_prices = {"ETH": 4000.0, "USDC": 1.0}
pool_il = ILCalculator.compute_pool_il(pool, current_prices)
assert abs(pool_il["price_ratio"] - 2.0) < 0.01, f"Price ratio: {pool_il['price_ratio']}"
assert abs(pool_il["il_pct"] - (-5.72)) < 0.1, f"IL pct: {pool_il['il_pct']}"
assert pool_il["il_decimal"] < 0
# Verify it matches direct formula
direct_il = ILCalculator.compute_il(pool_il["price_ratio"])
assert abs(pool_il["il_decimal"] - direct_il) < 1e-6
print(f"[OK] compute_pool_il: ETH 2000->4000, IL={pool_il['il_pct']:.4f}% (matches direct formula)")

# No IL when price unchanged
pool_il_flat = ILCalculator.compute_pool_il(pool, {"ETH": 2000.0, "USDC": 1.0})
assert abs(pool_il_flat["il_pct"]) < 0.01
print("[OK] Pool IL ~ 0 when price unchanged")

# -----------------------------------------------------------------------
# 7. IL lookup table
# -----------------------------------------------------------------------
table = ILCalculator.il_table()
assert len(table) >= 5
for entry in table:
    assert entry["il_pct"] <= 0.0
entry_1 = next(e for e in table if e["price_ratio"] == 1.0)
assert abs(entry_1["il_pct"]) < 1e-8
print(f"[OK] IL table: {len(table)} entries, all <= 0, 1.0->0%")

# -----------------------------------------------------------------------
# 8. Interface verification
# -----------------------------------------------------------------------
assert hasattr(ILCalculator, 'compute_il')
assert hasattr(ILCalculator, 'compute_pool_il')
assert hasattr(ILCalculator, 'il_table')
assert hasattr(LiquidityFlowAnalyzer, 'detect_rugpull_risk')
assert hasattr(LiquidityFlowAnalyzer, 'compute_rugpull_score_from_signals')
print("[OK] ILCalculator: all 3 methods present")
print("[OK] LiquidityFlowAnalyzer: both rugpull methods present")

# DEXAnalyticsEngine instantiation (no network)
engine = DEXAnalyticsEngine()
assert hasattr(engine, '_llama') and hasattr(engine, '_flow')
print("[OK] DEXAnalyticsEngine instantiates correctly")

print("\n[PASS] dim_110: DEX Analytics — rugpull detection + IL modeling")
PYEOF
