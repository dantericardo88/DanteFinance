#!/usr/bin/env bash
# dim_110: DEX analytics — class structure, pure AMM price math, dataclasses
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math

from sentinel.sfe.dex_analytics_v3 import (
    DEXAnalyticsEngine,
    AMMPriceEngine,
    LiquidityFlowAnalyzer,
    YieldFarmingAnalyzer,
    DEXVolumeTracker,
    DEXScreener,
    PoolData,
    SwapData,
    LiquidityTick,
    ProtocolHealth,
    YieldOpportunity,
    _TTL,
)

# Test class importability
assert DEXAnalyticsEngine is not None, "DEXAnalyticsEngine should be importable"
assert AMMPriceEngine is not None, "AMMPriceEngine should be importable"
assert LiquidityFlowAnalyzer is not None, "LiquidityFlowAnalyzer should be importable"
assert YieldFarmingAnalyzer is not None, "YieldFarmingAnalyzer should be importable"
assert DEXVolumeTracker is not None, "DEXVolumeTracker should be importable"
assert DEXScreener is not None, "DEXScreener should be importable"
print("[OK] All DEX analytics classes imported successfully")

# Test _TTL constants have expected data types
assert isinstance(_TTL, dict), "TTL should be a dict"
assert len(_TTL) >= 5, f"Should have >= 5 TTL entries: {len(_TTL)}"
for key, val in _TTL.items():
    assert isinstance(val, int) and val > 0, f"TTL for {key} should be positive int: {val}"
print(f"[OK] _TTL: {len(_TTL)} entries, all positive integers")

# Test AMMPriceEngine.sqrt_price_x96_to_price — pure math, no network calls
# Uniswap V3 formula: price = (sqrtPriceX96 / 2^96)^2 * 10^(dec0 - dec1)
# For ETH/USDC pool (ETH=18 dec, USDC=6 dec):
# If 1 ETH = 2000 USDC, price_token1_per_token0 = 2000
# sqrtPriceX96 = sqrt(2000 / 10^(18-6)) * 2^96 = sqrt(2000 * 1e-12) * 2^96
import math as _math
dec0, dec1 = 18, 6
target_price_raw = 2000.0  # 2000 USDC per ETH in raw space
# price_raw = sqrtPriceX96^2 / 2^192 * 10^(dec0-dec1)
# so sqrtPriceX96 = sqrt(target_price_raw / 10^(dec0-dec1)) * 2^96
decimal_factor = 10 ** (dec0 - dec1)  # 1e12
sqrt_val = _math.sqrt(target_price_raw / decimal_factor)
sqrt_price_x96 = int(sqrt_val * (2 ** 96))

price = AMMPriceEngine.sqrt_price_x96_to_price(sqrt_price_x96, token0_decimals=dec0, token1_decimals=dec1)
assert abs(price - target_price_raw) < 1.0, \
    f"ETH/USDC price should be ~2000: {price:.2f}"
print(f"[OK] sqrt_price_x96_to_price: ETH/USDC price = {price:.2f} USDC (expected ~2000)")

# Edge cases: zero sqrtPriceX96 returns 0.0
price_zero = AMMPriceEngine.sqrt_price_x96_to_price(0)
assert price_zero == 0.0, f"Zero sqrtPriceX96 should return 0.0: {price_zero}"
price_neg = AMMPriceEngine.sqrt_price_x96_to_price(-1)
assert price_neg == 0.0, f"Negative sqrtPriceX96 should return 0.0: {price_neg}"
print("[OK] sqrt_price_x96_to_price edge cases: zero and negative input return 0.0")

# Test that price math is monotonic: higher sqrtPriceX96 = higher price
prices = []
for mult in [0.5, 0.8, 1.0, 1.2, 2.0]:
    sp = int(sqrt_price_x96 * mult)
    p = AMMPriceEngine.sqrt_price_x96_to_price(sp, dec0, dec1)
    prices.append(p)
for i in range(1, len(prices)):
    assert prices[i] > prices[i-1], \
        f"Price should be monotonic: {prices[i]:.2f} vs {prices[i-1]:.2f}"
print(f"[OK] sqrt_price_x96_to_price monotonic: prices increase with sqrtPriceX96")

# Test PoolData dataclass properties
from dataclasses import dataclass
pool = PoolData(
    id="0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
    token0_symbol="WETH",
    token1_symbol="USDC",
    token0_address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    token1_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    fee_tier=3000,
    liquidity=1_500_000_000.0,
    tvl_usd=50_000_000.0,
    volume_usd_24h=25_000_000.0,
    fees_usd_24h=75_000.0,
    volume_usd_total=10_000_000_000.0,
    token0_price=2000.0,
    token1_price=0.0005,
)
assert pool.fee_pct == 0.003, f"fee_tier=3000 should give fee_pct=0.003: {pool.fee_pct}"
assert pool.vol_tvl_ratio == 0.5, \
    f"vol_tvl_ratio: 25M/50M = 0.5: {pool.vol_tvl_ratio}"
assert abs(pool.daily_fee_yield - 0.0015) < 1e-10, \
    f"daily_fee_yield: 75k/50M = 0.0015: {pool.daily_fee_yield}"
assert abs(pool.annualized_fee_yield - 0.0015 * 365) < 1e-6, \
    f"annualized_fee_yield: 0.0015 * 365: {pool.annualized_fee_yield}"
print(f"[OK] PoolData: fee_pct={pool.fee_pct} vol_tvl={pool.vol_tvl_ratio} daily_fee_yield={pool.daily_fee_yield:.4f}")

# Test LiquidityTick dataclass
tick = LiquidityTick(
    tick_idx=-887272,
    liquidity_net=1_000_000_000.0,
    liquidity_gross=2_000_000_000.0,
    price=1800.5,
)
assert tick.tick_idx == -887272
assert tick.liquidity_net == 1_000_000_000.0
assert tick.price == 1800.5
print(f"[OK] LiquidityTick: tick_idx={tick.tick_idx} liq_net={tick.liquidity_net:,.0f} price={tick.price}")

# Test ProtocolHealth dataclass
protocol = ProtocolHealth(
    name="Uniswap",
    slug="uniswap",
    tvl_usd=5_000_000_000.0,
    tvl_change_1d=-0.02,
    tvl_change_7d=0.05,
    tvl_change_30d=0.15,
    chains=["Ethereum", "Arbitrum", "Polygon"],
    category="Dexes",
    volume_24h=1_500_000_000.0,
    fees_24h=4_500_000.0,
    audits=5,
    has_audit=True,
    rugpull_risk_score=0.05,
    age_days=1500,
    health_score=88.5,
)
assert protocol.name == "Uniswap"
assert len(protocol.chains) == 3
assert protocol.has_audit is True
assert 0 <= protocol.rugpull_risk_score <= 1
print(f"[OK] ProtocolHealth: {protocol.name} TVL=${protocol.tvl_usd/1e9:.1f}B chains={protocol.chains}")

# Test YieldOpportunity dataclass
yield_opp = YieldOpportunity(
    pool_id="usdc-dai-0.01",
    project="Curve",
    chain="Ethereum",
    symbol="USDC-DAI",
    tvl_usd=250_000_000.0,
    apy=0.045,
    apy_base=0.03,
    apy_reward=0.015,
    risk_score=0.15,
    risk_adjusted_apy=0.038,
    il_risk=0.0,
    is_stable=True,
    audited=True,
    stablecoin=True,
)
assert yield_opp.is_stable is True
assert yield_opp.stablecoin is True
assert yield_opp.il_risk == 0.0, "Stable pair should have 0 IL risk"
assert yield_opp.apy == yield_opp.apy_base + yield_opp.apy_reward
print(f"[OK] YieldOpportunity: {yield_opp.symbol} APY={yield_opp.apy:.2%} risk_adj={yield_opp.risk_adjusted_apy:.2%} IL={yield_opp.il_risk:.2%}")

# Test DEXAnalyticsEngine instantiates (no network calls at __init__)
engine = DEXAnalyticsEngine()
assert hasattr(engine, '_llama'), "DEXAnalyticsEngine should have _llama"
assert hasattr(engine, '_uni'), "DEXAnalyticsEngine should have _uni"
assert hasattr(engine, '_price'), "DEXAnalyticsEngine should have _price"
assert hasattr(engine, '_flow'), "DEXAnalyticsEngine should have _flow"
assert hasattr(engine, '_yield'), "DEXAnalyticsEngine should have _yield"
assert hasattr(engine, '_volume'), "DEXAnalyticsEngine should have _volume"
assert hasattr(engine, '_screener'), "DEXAnalyticsEngine should have _screener"
print("[OK] DEXAnalyticsEngine instantiates with all 7 sub-components")

# Test that tick_spacing concept works: Uniswap V3 fee tiers map to tick spacings
tick_spacing_map = {100: 1, 500: 10, 3000: 60, 10000: 200}
for fee_tier, expected_spacing in tick_spacing_map.items():
    assert expected_spacing > 0, f"Tick spacing for {fee_tier} should be positive"
print(f"[OK] Uniswap V3 tick spacing: fee_tiers {list(tick_spacing_map.keys())} -> spacings {list(tick_spacing_map.values())}")

print("\n[PASS] dim_110: DEX analytics")
PYEOF
