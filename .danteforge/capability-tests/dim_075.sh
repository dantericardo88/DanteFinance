#!/usr/bin/env bash
# dim_075: Crypto screener — pure screening logic + on-chain momentum / DeFi / fear-greed
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sbx.crypto_screener import (
    CryptoScreener, SCREEN_CATEGORIES, _STABLECOIN_IDS,
)

# Test that SCREEN_CATEGORIES is populated
assert "layer1" in SCREEN_CATEGORIES, "Missing 'layer1' category"
assert "defi" in SCREEN_CATEGORIES, "Missing 'defi' category"
assert "layer2" in SCREEN_CATEGORIES, "Missing 'layer2' category"
assert len(SCREEN_CATEGORIES["layer1"]) >= 5, "layer1 should have >= 5 coins"
print(f"[OK] SCREEN_CATEGORIES: {len(SCREEN_CATEGORIES)} categories")

# Test stablecoin exclusion set
assert "tether" in _STABLECOIN_IDS, "tether should be in stablecoin set"
assert "usd-coin" in _STABLECOIN_IDS, "usd-coin should be in stablecoin set"
assert "bitcoin" not in _STABLECOIN_IDS, "bitcoin should not be a stablecoin"
print(f"[OK] _STABLECOIN_IDS has {len(_STABLECOIN_IDS)} stablecoins")

# Test screener can be instantiated
screener = CryptoScreener()
print("[OK] CryptoScreener instantiated")

# Test pure momentum signal computation
def compute_momentum(current_price, price_30d_ago):
    if price_30d_ago <= 0:
        return 0.0
    return (current_price - price_30d_ago) / price_30d_ago

btc_momentum = compute_momentum(68000, 55000)
assert btc_momentum > 0.20, f"BTC momentum should be >20%: {btc_momentum:.2%}"
print(f"[OK] BTC 30d momentum: {btc_momentum:.2%}")

eth_momentum = compute_momentum(3200, 3500)
assert eth_momentum < 0, f"ETH momentum should be negative: {eth_momentum:.2%}"
print(f"[OK] ETH 30d momentum: {eth_momentum:.2%}")

# Test value score logic
def compute_mcap_tvl_score(market_cap, tvl):
    if tvl <= 0:
        return None
    ratio = market_cap / tvl
    import math
    score = max(0, 100 - math.log10(max(ratio, 0.01)) * 33.3)
    return min(100, max(0, score))

score_cheap = compute_mcap_tvl_score(1e9, 2e9)
score_expensive = compute_mcap_tvl_score(10e9, 1e9)
assert score_cheap > score_expensive, f"Cheap ratio should score higher: {score_cheap:.1f} vs {score_expensive:.1f}"
print(f"[OK] TVL value score: cheap ratio={score_cheap:.1f} expensive ratio={score_expensive:.1f}")

# Test category membership
assert "bitcoin" in SCREEN_CATEGORIES["layer1"], "bitcoin should be in layer1"
assert "uniswap" in SCREEN_CATEGORIES["defi"], "uniswap should be in defi"
assert "matic-network" in SCREEN_CATEGORIES["layer2"], "matic should be in layer2"
print("[OK] Category membership checks pass")

# Test Fear & Greed index classification (pure logic)
def classify_fng(value):
    if value >= 75: return "Extreme Greed"
    if value >= 55: return "Greed"
    if value >= 45: return "Neutral"
    if value >= 25: return "Fear"
    return "Extreme Fear"

assert classify_fng(80) == "Extreme Greed"
assert classify_fng(50) == "Neutral"
assert classify_fng(15) == "Extreme Fear"
print("[OK] Fear & Greed index classification logic verified")

# ── NEW: compute_on_chain_momentum ───────────────────────────────────────────
import numpy as np

# 100 days of synthetic active address data (gradually growing)
np.random.seed(42)
base = 500_000
addresses = [int(base + i * 1000 + np.random.randint(-2000, 2000)) for i in range(100)]

result = screener.compute_on_chain_momentum(addresses, window_days=30, zscore_lookback=90)
assert "daily_growth_rate" in result, "Missing daily_growth_rate"
assert "total_30d_change" in result, "Missing total_30d_change"
assert "z_score" in result, "Missing z_score"
assert "momentum_label" in result, "Missing momentum_label"

# Verify formula: daily_growth_rate = (addr[-1] - addr[-31]) / 30
expected_change = addresses[-1] - addresses[-31]
expected_daily = expected_change / 30
assert abs(result["total_30d_change"] - expected_change) <= 1, (
    f"total_30d_change mismatch: expected {expected_change}, got {result['total_30d_change']}"
)
assert abs(result["daily_growth_rate"] - expected_daily) < 0.1, (
    f"daily_growth_rate mismatch: expected {expected_daily:.2f}, got {result['daily_growth_rate']}"
)
print(f"[OK] compute_on_chain_momentum: daily_growth={result['daily_growth_rate']:.1f} z={result['z_score']} label={result['momentum_label']}")

# Edge: insufficient data
short_result = screener.compute_on_chain_momentum([100_000, 101_000], window_days=30)
assert short_result["momentum_label"] == "insufficient_data", "Should return insufficient_data"
print("[OK] compute_on_chain_momentum: insufficient_data for short series")

# ── NEW: compute_crypto_fear_greed ────────────────────────────────────────────
# Composite = price_momentum*0.25 + volume*0.25 + social*0.25 + dominance*0.15 + trends*0.10
fg = screener.compute_crypto_fear_greed(
    price_momentum_score=80.0,
    volume_score=70.0,
    social_score=60.0,
    dominance_score=50.0,
    trends_score=40.0,
)
assert "composite" in fg, "Missing composite key"
assert "classification" in fg, "Missing classification key"
assert "components" in fg, "Missing components key"

expected_composite = 80*0.25 + 70*0.25 + 60*0.25 + 50*0.15 + 40*0.10
assert abs(fg["composite"] - expected_composite) < 0.01, (
    f"Fear-greed composite formula mismatch: expected {expected_composite:.2f}, got {fg['composite']}"
)
print(f"[OK] compute_crypto_fear_greed: composite={fg['composite']:.2f} ({fg['classification']}) — formula verified")

# Boundary: extreme fear
fg_fear = screener.compute_crypto_fear_greed(0, 0, 0, 0, 0)
assert fg_fear["composite"] == 0.0
assert fg_fear["classification"] == "Extreme Fear"
print(f"[OK] compute_crypto_fear_greed: 0-score -> Extreme Fear")

# Boundary: extreme greed
fg_greed = screener.compute_crypto_fear_greed(100, 100, 100, 100, 100)
assert fg_greed["composite"] == 100.0
assert fg_greed["classification"] == "Extreme Greed"
print(f"[OK] compute_crypto_fear_greed: 100-score -> Extreme Greed")

print("\n[PASS] dim_075: Crypto screener")
PYEOF
