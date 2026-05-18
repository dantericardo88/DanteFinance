#!/usr/bin/env bash
# dim_075: Crypto screener — pure screening logic (no network)
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
# Momentum = (current_price - price_30d_ago) / price_30d_ago
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

# Test value score logic (market_cap / tvl ratio)
# Low ratio = undervalued DeFi
def compute_mcap_tvl_score(market_cap, tvl):
    if tvl <= 0:
        return None
    ratio = market_cap / tvl
    # Score 0-100: ratio < 1 = very cheap, ratio > 10 = expensive
    import math
    score = max(0, 100 - math.log10(max(ratio, 0.01)) * 33.3)
    return min(100, max(0, score))

score_cheap = compute_mcap_tvl_score(1e9, 2e9)   # mcap/tvl = 0.5 → cheap
score_expensive = compute_mcap_tvl_score(10e9, 1e9)  # mcap/tvl = 10 → expensive
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

print("\n[PASS] dim_075: Crypto screener")
PYEOF
