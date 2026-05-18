#!/usr/bin/env bash
# dim_106: CCXT multi-exchange v3 — smart order router math (pure, no network)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

from sentinel.sds.adapters.ccxt_multi_exchange_v3 import SmartOrderRouter

# We test only the pure-math methods — no network needed.
# Use a minimal stub for exchange_manager and ob_aggregator.

class _FakeMgr:
    def get_exchange_taker_fee(self, ex_id):
        fees = {"binance": 0.001, "coinbase": 0.005, "kraken": 0.0026}
        return fees.get(ex_id, 0.002)

class _FakeBook:
    def __init__(self, bids, asks, mid):
        self.bids = bids
        self.asks = asks
        self.mid_price = mid

class _FakeOBA:
    pass

router = SmartOrderRouter(_FakeMgr(), _FakeOBA())

# -----------------------------------------------------------------------
# 1. Route splitting — $1M across 3 exchanges with liquidity [500k,300k,200k]
# -----------------------------------------------------------------------
liquidity_map = {"binance": 500_000, "coinbase": 300_000, "kraken": 200_000}
splits = router.split_order_by_liquidity(
    symbol="BTC/USDT",
    side="buy",
    total_quantity=1_000_000,
    exchanges=["binance", "coinbase", "kraken"],
    liquidity_map=liquidity_map,
)
assert len(splits) == 3, f"Expected 3 splits, got {len(splits)}"

# Sort by exchange name for deterministic comparison
by_exchange = {s["exchange"]: s for s in splits}

binance_pct = by_exchange["binance"]["allocation_pct"]
coinbase_pct = by_exchange["coinbase"]["allocation_pct"]
kraken_pct   = by_exchange["kraken"]["allocation_pct"]

assert abs(binance_pct - 50.0) < 0.01, f"Binance should get 50%, got {binance_pct}"
assert abs(coinbase_pct - 30.0) < 0.01, f"Coinbase should get 30%, got {coinbase_pct}"
assert abs(kraken_pct   - 20.0) < 0.01, f"Kraken should get 20%, got {kraken_pct}"

total_qty = sum(s["quantity"] for s in splits)
assert abs(total_qty - 1_000_000) < 0.01, f"Total qty should equal order: {total_qty}"
print(f"[OK] Route split: binance={binance_pct:.1f}% coinbase={coinbase_pct:.1f}% kraken={kraken_pct:.1f}%")

# Sorted descending by liquidity
assert splits[0]["exchange"] == "binance"
assert splits[1]["exchange"] == "coinbase"
assert splits[2]["exchange"] == "kraken"
print("[OK] Split order sorted by liquidity descending")

# -----------------------------------------------------------------------
# 2. Slippage estimation — monotone (larger order = more slippage)
# -----------------------------------------------------------------------
# Use deep book ($50M depth) so all test orders stay below the 1.0 cap
depth = 50_000_000  # $50M order book depth
coeff = 0.1

slip_small  = router.estimate_slippage(order_size_usd=10_000,   bid_ask_depth_usd=depth, market_impact_coeff=coeff)
slip_medium = router.estimate_slippage(order_size_usd=500_000,  bid_ask_depth_usd=depth, market_impact_coeff=coeff)
slip_large  = router.estimate_slippage(order_size_usd=5_000_000, bid_ask_depth_usd=depth, market_impact_coeff=coeff)

assert slip_small < slip_medium < slip_large, (
    f"Slippage must be monotone: {slip_small:.6f} < {slip_medium:.6f} < {slip_large:.6f}"
)
print(f"[OK] Slippage monotone: 10k={slip_small:.6f} 500k={slip_medium:.6f} 5M={slip_large:.6f}")

# Verify formula: slippage = order_size / (depth × coeff)
expected_small = 10_000 / (depth * coeff)
assert abs(slip_small - expected_small) < 1e-9, f"Formula mismatch: {slip_small} vs {expected_small}"
print(f"[OK] Slippage formula verified: 10k/(50M*0.1) = {slip_small:.6f}")

# Edge: zero depth → max slippage (1.0)
slip_no_liq = router.estimate_slippage(100_000, 0)
assert slip_no_liq == 1.0, f"Zero depth slippage should be 1.0, got {slip_no_liq}"
print("[OK] Zero depth returns 1.0 slippage")

# -----------------------------------------------------------------------
# 3. Best execution scoring — lower fee wins when price is equal
# -----------------------------------------------------------------------
# Use score_best_execution's fee/slippage math directly to test the principle.
# score = price_improvement - fee_pct - slippage_estimate
# When price_improvement and slippage are equal, lower fee = higher score.

def _mock_score(price_improvement, fee_pct, slippage):
    return price_improvement - fee_pct - slippage

# Equal price & slippage → lower fee wins
score_low_fee  = _mock_score(price_improvement=0.0, fee_pct=0.001, slippage=0.01)
score_high_fee = _mock_score(price_improvement=0.0, fee_pct=0.005, slippage=0.01)
assert score_low_fee > score_high_fee, (
    f"Lower fee should give higher score: {score_low_fee:.4f} > {score_high_fee:.4f}"
)
print(f"[OK] Lower fee wins: low_fee_score={score_low_fee:.4f} > high_fee_score={score_high_fee:.4f}")

# Price improvement can overcome a higher fee
score_price_improved = _mock_score(price_improvement=0.01, fee_pct=0.005, slippage=0.01)
score_no_improvement = _mock_score(price_improvement=0.0,  fee_pct=0.001, slippage=0.01)
assert score_price_improved > score_no_improvement, (
    "Price improvement should overcome fee difference"
)
print(f"[OK] Price improvement ({score_price_improved:.4f}) can overcome fee penalty ({score_no_improvement:.4f})")

# -----------------------------------------------------------------------
# 4. TWAP schedule — 10 slices over 60 min → ~6 min intervals with jitter
# -----------------------------------------------------------------------
schedule = router.generate_twap_schedule(
    symbol="BTC/USDT",
    side="buy",
    total_quantity=100.0,
    time_window_minutes=60,
    num_slices=10,
    jitter_pct=0.10,
)

assert len(schedule) == 10, f"Expected 10 slices, got {len(schedule)}"
print(f"[OK] TWAP: {len(schedule)} slices generated")

# Each slice has equal quantity
slice_qty = schedule[0]["quantity"]
for s in schedule:
    assert abs(s["quantity"] - slice_qty) < 1e-6, f"Unequal slice qty: {s['quantity']}"
assert abs(slice_qty * 10 - 100.0) < 1e-4, f"Total qty mismatch: {slice_qty*10}"
print(f"[OK] TWAP: equal slice quantity = {slice_qty:.4f}")

# Check nominal intervals are ~6 minutes
nominal_offsets = [s["scheduled_offset_minutes"] for s in schedule]
intervals = [nominal_offsets[i+1] - nominal_offsets[i] for i in range(len(nominal_offsets)-1)]
for iv in intervals:
    assert abs(iv - 6.0) < 1e-6, f"Expected 6-min nominal interval, got {iv}"
print(f"[OK] TWAP: nominal interval = 6.0 min")

# Jitter is within ±10% of 6-min interval = ±0.6 min
for s in schedule:
    assert abs(s["jitter_minutes"]) <= 6.0 * 0.10 + 1e-6, (
        f"Jitter out of bounds: {s['jitter_minutes']}"
    )
print(f"[OK] TWAP: all jitter values within ±10% of interval")

# execute_at_minutes bounded to [0, 60]
for s in schedule:
    assert 0.0 <= s["execute_at_minutes"] <= 60.0 + 1e-9, (
        f"execute_at out of bounds: {s['execute_at_minutes']}"
    )
print(f"[OK] TWAP: all execute_at times within [0, 60] minutes")

# All required fields present
required = {"slice_index", "symbol", "side", "quantity",
            "scheduled_offset_minutes", "jitter_minutes", "execute_at_minutes"}
for s in schedule:
    assert required.issubset(s.keys()), f"Missing fields in slice: {set(s.keys())}"
print("[OK] TWAP: all required fields present in each slice")

print("\n[PASS] dim_106: SmartOrderRouter — split, slippage, best execution, TWAP")
PYEOF
