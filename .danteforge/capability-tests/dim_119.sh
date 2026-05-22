#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
python - <<'PYEOF'
import sys, os, numpy as np
sys.path.insert(0, os.getcwd())
np.random.seed(42)

from sentinel.sma.lob_microstructure_v3 import (
    LOBSnapshot, OrderBookLevel, LOBAnalytics, MicrostructureMetrics,
    LOBSimulator, quoted_spread, order_book_imbalance, kyle_lambda,
    amihud_illiquidity, vpin, roll_spread
)

# 1. Build a LOB snapshot
bids = [OrderBookLevel(99.95, 100), OrderBookLevel(99.90, 200), OrderBookLevel(99.85, 300)]
asks = [OrderBookLevel(100.05, 150), OrderBookLevel(100.10, 250), OrderBookLevel(100.15, 400)]
snap = LOBSnapshot(bids=bids, asks=asks)

assert snap.best_bid == 99.95
assert snap.best_ask == 100.05
assert abs(snap.midpoint - 100.0) < 1e-6
assert abs(snap.quoted_spread - 0.10) < 1e-6
assert snap.spread_bps > 0
obi = snap.imbalance  # (100+200+300 - 150+250+400) / total
assert -1 <= obi <= 1
wmid = snap.weighted_mid
assert 99.95 <= wmid <= 100.05  # between bid and ask

# 2. LOBSimulator
sim = LOBSimulator(mid_price=100.0, spread_bps=5.0, depth_levels=5, seed=42)
snaps = sim.generate_sequence(n=50)
assert len(snaps) == 50
assert all(s.quoted_spread > 0 for s in snaps)

# 3. LOBAnalytics
analytics = LOBAnalytics(snaps)
tws = analytics.time_weighted_spread()
assert tws > 0
mi = analytics.market_impact(order_size=1000.0, side='buy')
assert mi >= 0  # bps cost >= 0

# 4. Microstructure metrics
trade_flow = sim.simulate_trade_flow(n_trades=200)
prices = trade_flow['prices']
volumes = trade_flow['volumes']
directions = trade_flow['directions']
midpoints = trade_flow['midpoints']
future_mids = np.roll(midpoints, -5); future_mids[-5:] = midpoints[-5:]

m = MicrostructureMetrics()
eff_spread = m.effective_spread(prices, midpoints, directions)
assert eff_spread >= 0
kl = m.kyle_lambda(np.diff(midpoints), directions[1:] * volumes[1:])
assert np.isfinite(kl)
returns = np.diff(prices) / prices[:-1]
illiq = m.amihud_illiquidity(returns, volumes[1:])
assert illiq >= 0

# 5. Module-level functions
kl2 = kyle_lambda(np.diff(prices), directions[1:] * volumes[1:])
assert np.isfinite(kl2)
illiq2 = amihud_illiquidity(returns, volumes[1:])
assert illiq2 >= 0
rs = roll_spread(prices)
assert np.isfinite(rs)

# 6. VPIN
vpin_val = vpin(prices, volumes, bucket_size=500.0)
assert 0 <= vpin_val <= 1, f"VPIN should be in [0,1], got {vpin_val:.4f}"

print(f"Quoted spread: {snap.spread_bps:.2f} bps, OBI: {obi:.3f}")
print(f"Kyle lambda: {kl:.6f}, Amihud: {illiq:.8f}, VPIN: {vpin_val:.4f}")
print("[PASS] dim_119: LOB microstructure analytics")
PYEOF
