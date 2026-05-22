#!/bin/bash
# dim_010 — Order book / Level 2 depth analytics
# Exit 0 = dimension verified  |  Exit 1 = not verified
set -euo pipefail
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
import math
import numpy as np

sys.path.insert(0, ".")
from sentinel.sma.order_book_depth_v3 import (
    OrderBookLevel, OrderBook, DepthAnalytics,
    OrderBookSimulator, DepthSignalGenerator,
    depth_weighted_mid, price_impact_bps, book_imbalance, detect_walls,
)

errors = []

# -----------------------------------------------------------------------
# Build the reference order book
# -----------------------------------------------------------------------
# 5 bid levels (descending price)
bids = [
    OrderBookLevel(price=99.95, size=100),
    OrderBookLevel(price=99.90, size=200),
    OrderBookLevel(price=99.85, size=300),
    OrderBookLevel(price=99.80, size=150),
    OrderBookLevel(price=99.75, size=250),
]
# 5 ask levels (ascending price)
asks = [
    OrderBookLevel(price=100.05, size=120),
    OrderBookLevel(price=100.10, size=180),
    OrderBookLevel(price=100.15, size=220),
    OrderBookLevel(price=100.20, size=300),
    OrderBookLevel(price=100.25, size=400),
]
book = OrderBook(bids=bids, asks=asks)

# -----------------------------------------------------------------------
# Basic book properties
# -----------------------------------------------------------------------
if not math.isclose(book.best_bid, 99.95, abs_tol=1e-9):
    errors.append(f"best_bid: expected 99.95, got {book.best_bid}")

if not math.isclose(book.best_ask, 100.05, abs_tol=1e-9):
    errors.append(f"best_ask: expected 100.05, got {book.best_ask}")

spread = book.spread
if not math.isclose(spread, 0.10, abs_tol=1e-9):
    errors.append(f"spread: expected 0.10, got {spread}")

# -----------------------------------------------------------------------
# DepthAnalytics
# -----------------------------------------------------------------------
da = DepthAnalytics(book)

# bid_vwap(5): weighted avg of bid prices
bv = da.bid_vwap(5)
expected_bid_vwap = (99.95*100 + 99.90*200 + 99.85*300 + 99.80*150 + 99.75*250) / (100+200+300+150+250)
if not math.isclose(bv, expected_bid_vwap, rel_tol=1e-9):
    errors.append(f"bid_vwap: expected {expected_bid_vwap:.6f}, got {bv:.6f}")

# ask_vwap(5): weighted avg of ask prices
av = da.ask_vwap(5)
expected_ask_vwap = (100.05*120 + 100.10*180 + 100.15*220 + 100.20*300 + 100.25*400) / (120+180+220+300+400)
if not math.isclose(av, expected_ask_vwap, rel_tol=1e-9):
    errors.append(f"ask_vwap: expected {expected_ask_vwap:.6f}, got {av:.6f}")

# depth_weighted_mid must be between bid_vwap and ask_vwap
dwm = da.depth_weighted_mid(5)
if not (bv <= dwm <= av):
    errors.append(f"depth_weighted_mid {dwm:.6f} not in [{bv:.6f}, {av:.6f}]")

# price_impact_buy(500): cost in bps > 0 (walk up ask side)
# asks: 120@100.05, 180@100.10, 200@100.15 (need 200 more) → partial 200.15
pib = da.price_impact_buy(500)
if not (pib > 0):
    errors.append(f"price_impact_buy(500) should be > 0, got {pib}")

# price_impact_buy must be in reasonable range (< 100 bps for 500 shares)
if pib > 100:
    errors.append(f"price_impact_buy(500)={pib:.2f} bps seems unreasonably large")

# cumulative_bid_depth(5): returns two arrays of length 5
prices, cum_sizes = da.cumulative_bid_depth(5)
if len(prices) != 5:
    errors.append(f"cumulative_bid_depth prices length: expected 5, got {len(prices)}")
if len(cum_sizes) != 5:
    errors.append(f"cumulative_bid_depth cum_sizes length: expected 5, got {len(cum_sizes)}")
if not (cum_sizes[-1] > 0):
    errors.append(f"cumulative_bid_depth total should be > 0")
# Verify monotone increasing
if not np.all(np.diff(cum_sizes) >= 0):
    errors.append("cumulative_bid_depth not monotone increasing")

# book_skew: finite value (total bids vs asks)
skew = da.book_skew(5)
if not math.isfinite(skew):
    errors.append(f"book_skew is not finite: {skew}")
# bids sum=1000, asks sum=1220 → more asks → negative skew
total_bid = 100+200+300+150+250
total_ask = 120+180+220+300+400
expected_skew = (total_bid - total_ask) / (total_bid + total_ask)
if not math.isclose(skew, expected_skew, rel_tol=1e-9):
    errors.append(f"book_skew: expected {expected_skew:.6f}, got {skew:.6f}")

# aggregate_imbalance: in [-1, 1]
ai = da.aggregate_imbalance(5)
if not (-1.0 <= ai <= 1.0):
    errors.append(f"aggregate_imbalance {ai:.6f} out of [-1,1]")
if not math.isfinite(ai):
    errors.append(f"aggregate_imbalance is not finite: {ai}")

# bid_wall / ask_wall: at least one detected (price or None)
bwall = da.bid_wall(threshold_multiple=2.0)
awall = da.ask_wall(threshold_multiple=2.0)
# Both return either float or None — just verify type
if bwall is not None and not isinstance(bwall, float):
    errors.append(f"bid_wall should be float or None, got {type(bwall)}")
if awall is not None and not isinstance(awall, float):
    errors.append(f"ask_wall should be float or None, got {type(awall)}")

# depth_map
dmap = da.depth_map(5)
required_keys = {"bid_prices", "bid_sizes", "ask_prices", "ask_sizes", "cum_bid", "cum_ask"}
missing = required_keys - set(dmap.keys())
if missing:
    errors.append(f"depth_map missing keys: {missing}")

# -----------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------
dwm2 = depth_weighted_mid(book, levels=5)
if not math.isclose(dwm2, dwm, rel_tol=1e-9):
    errors.append(f"depth_weighted_mid convenience fn mismatch: {dwm2} vs {dwm}")

pib2 = price_impact_bps(book, 500, side="buy")
if not math.isclose(pib2, pib, rel_tol=1e-9):
    errors.append(f"price_impact_bps convenience fn mismatch: {pib2} vs {pib}")

bi = book_imbalance(book, levels=5)
if not (-1.0 <= bi <= 1.0):
    errors.append(f"book_imbalance out of [-1,1]: {bi}")

walls = detect_walls(book, threshold=2.0)
if "bid_wall" not in walls or "ask_wall" not in walls:
    errors.append(f"detect_walls missing keys: {walls.keys()}")

# -----------------------------------------------------------------------
# OrderBookSimulator
# -----------------------------------------------------------------------
sim = OrderBookSimulator(mid_price=100.0, n_levels=10, seed=42)

gen_book = sim.generate(spread_bps=5.0)
if not isinstance(gen_book, OrderBook):
    errors.append("generate() did not return OrderBook")
else:
    if gen_book.spread <= 0:
        errors.append(f"generated book spread should be > 0, got {gen_book.spread}")
    if gen_book.best_bid >= gen_book.best_ask:
        errors.append("generated book best_bid >= best_ask (crossed market)")
    if len(gen_book.bids) == 0 or len(gen_book.asks) == 0:
        errors.append("generated book has empty bids or asks")

# generate_skewed(buy_pressure=0.5): bid depth > ask depth
skewed_book = sim.generate_skewed(buy_pressure=0.5)
if not isinstance(skewed_book, OrderBook):
    errors.append("generate_skewed() did not return OrderBook")
else:
    if skewed_book.total_bid_depth <= skewed_book.total_ask_depth:
        errors.append(
            f"skewed book (buy_pressure=0.5): bid_depth={skewed_book.total_bid_depth:.0f} "
            f"should exceed ask_depth={skewed_book.total_ask_depth:.0f}"
        )

# -----------------------------------------------------------------------
# DepthSignalGenerator
# -----------------------------------------------------------------------
dsg = DepthSignalGenerator()

imb_sig = dsg.imbalance_signal(book, levels=5)
if not (-1.0 <= imb_sig <= 1.0):
    errors.append(f"imbalance_signal out of [-1,1]: {imb_sig}")

wall_sig = dsg.wall_signal(book)
if "bid_wall" not in wall_sig or "ask_wall" not in wall_sig:
    errors.append(f"wall_signal missing keys: {wall_sig.keys()}")

impact_sig = dsg.impact_cost_signal(book, trade_size=500.0)
if not math.isfinite(impact_sig):
    errors.append(f"impact_cost_signal not finite: {impact_sig}")

# -----------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------
if errors:
    print("FAIL dim_010:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print("[PASS] dim_010: Order book / Level 2 depth analytics")
PYEOF
