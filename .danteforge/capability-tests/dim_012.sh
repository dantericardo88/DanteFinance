#!/bin/bash
# dim_012 — Tick-level trade data (TAQ) analytics
# Exit 0 = dimension verified  |  Exit 1 = not verified
set -euo pipefail
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
import math
import numpy as np

sys.path.insert(0, ".")
from sentinel.sma.tick_data_v3 import (
    Trade, Quote, NBBO,
    NBBOCalculator, TradeClassifier, IntradayAnalytics,
    TAQSimulator,
    nbbo_spread, effective_spread_bps,
    volume_weighted_price, intraday_vwap,
    rogers_satchell_vol, classify_trades,
)

errors = []

# -----------------------------------------------------------------------
# TAQSimulator: generate_trades
# -----------------------------------------------------------------------
sim = TAQSimulator(mid_price=100.0, seed=42)

trades = sim.generate_trades(200)
if len(trades) != 200:
    errors.append(f"generate_trades(200): expected 200, got {len(trades)}")
for t in trades:
    if not isinstance(t, Trade):
        errors.append("generate_trades returned non-Trade object")
        break
    if t.price <= 0:
        errors.append(f"Trade price <= 0: {t.price}")
        break

# -----------------------------------------------------------------------
# TAQSimulator: generate_quotes
# -----------------------------------------------------------------------
quotes = sim.generate_quotes(400)
if len(quotes) != 400:
    errors.append(f"generate_quotes(400): expected 400, got {len(quotes)}")
neg_spread_count = sum(1 for q in quotes if q.spread <= 0)
if neg_spread_count > 0:
    errors.append(f"{neg_spread_count} quotes have spread <= 0")

# -----------------------------------------------------------------------
# TradeClassifier.tick_rule
# -----------------------------------------------------------------------
classifier = TradeClassifier()
test_prices = np.array([100.0, 100.1, 100.05, 100.2])
tick_result = classifier.tick_rule(test_prices)
if tick_result.shape != (4,):
    errors.append(f"tick_rule shape: expected (4,), got {tick_result.shape}")
valid_vals = {-1, 0, 1}
bad_vals = set(tick_result.tolist()) - valid_vals
if bad_vals:
    errors.append(f"tick_rule contains invalid values: {bad_vals}")
# prices: 100->100.1 (+1), 100.1->100.05 (-1), 100.05->100.2 (+1)
if tick_result[1] != 1:
    errors.append(f"tick_rule[1] should be +1 (uptick), got {tick_result[1]}")
if tick_result[2] != -1:
    errors.append(f"tick_rule[2] should be -1 (downtick), got {tick_result[2]}")
if tick_result[3] != 1:
    errors.append(f"tick_rule[3] should be +1 (uptick), got {tick_result[3]}")

# -----------------------------------------------------------------------
# IntradayAnalytics.vwap_series
# -----------------------------------------------------------------------
analytics = IntradayAnalytics()
ts_arr, vwap_arr = analytics.vwap_series(trades)
if not isinstance(ts_arr, np.ndarray):
    errors.append("vwap_series[0] should be np.ndarray")
if not isinstance(vwap_arr, np.ndarray):
    errors.append("vwap_series[1] should be np.ndarray")
if len(ts_arr) != len(trades) or len(vwap_arr) != len(trades):
    errors.append(f"vwap_series length mismatch: {len(ts_arr)}, {len(vwap_arr)} vs {len(trades)}")

# Last element of cum_vwap should equal the overall VWAP
overall_vwap = volume_weighted_price(trades)
if len(vwap_arr) > 0:
    last_vwap = float(vwap_arr[-1])
    if not math.isclose(last_vwap, overall_vwap, rel_tol=1e-9):
        errors.append(f"vwap_series last element {last_vwap:.6f} != overall_vwap {overall_vwap:.6f}")

# -----------------------------------------------------------------------
# IntradayAnalytics.volume_profile
# -----------------------------------------------------------------------
vp = analytics.volume_profile(trades, n_buckets=13)
if vp.shape != (13,):
    errors.append(f"volume_profile shape: expected (13,), got {vp.shape}")
if vp.sum() <= 0:
    errors.append(f"volume_profile sum should be > 0, got {vp.sum()}")

# -----------------------------------------------------------------------
# IntradayAnalytics.trade_size_distribution
# -----------------------------------------------------------------------
dist = analytics.trade_size_distribution(trades)
required_keys = {"mean", "median", "std", "odd_lot_pct",
                 "institutional_pct", "large_block_pct", "total_volume", "n_trades"}
missing = required_keys - set(dist.keys())
if missing:
    errors.append(f"trade_size_distribution missing keys: {missing}")
if "mean" in dist and dist["mean"] <= 0:
    errors.append(f"trade_size_distribution mean should be > 0, got {dist['mean']}")

# -----------------------------------------------------------------------
# IntradayAnalytics.rogers_satchell_vol
# -----------------------------------------------------------------------
# Generate synthetic OHLC from trade prices
prices_arr = np.array([t.price for t in trades])
n_bars = 20
bar_size = len(prices_arr) // n_bars
opens_arr  = np.array([prices_arr[i*bar_size] for i in range(n_bars)])
closes_arr = np.array([prices_arr[min((i+1)*bar_size-1, len(prices_arr)-1)] for i in range(n_bars)])
highs_arr  = np.array([prices_arr[i*bar_size:(i+1)*bar_size].max() for i in range(n_bars)])
lows_arr   = np.array([prices_arr[i*bar_size:(i+1)*bar_size].min() for i in range(n_bars)])

rs_vol = analytics.rogers_satchell_vol(highs_arr, lows_arr, opens_arr, closes_arr)
if rs_vol <= 0:
    errors.append(f"rogers_satchell_vol should be > 0, got {rs_vol}")
if not math.isfinite(rs_vol):
    errors.append(f"rogers_satchell_vol is not finite: {rs_vol}")

# Also test module-level convenience function
rs_vol2 = rogers_satchell_vol(highs_arr, lows_arr, opens_arr, closes_arr)
if not math.isclose(rs_vol2, rs_vol, rel_tol=1e-9):
    errors.append(f"rogers_satchell_vol convenience fn mismatch: {rs_vol2} vs {rs_vol}")

# -----------------------------------------------------------------------
# NBBO from 2 venues (NYSE / NASDAQ)
# -----------------------------------------------------------------------
# Build explicit quotes for two venues so we can assert NBBO properties
nyse_q = Quote(timestamp=1000.0, bid=99.98, ask=100.03, bid_size=500, ask_size=300, venue="NYSE")
nasdaq_q = Quote(timestamp=1000.0, bid=99.97, ask=100.02, bid_size=400, ask_size=200, venue="NASDAQ")

nyse_spread = nyse_q.spread    # 0.05
nasdaq_spread = nasdaq_q.spread  # 0.05

# NBBO: bid = max(99.98, 99.97) = 99.98, ask = min(100.03, 100.02) = 100.02
calc = NBBOCalculator()
nbbos = calc.from_quotes({
    "NYSE": [nyse_q],
    "NASDAQ": [nasdaq_q],
})
if len(nbbos) == 0:
    errors.append("NBBOCalculator.from_quotes returned empty list")
else:
    nbbo = nbbos[-1]
    if not math.isclose(nbbo.bid, 99.98, abs_tol=1e-9):
        errors.append(f"NBBO bid: expected 99.98, got {nbbo.bid}")
    if not math.isclose(nbbo.ask, 100.02, abs_tol=1e-9):
        errors.append(f"NBBO ask: expected 100.02, got {nbbo.ask}")
    nbbo_sp = nbbo.spread
    if nbbo_sp > nyse_spread + 1e-9:
        errors.append(f"NBBO spread {nbbo_sp} > NYSE spread {nyse_spread}")
    if nbbo_sp > nasdaq_spread + 1e-9:
        errors.append(f"NBBO spread {nbbo_sp} > NASDAQ spread {nasdaq_spread}")

# nbbo_spread convenience function
instant_spread = nbbo_spread({"NYSE": nyse_q, "NASDAQ": nasdaq_q})
if not math.isclose(instant_spread, 100.02 - 99.98, abs_tol=1e-9):
    errors.append(f"nbbo_spread: expected {100.02 - 99.98}, got {instant_spread}")

# -----------------------------------------------------------------------
# effective_spread_bps: trade at ask vs midpoint → positive bps
# -----------------------------------------------------------------------
# midpoint = (99.98 + 100.02) / 2 = 100.00
# trade at ask (100.02), direction = +1 (buy)
eff_bps = effective_spread_bps(trade_price=100.02, midpoint=100.00, direction=1)
if eff_bps <= 0:
    errors.append(f"effective_spread_bps for trade at ask should be > 0, got {eff_bps}")
expected_eff = 2.0 * 1 * (100.02 - 100.00) / 100.00 * 10_000
if not math.isclose(eff_bps, expected_eff, rel_tol=1e-9):
    errors.append(f"effective_spread_bps: expected {expected_eff:.4f}, got {eff_bps:.4f}")

# -----------------------------------------------------------------------
# volume_weighted_price: assert between min and max trade prices
# -----------------------------------------------------------------------
vwap_val = volume_weighted_price(trades)
min_price = min(t.price for t in trades)
max_price = max(t.price for t in trades)
if not (min_price <= vwap_val <= max_price):
    errors.append(
        f"VWAP {vwap_val:.4f} not in [{min_price:.4f}, {max_price:.4f}]"
    )

# -----------------------------------------------------------------------
# Lee-Ready classify on simple cases
# -----------------------------------------------------------------------
mid_q = Quote(timestamp=100.0, bid=99.95, ask=100.05, bid_size=100, ask_size=100, venue="NYSE")
mid_price = mid_q.midpoint  # 100.0

buy_trade  = Trade(timestamp=100.1, price=100.03, size=100)   # above mid → buy
sell_trade = Trade(timestamp=100.1, price=99.97,  size=100)   # below mid → sell

buy_dir  = classifier.classify_lee_ready(buy_trade, mid_q)
sell_dir = classifier.classify_lee_ready(sell_trade, mid_q)
if buy_dir != 1:
    errors.append(f"Lee-Ready buy classification: expected +1, got {buy_dir}")
if sell_dir != -1:
    errors.append(f"Lee-Ready sell classification: expected -1, got {sell_dir}")

# -----------------------------------------------------------------------
# Full session
# -----------------------------------------------------------------------
session = sim.generate_taq_session(n_trades=100)
if "trades" not in session or "quotes" not in session or "nbbo" not in session:
    errors.append(f"generate_taq_session missing keys: {session.keys()}")
else:
    if len(session["trades"]) != 100:
        errors.append(f"session trades count: expected 100, got {len(session['trades'])}")
    if len(session["quotes"]) == 0:
        errors.append("session quotes is empty")

# -----------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------
if errors:
    print("FAIL dim_012:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print("[PASS] dim_012: Tick-level trade data (TAQ) analytics")
PYEOF
