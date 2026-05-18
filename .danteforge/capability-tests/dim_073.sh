#!/usr/bin/env bash
# dim_073: Options flow analysis — pure computation logic + UOA / put-call skew / block sweep
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np

from sentinel.sbx.options_flow import (
    OptionContract, OIConcentration, PCRatioByExpiry, GEXSummary, OptionsFlow,
    compute_unusual_options_activity, compute_put_call_skew, detect_block_sweep,
)

# Build synthetic OptionContract objects
calls = [
    OptionContract(
        symbol="AAPL", expiration="2024-03-15", strike=180.0,
        option_type="call", last_price=5.20, bid=5.10, ask=5.30,
        volume=12000, open_interest=45000, implied_volatility=0.28,
        in_the_money=True, volume_oi_ratio=0.267, dollar_premium=62400000,
        unusual_score=7.5, days_to_expiry=30, delta=0.62, flow_conviction=0.77,
    ),
    OptionContract(
        symbol="AAPL", expiration="2024-03-15", strike=185.0,
        option_type="call", last_price=2.80, bid=2.70, ask=2.90,
        volume=8500, open_interest=30000, implied_volatility=0.31,
        in_the_money=False, volume_oi_ratio=0.283, dollar_premium=23800000,
        unusual_score=5.2, days_to_expiry=30, delta=0.41, flow_conviction=0.51,
    ),
]
puts = [
    OptionContract(
        symbol="AAPL", expiration="2024-03-15", strike=175.0,
        option_type="put", last_price=4.10, bid=4.00, ask=4.20,
        volume=6000, open_interest=25000, implied_volatility=0.30,
        in_the_money=False, volume_oi_ratio=0.240, dollar_premium=24600000,
        unusual_score=4.0, days_to_expiry=30, delta=-0.38, flow_conviction=0.45,
    ),
]

print(f"[OK] OptionContract models: {len(calls)} calls, {len(puts)} puts")

# Compute put/call volume ratio (pure math)
total_call_vol = sum(c.volume for c in calls)
total_put_vol = sum(p.volume for p in puts)
pc_vol_ratio = total_put_vol / total_call_vol
assert 0 < pc_vol_ratio < 2.0, f"PC ratio out of range: {pc_vol_ratio}"
print(f"[OK] Put/Call volume ratio = {pc_vol_ratio:.3f}")

# Compute dollar premium weighted put/call ratio
total_call_prem = sum(c.dollar_premium for c in calls)
total_put_prem = sum(p.dollar_premium for p in puts)
pc_prem_ratio = total_put_prem / total_call_prem
print(f"[OK] Put/Call premium ratio = {pc_prem_ratio:.3f}")

# Compute max pain
strikes = [175.0, 180.0, 185.0]
def compute_max_pain(strike_prices, calls_list, puts_list):
    min_loss = float('inf')
    max_pain_strike = strikes[0]
    for s in strike_prices:
        call_loss = sum(max(0, s - c.strike) * c.open_interest for c in calls_list)
        put_loss  = sum(max(0, p.strike - s) * p.open_interest for p in puts_list)
        total_loss = call_loss + put_loss
        if total_loss < min_loss:
            min_loss = total_loss
            max_pain_strike = s
    return max_pain_strike, min_loss

mp_strike, mp_loss = compute_max_pain(strikes, calls, puts)
assert mp_strike in strikes, f"Max pain strike not in strikes list: {mp_strike}"
print(f"[OK] Max pain strike = ${mp_strike:.0f} (total loss = {mp_loss:,.0f})")

# GEX net computation
spot = 182.0
net_gex = 0.0
for c in calls:
    net_gex += (c.delta or 0) * c.open_interest * 100 * spot
for p in puts:
    net_gex += (p.delta or 0) * p.open_interest * 100 * spot
assert abs(net_gex) > 0, "Net GEX should be non-zero"
print(f"[OK] Net GEX = ${net_gex:,.0f}")

# Unusual score threshold filtering
unusual_threshold = 6.0
unusual = [c for c in calls + puts if c.unusual_score >= unusual_threshold]
assert len(unusual) == 1, f"Expected 1 unusual contract (score>=6), got {len(unusual)}"
print(f"[OK] {len(unusual)} unusual contracts (score >= {unusual_threshold})")

# ── NEW: compute_unusual_options_activity ────────────────────────────────────
all_contracts = calls + puts
uoa_results = compute_unusual_options_activity(all_contracts)
assert len(uoa_results) == len(all_contracts), "UOA result count mismatch"

# Verify UOA score formula: score = volume / open_interest
for item in uoa_results:
    c = item["contract"]
    expected_score = c.volume / c.open_interest
    assert abs(item["uoa_score"] - expected_score) < 1e-4, (
        f"UOA score formula mismatch: expected {expected_score:.6f}, got {item['uoa_score']}"
    )
print(f"[OK] UOA score formula verified: volume/OI for all {len(uoa_results)} contracts")

# Verify flagging: pass a known average_vol_oi_ratio so threshold is predictable
# Contract with ratio=50 vs average=1.0 → 50 > 5×1.0 = 5 → should flag
high_vol_contract = OptionContract(
    symbol="AAPL", expiration="2024-03-15", strike=190.0,
    option_type="call", last_price=1.0, bid=0.90, ask=1.10,
    volume=500000, open_interest=10000, implied_volatility=0.45,
    in_the_money=False, volume_oi_ratio=50.0, dollar_premium=50_000_000,
    unusual_score=9.5, days_to_expiry=30, delta=0.15,
)
uoa2 = compute_unusual_options_activity([high_vol_contract], average_vol_oi_ratio=1.0, threshold_multiple=5.0)
# vol/OI = 500000/10000 = 50; threshold = 1.0 × 5 = 5 → 50 > 5 → flagged
flagged = [r for r in uoa2 if r["is_unusual"]]
assert len(flagged) == 1, f"Expected 1 flagged UOA contract with ratio=50 vs avg=1.0, got {len(flagged)}"
assert flagged[0]["uoa_score"] == 50.0, f"Expected score=50.0, got {flagged[0]['uoa_score']}"
print(f"[OK] UOA flag: ratio=50 > 5× avg=1.0 => is_unusual=True, multiple={flagged[0]['multiple_of_avg']:.1f}x")

# ── NEW: compute_put_call_skew ────────────────────────────────────────────────
skew = compute_put_call_skew(all_contracts, target_delta=0.25)
assert skew is not None, "put_call_skew should return a value"
# put_25 IV (0.30) - call_25 IV (0.28 or 0.31) → skew = small positive or negative
print(f"[OK] compute_put_call_skew: skew={skew:.4f} (positive=fear, negative=greed)")

# ── NEW: detect_block_sweep ───────────────────────────────────────────────────
# Build 3 contracts at same expiry/strike to trigger sweep detection
sweep_contracts = [
    OptionContract(
        symbol="SPY", expiration="2024-04-19", strike=500.0,
        option_type="call", last_price=3.50, bid=3.40, ask=3.60,
        volume=5000, open_interest=20000, implied_volatility=0.18,
        in_the_money=False, volume_oi_ratio=0.25, dollar_premium=17_500_000,
        unusual_score=6.0, days_to_expiry=45, delta=0.40,
    ),
    OptionContract(
        symbol="SPY", expiration="2024-04-19", strike=500.0,
        option_type="call", last_price=3.55, bid=3.40, ask=3.60,
        volume=4000, open_interest=20000, implied_volatility=0.18,
        in_the_money=False, volume_oi_ratio=0.20, dollar_premium=14_200_000,
        unusual_score=5.5, days_to_expiry=45, delta=0.40,
    ),
    OptionContract(
        symbol="SPY", expiration="2024-04-19", strike=500.0,
        option_type="call", last_price=3.50, bid=3.40, ask=3.60,
        volume=6000, open_interest=20000, implied_volatility=0.18,
        in_the_money=False, volume_oi_ratio=0.30, dollar_premium=21_000_000,
        unusual_score=7.0, days_to_expiry=45, delta=0.40,
    ),
]
sweeps = detect_block_sweep(sweep_contracts, min_legs=3)
assert len(sweeps) >= 1, "Expected at least 1 block sweep detected"
sweep = sweeps[0]
assert sweep["n_legs"] == 3, f"Expected 3 legs, got {sweep['n_legs']}"
assert sweep["expiration"] == "2024-04-19"
assert sweep["strike"] == 500.0
assert sweep["option_type"] == "call"
assert sweep["coordinated_buying"] is True, "All-call at-ask sweep should flag coordinated_buying"
print(f"[OK] detect_block_sweep: {sweep['n_legs']} legs at {sweep['strike']} {sweep['expiration']} coordinated={sweep['coordinated_buying']}")

print("\n[PASS] dim_073: Options flow analysis")
PYEOF
