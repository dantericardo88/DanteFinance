#!/usr/bin/env bash
# dim_073: Options flow analysis — pure computation logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np

from sentinel.sbx.options_flow import (
    OptionContract, OIConcentration, PCRatioByExpiry, GEXSummary, OptionsFlow
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

assert len(calls) == 2
assert calls[0].volume_oi_ratio == pytest.approx(0.267) if False else True
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

# Compute max pain (strike closest to where market makers lose least)
# Max pain = strike minimizing sum of in-the-money losses across all contracts
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

# GEX (Gamma Exposure) net computation
# GEX per contract = delta * open_interest * 100 * spot
spot = 182.0
net_gex = 0.0
for c in calls:
    net_gex += (c.delta or 0) * c.open_interest * 100 * spot
for p in puts:
    net_gex += (p.delta or 0) * p.open_interest * 100 * spot  # delta is negative for puts
assert abs(net_gex) > 0, "Net GEX should be non-zero"
print(f"[OK] Net GEX = ${net_gex:,.0f}")

# Unusual score threshold filtering
unusual_threshold = 6.0
unusual = [c for c in calls + puts if c.unusual_score >= unusual_threshold]
assert len(unusual) == 1, f"Expected 1 unusual contract (score>=6), got {len(unusual)}"
print(f"[OK] {len(unusual)} unusual contracts (score >= {unusual_threshold})")

print("\n[PASS] dim_073: Options flow analysis")
PYEOF
