#!/bin/bash
# dim_006: FX spot/forwards/vol surface — FXOptionsAnalytics (Garman-Kohlhagen), ForwardCurveBuilder
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

# Test 1: FX module constants
from sentinel.sfe.fx_surface_v3 import _G10, _TENORS, _FRED_FX_SERIES
assert "USD" in _G10 and "EUR" in _G10 and "JPY" in _G10
assert len(_G10) == 10, f"G10 should have 10 currencies, got {len(_G10)}"
assert "1M" in _TENORS and "1Y" in _TENORS
assert "EURUSD" in _FRED_FX_SERIES and "USDJPY" in _FRED_FX_SERIES
print(f"[OK] G10 set has 10 currencies, _TENORS has {len(_TENORS)} tenors, FRED FX series present")

# Test 2: FXOptionsAnalytics - Garman-Kohlhagen pricing (pure computation)
from sentinel.sfe.fx_surface_v3 import FXOptionsAnalytics
gk = FXOptionsAnalytics()

# EURUSD ATM call: spot=1.10, strike=1.10, T=0.25yr, vol=7%, r_dom=5%, r_for=4%
call = gk.price_vanilla(spot=1.10, strike=1.10, T=0.25, vol=0.07,
                         r_dom=0.05, r_for=0.04, call_put="call")
assert call.value > 0, f"Call value must be positive: {call.value}"
assert 0 < call.delta < 1, f"Call delta must be in (0,1): {call.delta}"
assert call.gamma > 0, f"Gamma must be positive: {call.gamma}"
assert call.vega > 0, f"Vega must be positive: {call.vega}"
print(f"[OK] GK call: value={call.value:.5f}, delta={call.delta:.4f}, vega={call.vega:.5f}")

# Test 3: Put option
put = gk.price_vanilla(spot=1.10, strike=1.10, T=0.25, vol=0.07,
                        r_dom=0.05, r_for=0.04, call_put="put")
assert put.value > 0, f"Put value must be positive: {put.value}"
assert -1 < put.delta < 0, f"Put delta must be in (-1,0): {put.delta}"
# Put-call parity for GK: C - P = S*e^(-r_f*T) - K*e^(-r_d*T)
parity_lhs = call.value - put.value
parity_rhs = 1.10 * math.exp(-0.04 * 0.25) - 1.10 * math.exp(-0.05 * 0.25)
assert abs(parity_lhs - parity_rhs) < 0.0001, \
    f"GK put-call parity failed: LHS={parity_lhs:.5f}, RHS={parity_rhs:.5f}"
print(f"[OK] GK put: value={put.value:.5f}, put-call parity verified")

# Test 4: ForwardCurveBuilder - CIP forward rates (pure computation)
from sentinel.sfe.fx_surface_v3 import ForwardCurveBuilder
# Instantiate without calling FRED (pass rates directly to build_forward_curve)
builder = ForwardCurveBuilder.__new__(ForwardCurveBuilder)

# Manually call build_forward_curve: EURUSD spot=1.10, EUR rate=4%, USD rate=5.25%
curve = builder.build_forward_curve(spot=1.10, base_rate=4.0, quote_rate=5.25,
                                     pair="EURUSD")
assert curve.pair == "EURUSD"
assert abs(curve.spot - 1.10) < 1e-6
# USD rates higher than EUR → forward EUR should trade at premium (USD at discount)
# F = S * exp((r_usd - r_eur) * T) → F > S when r_usd > r_eur
assert curve.tenors["1Y"] > 1.10, \
    f"EURUSD 1Y fwd should be > spot with USD rate > EUR rate: {curve.tenors['1Y']}"
assert "ON" in curve.tenors and "1M" in curve.tenors and "1Y" in curve.tenors
print(f"[OK] EURUSD fwd curve: spot={curve.spot}, 1M={curve.tenors['1M']:.5f}, 1Y={curve.tenors['1Y']:.5f}")

# Test 5: MAJOR_PAIRS constant from fx_adapter
from sentinel.sds.adapters.fx_adapter import MAJOR_PAIRS, FXQuote, FXBar
assert "EURUSD" in MAJOR_PAIRS
assert "USDJPY" in MAJOR_PAIRS
assert len(MAJOR_PAIRS) >= 8
# FXQuote model instantiation
from decimal import Decimal
from datetime import datetime, timezone
q = FXQuote(pair="EURUSD", base="EUR", quote="USD",
            rate=Decimal("1.0850"), timestamp=datetime.now(timezone.utc))
assert q.pair == "EURUSD"
assert q.source == "frankfurter"
print(f"[OK] MAJOR_PAIRS has {len(MAJOR_PAIRS)} pairs; FXQuote instantiates correctly")

print("[PASS]")
PYEOF
