#!/bin/bash
# dim_006: FX spot/forwards/vol surface — FXOptionsAnalytics (Garman-Kohlhagen), ForwardCurveBuilder,
#          Garman-Klass vol estimator, implied forward rates, carry signal
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
# USD rates higher than EUR -> forward EUR should trade at premium (USD at discount)
# F = S * exp((r_usd - r_eur) * T) -> F > S when r_usd > r_eur
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

# Test 6: Implied forward rate — pure math
# F = S * exp((r_d - r_f) * T)
# EURUSD spot=1.10, USD=5.25%, EUR=4.0%, T=1Y
# F = 1.10 * exp((0.0525 - 0.04) * 1.0) = 1.10 * exp(0.0125) ≈ 1.1138
from sentinel.sfe.fx_surface_v3 import compute_implied_forward, build_implied_forward_curve
import math as _math
spot = 1.10
r_dom = 0.0525   # USD
r_for = 0.04     # EUR
T = 1.0
expected_fwd = spot * _math.exp((r_dom - r_for) * T)
fwd = compute_implied_forward(spot, r_dom, r_for, T)
assert abs(fwd - expected_fwd) < 1e-9, f"Implied forward mismatch: {fwd:.6f} vs {expected_fwd:.6f}"
assert fwd > spot, f"USD rate > EUR rate -> 1Y forward should exceed spot: {fwd:.6f}"
print(f"[OK] Implied 1Y EURUSD forward: spot={spot}, r_usd={r_dom*100:.2f}%, r_eur={r_for*100:.2f}% -> fwd={fwd:.6f}")

# Build full forward curve using rate fallbacks
curve2 = build_implied_forward_curve(spot=1.10, pair="EURUSD")
assert "1Y" in curve2 and "1M" in curve2 and "ON" in curve2
assert curve2["1Y"] > curve2["1M"] > spot, \
    f"Forward curve should be upward-sloping: ON={curve2['ON']:.5f}, 1M={curve2['1M']:.5f}, 1Y={curve2['1Y']:.5f}"
print(f"[OK] Full implied forward curve: 1M={curve2['1M']:.5f}, 3M={curve2['3M']:.5f}, 1Y={curve2['1Y']:.5f}")

# Test 7: Garman-Klass vol estimator — pure math
# Synthetic OHLCV where we know the vol: constant bar with H=102, L=98, C=100, O=100
# GK = sqrt(0.5*(ln(102/98))^2 - (2*ln2-1)*(ln(100/100))^2) = sqrt(0.5*(ln(102/98))^2)
import numpy as np
from sentinel.sfe.fx_surface_v3 import garman_klass_vol, _GK_CONST

H = np.array([102.0] * 10)
L = np.array([98.0] * 10)
C = np.array([100.0] * 10)
O = np.array([100.0] * 10)

gk_vol = garman_klass_vol(H, L, C, O, annualize=False)  # per-period
expected_per_period = _math.sqrt(0.5 * (_math.log(102.0/98.0))**2 - _GK_CONST * 0.0)
assert abs(gk_vol - expected_per_period) < 1e-6, \
    f"GK vol mismatch: {gk_vol:.8f} vs {expected_per_period:.8f}"
assert gk_vol > 0, "GK vol must be positive"

gk_vol_ann = garman_klass_vol(H, L, C, O, annualize=True)
assert abs(gk_vol_ann - gk_vol * _math.sqrt(252)) < 1e-6, "Annualized GK vol should be gk_vol * sqrt(252)"
print(f"[OK] Garman-Klass vol: per-period={gk_vol:.6f}, annualized={gk_vol_ann:.4f}")

# Verify GK vol > 0 for realistic price bars (H > L)
H2 = np.array([1.1050, 1.1080, 1.1020, 1.1060, 1.1040])
L2 = np.array([1.0990, 1.1010, 1.0970, 1.1000, 1.0985])
C2 = np.array([1.1020, 1.1050, 1.0990, 1.1030, 1.1010])
O2 = np.array([1.1010, 1.1020, 1.1040, 1.1010, 1.1020])
gk_vol2 = garman_klass_vol(H2, L2, C2, O2)
assert gk_vol2 > 0, f"GK vol should be positive for realistic bars: {gk_vol2}"
print(f"[OK] Garman-Klass vol on EURUSD-like bars: annualized={gk_vol2:.2%}")

# Test 8: FXCarrySignal — carry trade ranking (pure math, no network)
from sentinel.sfe.fx_surface_v3 import FXCarrySignal, _G10_RATE_FALLBACK

signal = FXCarrySignal()
ladder = signal.rank_g10_carry()

# Validate structure
assert "currency" in ladder.columns and "diff_vs_usd" in ladder.columns
assert "carry_signal" in ladder.columns and "rank" in ladder.columns
assert len(ladder) == 9, f"Should have 9 G10 currencies (excl. USD): {len(ladder)}"

# JPY should have the most negative carry vs USD (lowest rate in G10)
jpy_row = ladder[ladder["currency"] == "JPY"].iloc[0]
assert jpy_row["diff_vs_usd"] < 0, f"JPY should have negative carry vs USD: {jpy_row['diff_vs_usd']}"
assert jpy_row["carry_signal"] == "SHORT", f"JPY should be SHORT signal"

# AUD should have positive carry vs USD (AUD rate > USD? — depends on current rates)
# With fallback: AUD=4.35, USD=5.25 -> AUD < USD -> actually negative vs USD
# But NZD=5.50 > USD=5.25 -> NZD is top carry
nzd_row = ladder[ladder["currency"] == "NZD"].iloc[0]
assert nzd_row["diff_vs_usd"] > 0, f"NZD (5.50%) should have positive carry vs USD (5.25%): {nzd_row['diff_vs_usd']}"
assert nzd_row["carry_signal"] == "LONG"
assert nzd_row["rank"] == 1, f"NZD should rank #1 highest carry: rank={nzd_row['rank']}"

# JPY should rank last (lowest carry)
jpy_rank = int(jpy_row["rank"])
assert jpy_rank == len(ladder), f"JPY should be ranked last (#{len(ladder)}): got rank={jpy_rank}"

print(f"[OK] G10 carry ladder: {len(ladder)} currencies")
print(f"  Top carry: {ladder.iloc[0]['currency']} (+{ladder.iloc[0]['diff_vs_usd']:.2f}% vs USD)")
print(f"  Low carry: {ladder.iloc[-1]['currency']} ({ladder.iloc[-1]['diff_vs_usd']:.2f}% vs USD)")

# Verify JPY negative, NZD positive
print(f"  JPY carry vs USD: {jpy_row['diff_vs_usd']:.2f}% -> {jpy_row['carry_signal']}")
print(f"  NZD carry vs USD: {nzd_row['diff_vs_usd']:.2f}% -> {nzd_row['carry_signal']}")

# Test 9: Carry-implied forward via FXCarrySignal (pure math)
# USDJPY: USD=5.25%, JPY=0.10%, spot=150.0, T=1Y
# F = 150.0 * exp((r_JPY - r_USD) * 1.0) = 150.0 * exp(-0.0515) ≈ 142.28
# But USDJPY: base=USD, quote=JPY -> F = spot * exp((r_quote - r_base)*T)
# = 150.0 * exp((0.001 - 0.0525)*1.0) = 150.0 * exp(-0.0515)
spot_usdjpy = 150.0
fwd_usdjpy = signal.compute_implied_carry_forward(spot_usdjpy, "USDJPY", "1Y")
expected_usdjpy = spot_usdjpy * _math.exp((_G10_RATE_FALLBACK["JPY"]/100 - _G10_RATE_FALLBACK["USD"]/100) * 1.0)
assert abs(fwd_usdjpy - expected_usdjpy) < 0.001, \
    f"USDJPY 1Y fwd mismatch: {fwd_usdjpy:.4f} vs {expected_usdjpy:.4f}"
# USDJPY 1Y forward should be BELOW spot (USD earns more than JPY)
assert fwd_usdjpy < spot_usdjpy, \
    f"USDJPY 1Y fwd should be < spot (JPY at premium, USD at discount): {fwd_usdjpy:.4f} < {spot_usdjpy}"
print(f"[OK] USDJPY carry-implied 1Y forward: spot={spot_usdjpy}, fwd={fwd_usdjpy:.4f} (JPY at fwd premium)")

print("\n[PASS]")
PYEOF
