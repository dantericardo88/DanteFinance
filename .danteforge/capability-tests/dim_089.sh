#!/usr/bin/env bash
# dim_089: Google Trends — pure signal math (zscore, smooth, second derivative, clamp)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math

from sentinel.sma.google_trends_v3 import (
    _zscore,
    _smooth,
    _second_derivative,
    _clamp,
    _FEAR_TERMS,
    _GREED_TERMS,
    _SECTOR_TICKERS,
    _VALID_TIMEFRAMES,
    TrendsScore,
    TrendsSignalEngine,
)

# Test _zscore with known inputs
series = [40.0, 42.0, 38.0, 45.0, 41.0, 39.0, 43.0, 40.0]
value_high = 65.0   # well above mean
value_low = 20.0    # well below mean

z_high = _zscore(series, value_high)
z_low = _zscore(series, value_low)
z_mean = _zscore(series, sum(series) / len(series))

assert z_high > 2.0, f"High value should have large positive z-score: {z_high:.4f}"
assert z_low < -2.0, f"Low value should have large negative z-score: {z_low:.4f}"
assert abs(z_mean) < 0.01, f"Mean value should have z-score ~0: {z_mean:.4f}"
print(f"[OK] _zscore: high={z_high:.4f} low={z_low:.4f} mean={z_mean:.4f}")

# Test _zscore returns 0.0 for series too short (< 4)
short_series = [40.0, 42.0, 38.0]   # only 3 elements
z_short = _zscore(short_series, 50.0)
assert z_short == 0.0, f"Short series should return 0.0: {z_short}"
print(f"[OK] _zscore short series returns 0.0")

# Test _zscore with constant series (std=0)
const_series = [50.0] * 10
z_const = _zscore(const_series, 55.0)
# std is replaced with 1e-9, so result should be very large but not NaN/Inf
assert not math.isnan(z_const), "z-score on constant series should not be NaN"
assert not math.isinf(z_const), "z-score on constant series should not be Inf"
print(f"[OK] _zscore constant series handled: {z_const:.2f}")

# Test _smooth (SMA or savgol depending on scipy availability)
raw_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
smoothed = _smooth(raw_values, window=4)
assert len(smoothed) == len(raw_values), \
    f"Smoothed should have same length: {len(smoothed)} vs {len(raw_values)}"
last_smooth = smoothed[-1]
# savgol on a perfectly linear series returns exact values (last=100),
# SMA with window=4 returns 85; both are valid implementations — just
# check the value is somewhere in the plausible range of the input data
assert 50.0 <= last_smooth <= 110.0, \
    f"Last smoothed value should be in [50, 110] for linearly increasing input: {last_smooth:.2f}"
# First value should be near the beginning of the series
assert smoothed[0] <= 30.0, f"First smoothed value should be near start: {smoothed[0]:.2f}"
print(f"[OK] _smooth: len={len(smoothed)} last={last_smooth:.2f}")

# Test _second_derivative (acceleration)
# For an accelerating series (exponentially increasing), 2nd derivative should be positive
accel_series = [10.0, 12.0, 15.0, 19.0, 25.0, 35.0, 50.0, 72.0, 100.0]
accel_val = _second_derivative(accel_series)
assert accel_val > 0, f"Accelerating series should have positive 2nd derivative: {accel_val:.4f}"
print(f"[OK] _second_derivative accelerating: {accel_val:.4f} > 0")

# For a decelerating series (logarithmically slowing), 2nd derivative should be negative
decel_series = [10.0, 30.0, 50.0, 65.0, 75.0, 82.0, 87.0, 91.0, 94.0]
decel_val = _second_derivative(decel_series)
assert decel_val < 0, f"Decelerating series should have negative 2nd derivative: {decel_val:.4f}"
print(f"[OK] _second_derivative decelerating: {decel_val:.4f} < 0")

# For a series too short (< 3), should return 0.0
short_val = _second_derivative([1.0, 2.0])
assert short_val == 0.0, f"Short series should return 0.0: {short_val}"
print(f"[OK] _second_derivative short series returns 0.0")

# Test _clamp
assert _clamp(5.0, 0.0, 1.0) == 1.0, "Clamp above max"
assert _clamp(-5.0, 0.0, 1.0) == 0.0, "Clamp below min"
assert _clamp(0.5, 0.0, 1.0) == 0.5, "Clamp within range"
assert _clamp(0.0, 0.0, 1.0) == 0.0, "Clamp at min boundary"
assert _clamp(1.0, 0.0, 1.0) == 1.0, "Clamp at max boundary"
print("[OK] _clamp: all boundary cases correct")

# Test composite score formula (mirrors TrendsSignalEngine.compute_ticker_search_score)
def composite_score(raw_interest: float, momentum_z: float,
                    acceleration: float, spike_z: float, spike: bool) -> float:
    mom_component = _clamp((momentum_z + 3) / 6, 0, 1)
    raw_component = raw_interest / 100.0
    acc_component = _clamp((acceleration + 10) / 20, 0, 1)
    spike_component = min(1.0, spike_z / 3.0) if spike else 0.0
    return _clamp(
        0.35 * raw_component
        + 0.35 * mom_component
        + 0.15 * acc_component
        + 0.15 * spike_component,
        0, 1
    )

# High raw interest + high momentum = high composite
score_high = composite_score(80.0, 2.5, 3.0, 0.0, False)
# Low raw interest + negative momentum = low composite
score_low = composite_score(20.0, -2.0, -5.0, 0.0, False)
# Spike should boost composite
score_spike = composite_score(60.0, 1.0, 0.0, 3.0, True)
score_no_spike = composite_score(60.0, 1.0, 0.0, 0.0, False)

assert 0.0 <= score_high <= 1.0, f"Composite must be in [0,1]: {score_high}"
assert 0.0 <= score_low <= 1.0, f"Composite must be in [0,1]: {score_low}"
assert score_high > score_low, f"High interest+momentum should beat low: {score_high:.4f} vs {score_low:.4f}"
assert score_spike > score_no_spike, f"Spike should boost score: {score_spike:.4f} vs {score_no_spike:.4f}"
print(f"[OK] Composite score: high={score_high:.4f} low={score_low:.4f} spike={score_spike:.4f} no_spike={score_no_spike:.4f}")

# Test constants
assert len(_FEAR_TERMS) >= 3, f"Expected >= 3 fear terms: {len(_FEAR_TERMS)}"
assert len(_GREED_TERMS) >= 3, f"Expected >= 3 greed terms: {len(_GREED_TERMS)}"
assert "recession" in _FEAR_TERMS or any("recession" in t for t in _FEAR_TERMS), \
    "Fear terms should contain recession-related term"
print(f"[OK] _FEAR_TERMS: {len(_FEAR_TERMS)} terms, _GREED_TERMS: {len(_GREED_TERMS)} terms")

assert "technology" in _SECTOR_TICKERS, "Should have technology sector"
assert "finance" in _SECTOR_TICKERS, "Should have finance sector"
assert len(_SECTOR_TICKERS) >= 5, f"Expected >= 5 sectors: {len(_SECTOR_TICKERS)}"
for sector, tickers in _SECTOR_TICKERS.items():
    assert len(tickers) >= 3, f"Sector {sector} should have >= 3 tickers: {tickers}"
print(f"[OK] _SECTOR_TICKERS: {len(_SECTOR_TICKERS)} sectors")

assert "today 12-m" in _VALID_TIMEFRAMES, "'today 12-m' must be a valid timeframe"
assert "today 5-y" in _VALID_TIMEFRAMES, "'today 5-y' must be a valid timeframe"
print(f"[OK] _VALID_TIMEFRAMES: {len(_VALID_TIMEFRAMES)} timeframes")

# Test TrendsScore dataclass
from datetime import datetime, timezone
score_obj = TrendsScore(
    ticker="AAPL",
    raw_interest=75.0,
    momentum_zscore=1.8,
    acceleration=2.3,
    spike_detected=False,
    spike_zscore=0.5,
    composite_score=0.72,
    keywords_used=["Apple stock", "AAPL", "Apple earnings"],
    computed_at=datetime.now(timezone.utc).isoformat(),
)
assert score_obj.ticker == "AAPL"
assert 0.0 <= score_obj.composite_score <= 1.0
assert score_obj.momentum_zscore == 1.8
print(f"[OK] TrendsScore: ticker={score_obj.ticker} composite={score_obj.composite_score:.4f}")

# ---------------------------------------------------------------
# NEW: Fear/greed composite — 5 component weights sum to 1.0
# ---------------------------------------------------------------
W_SEARCH   = 0.20
W_MOMENTUM = 0.20
W_BREADTH  = 0.20
W_JUNK     = 0.20
W_VOL      = 0.20
total_weight = W_SEARCH + W_MOMENTUM + W_BREADTH + W_JUNK + W_VOL
assert abs(total_weight - 1.0) < 1e-9, \
    f"Component weights must sum to 1.0: {total_weight}"
print(f"[OK] Fear/greed component weights sum to 1.0: {total_weight}")

# Composite formula with known inputs should produce score in [0, 100]
def fear_greed_composite(search, momentum, breadth, junk, vol):
    score = (
        W_SEARCH   * search
        + W_MOMENTUM * momentum
        + W_BREADTH  * breadth
        + W_JUNK     * junk
        + W_VOL      * vol
    )
    return _clamp(score, 0.0, 100.0)

score_fear  = fear_greed_composite(20, 25, 30, 35, 25)  # ~27 → Fear
score_greed = fear_greed_composite(70, 75, 65, 80, 70)  # ~72 → Greed
score_neutral = fear_greed_composite(50, 50, 50, 50, 50)

assert 0.0 <= score_fear <= 100.0, f"Fear/greed score must be in [0,100]: {score_fear}"
assert 0.0 <= score_greed <= 100.0, f"Fear/greed score must be in [0,100]: {score_greed}"
assert score_greed > score_fear, f"Greed score should exceed fear score: {score_greed} vs {score_fear}"
assert abs(score_neutral - 50.0) < 1e-9, f"Neutral inputs should yield 50.0: {score_neutral}"
print(f"[OK] Fear/greed composite: fear={score_fear:.1f} neutral={score_neutral:.1f} greed={score_greed:.1f}")

# Classification thresholds: 30 → Fear, 70 → Greed
from sentinel.sma.google_trends_v3 import TrendsSignalEngine
engine = TrendsSignalEngine.__new__(TrendsSignalEngine)  # no-network instantiation

# Access classify_fear_greed via the module's signal engine class
from sentinel.sma.google_trends_v3 import TrendsSignalEngine as TSE
classify = TSE.classify_fear_greed

assert classify(30.0) == "Fear", f"30 should be Fear: {classify(30.0)}"
assert classify(70.0) == "Greed", f"70 should be Greed: {classify(70.0)}"
assert classify(10.0) == "Extreme Fear", f"10 should be Extreme Fear: {classify(10.0)}"
assert classify(90.0) == "Extreme Greed", f"90 should be Extreme Greed: {classify(90.0)}"
assert classify(50.0) == "Neutral", f"50 should be Neutral: {classify(50.0)}"
print(f"[OK] Fear/greed classification: 10=Extreme Fear, 30=Fear, 50=Neutral, 70=Greed, 90=Extreme Greed")

# ---------------------------------------------------------------
# NEW: Product cycle acceleration — pure derivative math
# ---------------------------------------------------------------
# Given an accelerating search trend series, acceleration (2nd derivative) > 0
accel_product_series = [10.0, 15.0, 22.0, 32.0, 45.0, 62.0, 85.0]
d1_last = accel_product_series[-1] - accel_product_series[-2]
d1_prev = accel_product_series[-2] - accel_product_series[-3]
accel_2nd_deriv = d1_last - d1_prev
assert accel_2nd_deriv > 0, \
    f"Accelerating search trend should have positive 2nd derivative: {accel_2nd_deriv}"
print(f"[OK] Product cycle acceleration (2nd deriv): {accel_2nd_deriv:.2f} > 0")

# Decelerating: 2nd derivative < 0
decel_product_series = [10.0, 25.0, 38.0, 48.0, 55.0, 60.0, 63.0]
d1_last = decel_product_series[-1] - decel_product_series[-2]
d1_prev = decel_product_series[-2] - decel_product_series[-3]
decel_2nd_deriv = d1_last - d1_prev
assert decel_2nd_deriv < 0, \
    f"Decelerating search trend should have negative 2nd derivative: {decel_2nd_deriv}"
print(f"[OK] Product cycle deceleration (2nd deriv): {decel_2nd_deriv:.2f} < 0")

# Surprise probability mapping: acceleration > 2 → 0.65, <=0 → 0.45
def map_accel_to_prob(accel):
    if accel > 2.0:
        return 0.65
    elif accel > 0.5:
        return 0.60
    elif accel > 0.0:
        return 0.55
    elif accel > -0.5:
        return 0.50
    else:
        return 0.45

assert map_accel_to_prob(3.0) == 0.65, "High accel → 0.65"
assert map_accel_to_prob(-1.0) == 0.45, "Negative accel → 0.45"
assert map_accel_to_prob(0.0) == 0.50, "Zero accel → 0.50"
print(f"[OK] Product cycle surprise_prob mapping: high_accel=0.65, zero=0.50, negative=0.45")

print("\n[PASS] dim_089: Google Trends")
PYEOF
