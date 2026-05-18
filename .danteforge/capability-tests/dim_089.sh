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

print("\n[PASS] dim_089: Google Trends")
PYEOF
