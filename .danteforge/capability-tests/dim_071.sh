#!/usr/bin/env bash
# dim_071: Technical screener — pure TA indicator logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.sbx.technical_screener import TechnicalScreener

screener = TechnicalScreener()

# Build synthetic price series (uptrend)
np.random.seed(42)
n = 60
prices = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
closes = pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="B"))

# Test compute_sma
sma20 = screener.compute_sma(closes, 20)
assert 90 < sma20 < 130, f"SMA20 out of range: {sma20}"
print(f"[OK] compute_sma(20) = {sma20:.2f}")

# Test compute_ema
ema12 = screener.compute_ema(closes, 12)
assert 90 < ema12 < 130, f"EMA12 out of range: {ema12}"
print(f"[OK] compute_ema(12) = {ema12:.2f}")

# Test compute_rsi
rsi = screener.compute_rsi(closes, 14)
assert 0 <= rsi <= 100, f"RSI out of [0,100]: {rsi}"
print(f"[OK] compute_rsi(14) = {rsi:.2f}")

# Test compute_macd
macd_line, signal_line, histogram = screener.compute_macd(closes)
assert abs(macd_line) < 20, f"MACD out of range: {macd_line}"
assert abs(histogram) < 10, f"MACD histogram out of range: {histogram}"
print(f"[OK] compute_macd: line={macd_line:.4f} signal={signal_line:.4f} hist={histogram:.4f}")

# Test compute_bollinger_bands
upper, middle, lower = screener.compute_bollinger_bands(closes, 20)
assert upper > middle > lower, "Bollinger bands must be ordered: upper > middle > lower"
width = (upper - lower) / middle
assert 0 < width < 0.5, f"Bollinger width unexpected: {width}"
print(f"[OK] compute_bollinger_bands: upper={upper:.2f} mid={middle:.2f} lower={lower:.2f}")

# Test ATR
highs = closes * 1.005
lows  = closes * 0.995
atr = screener.compute_atr(highs, lows, closes, 14)
assert atr > 0, f"ATR should be positive: {atr}"
print(f"[OK] compute_atr(14) = {atr:.4f}")

# RSI range checks with realistic mixed-direction price
# Use the existing `closes` series which has both ups and downs
assert not np.isnan(rsi), f"RSI should not be NaN for mixed-direction series: {rsi}"
assert 0 <= rsi <= 100, f"RSI must be in [0,100]: {rsi}"

# Verify RSI bounds
oversold_series = closes.copy()
# Introduce a 30-day decline at end
oversold_series = pd.concat([closes, pd.Series(
    closes.iloc[-1] * np.cumprod(1 + np.full(30, -0.01)),
    index=pd.date_range(closes.index[-1] + pd.Timedelta(days=1), periods=30, freq="B")
)])
rsi_os = screener.compute_rsi(oversold_series, 14)
assert 0 <= rsi_os <= 100, f"RSI boundary violated: {rsi_os}"
print(f"[OK] RSI is in valid range [0,100]: {rsi:.2f} / declined period: {rsi_os:.2f}")

print("\n[PASS] dim_071: Technical screener")
PYEOF
