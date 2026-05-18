#!/bin/bash
# dim_004: Options chain — Black-Scholes Greeks, max pain computation
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

from sentinel.sbx.options_analytics import bs_greeks, _norm_cdf, _norm_pdf

# Test 1: _norm_cdf sanity checks
assert abs(_norm_cdf(0.0) - 0.5) < 1e-10, "CDF(0) should be 0.5"
assert _norm_cdf(10.0) > 0.9999, "CDF(10) should be ~1"
assert _norm_cdf(-10.0) < 0.0001, "CDF(-10) should be ~0"
print("[OK] _norm_cdf produces correct values at 0, +10, -10")

# Test 2: _norm_pdf sanity checks
assert abs(_norm_pdf(0.0) - 1.0 / math.sqrt(2 * math.pi)) < 1e-10, "PDF(0) mismatch"
assert _norm_pdf(10.0) < 1e-10, "PDF(10) should be ~0"
print("[OK] _norm_pdf produces correct values")

# Test 3: BS call greeks — ATM
g_call = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.20, "call")
assert g_call["delta"] is not None and 0 < g_call["delta"] < 1, \
    f"Call delta must be in (0,1): {g_call['delta']}"
assert g_call["gamma"] is not None and g_call["gamma"] > 0, \
    f"Gamma must be positive: {g_call['gamma']}"
assert g_call["vega"] is not None and g_call["vega"] > 0, \
    f"Vega must be positive: {g_call['vega']}"
assert g_call["theta"] is not None and g_call["theta"] < 0, \
    f"Theta must be negative (time decay): {g_call['theta']}"
# ATM call delta near 0.5–0.6
assert 0.50 < g_call["delta"] < 0.65, f"ATM call delta out of range: {g_call['delta']}"
print(f"[OK] ATM call greeks: delta={g_call['delta']:.4f}, gamma={g_call['gamma']:.6f}")

# Test 4: BS put greeks — ATM
g_put = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.20, "put")
assert g_put["delta"] is not None and -1 < g_put["delta"] < 0, \
    f"Put delta must be in (-1,0): {g_put['delta']}"
# Put-call delta parity: delta_call + |delta_put| ~= 1
delta_sum = g_call["delta"] + abs(g_put["delta"])
assert abs(delta_sum - 1.0) < 0.001, f"Put-call delta parity failed: {delta_sum}"
print(f"[OK] ATM put greeks: delta={g_put['delta']:.4f}, parity check passed")

# Test 5: Invalid inputs return None for all greeks
g_bad = bs_greeks(100, 100, -1, 0.05, 0.20, "call")  # T <= 0
assert all(v is None for v in g_bad.values()), "Invalid T should return all None"
g_bad2 = bs_greeks(100, 100, 0.25, 0.05, 0.0, "call")  # sigma = 0
assert all(v is None for v in g_bad2.values()), "sigma=0 should return all None"
print("[OK] Invalid inputs return all-None greeks dict")

# Test 6: Max pain computation
from sentinel.sbx.options_analytics import compute_max_pain, NormalizedContract
from decimal import Decimal
from datetime import date

contracts = [
    NormalizedContract(ticker="AAPL", expiry=date(2024,3,15), strike=Decimal("100"),
        contract_type="call", iv=0.25, open_interest=500, volume=100,
        delta=0.5, gamma=0.01, theta=-0.05, vega=0.1, close_price=None,
        underlying_price=102.0, moneyness=0.98),
    NormalizedContract(ticker="AAPL", expiry=date(2024,3,15), strike=Decimal("105"),
        contract_type="call", iv=0.22, open_interest=300, volume=80,
        delta=0.3, gamma=0.008, theta=-0.04, vega=0.09, close_price=None,
        underlying_price=102.0, moneyness=1.029),
    NormalizedContract(ticker="AAPL", expiry=date(2024,3,15), strike=Decimal("95"),
        contract_type="put", iv=0.28, open_interest=400, volume=90,
        delta=-0.4, gamma=0.01, theta=-0.04, vega=0.1, close_price=None,
        underlying_price=102.0, moneyness=0.931),
    NormalizedContract(ticker="AAPL", expiry=date(2024,3,15), strike=Decimal("100"),
        contract_type="put", iv=0.26, open_interest=200, volume=60,
        delta=-0.5, gamma=0.01, theta=-0.05, vega=0.1, close_price=None,
        underlying_price=102.0, moneyness=0.98),
]
result = compute_max_pain(contracts)
assert result.ticker == "AAPL", "Ticker mismatch"
assert result.max_pain_strike in [95.0, 100.0, 105.0], \
    f"Max pain strike not one of our strikes: {result.max_pain_strike}"
assert isinstance(result.pain_table, dict) and len(result.pain_table) == 3, \
    f"Expected 3 strikes in pain table: {result.pain_table}"
print(f"[OK] Max pain = {result.max_pain_strike}, distance = {result.distance_pct:.2f}%")

print("[PASS]")
PYEOF
