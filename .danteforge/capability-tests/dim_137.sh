#!/usr/bin/env bash
# dim_137: SPAN margin and portfolio margin simulation
# Tests SPANEngine, MarginAnalytics, and convenience functions
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# 1. Import verification
# ---------------------------------------------------------------------------
from sentinel.spm.span_margin_v3 import (
    Position,
    SPANParams,
    SPANEngine,
    MarginAnalytics,
    black_scholes_price,
    option_delta,
    compute_span_margin,
    compute_portfolio_margin,
)
print("[OK] All span_margin_v3 classes and functions imported")

# ---------------------------------------------------------------------------
# 2. Black-Scholes sanity check
# ---------------------------------------------------------------------------
# Call-put parity: C - P = S - K*exp(-rT)
S, K, T, r, sigma = 450.0, 460.0, 0.25, 0.05, 0.20
call_price = black_scholes_price(S, K, T, r, sigma, "call")
put_price  = black_scholes_price(S, K, T, r, sigma, "put")
parity_lhs = call_price - put_price
parity_rhs = S - K * math.exp(-r * T)
assert abs(parity_lhs - parity_rhs) < 0.01, \
    f"Put-call parity violated: C-P={parity_lhs:.4f}, S-K*e^-rT={parity_rhs:.4f}"
assert call_price > 0
assert put_price  > 0
print(f"[OK] BS call={call_price:.4f}, put={put_price:.4f}, parity error={abs(parity_lhs-parity_rhs):.6f}")

# Call delta in (0, 1), put delta in (-1, 0)
call_delta = option_delta(S, K, T, r, sigma, "call")
put_delta  = option_delta(S, K, T, r, sigma, "put")
assert 0.0 < call_delta < 1.0, f"Call delta out of range: {call_delta}"
assert -1.0 < put_delta < 0.0, f"Put delta out of range: {put_delta}"
print(f"[OK] call delta={call_delta:.4f}, put delta={put_delta:.4f}")

# ---------------------------------------------------------------------------
# 3. Build mixed portfolio:
#    - Long 100 shares of SPY @ $450
#    - Long 2 call options (K=460, T=0.25, σ=0.20)
#    - Short 1 put option  (K=440, T=0.25, σ=0.22)
# ---------------------------------------------------------------------------
spy_long  = Position(ticker="SPY",   underlying_price=450.0, quantity=100, is_option=False)
call_long = Position(ticker="SPY",   underlying_price=450.0, quantity=2,
                     is_option=True, option_type="call", strike=460.0,
                     expiry=0.25, implied_vol=0.20)
put_short = Position(ticker="SPY",   underlying_price=450.0, quantity=-1,
                     is_option=True, option_type="put", strike=440.0,
                     expiry=0.25, implied_vol=0.22)

mixed_portfolio = [spy_long, call_long, put_short]
print("[OK] Mixed portfolio constructed: 100 SPY long + 2 calls long + 1 put short")

# ---------------------------------------------------------------------------
# 4. SPAN margin on mixed portfolio
# ---------------------------------------------------------------------------
engine = SPANEngine()
params = SPANParams()
span_result = engine.span_margin(mixed_portfolio, params)

assert span_result["total_span_margin"] > 0, \
    f"SPAN total margin must be > 0, got {span_result['total_span_margin']}"
assert span_result["scanning_risk"] >= 0, \
    f"Scanning risk must be >= 0, got {span_result['scanning_risk']}"
assert "spread_charge" in span_result
assert "delivery_charge" in span_result
assert "short_option_min" in span_result
assert "span_credit" in span_result

print(f"[OK] SPAN margin decomposition:")
print(f"       scanning_risk     = ${span_result['scanning_risk']:,.2f}")
print(f"       short_option_min  = ${span_result['short_option_min']:,.2f}")
print(f"       spread_charge     = ${span_result['spread_charge']:,.2f}")
print(f"       delivery_charge   = ${span_result['delivery_charge']:,.2f}")
print(f"       span_credit       = ${span_result['span_credit']:,.2f}")
print(f"       TOTAL SPAN MARGIN = ${span_result['total_span_margin']:,.2f}")

# ---------------------------------------------------------------------------
# 5. Portfolio margin on mixed portfolio
# ---------------------------------------------------------------------------
pm_result = engine.portfolio_margin(mixed_portfolio)

assert pm_result["portfolio_margin"] > 0, \
    f"Portfolio margin must be > 0, got {pm_result['portfolio_margin']}"
assert "scenario_losses" in pm_result
assert len(pm_result["scenario_losses"]) == 10, \
    f"Expected 10 PM scenarios, got {len(pm_result['scenario_losses'])}"
assert pm_result["worst_scenario_loss"] >= 0
assert pm_result["floor_margin"] > 0

print(f"[OK] Portfolio margin:")
print(f"       worst_scenario_loss = ${pm_result['worst_scenario_loss']:,.2f}")
print(f"       floor_margin        = ${pm_result['floor_margin']:,.2f}")
print(f"       TOTAL PM            = ${pm_result['portfolio_margin']:,.2f}")

# ---------------------------------------------------------------------------
# 6. Margin analytics
# ---------------------------------------------------------------------------
analytics = MarginAnalytics()

efficiency = analytics.margin_efficiency(mixed_portfolio, params)
assert math.isfinite(efficiency) and efficiency > 0, \
    f"Margin efficiency must be finite > 0, got {efficiency}"

portfolio_value = 100_000.0
excess_liq = analytics.excess_liquidity(portfolio_value, mixed_portfolio, params)
assert excess_liq > 0, \
    f"Excess liquidity must be > 0 for $100k portfolio, got {excess_liq:.2f}"

utilization = analytics.margin_utilization(portfolio_value, mixed_portfolio, params)
assert 0.0 < utilization < 1.0, \
    f"Margin utilization must be in (0,1) for $100k portfolio, got {utilization:.4f}"

print(f"[OK] Margin analytics:")
print(f"       margin_efficiency   = {efficiency:.4f}")
print(f"       excess_liquidity    = ${excess_liq:,.2f}")
print(f"       margin_utilization  = {utilization:.4f} ({utilization*100:.2f}%)")

# ---------------------------------------------------------------------------
# 7. Convenience functions (module-level API)
# ---------------------------------------------------------------------------
span_total = compute_span_margin(mixed_portfolio, params)
pm_total   = compute_portfolio_margin(mixed_portfolio)

assert span_total == span_result["total_span_margin"], \
    f"compute_span_margin mismatch: {span_total} vs {span_result['total_span_margin']}"
assert pm_total == pm_result["portfolio_margin"], \
    f"compute_portfolio_margin mismatch: {pm_total} vs {pm_result['portfolio_margin']}"

print(f"[OK] Convenience functions: SPAN=${span_total:,.2f}, PM=${pm_total:,.2f}")

# ---------------------------------------------------------------------------
# 8. Extreme scenario: short 10 deep-ITM puts → high margin
# ---------------------------------------------------------------------------
deep_itm_puts = [
    Position(ticker="SPX", underlying_price=4500.0, quantity=-10,
             is_option=True, option_type="put", strike=4600.0,
             expiry=0.25, implied_vol=0.25)
]

span_extreme = compute_span_margin(deep_itm_puts)
assert span_extreme > 0, f"Extreme short put margin must be > 0, got {span_extreme}"

# Absolute margin should be substantial (deep ITM puts have high risk)
assert span_extreme > 1000, \
    f"Short 10 deep-ITM puts: margin should be > $1000, got {span_extreme:.2f}"

print(f"[OK] Extreme scenario (short 10 deep-ITM puts): SPAN margin = ${span_extreme:,.2f}")

# ---------------------------------------------------------------------------
# 9. Short position has higher margin than equivalent long position
# ---------------------------------------------------------------------------
short_call = [Position(ticker="AAPL", underlying_price=200.0, quantity=-5,
                       is_option=True, option_type="call", strike=210.0,
                       expiry=0.5, implied_vol=0.30)]
long_call  = [Position(ticker="AAPL", underlying_price=200.0, quantity=+5,
                       is_option=True, option_type="call", strike=210.0,
                       expiry=0.5, implied_vol=0.30)]

span_short = compute_span_margin(short_call)
span_long  = compute_span_margin(long_call)

assert span_short > span_long, \
    (f"Short position margin ({span_short:.2f}) must exceed "
     f"long position margin ({span_long:.2f})")

print(f"[OK] Short call margin=${span_short:,.2f} > long call margin=${span_long:,.2f}")

# ---------------------------------------------------------------------------
# 10. Scanning risk: verify it equals max loss across 16 scenarios
# ---------------------------------------------------------------------------
# Manually compute worst-case for a simple long equity position
spy_only = [Position(ticker="SPY", underlying_price=400.0, quantity=10, is_option=False)]
sr = engine.scanning_risk(spy_only, params)

# For long equity, worst scenario is extreme down: -2σ = -10% price drop
# Loss = 10 * 400 * 0.10 = 400, weighted at 35% → 140
# Or standard down 1σ = -5% → loss = 200 (full weight)
# Scanning risk should be ~ 200 for 10 shares at $400 with 5% price range
expected_sr = 10 * 400 * params.price_scan_range  # full 1σ down = $200
assert abs(sr - expected_sr) < 5.0, \
    f"SPY scanning risk: expected ~{expected_sr:.2f}, got {sr:.2f}"
print(f"[OK] Scanning risk for long 10 SPY @ $400: ${sr:.2f} (expected ~${expected_sr:.2f})")

# ---------------------------------------------------------------------------
# 11. SPAN params customization
# ---------------------------------------------------------------------------
tight_params = SPANParams(price_scan_range=0.02, vol_scan_range=0.01)
wide_params  = SPANParams(price_scan_range=0.10, vol_scan_range=0.05)

span_tight = compute_span_margin(mixed_portfolio, tight_params)
span_wide  = compute_span_margin(mixed_portfolio, wide_params)

assert span_wide >= span_tight, \
    f"Wider scan range must produce >= margin: wide={span_wide:.2f}, tight={span_tight:.2f}"
print(f"[OK] Wide params margin=${span_wide:,.2f} >= tight params margin=${span_tight:,.2f}")

print("\n[PASS] dim_137: SPAN/portfolio-margin simulation -- all checks passed")
PYEOF
