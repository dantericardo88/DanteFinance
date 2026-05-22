#!/usr/bin/env bash
# dim_145: Tail risk hedging (VIX options / put spreads / hedge effectiveness)
# Comprehensive capability verification for sentinel.sbx.tail_risk_hedge_v3
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from sentinel.sbx.tail_risk_hedge_v3 import (
    HedgeInstrument,
    TailRiskMetrics,
    PutSpreadStrategy,
    VIXHedge,
    HedgeEffectiveness,
    compute_var,
    compute_cvar,
    bs_put_price,
    optimal_hedge_ratio,
    # Legacy aliases (must also be importable)
    TailRiskHedge,
    VIXOptionHedge,
    PutSpreadHedge,
    compute_hedge_cost,
)
print("[OK] All imports successful from sentinel.sbx.tail_risk_hedge_v3")

# ---------------------------------------------------------------------------
# 2. Generate fat-tailed returns
#    1 000 daily returns: mixture
#      95% ~ N(0.0004, 0.01)   -- normal market
#       5% ~ N(-0.05, 0.03)    -- crash regime
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)
n = 1000
regime = rng.random(n) < 0.05   # True = crash day

normal_returns = rng.normal(0.0004, 0.01, n)
crash_returns  = rng.normal(-0.05, 0.03, n)
returns = np.where(regime, crash_returns, normal_returns)

print(f"[OK] Generated {n} fat-tailed returns  "
      f"(crash days={regime.sum()}, mean={returns.mean():.5f}, "
      f"min={returns.min():.4f})")

# ---------------------------------------------------------------------------
# 3. VaR and CVaR
# ---------------------------------------------------------------------------
var_95  = compute_var(returns, 0.95)
var_99  = compute_var(returns, 0.99)
cvar_95 = compute_cvar(returns, 0.95)
cvar_99 = compute_cvar(returns, 0.99)

print(f"[OK] VaR_95 = {var_95:.6f}, VaR_99 = {var_99:.6f}")
print(f"[OK] CVaR_95 = {cvar_95:.6f}, CVaR_99 = {cvar_99:.6f}")

# CVaR must exceed VaR (expected shortfall is always worse than VaR)
assert cvar_95 > var_95, (
    f"FAIL: CVaR_95 ({cvar_95:.6f}) should be > VaR_95 ({var_95:.6f})"
)
print("[OK] CVaR_95 > VaR_95 (expected shortfall exceeds VaR)")

# Higher confidence = worse tail
assert cvar_99 > cvar_95, (
    f"FAIL: CVaR_99 ({cvar_99:.6f}) should be > CVaR_95 ({cvar_95:.6f})"
)
print("[OK] CVaR_99 > CVaR_95 (99% tail worse than 95% tail)")

# ---------------------------------------------------------------------------
# 4. TailRiskMetrics via HedgeEffectiveness
# ---------------------------------------------------------------------------
engine = HedgeEffectiveness()
metrics = engine.tail_risk_metrics(returns)

assert isinstance(metrics, TailRiskMetrics), "tail_risk_metrics must return TailRiskMetrics"
assert metrics.var_95 > 0, "VaR_95 must be positive"
assert metrics.cvar_95 > metrics.var_95, "CVaR_95 > VaR_95 in TailRiskMetrics"
assert metrics.max_drawdown >= 0, "max_drawdown must be non-negative"
print(f"[OK] TailRiskMetrics: var95={metrics.var_95:.4f}, cvar95={metrics.cvar_95:.4f}, "
      f"max_dd={metrics.max_drawdown:.4f}, skew={metrics.skewness:.3f}, "
      f"exc_kurt={metrics.excess_kurtosis:.3f}")

# ---------------------------------------------------------------------------
# 5. Black-Scholes put price sanity
# ---------------------------------------------------------------------------
S, K_pp, T, sigma = 100.0, 95.0, 0.25, 0.2
r = 0.05
prem = bs_put_price(S, K_pp, T, r, sigma)

assert prem > 0, f"FAIL: put premium must be > 0, got {prem}"
assert prem < 10, f"FAIL: put premium must be < 10, got {prem}"
print(f"[OK] bs_put_price(S=100, K=95, T=0.25, sigma=0.20) = {prem:.4f}  (in (0, 10))")

# ---------------------------------------------------------------------------
# 6. Protective put
# ---------------------------------------------------------------------------
strat = PutSpreadStrategy(S=100.0, r=0.05)
pp = strat.protective_put(K=95.0, T=0.25, sigma=0.2)

assert "premium" in pp and "breakeven" in pp and "max_loss" in pp, \
    f"FAIL: protective_put missing keys: {pp.keys()}"
assert pp["premium"] > 0, f"FAIL: protective_put premium should be > 0"
assert pp["premium"] < 10, f"FAIL: protective_put premium should be < 10"
assert pp["unlimited_upside"] is True, "FAIL: protective put has unlimited upside"
print(f"[OK] protective_put: premium={pp['premium']:.4f}, "
      f"breakeven={pp['breakeven']:.4f}, max_loss={pp['max_loss']:.4f}")

# ---------------------------------------------------------------------------
# 7. Bear put spread: K_long=95, K_short=90
#    net_debit < premium of protective put at K=95
# ---------------------------------------------------------------------------
bps = strat.bear_put_spread(K_long=95.0, K_short=90.0, T=0.25, sigma=0.2)

assert "net_debit" in bps and "max_profit" in bps and "breakeven" in bps, \
    f"FAIL: bear_put_spread missing keys: {bps.keys()}"
assert bps["net_debit"] < pp["premium"], (
    f"FAIL: bear put spread net_debit ({bps['net_debit']:.4f}) should be "
    f"cheaper than protective put ({pp['premium']:.4f})"
)
print(f"[OK] bear_put_spread net_debit={bps['net_debit']:.4f} "
      f"< protective put premium={pp['premium']:.4f}")

# max_profit = K_long - K_short - net_debit
expected_max_profit = 95.0 - 90.0 - bps["net_debit"]
assert abs(bps["max_profit"] - expected_max_profit) < 1e-10, (
    f"FAIL: max_profit={bps['max_profit']:.6f} != "
    f"K_long-K_short-net_debit={expected_max_profit:.6f}"
)
print(f"[OK] max_profit = K_long - K_short - net_debit = {bps['max_profit']:.4f}")

# ---------------------------------------------------------------------------
# 8. VIX hedge: crisis simulation
#    Portfolio -30%, VIX spikes to 80, strike K_vix=25, premium=5000
# ---------------------------------------------------------------------------
vix_hedge = VIXHedge()
pnl = vix_hedge.crisis_pnl(
    portfolio_loss_pct=-0.30,
    vix_spike=80.0,
    vix_strike=25.0,
    vix_premium=5_000.0,
)

assert "hedge_pnl" in pnl and "hedge_payoff" in pnl, \
    f"FAIL: crisis_pnl missing keys: {pnl.keys()}"

# Payoff = (80 - 25) * 1000 = 55 000, net = 55 000 - 5 000 = 50 000
assert pnl["hedge_pnl"] > 0, (
    f"FAIL: hedge_pnl should be positive (crisis VIX spike), got {pnl['hedge_pnl']}"
)
assert abs(pnl["hedge_payoff"] - 55_000.0) < 1e-6, (
    f"FAIL: hedge_payoff should be 55000, got {pnl['hedge_payoff']}"
)
assert abs(pnl["hedge_pnl"] - 50_000.0) < 1e-6, (
    f"FAIL: hedge_pnl should be 50000, got {pnl['hedge_pnl']}"
)
print(f"[OK] VIX crisis hedge_payoff={pnl['hedge_payoff']:,.0f}, "
      f"hedge_pnl={pnl['hedge_pnl']:,.0f} (> 0)")

# ---------------------------------------------------------------------------
# 9. Optimal hedge ratio: two correlated series (rho ~= -0.7)
#    Long portfolio + negatively correlated hedge -> h* < 0
# ---------------------------------------------------------------------------
rng2 = np.random.default_rng(seed=99)
port_r = rng2.normal(0.001, 0.015, 500)
# Create hedge with ~-0.7 correlation
noise = rng2.normal(0, 0.01, 500)
hedge_r = -0.7 * (port_r / 0.015) * 0.02 + noise   # corr ~= -0.7

actual_corr = float(np.corrcoef(port_r, hedge_r)[0, 1])
h_star = optimal_hedge_ratio(port_r, hedge_r)

assert h_star < 0, (
    f"FAIL: hedge_ratio should be < 0 for inverse hedge, got {h_star:.4f} "
    f"(correlation = {actual_corr:.4f})"
)
print(f"[OK] optimal_hedge_ratio = {h_star:.4f}  "
      f"(correlation = {actual_corr:.4f}, negative as expected)")

# ---------------------------------------------------------------------------
# 10. Hedge effectiveness: R-squared > 0 (some protection)
# ---------------------------------------------------------------------------
eff = engine.hedge_effectiveness(
    portfolio_returns=port_r,
    hedge_returns=hedge_r,
    hedge_cost=0.01,
)

assert "r_squared" in eff, "FAIL: hedge_effectiveness must return r_squared"
assert eff["r_squared"] > 0, (
    f"FAIL: r_squared should be > 0, got {eff['r_squared']:.6f}"
)
assert 0 <= eff["r_squared"] <= 1, \
    f"FAIL: r_squared must be in [0,1], got {eff['r_squared']:.6f}"
print(f"[OK] hedge_effectiveness r_squared={eff['r_squared']:.4f}  "
      f"cvar_reduction_95={eff['cvar_reduction_95']:.6f}")

# ---------------------------------------------------------------------------
# 11. Legacy alias checks
# ---------------------------------------------------------------------------
trh = TailRiskHedge()
assert trh is not None, "FAIL: TailRiskHedge not instantiable"

voh = VIXOptionHedge()
assert voh is not None, "FAIL: VIXOptionHedge not instantiable"
payoff_check = voh.vix_call_payoff(vix_at_expiry=50.0, strike=25.0)
assert abs(payoff_check - 25_000.0) < 1e-6, \
    f"FAIL: VIXOptionHedge.vix_call_payoff = {payoff_check} expected 25000"

psh = PutSpreadHedge(S=100.0)
assert psh is not None, "FAIL: PutSpreadHedge not instantiable"

hi = HedgeInstrument(instrument_type="vix_call", cost=5_000.0, notional=1_000_000.0,
                     payoff_params={"strike": 25.0})
cost_info = compute_hedge_cost(hi, portfolio_size=1_000_000.0)
assert "cost" in cost_info and "cost_as_pct_portfolio" in cost_info, \
    f"FAIL: compute_hedge_cost missing keys: {cost_info.keys()}"
assert abs(cost_info["cost_as_pct_portfolio"] - 0.005) < 1e-10, \
    f"FAIL: cost pct = {cost_info['cost_as_pct_portfolio']}"
print("[OK] Legacy aliases TailRiskHedge, VIXOptionHedge, PutSpreadHedge, "
      "compute_hedge_cost all verified")

# ---------------------------------------------------------------------------
# 12. Print summary
# ---------------------------------------------------------------------------
print("\n--- Tail Risk Hedging Summary ---")
print(f"  Returns (n={n}): mean={returns.mean():+.5f}  std={returns.std():.5f}")
print(f"  VaR_95        : {var_95:.5f}  ({var_95*100:.3f}%)")
print(f"  VaR_99        : {var_99:.5f}  ({var_99*100:.3f}%)")
print(f"  CVaR_95       : {cvar_95:.5f}  ({cvar_95*100:.3f}%)")
print(f"  CVaR_99       : {cvar_99:.5f}  ({cvar_99*100:.3f}%)")
print(f"  Max drawdown  : {metrics.max_drawdown:.5f}")
print(f"  Skewness      : {metrics.skewness:.4f}")
print(f"  Excess kurtosis: {metrics.excess_kurtosis:.4f}")
print(f"  Protective put premium (K=95, T=0.25, sigma=0.2): {pp['premium']:.4f}")
print(f"  Bear put spread net_debit (K_long=95, K_short=90): {bps['net_debit']:.4f}")
print(f"  Bear put spread max_profit: {bps['max_profit']:.4f}")
print(f"  VIX crisis hedge_pnl (VIX=80, K=25): {pnl['hedge_pnl']:,.0f}")
print(f"  Optimal hedge ratio (rho~=-0.7): {h_star:.4f}")
print(f"  Hedge R-squared: {eff['r_squared']:.4f}")

print("\n[PASS] dim_145: Tail risk hedging")
PYEOF
