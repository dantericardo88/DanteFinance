#!/usr/bin/env bash
# dim_144: Multi-Factor Return Attribution Engine — capability verification
# Tests the Barra-style decomposition of portfolio returns into systematic
# (factor-driven) and idiosyncratic (stock-specific) components.
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from sentinel.spm.factor_attribution_v3 import (
    FactorExposure,
    SystematicReturn,
    IdiosyncraticReturn,
    AttributionResult,
    MultiFactorAttribution,
    decompose_returns,
    SUPPORTED_FACTORS,
)
print("[OK] All classes and functions imported from factor_attribution_v3")

# ---------------------------------------------------------------------------
# 2. Verify SUPPORTED_FACTORS contains all 6 required factors
# ---------------------------------------------------------------------------
required_factors = {"market", "size", "value", "momentum", "quality", "low_vol"}
assert required_factors.issubset(set(SUPPORTED_FACTORS)), \
    f"Missing factors: {required_factors - set(SUPPORTED_FACTORS)}"
print(f"[OK] SUPPORTED_FACTORS has all 6 required factors: {SUPPORTED_FACTORS}")

# ---------------------------------------------------------------------------
# 3. Build the 3-stock test portfolio
#
#   AAPL: weight=0.5, return=+5%,  beta=1.2, momentum=1.0,  value=-0.5
#   MSFT: weight=0.3, return=+3%,  beta=1.1, momentum=0.5,  value=-0.3
#   JPM:  weight=0.2, return=+1%,  beta=0.9, momentum=-0.2, value=0.8
#
#   Factor returns: market=2%, size=-0.5%, value=1%, momentum=1.5%,
#                   quality=0.3%, low_vol=-0.2%
# ---------------------------------------------------------------------------
weights = {"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}
stock_returns = {"AAPL": 0.05, "MSFT": 0.03, "JPM": 0.01}
factor_returns = {
    "market":   0.02,
    "size":    -0.005,
    "value":    0.01,
    "momentum": 0.015,
    "quality":  0.003,
    "low_vol": -0.002,
}
exposures = {
    "AAPL": FactorExposure(
        ticker="AAPL",
        market_beta=1.2,
        size=-0.3,
        value=-0.5,
        momentum=1.0,
        quality=0.8,
        low_vol=-0.4,
    ),
    "MSFT": FactorExposure(
        ticker="MSFT",
        market_beta=1.1,
        size=-0.2,
        value=-0.3,
        momentum=0.5,
        quality=0.9,
        low_vol=-0.3,
    ),
    "JPM": FactorExposure(
        ticker="JPM",
        market_beta=0.9,
        size=0.4,
        value=0.8,
        momentum=-0.2,
        quality=0.5,
        low_vol=0.6,
    ),
}

# ---------------------------------------------------------------------------
# 4. Run attribution
# ---------------------------------------------------------------------------
engine = MultiFactorAttribution()
result = engine.compute(weights, stock_returns, factor_returns, exposures)
print("[OK] MultiFactorAttribution.compute() executed successfully")

# ---------------------------------------------------------------------------
# 5. Check 1: Portfolio total return = sum(w*r) = 0.5*5% + 0.3*3% + 0.2*1% = 3.6%
# ---------------------------------------------------------------------------
expected_total = 0.5 * 0.05 + 0.3 * 0.03 + 0.2 * 0.01   # = 0.036
assert abs(result.total_portfolio_return - expected_total) < 1e-9, \
    f"Total return mismatch: {result.total_portfolio_return:.6f} vs {expected_total:.6f}"
print(f"[OK] Portfolio total return = {result.total_portfolio_return*100:.4f}% (expected {expected_total*100:.4f}%)")

# ---------------------------------------------------------------------------
# 6. Check 2: factor_return + specific_return = total_return (decomposition identity)
# ---------------------------------------------------------------------------
decomp_sum = result.systematic.total_factor_return + result.idiosyncratic.total_specific_return
assert abs(decomp_sum - result.total_portfolio_return) < 1e-6, (
    f"Decomposition identity violated: "
    f"factor={result.systematic.total_factor_return:.8f} + "
    f"specific={result.idiosyncratic.total_specific_return:.8f} = {decomp_sum:.8f} "
    f"!= total={result.total_portfolio_return:.8f}"
)
print(f"[OK] factor_return + specific_return = total_return: "
      f"{result.systematic.total_factor_return*100:.4f}% + "
      f"{result.idiosyncratic.total_specific_return*100:.4f}% = "
      f"{result.total_portfolio_return*100:.4f}%")

# ---------------------------------------------------------------------------
# 7. Check 3: Market contribution = portfolio_beta * market_return
#    portfolio_beta = 0.5*1.2 + 0.3*1.1 + 0.2*0.9 = 0.6 + 0.33 + 0.18 = 1.11
#    market_contribution = 1.11 * 0.02 = 0.0222
# ---------------------------------------------------------------------------
portfolio_beta = 0.5 * 1.2 + 0.3 * 1.1 + 0.2 * 0.9
expected_market_contrib = portfolio_beta * factor_returns["market"]
assert abs(result.systematic.market_contribution - expected_market_contrib) < 1e-9, (
    f"Market contribution mismatch: "
    f"{result.systematic.market_contribution:.8f} vs {expected_market_contrib:.8f}"
)
print(f"[OK] Market contribution = portfolio_beta({portfolio_beta:.3f}) x "
      f"market_return({factor_returns['market']*100:.1f}%) = "
      f"{result.systematic.market_contribution*100:.4f}%")

# ---------------------------------------------------------------------------
# 8. Check 4: R-squared in [0, 1]
# ---------------------------------------------------------------------------
assert 0.0 <= result.r_squared <= 1.0, \
    f"R-squared out of range: {result.r_squared}"
print(f"[OK] R-squared = {result.r_squared:.4f} (in [0, 1])")

# ---------------------------------------------------------------------------
# 9. Check 5: SystematicReturn.factor_breakdown has all 6 factors
# ---------------------------------------------------------------------------
assert isinstance(result.systematic.factor_breakdown, dict), \
    "factor_breakdown must be a dict"
missing_factors = required_factors - set(result.systematic.factor_breakdown.keys())
assert len(missing_factors) == 0, \
    f"factor_breakdown missing factors: {missing_factors}"
print(f"[OK] SystematicReturn.factor_breakdown has all 6 factors: "
      f"{list(result.systematic.factor_breakdown.keys())}")

# ---------------------------------------------------------------------------
# 10. Check 6: IdiosyncraticReturn.stock_contributions has all 3 tickers
# ---------------------------------------------------------------------------
assert isinstance(result.idiosyncratic.stock_contributions, dict), \
    "stock_contributions must be a dict"
missing_tickers = {"AAPL", "MSFT", "JPM"} - set(result.idiosyncratic.stock_contributions.keys())
assert len(missing_tickers) == 0, \
    f"stock_contributions missing tickers: {missing_tickers}"
print(f"[OK] IdiosyncraticReturn.stock_contributions has all 3 tickers: "
      f"{list(result.idiosyncratic.stock_contributions.keys())}")

# ---------------------------------------------------------------------------
# 11. Check 7: sum of all contributions = total portfolio return (within 1e-6)
#    All contributions = factor_breakdown values + stock_contributions values
# ---------------------------------------------------------------------------
sum_factor_contribs = sum(result.systematic.factor_breakdown.values())
sum_stock_contribs  = sum(result.idiosyncratic.stock_contributions.values())
grand_total = sum_factor_contribs + sum_stock_contribs
assert abs(grand_total - result.total_portfolio_return) < 1e-6, (
    f"Grand total mismatch: "
    f"sum_factors({sum_factor_contribs:.8f}) + sum_stocks({sum_stock_contribs:.8f}) = "
    f"{grand_total:.8f} != total({result.total_portfolio_return:.8f})"
)
print(f"[OK] Sum of all contributions = total portfolio return "
      f"({grand_total*100:.4f}% within 1e-6)")

# ---------------------------------------------------------------------------
# 12. AttributionResult.verify_decomposition() helper
# ---------------------------------------------------------------------------
assert result.verify_decomposition(), \
    "AttributionResult.verify_decomposition() returned False"
print("[OK] AttributionResult.verify_decomposition() passed")

# ---------------------------------------------------------------------------
# 13. decompose_returns() convenience function
# ---------------------------------------------------------------------------
result2 = decompose_returns(
    portfolio_returns=stock_returns,
    factor_returns=factor_returns,
    factor_exposures=exposures,
    weights=weights,
)
assert abs(result2.total_portfolio_return - result.total_portfolio_return) < 1e-9, \
    "decompose_returns() should match MultiFactorAttribution.compute()"
assert result2.verify_decomposition(), "decompose_returns() decomposition identity failed"
print("[OK] decompose_returns() convenience function works correctly")

# ---------------------------------------------------------------------------
# 14. Edge case: equal weights inferred when weights=None
# ---------------------------------------------------------------------------
result_eq = decompose_returns(
    portfolio_returns=stock_returns,
    factor_returns=factor_returns,
    factor_exposures=exposures,
    weights=None,
)
expected_eq = (0.05 + 0.03 + 0.01) / 3
assert abs(result_eq.total_portfolio_return - expected_eq) < 1e-9, \
    f"Equal-weight inference failed: {result_eq.total_portfolio_return:.6f} vs {expected_eq:.6f}"
assert result_eq.verify_decomposition(), "Equal-weight decomposition identity failed"
print(f"[OK] Equal-weight inference: total = {result_eq.total_portfolio_return*100:.4f}%")

# ---------------------------------------------------------------------------
# 15. Dataclass field checks
# ---------------------------------------------------------------------------
fe = FactorExposure("TEST", market_beta=1.5, size=-1.0, value=0.5,
                    momentum=2.0, quality=1.2, low_vol=-0.8)
assert fe.ticker == "TEST"
assert fe.market_beta == 1.5
assert fe.momentum == 2.0

sr = SystematicReturn(
    total_factor_return=0.01,
    market_contribution=0.005,
    size_contribution=0.002,
    value_contribution=0.001,
    momentum_contribution=0.001,
    quality_contribution=0.0005,
    low_vol_contribution=0.0005,
)
assert "market" in sr.factor_breakdown
assert abs(sr.factor_breakdown["market"] - 0.005) < 1e-12

ir = IdiosyncraticReturn(
    total_specific_return=0.003,
    stock_contributions={"AAPL": 0.002, "MSFT": 0.001},
)
assert ir.total_specific_return == 0.003
print("[OK] FactorExposure, SystematicReturn, IdiosyncraticReturn dataclasses verified")

# ---------------------------------------------------------------------------
# 16. MultiFactorAttribution with custom factor subset
# ---------------------------------------------------------------------------
engine_sub = MultiFactorAttribution(factors=["market", "momentum"])
result_sub = engine_sub.compute(weights, stock_returns, factor_returns, exposures)
assert result_sub.verify_decomposition(), "Subset-factor decomposition identity failed"
print("[OK] MultiFactorAttribution with custom factor subset works")

# ---------------------------------------------------------------------------
# 17. Top/Bottom contributors are populated
# ---------------------------------------------------------------------------
assert isinstance(result.idiosyncratic.top_contributors, list), \
    "top_contributors must be a list"
assert len(result.idiosyncratic.top_contributors) > 0, \
    "top_contributors should not be empty for a 3-stock portfolio"
for entry in result.idiosyncratic.top_contributors:
    assert len(entry) == 3, f"top_contributors entry should be 3-tuple: {entry}"
    assert isinstance(entry[0], str), "First element should be ticker string"
    assert isinstance(entry[1], float), "Second element should be raw epsilon (float)"
    assert isinstance(entry[2], float), "Third element should be weighted contribution (float)"
print(f"[OK] top_contributors: {[(t, f'{c*100:.3f}%') for t, _, c in result.idiosyncratic.top_contributors]}")

# ---------------------------------------------------------------------------
# 18. Print attribution summary
# ---------------------------------------------------------------------------
print("\n--- Attribution Summary ---")
print(f"  Total portfolio return : {result.total_portfolio_return*100:.4f}%")
print(f"  Systematic (factors)   : {result.systematic.total_factor_return*100:.4f}%")
print(f"  Idiosyncratic (specific): {result.idiosyncratic.total_specific_return*100:.4f}%")
print(f"  R-squared              : {result.r_squared:.4f}")
print(f"  Factor breakdown:")
for factor, contrib in result.systematic.factor_breakdown.items():
    print(f"    {factor:12s}: {contrib*100:+.4f}%")
print(f"  Stock contributions:")
for ticker, contrib in result.idiosyncratic.stock_contributions.items():
    print(f"    {ticker:6s}: {contrib*100:+.4f}%")

print("\n[PASS] dim_144")
PYEOF
