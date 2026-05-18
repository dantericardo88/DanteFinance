#!/bin/bash
# dim_068: Factor research v3 — FactorLibrary, pure math helpers, dataclasses
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from sentinel.sai.factor_research_v3 import (
    FactorLibrary,
    FactorComputer,
    FactorTester,
    FactorCombiner,
    FactorResearchEngine,
    FactorHypothesis,
    QuintileResult,
    CompositeSignal,
    FactorScanResult,
    BacktestResult,
    _winsorize,
    _SP500_PROXY,
    _SECTOR_MAP,
)

# 1. FactorLibrary FACTORS catalog
lib = FactorLibrary()
factors = lib.FACTORS
assert len(factors) >= 20, f"Expected >=20 factors, got {len(factors)}"
# Verify key factors exist
assert "pe_ratio" in factors
assert "pb_ratio" in factors
assert "roe" in factors or "return_on_equity" in factors or any("roe" in k for k in factors)
assert "momentum_12_1" in factors or any("momentum" in k for k in factors)

# Check factor structure
pe = factors["pe_ratio"]
assert "category" in pe
assert "direction" in pe
assert pe["category"] == "value"
assert pe["direction"] == "lower_is_better"
print(f"[OK] FactorLibrary.FACTORS: {len(factors)} factors, pe_ratio structure verified")

# 2. Factor categories
categories = set(f["category"] for f in factors.values())
assert len(categories) >= 4, f"Expected >=4 categories, got {categories}"
print(f"[OK] Factor categories: {sorted(categories)}")

# 3. _winsorize pure math
values = list(range(100))  # 0-99
winsorized = _winsorize(values, lower=0.05, upper=0.95)
assert len(winsorized) == 100
# lower 5% clipped to value at index 5 = 5, upper 95% clipped to value at index 95 = 95
assert min(winsorized) == 5, f"Expected lower clip=5, got {min(winsorized)}"
assert max(winsorized) == 95, f"Expected upper clip=95, got {max(winsorized)}"
print(f"[OK] _winsorize(): min={min(winsorized)}, max={max(winsorized)} (expected 5 and 95)")

# 4. _winsorize edge cases
assert _winsorize([]) == []
assert _winsorize([42.0]) == [42.0]
print(f"[OK] _winsorize edge cases: empty=[], single=[42.0]")

# 5. _SP500_PROXY universe
assert isinstance(_SP500_PROXY, list)
assert len(_SP500_PROXY) >= 20
assert "AAPL" in _SP500_PROXY
assert "MSFT" in _SP500_PROXY
assert "NVDA" in _SP500_PROXY
print(f"[OK] _SP500_PROXY: {len(_SP500_PROXY)} tickers, AAPL/MSFT/NVDA present")

# 6. _SECTOR_MAP coverage
assert isinstance(_SECTOR_MAP, dict)
assert len(_SECTOR_MAP) >= 20
assert _SECTOR_MAP["AAPL"] == "Technology"
assert _SECTOR_MAP["JPM"] == "Financials"
print(f"[OK] _SECTOR_MAP: {len(_SECTOR_MAP)} tickers, AAPL=Technology, JPM=Financials")

# 7. Dataclass: FactorHypothesis
fh = FactorHypothesis(
    factor_name="pe_ratio",
    category="value",
    rationale="Lower P/E = cheaper stock",
    expected_direction="lower_is_better",
    regime_fit=["recession", "recovery"],
    confidence=0.75,
)
assert fh.factor_name == "pe_ratio"
assert fh.confidence == 0.75
assert "recession" in fh.regime_fit
print(f"[OK] FactorHypothesis: factor={fh.factor_name}, confidence={fh.confidence}")

# 8. Dataclass: QuintileResult
qr = QuintileResult(
    factor_name="pe_ratio",
    start_date="2020-01-01",
    end_date="2024-12-31",
    quintile_returns=[0.05, 0.08, 0.10, 0.13, 0.18],
    spread_q1_q5=0.13,
    hit_rate=0.62,
    sharpe=0.85,
    n_periods=20,
)
assert len(qr.quintile_returns) == 5
assert abs(qr.spread_q1_q5 - 0.13) < 1e-9
print(f"[OK] QuintileResult: {qr.factor_name}, Q1-Q5 spread={qr.spread_q1_q5:.2f}, sharpe={qr.sharpe}")

# 9. Dataclass: CompositeSignal
cs = CompositeSignal(
    tickers=["AAPL", "MSFT", "GOOG"],
    scores=[0.8, 0.6, 0.4],
    weights={"pe_ratio": 0.4, "momentum_12_1": 0.3, "roe": 0.3},
    method="ic_weighted",
    date="2024-03-15",
    icir=1.2,
)
assert len(cs.tickers) == 3
assert cs.method == "ic_weighted"
assert abs(cs.icir - 1.2) < 1e-9
print(f"[OK] CompositeSignal: {len(cs.tickers)} tickers, method={cs.method}, ICIR={cs.icir}")

# 10. FactorCombiner pure computation (no network)
combiner = FactorCombiner()
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
fac1 = pd.Series([0.5, 0.3, -0.1, -0.3, 0.2], index=tickers, name="pe_ratio")
fac2 = pd.Series([0.2, 0.4, 0.1, -0.2, 0.3], index=tickers, name="momentum")
composite = combiner.equal_weight({"pe_ratio": fac1, "momentum": fac2})
assert len(composite) == 5
assert composite.index.tolist() == tickers
expected_aapl = (0.5 + 0.2) / 2
assert abs(composite["AAPL"] - expected_aapl) < 1e-9
print(f"[OK] FactorCombiner.equal_weight(): {len(composite)} stocks, AAPL={composite['AAPL']:.4f} (expected {expected_aapl:.4f})")

# 11. FactorResearchEngine instantiation (no network)
engine = FactorResearchEngine()
assert hasattr(engine, 'library')
assert hasattr(engine, 'computer')
assert hasattr(engine, 'tester')
assert hasattr(engine, 'combiner')
assert isinstance(engine.library, FactorLibrary)
print(f"[OK] FactorResearchEngine: library={type(engine.library).__name__}, computer={type(engine.computer).__name__}")

# 12. FactorScanResult defaults
scan = FactorScanResult(date="2024-03-15", universe=tickers, factors_computed=15, top_factors=["pe_ratio", "roe"])
assert scan.factors_computed == 15
assert len(scan.top_factors) == 2
assert scan.composite_signal is None
print(f"[OK] FactorScanResult: factors_computed={scan.factors_computed}, top={scan.top_factors}")

from sentinel.sai.factor_research_v3 import _spearman_corr

# ---- New math: compute_factor_ic_series (single period IC) ----
tester = FactorTester()

# Known rank data: perfectly correlated -> IC = +1.0
fv_perfect = [1.0, 2.0, 3.0, 4.0, 5.0]
fr_perfect  = [1.0, 2.0, 3.0, 4.0, 5.0]
ic_perfect  = tester.compute_factor_ic_series(fv_perfect, fr_perfect)
assert abs(ic_perfect - 1.0) < 1e-9, f"Perfect correlation IC should be 1.0, got {ic_perfect}"
print(f"[OK] compute_factor_ic_series (perfect rank) = {ic_perfect:.8f}  (expected 1.0)")

# Perfectly anti-correlated -> IC = -1.0
fv_anti = [1.0, 2.0, 3.0, 4.0, 5.0]
fr_anti  = [5.0, 4.0, 3.0, 2.0, 1.0]
ic_anti  = tester.compute_factor_ic_series(fv_anti, fr_anti)
assert abs(ic_anti - (-1.0)) < 1e-9, f"Anti-correlated IC should be -1.0, got {ic_anti}"
print(f"[OK] compute_factor_ic_series (anti rank) = {ic_anti:.8f}  (expected -1.0)")

# Spearman on ranked data is same as _spearman_corr directly
fv_rand = [3.0, 1.0, 4.0, 2.0, 5.0]
fr_rand  = [2.0, 5.0, 3.0, 1.0, 4.0]
ic_via_method   = tester.compute_factor_ic_series(fv_rand, fr_rand)
ic_via_spearman = _spearman_corr(fv_rand, fr_rand)
assert abs(ic_via_method - ic_via_spearman) < 1e-9, \
    f"IC series math should match _spearman_corr: {ic_via_method} vs {ic_via_spearman}"
print(f"[OK] compute_factor_ic_series matches _spearman_corr = {ic_via_method:.8f}")

# Length mismatch raises ValueError
import traceback as _tb
try:
    tester.compute_factor_ic_series([1.0, 2.0], [1.0, 2.0, 3.0])
    assert False, "Should have raised ValueError"
except ValueError:
    print("[OK] compute_factor_ic_series raises ValueError on length mismatch")

# Too few obs returns 0.0
assert tester.compute_factor_ic_series([1.0, 2.0], [2.0, 1.0]) == 0.0
print("[OK] compute_factor_ic_series returns 0.0 for n < 3")

# ---- New math: compute_factor_turnover ----
# Identical ranks -> no change -> turnover = 0
ranks_same = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
turnover_zero = tester.compute_factor_turnover(ranks_same, ranks_same, n_long=3)
assert abs(turnover_zero - 0.0) < 1e-9, f"Identical ranks -> turnover 0, got {turnover_zero}"
print(f"[OK] compute_factor_turnover (same ranks) = {turnover_zero:.4f}  (expected 0.0)")

# Completely reversed ranks -> maximum turnover = 1.0
ranks_t  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
ranks_t1 = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # fully reversed
turnover_max = tester.compute_factor_turnover(ranks_t, ranks_t1, n_long=3)
assert turnover_max == 1.0, f"Fully reversed ranks -> turnover 1.0, got {turnover_max}"
print(f"[OK] compute_factor_turnover (fully reversed) = {turnover_max:.4f}  (expected 1.0)")

# Result is in [0, 1]
import random as _rand
_rand.seed(7)
rr_t  = [float(i) for i in range(1, 11)]
rr_t1 = rr_t[:]
_rand.shuffle(rr_t1)
to_partial = tester.compute_factor_turnover(rr_t, rr_t1, n_long=5)
assert 0.0 <= to_partial <= 1.0, f"Turnover must be in [0,1]: {to_partial}"
print(f"[OK] compute_factor_turnover (random) = {to_partial:.4f}  (in [0,1])")

# ---- New math: compute_gross_profitability_factor (Novy-Marx 2013) ----
revenue = 10_000_000.0
cogs    =  6_000_000.0
assets  = 20_000_000.0
gp_factor = tester.compute_gross_profitability_factor(revenue, cogs, assets)
expected_gp = (revenue - cogs) / assets   # = 4_000_000 / 20_000_000 = 0.20
assert abs(gp_factor - expected_gp) < 1e-9, \
    f"GP factor expected {expected_gp}, got {gp_factor}"
print(f"[OK] compute_gross_profitability_factor = {gp_factor:.8f}  (expected {expected_gp:.8f})")

# Zero or negative assets returns 0.0
assert tester.compute_gross_profitability_factor(1e6, 0.5e6, 0.0) == 0.0
assert tester.compute_gross_profitability_factor(1e6, 0.5e6, -1.0) == 0.0
print("[OK] compute_gross_profitability_factor returns 0.0 for non-positive assets")

# GP can be negative when COGS > revenue
gp_neg = tester.compute_gross_profitability_factor(1_000_000, 1_500_000, 5_000_000)
assert gp_neg < 0.0, f"Negative GP should be negative: {gp_neg}"
print(f"[OK] compute_gross_profitability_factor (COGS > revenue) = {gp_neg:.4f}  (< 0)")

print("\n[PASS] dim_068: Factor research v3 -- FactorLibrary + IC/Turnover/GP factor verified")
PYEOF
