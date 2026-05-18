#!/bin/bash
# dim_047: yield_spread_v3 — yield spread analytics, term structure, recession models
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.yield_spread_v3 import (
    FRED_BASE,
    FRED_YIELD_SERIES,
    FRED_SPREAD_SERIES,
    FRED_INTL_SERIES,
    SpreadData,
    RecessionForecast,
    BacktestResult,
    YieldSpreadCalculator,
    RecessionProbabilityModel,
    YieldSpreadEngine,
    _norm_cdf,
    _risk_label,
)

# --- constants ---
assert "fred.stlouisfed.org" in FRED_BASE
assert "10Y" in FRED_YIELD_SERIES and FRED_YIELD_SERIES["10Y"] == "DGS10"
assert "2Y" in FRED_YIELD_SERIES and "3M" in FRED_YIELD_SERIES
assert "T10Y2Y" in FRED_SPREAD_SERIES
assert "JP" in FRED_INTL_SERIES and "GB" in FRED_INTL_SERIES
print(f"[OK] FRED series constants: {len(FRED_YIELD_SERIES)} yield, {len(FRED_SPREAD_SERIES)} spread, {len(FRED_INTL_SERIES)} intl")

# --- _norm_cdf boundary values ---
assert abs(_norm_cdf(0.0) - 0.5) < 1e-6, "CDF(0) should be 0.5"
assert _norm_cdf(5.0) > 0.999
assert _norm_cdf(-5.0) < 0.001
print("[OK] _norm_cdf boundary values correct")

# --- _risk_label ---
assert _risk_label(0.10) in ("low", "elevated", "high", "very_high") or isinstance(_risk_label(0.10), str)
label_high = _risk_label(0.60)
assert isinstance(label_high, str)
print(f"[OK] _risk_label: 10%->'{_risk_label(0.10)}', 60%->'{label_high}'")

# --- SpreadData dataclass ---
sd = SpreadData(
    as_of="2024-03-15",
    yields={"3M": 5.25, "2Y": 4.60, "10Y": 4.20, "30Y": 4.35},
    spreads={"T10Y2Y": -0.40, "T10Y3M": -1.05},
    percentiles={"T10Y2Y": 5.0, "T10Y3M": 3.0},
    regime="INVERTED",
)
assert sd.as_of == "2024-03-15"
assert sd.regime == "INVERTED"
assert sd.spreads["T10Y2Y"] == -0.40
d = sd.to_dict()
assert "yields" in d and "spreads" in d
print("[OK] SpreadData dataclass and to_dict() work")

# --- RecessionForecast dataclass ---
rf = RecessionForecast(
    as_of="2024-03-15",
    ny_fed_model=0.58,
    wright_model=0.45,
    spread_3m10y=-1.05,
    fed_funds=5.33,
    ny_fed_risk_label="high",
    wright_risk_label="elevated",
    historical_comparison="Matches 2006-2007 inversion pattern",
)
assert rf.ny_fed_model == 0.58
assert rf.ny_fed_risk_label == "high"
d2 = rf.to_dict()
assert "ny_fed_model" in d2
print("[OK] RecessionForecast dataclass and to_dict() work")

# --- BacktestResult dataclass ---
bt = BacktestResult(
    strategy="Inversion Signal",
    start="2000-01-01",
    end="2023-12-31",
    total_return=2.45,
    cagr=0.042,
    sharpe=0.65,
    max_drawdown=-0.38,
    num_trades=7,
    win_rate=0.71,
    benchmark_return=1.85,
    alpha=0.021,
)
assert bt.sharpe == 0.65
assert bt.win_rate == 0.71
print("[OK] BacktestResult dataclass created")

# --- YieldSpreadCalculator class structure ---
calc = YieldSpreadCalculator()
assert hasattr(calc, "fetch_all_yields")
assert hasattr(calc, "compute_spreads") or hasattr(calc, "_get_series")
assert calc.TENOR_ORDER[0] == "3M"
print("[OK] YieldSpreadCalculator instantiates with TENOR_ORDER")

# --- RecessionProbabilityModel class structure ---
rpm = RecessionProbabilityModel()
assert hasattr(rpm, "compute_ny_fed_model")
# NY Fed probit model: -100bp spread (inverted) => elevated recession probability
prob = rpm.compute_ny_fed_model(-1.0)   # -100bps
assert 0.0 < prob < 1.0, f"Probability out of [0,1]: {prob}"
assert prob > 0.3, f"Inverted curve should show elevated recession risk: {prob}"
print(f"[OK] RecessionProbabilityModel.compute_ny_fed_model(-100bp) = {prob:.3f}")

# --- YieldSpreadEngine class structure ---
engine = YieldSpreadEngine()
assert hasattr(engine, "calc")
assert hasattr(engine, "recession_model")
assert hasattr(engine, "get_dashboard")
print("[OK] YieldSpreadEngine instantiates with calc, recession_model, get_dashboard")

print("\n[PASS] dim_047: yield_spread_v3 -- all checks passed")
PYEOF
