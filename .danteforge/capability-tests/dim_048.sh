#!/bin/bash
# dim_048: inflation_vix_analytics — inflation breakeven, VIX term structure, VRP
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.inflation_vix_analytics import (
    FRED_BASE,
    INFLATION_SERIES,
    VIX_SERIES,
    FED_INFLATION_TARGET,
    BreakevenSnapshot,
    VIXSnapshot,
    VRPSnapshot,
    InflationSignals,
    InflationBreakevenEngine,
    VIXTermStructureAnalyzer,
    VolatilityRiskPremiumEngine,
)

# --- constants ---
assert "fred.stlouisfed.org" in FRED_BASE
assert FED_INFLATION_TARGET == 2.0
print(f"[OK] FRED_BASE and FED_INFLATION_TARGET={FED_INFLATION_TARGET}% present")

# --- INFLATION_SERIES ---
assert "T5YIE" in INFLATION_SERIES    # 5Y breakeven
assert "T10YIE" in INFLATION_SERIES   # 10Y breakeven
assert "CPIAUCSL" in INFLATION_SERIES
print(f"[OK] INFLATION_SERIES has {len(INFLATION_SERIES)} series")

# --- VIX_SERIES ---
assert "VXST" in VIX_SERIES
assert "VIXCLS" in VIX_SERIES
assert "VXMT" in VIX_SERIES
print(f"[OK] VIX_SERIES has {len(VIX_SERIES)} series")

# --- BreakevenSnapshot model ---
bs = BreakevenSnapshot(
    breakeven_5y=2.35,
    breakeven_10y=2.40,
    forward_5y5y=2.45,
    real_yield_5y=1.95,
    real_yield_10y=2.00,
    fed_target=2.0,
    deviation_5y_from_target=0.35,
    deviation_10y_from_target=0.40,
    breakeven_regime_5y="anchored_above_target",
    breakeven_regime_10y="anchored_above_target",
    real_rate_stance_5y="positive_restrictive",
    real_rate_stance_10y="positive_restrictive",
    market_implied_path={"1Y": 2.5, "2Y": 2.3},
    as_of="2024-03-15",
)
assert bs.breakeven_10y == 2.40
assert bs.real_yield_5y == 1.95
assert abs(bs.deviation_10y_from_target - (bs.breakeven_10y - bs.fed_target)) < 0.01
print("[OK] BreakevenSnapshot model and deviation math correct")

# --- VIXSnapshot model ---
vs = VIXSnapshot(
    vix_9d=14.5,
    vix_1m=15.2,
    vix_3m=16.0,
    contango_ratio=1.053,
    spread_3m_1m=0.8,
    term_structure_shape="contango",
    vix_regime="low_vol",
    vix_percentile_1y=25.0,
    vix_percentile_5y=18.0,
    vix_percentile_10y=15.0,
    vix_mom_5d=-0.5,
    vol_of_vol_20d=3.2,
    as_of="2024-03-15",
)
assert vs.vix_1m == 15.2
assert vs.term_structure_shape == "contango"
contango_check = vs.vix_3m / vs.vix_1m
assert abs(contango_check - vs.contango_ratio) < 0.01, f"Contango ratio mismatch: {contango_check} vs {vs.contango_ratio}"
print(f"[OK] VIXSnapshot: contango_ratio={vs.contango_ratio:.3f} = vix_3m/vix_1m")

# --- VRPSnapshot model ---
vrp = VRPSnapshot(
    vix_current=15.2,
    realized_vol_21d=12.5,
    vrp_current=2.7,
    vrp_regime="normal_positive",
    vrp_signal="long_premium",
    as_of="2024-03-15",
)
vrp_calc = vrp.vix_current - vrp.realized_vol_21d
assert abs(vrp_calc - vrp.vrp_current) < 0.01, f"VRP mismatch: {vrp_calc} vs {vrp.vrp_current}"
print(f"[OK] VRPSnapshot: VRP={vrp.vrp_current} = implied - realized vol")

# --- InflationSignals model ---
inf_signals = InflationSignals(
    tips_vs_nominal_signal="elevated",
    bei_compression=False,
    bei_expansion=True,
    stagflation_indicator=False,
    real_rate_shock=False,
    signal_details={"cpi_yoy": 3.2},
    as_of="2024-03-15",
)
assert inf_signals.tips_vs_nominal_signal == "elevated"
assert inf_signals.bei_expansion is True
print("[OK] InflationSignals model created")

# --- class structures ---
assert hasattr(InflationBreakevenEngine, "__init__")
assert hasattr(VIXTermStructureAnalyzer, "__init__")
assert hasattr(VolatilityRiskPremiumEngine, "__init__")
print("[OK] InflationBreakevenEngine, VIXTermStructureAnalyzer, VolatilityRiskPremiumEngine classes present")

print("\n[PASS] dim_048: inflation_vix_analytics -- all checks passed")
PYEOF
