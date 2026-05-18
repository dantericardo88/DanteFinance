#!/bin/bash
# dim_048: inflation_vix_analytics -- VIX term structure, inflation regime, breakeven momentum, VRP
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
    INFLATION_REGIMES,
    BreakevenSnapshot,
    VIXSnapshot,
    VRPSnapshot,
    InflationSignals,
    VIXFullTermStructure,
    VIXSkewAndVVIX,
    InflationRegimeClassifier,
    BreakevenMomentum,
    compute_vrp,
)

# --- constants ---
assert "fred.stlouisfed.org" in FRED_BASE
assert FED_INFLATION_TARGET == 2.0
print(f"[OK] FRED_BASE and FED_INFLATION_TARGET={FED_INFLATION_TARGET}% present")

# --- INFLATION_SERIES ---
assert "T5YIE" in INFLATION_SERIES
assert "T10YIE" in INFLATION_SERIES
assert "CPIAUCSL" in INFLATION_SERIES
print(f"[OK] INFLATION_SERIES has {len(INFLATION_SERIES)} series")

# --- VIX_SERIES ---
assert "VXST" in VIX_SERIES
assert "VIXCLS" in VIX_SERIES
assert "VXMT" in VIX_SERIES
print(f"[OK] VIX_SERIES has {len(VIX_SERIES)} series")

# --- VIX term structure: pure-math contango/backwardation ---
vts = VIXFullTermStructure()

# VIX9D < VIX1M < VIX3M -> contango
shape_c = vts.classify_shape(vix_9d=12.0, vix_1m=14.5, vix_3m=16.0)
assert shape_c == "contango", f"Expected contango, got {shape_c}"
print(f"[OK] VIX term structure: VIX9D=12 < VIX=14.5 < VIX3M=16 -> '{shape_c}'")

# VIX9D > VIX1M > VIX3M -> backwardation (fear spike)
shape_b = vts.classify_shape(vix_9d=35.0, vix_1m=28.0, vix_3m=22.0)
assert shape_b == "backwardation", f"Expected backwardation, got {shape_b}"
print(f"[OK] VIX term structure: VIX9D=35 > VIX=28 > VIX3M=22 -> '{shape_b}'")

# Contango ratio: VIX3M / VIX9D must be > 1 for contango
contango_ratio = 16.0 / 12.0
assert contango_ratio > 1.0, f"Contango ratio should be > 1, got {contango_ratio}"
print(f"[OK] Contango ratio: {contango_ratio:.3f} > 1 (correct for contango)")

# --- VIX Skew (fear gauge): pure math ---
skew_engine = VIXSkewAndVVIX()
# fear_gauge = VIX - VVIX * 0.1
vix_val = 20.0
vvix_val = 85.0
fear = skew_engine.fear_gauge(vix_val, vvix_val)
expected_fear = vix_val - vvix_val * 0.1
assert abs(fear - expected_fear) < 0.01, f"Fear gauge: expected {expected_fear}, got {fear}"
print(f"[OK] VIX skew fear gauge: VIX={vix_val} - VVIX*0.1={vvix_val*0.1} = {fear}")

# --- Inflation regime classifier: pure math ---
clf = InflationRegimeClassifier()

# CPI=6%, GDP_growth=1% -> Stagflation (high inflation + low growth)
regime = clf.classify(cpi_pct=6.0, gdp_growth_pct=1.0)
assert regime == "Stagflation", f"Expected Stagflation, got {regime}"
print(f"[OK] Inflation regime: CPI=6%, GDP=1% -> '{regime}'")

# CPI=2%, GDP_growth=3% -> Goldilocks (low inflation + high growth)
regime2 = clf.classify(cpi_pct=2.0, gdp_growth_pct=3.0)
assert regime2 == "Goldilocks", f"Expected Goldilocks, got {regime2}"
print(f"[OK] Inflation regime: CPI=2%, GDP=3% -> '{regime2}'")

# CPI=5%, GDP_growth=3.5% -> Reflation (high inflation + high growth)
regime3 = clf.classify(cpi_pct=5.0, gdp_growth_pct=3.5)
assert regime3 == "Reflation", f"Expected Reflation, got {regime3}"
print(f"[OK] Inflation regime: CPI=5%, GDP=3.5% -> '{regime3}'")

# CPI=0.5%, GDP_growth=0.5% -> Deflation (low inflation + low growth)
regime4 = clf.classify(cpi_pct=0.5, gdp_growth_pct=0.5)
assert regime4 == "Deflation", f"Expected Deflation, got {regime4}"
print(f"[OK] Inflation regime: CPI=0.5%, GDP=0.5% -> '{regime4}'")

# All regime labels present in INFLATION_REGIMES
for r in ("Goldilocks", "Stagflation", "Deflation", "Reflation"):
    assert r in INFLATION_REGIMES, f"{r} not in INFLATION_REGIMES"
print(f"[OK] All 4 inflation regime labels verified in INFLATION_REGIMES")

# --- Breakeven momentum: pure math 5D vs 20D MA ---
# 20 values with upward trend: last 5 above first 15
base = [2.1, 2.15, 2.12, 2.18, 2.13, 2.20, 2.17, 2.19, 2.21, 2.18,
        2.22, 2.20, 2.24, 2.25, 2.23, 2.30, 2.32, 2.35, 2.38, 2.40]
result_pos = BreakevenMomentum.compute_from_series(base)
assert result_pos["signal"] == "positive", f"Expected positive momentum, got {result_pos['signal']}"
assert result_pos["ma5"] > result_pos["ma20"], f"ma5 should be > ma20 for positive momentum"
print(f"[OK] Breakeven momentum: rising series -> 5D MA={result_pos['ma5']:.4f} > 20D MA={result_pos['ma20']:.4f} -> '{result_pos['signal']}'")

# Downward trend -> negative momentum
base_down = [2.40, 2.38, 2.35, 2.32, 2.30, 2.28, 2.25, 2.22, 2.20, 2.19,
             2.17, 2.15, 2.13, 2.11, 2.09, 2.08, 2.06, 2.05, 2.04, 2.03]
result_neg = BreakevenMomentum.compute_from_series(base_down)
assert result_neg["signal"] == "negative", f"Expected negative momentum, got {result_neg['signal']}"
print(f"[OK] Breakeven momentum: falling series -> 5D MA < 20D MA -> '{result_neg['signal']}'")

# --- Vol Risk Premium: VIX > realized_vol -> positive VRP ---
vrp_result = compute_vrp(vix=18.5, realized_vol=13.2)
assert vrp_result["vrp"] > 0, f"VRP should be positive when VIX > realized_vol"
assert vrp_result["signal"] == "positive_vrp"
expected_vrp = 18.5 - 13.2
assert abs(vrp_result["vrp"] - expected_vrp) < 0.01
print(f"[OK] VRP: VIX=18.5 - realized_vol=13.2 = {vrp_result['vrp']} -> '{vrp_result['signal']}'")

# VIX < realized_vol -> negative VRP
vrp_neg = compute_vrp(vix=10.0, realized_vol=15.0)
assert vrp_neg["vrp"] < 0
assert vrp_neg["signal"] == "negative_vrp"
print(f"[OK] VRP negative when realized_vol > VIX: {vrp_neg['vrp']}")

print("\n[PASS] dim_048: inflation_vix_analytics -- VIX term structure, regime, momentum, VRP all verified")
PYEOF
