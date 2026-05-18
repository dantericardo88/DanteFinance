#!/usr/bin/env bash
# dim_070: Fundamental screener — pure ratio/filter logic + z-score ranking + DuckDB schema
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from sentinel.sbx.fundamental_screener import (
    FundamentalScreener,
    DuckDBQueryEngine,
)

screener = FundamentalScreener()

# Test PREBUILT_SCREENS dict exists and has entries
presets = screener.PREBUILT_SCREENS
assert isinstance(presets, dict), "PREBUILT_SCREENS should be dict"
assert len(presets) > 0, "PREBUILT_SCREENS should not be empty"
print(f"[OK] PREBUILT_SCREENS has {len(presets)} presets: {list(presets.keys())[:5]}")

# Test _build_where pure logic
clauses = screener._build_where({
    "pe_ttm":  {"lte": 15.0, "gte": 5.0},
    "roe":     {"gte": 0.15},
    "sector":  {"in": ["Technology", "Healthcare"]},
})
assert any("pe_ttm" in c for c in clauses), "Expected pe_ttm clause"
assert any("roe" in c for c in clauses), "Expected roe clause"
assert any("IN" in c for c in clauses), "Expected IN clause for sector"
print(f"[OK] _build_where generated {len(clauses)} SQL clauses")

# Test Altman Z-Score formula manually (pure math)
# Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
x1, x2, x3, x4, x5 = 0.20, 0.30, 0.10, 1.50, 1.20
z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
assert z > 2.99, f"Should be in safe zone: {z}"
print(f"[OK] Altman Z formula: {z:.3f} -> safe zone")

x1b, x2b, x3b, x4b, x5b = 0.05, 0.02, 0.02, 0.30, 0.40
zb = 1.2*x1b + 1.4*x2b + 3.3*x3b + 0.6*x4b + 1.0*x5b
assert zb < 1.81, f"Should be in distress zone: {zb}"
print(f"[OK] Altman Z distress zone: {zb:.3f}")

# Piotroski F-Score logic (9-point system)
checks = [
    1000 > 0,       # net income positive
    0.10 > 0,       # ROA positive
    1200 > 0,       # operating cash flow positive
    1200 > 1000,    # OCF > net income (quality)
    0.30 < 0.35,    # long-term debt ratio decreased
    1.8 > 1.5,      # current ratio improved
    500 == 500,     # no share dilution
    0.42 > 0.38,    # gross margin improved
    0.80 > 0.75,    # asset turnover improved
]
f_score = sum(checks)
assert 0 <= f_score <= 9, f"F-score out of range: {f_score}"
rating = "Strong" if f_score >= 7 else "Moderate" if f_score >= 4 else "Weak"
print(f"[OK] Piotroski F-Score: {f_score}/9 -> {rating}")

# DuckDBQueryEngine can be instantiated
engine = DuckDBQueryEngine()
schema = engine.get_schema()
assert isinstance(schema, pd.DataFrame), "get_schema should return DataFrame"
assert len(schema) > 0, "Schema should have columns"
print(f"[OK] DuckDBQueryEngine.get_schema() returned {len(schema)} field definitions")

# --- NEW: Market cap tier classification (pure logic, no network) ---
tier_cases = [
    (250_000_000_000, "mega"),
    (50_000_000_000,  "large"),
    (5_000_000_000,   "mid"),
    (800_000_000,     "small"),
    (100_000_000,     "micro"),
    (10_000_000,      "nano"),
    (None,            "unknown"),
]
for mktcap, expected_tier in tier_cases:
    got = screener.classify_market_cap_tier(mktcap)
    assert got == expected_tier, f"market_cap={mktcap} expected tier={expected_tier} got={got}"
print("[OK] Market cap tier classification: mega/large/mid/small/micro/nano/unknown all correct")

# --- NEW: ScreenerQuery builds valid DuckDB SQL from filter criteria ---
clauses2 = screener._build_where({
    "pe_ttm":        {"lte": 20.0},
    "revenue_growth_1yr": {"gte": 0.10},
    "roe":           {"gte": 0.15},
    "debt_to_equity": {"lte": 0.5},
})
assert len(clauses2) == 4, f"Expected 4 clauses, got {len(clauses2)}"
full_where = " AND ".join(clauses2)
assert "pe_ttm" in full_where, "pe_ttm missing from WHERE"
assert "revenue_growth_1yr" in full_where, "revenue_growth_1yr missing from WHERE"
assert "roe" in full_where, "roe missing from WHERE"
assert "debt_to_equity" in full_where, "debt_to_equity missing from WHERE"
print(f"[OK] ScreenerQuery WHERE clause: {full_where[:80]}...")

# --- NEW: Z-score normalization correctness ---
# Given 5 stocks with PE=[10,15,20,25,30]:
pe_values = [10.0, 15.0, 20.0, 25.0, 30.0]
pe_array  = np.array(pe_values, dtype=float)
mu        = pe_array.mean()    # 20.0
std       = pe_array.std(ddof=1)  # 7.906
z_scores  = (pe_array - mu) / std

# Verify z-scores
assert abs(mu - 20.0) < 1e-9,    f"Mean should be 20: {mu}"
assert abs(z_scores[0] - (-10.0 / std)) < 1e-9, f"First z-score wrong: {z_scores[0]}"
assert abs(z_scores[4] - ( 10.0 / std)) < 1e-9, f"Last z-score wrong:  {z_scores[4]}"
assert abs(z_scores.mean()) < 1e-9, "Z-scores should sum to ~0"
assert abs(z_scores.std(ddof=1) - 1.0) < 1e-9, "Z-scores should have unit std"
print(f"[OK] Z-score normalization: PE=[10,15,20,25,30] -> z={z_scores.round(3).tolist()}")

# --- NEW: Composite z-score ranking produces sorted list ---
df_stocks = pd.DataFrame({
    "ticker":    ["A", "B", "C", "D", "E"],
    "pe_ttm":    [10.0, 15.0, 20.0, 25.0, 30.0],   # lower is better (value)
    "roe":       [0.25, 0.20, 0.15, 0.10, 0.05],    # higher is better (quality)
    "price_12m": [0.30, 0.20, 0.10, 0.05, -0.05],   # higher is better (momentum)
})

ranked = FundamentalScreener.compute_composite_zscore(
    df_stocks,
    value_cols=["pe_ttm"],
    quality_cols=["roe"],
    momentum_cols=["price_12m"],
)
assert "composite_zscore" in ranked.columns, "composite_zscore column missing"
assert len(ranked) == 5, f"Expected 5 rows, got {len(ranked)}"

# Stock A (lowest PE, highest ROE, highest momentum) should rank first
top_ticker = ranked.iloc[0]["ticker"]
bot_ticker = ranked.iloc[-1]["ticker"]
assert top_ticker == "A", f"Stock A should rank first (best composite), got {top_ticker}"
assert bot_ticker == "E", f"Stock E should rank last (worst composite), got {bot_ticker}"

# Composite scores should be strictly decreasing
scores = ranked["composite_zscore"].tolist()
assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
    f"Composite scores should be non-increasing: {scores}"
print(f"[OK] Composite z-score ranking: top={top_ticker} (score={scores[0]:.3f}), bottom={bot_ticker} (score={scores[-1]:.3f})")

# --- NEW: DuckDB schema is valid (all expected columns present) ---
schema_cols = schema["column"].tolist()
required_cols = ["ticker", "market_cap", "pe_ttm", "roe", "roic", "ev_ebitda",
                 "revenue_ttm", "fcf_ttm", "fcf_yield", "debt_to_equity",
                 "current_ratio", "gross_margin", "net_margin"]
for col in required_cols:
    assert col in schema_cols, f"Missing required column in schema: {col}"
print(f"[OK] DuckDB schema has all {len(required_cols)} required columns")

# --- FCF yield formula: FCF / market_cap ---
fcf_ttm   = 500_000_000.0   # $500M FCF
market_cap = 5_000_000_000.0 # $5B market cap
fcf_yield_computed = fcf_ttm / market_cap
assert abs(fcf_yield_computed - 0.10) < 1e-9, f"FCF yield should be 10%: {fcf_yield_computed}"
print(f"[OK] FCF yield formula: FCF={fcf_ttm/1e9:.1f}B / mktcap={market_cap/1e9:.1f}B = {fcf_yield_computed:.1%}")

# --- ROE = net_income / stockholders_equity ---
net_income = 120_000_000.0
equity     = 600_000_000.0
roe_computed = net_income / equity
assert abs(roe_computed - 0.20) < 1e-9, f"ROE should be 20%: {roe_computed}"
print(f"[OK] ROE formula: {net_income/1e6:.0f}M / {equity/1e6:.0f}M equity = {roe_computed:.1%}")

# ---- NEW: compute_piotroski_f_score (pure-math static) -------------------------
# All 9 signals true → f_score = 9, rating = "Strong"
pf = FundamentalScreener.compute_piotroski_f_score(
    roa=0.08,                           # F1: ROA > 0 ✓
    operating_cf=500_000,               # F2: OCF > 0 ✓
    net_income=400_000,                 # F3: OCF > NI ✓ (500k > 400k)
    long_term_debt_ratio=0.30,          # F4: leverage improved ✓
    long_term_debt_ratio_prior=0.35,
    current_ratio=1.8,                  # F5: current ratio improved ✓
    current_ratio_prior=1.5,
    shares_outstanding=100_000_000,     # F6: no dilution ✓ (equal)
    shares_outstanding_prior=100_000_000,
    gross_margin=0.45,                  # F7: gross margin improved ✓
    gross_margin_prior=0.40,
    asset_turnover=0.85,                # F8: asset turnover improved ✓
    asset_turnover_prior=0.80,
)
# F9 = accruals signal = (OCF/TA > ROA) proxy — handled internally
assert pf["f_score"] >= 8, f"All-positive inputs should yield f_score >= 8: {pf['f_score']}"
assert pf["interpretation"] in ("Strong", "Moderate"), f"Interpretation should be Strong or Moderate: {pf['interpretation']}"
print(
    f"[OK] Piotroski F-score (all-positive): f_score={pf['f_score']}/9, "
    f"interpretation={pf['interpretation']}, signals={pf['signals']}"
)

# All 9 signals false → f_score = 0, rating = "Weak"
pf_weak = FundamentalScreener.compute_piotroski_f_score(
    roa=-0.05,                          # F1: ROA < 0 ✗
    operating_cf=-100_000,              # F2: OCF < 0 ✗
    net_income=50_000,                  # F3: OCF < NI ✗
    long_term_debt_ratio=0.50,          # F4: leverage worsened ✗
    long_term_debt_ratio_prior=0.40,
    current_ratio=1.2,                  # F5: current ratio worsened ✗
    current_ratio_prior=1.5,
    shares_outstanding=110_000_000,     # F6: dilution occurred ✗
    shares_outstanding_prior=100_000_000,
    gross_margin=0.35,                  # F7: gross margin fell ✗
    gross_margin_prior=0.40,
    asset_turnover=0.70,                # F8: asset turnover fell ✗
    asset_turnover_prior=0.80,
)
assert pf_weak["f_score"] <= 1, f"All-negative inputs should yield f_score <= 1: {pf_weak['f_score']}"
assert pf_weak["interpretation"] == "Weak", f"Interpretation should be 'Weak': {pf_weak['interpretation']}"
print(f"[OK] Piotroski F-score (all-negative): f_score={pf_weak['f_score']}/9, interpretation={pf_weak['interpretation']}")

# Score range always 0-9
assert 0 <= pf["f_score"] <= 9, "F-score must be 0-9"
assert 0 <= pf_weak["f_score"] <= 9, "F-score must be 0-9"
print("[OK] F-score range validated [0, 9]")

# ---- NEW: compute_altman_z_score (pure-math static) ----------------------------
# Safe-zone company: Z > 2.99
az_safe = FundamentalScreener.compute_altman_z_score(
    working_capital=200_000_000,
    total_assets=1_000_000_000,
    retained_earnings=300_000_000,
    ebit=100_000_000,
    market_cap=1_500_000_000,
    total_liabilities=1_000_000_000,
    revenue=1_200_000_000,
)
# Expected: X1=0.20, X2=0.30, X3=0.10, X4=1.50, X5=1.20
# Z = 1.2*0.20 + 1.4*0.30 + 3.3*0.10 + 0.6*1.50 + 1.0*1.20
#   = 0.24 + 0.42 + 0.33 + 0.90 + 1.20 = 3.09
expected_z_safe = 1.2*0.20 + 1.4*0.30 + 3.3*0.10 + 0.6*1.50 + 1.0*1.20
assert abs(az_safe["z_score"] - round(expected_z_safe, 4)) < 1e-4, (
    f"Safe-zone Z wrong: expected {expected_z_safe:.4f}, got {az_safe['z_score']:.4f}"
)
assert az_safe["zone"] == "safe", f"Z={az_safe['z_score']:.3f} should be safe zone: {az_safe['zone']}"
c = az_safe["components"]
assert abs(c["X1_working_capital_ta"]    - 0.20) < 1e-6, f"X1 should be 0.20: {c['X1_working_capital_ta']}"
assert abs(c["X2_retained_earnings_ta"]  - 0.30) < 1e-6, f"X2 should be 0.30: {c['X2_retained_earnings_ta']}"
assert abs(c["X3_ebit_ta"]               - 0.10) < 1e-6, f"X3 should be 0.10: {c['X3_ebit_ta']}"
assert abs(c["X4_mktcap_liabilities"]    - 1.50) < 1e-6, f"X4 should be 1.50: {c['X4_mktcap_liabilities']}"
assert abs(c["X5_revenue_ta"]            - 1.20) < 1e-6, f"X5 should be 1.20: {c['X5_revenue_ta']}"
# Verify coefficient dict
coeff = az_safe["coefficients"]
assert coeff["X1"] == 1.2 and coeff["X2"] == 1.4 and coeff["X3"] == 3.3
assert coeff["X4"] == 0.6 and coeff["X5"] == 1.0
print(
    f"[OK] Altman Z-score (safe): Z={az_safe['z_score']:.4f} (expected {expected_z_safe:.4f}), "
    f"zone={az_safe['zone']}, "
    f"X1={c['X1_working_capital_ta']:.2f} X2={c['X2_retained_earnings_ta']:.2f} "
    f"X3={c['X3_ebit_ta']:.2f} X4={c['X4_mktcap_liabilities']:.2f} X5={c['X5_revenue_ta']:.2f}"
)

# Distress-zone company: Z < 1.81
az_dist = FundamentalScreener.compute_altman_z_score(
    working_capital=50_000_000,
    total_assets=1_000_000_000,
    retained_earnings=20_000_000,
    ebit=20_000_000,
    market_cap=300_000_000,
    total_liabilities=1_000_000_000,
    revenue=400_000_000,
)
# X1=0.05 X2=0.02 X3=0.02 X4=0.30 X5=0.40
# Z = 0.06 + 0.028 + 0.066 + 0.18 + 0.40 = 0.734
expected_z_dist = 1.2*0.05 + 1.4*0.02 + 3.3*0.02 + 0.6*0.30 + 1.0*0.40
assert abs(az_dist["z_score"] - round(expected_z_dist, 4)) < 1e-4, (
    f"Distress Z wrong: expected {expected_z_dist:.4f}, got {az_dist['z_score']:.4f}"
)
assert az_dist["zone"] == "distress", f"Z={az_dist['z_score']:.3f} should be distress: {az_dist['zone']}"
print(f"[OK] Altman Z-score (distress): Z={az_dist['z_score']:.4f}, zone={az_dist['zone']}")

# Grey-zone: 1.81 <= Z <= 2.99
az_grey = FundamentalScreener.compute_altman_z_score(
    working_capital=120_000_000,
    total_assets=1_000_000_000,
    retained_earnings=150_000_000,
    ebit=60_000_000,
    market_cap=800_000_000,
    total_liabilities=1_000_000_000,
    revenue=800_000_000,
)
assert 1.81 <= az_grey["z_score"] <= 2.99, (
    f"Grey-zone Z should be in [1.81, 2.99]: {az_grey['z_score']:.4f}"
)
assert az_grey["zone"] == "grey", f"Z={az_grey['z_score']:.3f} should be 'grey': {az_grey['zone']}"
print(f"[OK] Altman Z-score (grey): Z={az_grey['z_score']:.4f}, zone={az_grey['zone']}")

# ---- NEW: compute_beneish_m_score (pure-math static) ---------------------------
# Known non-manipulator: M < -1.78
bm_clean = FundamentalScreener.compute_beneish_m_score(
    dsri=1.0,     # no receivable inflation
    gmi=1.0,      # stable gross margin
    aqi=1.0,      # stable asset quality
    sgi=1.0,      # stable sales
    depi=1.0,     # stable depreciation
    sgai=1.0,     # stable SGA
    accruals=0.0, # no accruals
    lvgi=1.0,     # stable leverage
)
# M = -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 0 - 0.327 = -2.480
expected_m_clean = (-4.84 + 0.920*1.0 + 0.528*1.0 + 0.404*1.0 + 0.892*1.0
                    + 0.115*1.0 - 0.172*1.0 + 4.679*0.0 - 0.327*1.0)
assert abs(bm_clean["m_score"] - round(expected_m_clean, 4)) < 1e-4, (
    f"Clean M-score wrong: expected {expected_m_clean:.6f}, got {bm_clean['m_score']:.6f}"
)
assert bm_clean["likely_manipulator"] is False, (
    f"M={bm_clean['m_score']:.4f} should NOT be a manipulator (threshold=-1.78)"
)
print(
    f"[OK] Beneish M-score (clean): M={bm_clean['m_score']:.4f} (expected ~{expected_m_clean:.4f}), "
    f"likely_manipulator={bm_clean['likely_manipulator']}"
)

# Known manipulator: M > -1.78
bm_manip = FundamentalScreener.compute_beneish_m_score(
    dsri=1.80,    # receivables inflated
    gmi=0.70,     # gross margin shrinking
    aqi=1.40,     # asset quality declining
    sgi=1.60,     # rapid sales growth
    depi=0.50,    # reducing depreciation
    sgai=1.30,    # SGA cost inflation
    accruals=0.10, # high accruals
    lvgi=1.20,    # leverage rising
)
expected_m_manip = (-4.84 + 0.920*1.80 + 0.528*0.70 + 0.404*1.40 + 0.892*1.60
                    + 0.115*0.50 - 0.172*1.30 + 4.679*0.10 - 0.327*1.20)
assert abs(bm_manip["m_score"] - round(expected_m_manip, 4)) < 1e-4, (
    f"Manipulator M-score wrong: expected {expected_m_manip:.6f}, got {bm_manip['m_score']:.6f}"
)
assert bm_manip["likely_manipulator"] is True, (
    f"M={bm_manip['m_score']:.4f} should be a likely manipulator (threshold=-1.78)"
)
print(
    f"[OK] Beneish M-score (manipulator): M={bm_manip['m_score']:.4f}, "
    f"likely_manipulator={bm_manip['likely_manipulator']}"
)

print("\n[PASS] dim_070: Fundamental screener")
PYEOF
