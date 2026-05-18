#!/bin/bash
# dim_015: Standardized cash flow — FCF computation, quality metrics (pure computation)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np

from sentinel.sfe.standardized_financials_v3 import XBRLConceptMapper

# Test 1: Cash flow items
mapper = XBRLConceptMapper()
cf_items = mapper.get_statement_items("cashflow")
assert "OperatingCF" in cf_items
assert "CapEx" in cf_items
assert "FinancingCF" in cf_items
print(f"[OK] Cash flow has {len(cf_items)} standard items")

# Test 2: FCF = OperatingCF - abs(CapEx) — pure computation
df = pd.DataFrame({
    "OperatingCF":     [100_000, 85_000, 70_000],
    "CapEx":           [-25_000, -20_000, -18_000],  # negative in XBRL
    "InvestingCF":     [-30_000, -25_000, -20_000],
    "FinancingCF":     [-20_000, -15_000, -10_000],
    "DividendsPaid":   [-5_000,  -4_500,  -4_000],
    "ShareRepurchases":[-10_000, -8_000,  -5_000],
    "DA":              [10_000,  9_000,   8_000],
    "SBC":             [5_000,   4_000,   3_000],
    "NetCashChange":   [50_000,  45_000,  42_000],
}, index=["FY2024", "FY2023", "FY2022"])

# FCF = OperatingCF - abs(CapEx)
df["FCF"] = df["OperatingCF"].fillna(0) - df["CapEx"].abs().fillna(0)
assert df.loc["FY2024", "FCF"] == 75_000, f"FCF wrong: {df.loc['FY2024', 'FCF']}"
assert df.loc["FY2023", "FCF"] == 65_000
print(f"[OK] FCF: FY2024={df.loc['FY2024','FCF']:,}, FY2023={df.loc['FY2023','FCF']:,}")

# Test 3: Cash conversion ratio (CFO / Net Income)
synthetic_net_income = pd.Series([80_000, 65_000, 55_000], index=df.index)
df["CashConversionRatio"] = df["OperatingCF"] / synthetic_net_income.replace(0, np.nan)
assert df.loc["FY2024", "CashConversionRatio"] == 100_000 / 80_000
print(f"[OK] Cash conversion FY2024 = {df.loc['FY2024','CashConversionRatio']:.3f}")

# Test 4: CapEx intensity (CapEx / Revenue)
synthetic_revenue = pd.Series([500_000, 450_000, 400_000], index=df.index)
capex_intensity = df["CapEx"].abs() / synthetic_revenue
assert abs(capex_intensity.iloc[0] - 0.05) < 1e-6  # 25,000 / 500,000 = 5%
print(f"[OK] CapEx intensity FY2024 = {capex_intensity.iloc[0]:.1%}")

# Test 5: XBRL concepts for cash flow items
cfo_concepts = mapper.get_all_concepts_for_item("OperatingCF")
assert len(cfo_concepts) >= 1, "OperatingCF should have at least 1 XBRL concept"
capex_concepts = mapper.get_all_concepts_for_item("CapEx")
assert len(capex_concepts) >= 1, "CapEx should have at least 1 XBRL concept"
print(f"[OK] XBRL: OperatingCF={len(cfo_concepts)} concepts, CapEx={len(capex_concepts)} concepts")

# Test 6: SBC % of CFO
df["SBC_pct_CFO"] = df["SBC"] / df["OperatingCF"] * 100
expected_sbc_pct = 5_000 / 100_000 * 100
assert abs(df.loc["FY2024", "SBC_pct_CFO"] - expected_sbc_pct) < 1e-6
print(f"[OK] SBC % of CFO FY2024 = {df.loc['FY2024','SBC_pct_CFO']:.1f}%")

print("[PASS]")
PYEOF
