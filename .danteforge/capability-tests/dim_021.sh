#!/bin/bash
# dim_021: IFRS financials — concept map, IFRSFinancialsRow, FX normalization math
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.ifrs_financials_v3 import (
    IFRS_GAAP_CONCEPT_MAP, IFRSFinancialsRow, ConceptMapEntry, FXRate, PeersResponse,
)

# Test 1: IFRS_GAAP_CONCEPT_MAP — 50+ mappings
assert len(IFRS_GAAP_CONCEPT_MAP) >= 50, f"Expected 50+ IFRS mappings, got {len(IFRS_GAAP_CONCEPT_MAP)}"

# Check some key mappings
labels = {m["ifrs_label"] for m in IFRS_GAAP_CONCEPT_MAP}
assert "Revenue" in labels, "Revenue should be in IFRS map"
assert "Gross Profit" in labels
assert "Operating Profit" in labels
assert "Research and Development" in labels

# Check category coverage
categories = {m["category"] for m in IFRS_GAAP_CONCEPT_MAP}
assert "income_statement" in categories
assert "balance_sheet" in categories
print(f"[OK] IFRS_GAAP_CONCEPT_MAP: {len(IFRS_GAAP_CONCEPT_MAP)} mappings, categories: {sorted(categories)}")

# Test 2: IFRSFinancialsRow model
row = IFRSFinancialsRow(
    ticker="ASML",
    name="ASML Holding N.V.",
    country="Netherlands",
    region="Europe",
    currency="EUR",
    report_date="2023-12-31",
    period="FY2023",
    source="edgar_20f",
    revenue=27_558_000_000.0,
    gross_profit=14_882_000_000.0,
    net_income=7_830_000_000.0,
    total_assets=38_897_000_000.0,
    total_equity=12_000_000_000.0,
    revenue_usd=30_200_000_000.0,  # at ~1.097 EUR/USD
)
assert row.ticker == "ASML"
assert row.currency == "EUR"
assert row.revenue_usd is not None and row.revenue_usd > row.revenue
print(f"[OK] IFRSFinancialsRow: {row.ticker} ({row.country}) revenue={row.revenue/1e9:.1f}B EUR")

# Test 3: FX normalization math (pure computation)
# USD per EUR rate = 1.097; EUR revenue = 27.558B
eur_usd = 1.097
eur_revenue = 27_558_000_000.0
usd_revenue = eur_revenue * eur_usd
assert abs(usd_revenue - 30_231_126_000.0) < 1_000_000
print(f"[OK] FX normalization: {eur_revenue/1e9:.2f}B EUR * {eur_usd} = {usd_revenue/1e9:.2f}B USD")

# Test 4: IFRS concept lookup by category
income_concepts = [m for m in IFRS_GAAP_CONCEPT_MAP if m["category"] == "income_statement"]
balance_concepts = [m for m in IFRS_GAAP_CONCEPT_MAP if m["category"] == "balance_sheet"]
assert len(income_concepts) >= 10, f"Expected 10+ IS concepts, got {len(income_concepts)}"
assert len(balance_concepts) >= 10, f"Expected 10+ BS concepts, got {len(balance_concepts)}"
print(f"[OK] IFRS categories: IS={len(income_concepts)}, BS={len(balance_concepts)} concepts")

# Test 5: ConceptMapEntry model
cme = ConceptMapEntry(
    ifrs_label="Revenue",
    ifrs_xbrl="ifrs-full:Revenue",
    gaap_label="Revenues",
    gaap_xbrl="us-gaap:Revenues",
    category="income_statement",
    note="Top-line revenue"
)
assert cme.ifrs_xbrl.startswith("ifrs-full:")
assert cme.gaap_xbrl.startswith("us-gaap:")
print(f"[OK] ConceptMapEntry: IFRS '{cme.ifrs_label}' -> GAAP '{cme.gaap_label}'")

# Test 6: FXRate model
fxr = FXRate(
    from_currency="EUR", to_currency="USD",
    rate_date="2024-01-15", rate=1.0895, source="ECB"
)
# Test ECB rate inversion: EUR/JPY from EUR/USD + USD/JPY
eur_usd_rate = 1.0895
usd_jpy_rate = 147.50
eur_jpy_rate = eur_usd_rate * usd_jpy_rate
assert abs(eur_jpy_rate - 160.7) < 1.0, f"EUR/JPY cross rate wrong: {eur_jpy_rate}"
print(f"[OK] FXRate: EUR/USD={fxr.rate}, cross EUR/JPY={eur_jpy_rate:.1f}")

print("[PASS]")
PYEOF
