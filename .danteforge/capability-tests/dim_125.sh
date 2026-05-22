#!/usr/bin/env bash
# dim_125: Quantitative Patent Analytics and IP Strategy
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import numpy as np

from sentinel.spm.patent_analytics_v3 import (
    Patent, PatentPortfolio, IPMoatScorer, CompetitiveIPAnalysis,
    h_index, tech_hhi, ip_moat_score, rd_efficiency,
)

FAILS = []

def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  [FAIL] " + msg)
    else:
        print("  [OK]   " + msg)

# ---------------------------------------------------------------------------
# Build 10-patent portfolio
# citations=[15,12,8,8,5,4,3,2,1,1], IPC=[G06F,G06F,H04L,H04L,G06N,G06N,A61K,H04W,G06F,H04L]
# ---------------------------------------------------------------------------
citations   = [15, 12, 8, 8, 5, 4, 3, 2, 1, 1]
ipc_classes = ['G06F','G06F','H04L','H04L','G06N','G06N','A61K','H04W','G06F','H04L']

patents = [
    Patent(
        patent_id=f"US{1000+i}",
        title=f"Patent {i}",
        filing_year=2015 + i,
        grant_year=2016 + i,
        ipc_class=ipc_classes[i],
        citations_received=citations[i],
        citations_made=5,
        n_jurisdictions=10 + i,
        claims=20 + i,
    )
    for i in range(10)
]

portfolio = PatentPortfolio(patents=patents, company="AlphaCorp", rd_spend_millions=50.0)

print("\n--- h-index ---")
hi = h_index(citations)
print(f"  h_index = {hi}")
# citations sorted desc: [15,12,8,8,5,4,3,2,1,1]
# 5 patents have >= 5 citations (15,12,8,8,5) -> h=5
check(hi == 5, f"h_index == 5 (got {hi}): 5 patents each with >= 5 citations")

print("\n--- PatentPortfolio basic properties ---")
check(portfolio.total_patents == 10,
      f"total_patents == 10 (got {portfolio.total_patents})")
check(portfolio.total_citations == sum(citations),
      f"total_citations == {sum(citations)} (got {portfolio.total_citations})")
check(portfolio.h_index() == 5,
      f"portfolio.h_index() == 5 (got {portfolio.h_index()})")

print("\n--- tech_hhi / tech_diversity ---")
hhi = tech_hhi(ipc_classes)
print(f"  tech_hhi = {hhi:.4f}")
check(hhi > 0, f"tech_hhi > 0 (got {hhi:.4f}): portfolio concentrated in G06F + H04L")

div = 1.0 - hhi
print(f"  tech_diversity = {div:.4f}")
check(div < 1.0, f"tech_diversity < 1.0 (got {div:.4f}): not perfectly diverse")
check(div > 0.0, f"tech_diversity > 0.0 (got {div:.4f}): multiple IPC classes exist")

portfolio_hhi = portfolio.tech_hhi()
portfolio_div = portfolio.tech_diversity()
check(abs(portfolio_hhi - hhi) < 1e-9,
      f"portfolio.tech_hhi() matches module-level tech_hhi()")
check(abs(portfolio_div - div) < 1e-9,
      "portfolio.tech_diversity() == 1 - tech_hhi()")

print("\n--- R&D efficiency ---")
eff = portfolio.rd_efficiency()
expected_patent_eff = 10 / 50.0   # 0.2 patents/M$
print(f"  patent_efficiency = {eff['patent_efficiency']:.4f} (expected {expected_patent_eff:.4f})")
check(abs(eff['patent_efficiency'] - expected_patent_eff) < 1e-9,
      f"patent_efficiency == 10/50 = 0.2 (got {eff['patent_efficiency']:.4f})")
check(eff['citation_efficiency'] > 0,
      f"citation_efficiency > 0 (got {eff['citation_efficiency']:.4f})")

eff2 = rd_efficiency(10, sum(citations), 50.0)
check(abs(eff2['patent_efficiency'] - 0.2) < 1e-9,
      f"rd_efficiency() convenience: patent_efficiency == 0.2")

print("\n--- IPMoatScorer ---")
scorer = IPMoatScorer(portfolio, current_year=2025)
moat = scorer.moat_score()
print(f"  moat_score = {moat:.4f}")
check(0 < moat < 1, f"moat_score in (0, 1) (got {moat:.4f})")

breakdown = scorer.moat_breakdown()
for key in ('citation_quality', 'tech_diversity', 'family_breadth',
            'forward_citation_rate', 'moat_score'):
    check(key in breakdown, f"moat_breakdown contains '{key}'")
    check(0 <= breakdown[key] <= 1,
          f"moat_breakdown['{key}'] in [0,1] (got {breakdown[key]:.4f})")

print("\n--- ip_moat_score convenience function ---")
moat2 = ip_moat_score(portfolio, current_year=2025)
check(moat2 > 0, f"ip_moat_score > 0 (got {moat2:.4f})")
check(abs(moat2 - moat) < 1e-9, "ip_moat_score matches IPMoatScorer.moat_score()")

print("\n--- technology_overlap (CompetitiveIPAnalysis) ---")
patents2 = [
    Patent("US9001","Shared IPC",2018,2019,"H04L",6,3,8,15),
    Patent("US9002","Shared IPC",2019,2020,"G06F",4,2,5,12),
    Patent("US9003","Unique IPC",2020,2021,"B60W",2,1,4,10),
]
portfolio2 = PatentPortfolio(patents=patents2, company="BetaCorp", rd_spend_millions=30.0)

cia = CompetitiveIPAnalysis()
overlap = cia.technology_overlap(portfolio, portfolio2)
print(f"  technology_overlap = {overlap:.4f}")
check(0 < overlap <= 1,
      f"technology_overlap in (0, 1] (got {overlap:.4f}): G06F and H04L are shared")

print("\n--- CompetitiveIPAnalysis.compare ---")
comparison = cia.compare([portfolio, portfolio2])
check("AlphaCorp" in comparison, "compare result has AlphaCorp")
check("BetaCorp" in comparison, "compare result has BetaCorp")
for key in ("h_index", "total_patents", "moat_score", "tech_focus"):
    check(key in comparison["AlphaCorp"], f"compare['AlphaCorp'] has '{key}'")

print("\n--- citation_network_centrality ---")
centrality = cia.citation_network_centrality([portfolio, portfolio2])
check("AlphaCorp" in centrality, "centrality has AlphaCorp")
check(centrality["AlphaCorp"] == 1.0,
      f"AlphaCorp has max centrality 1.0 (got {centrality['AlphaCorp']:.4f})")
check(0 < centrality["BetaCorp"] <= 1,
      f"BetaCorp centrality in (0,1] (got {centrality['BetaCorp']:.4f})")

# Final result
print()
if FAILS:
    print("[FAIL] dim_125: %d check(s) failed:" % len(FAILS))
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
else:
    print("[PASS] dim_125: Patent analytics")
PYEOF
