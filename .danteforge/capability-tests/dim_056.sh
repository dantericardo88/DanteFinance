#!/bin/bash
# dim_056: Query Expander v3 — financial ontology synonym expansion (no network)
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.query_expander_v3 import (
    FinancialOntology, QueryExpander, ExpandedQuery,
)

# 1. FinancialOntology structure
ont = FinancialOntology()
assert "total_revenue" in ont.METRIC_SYNONYMS
assert "net_income" in ont.METRIC_SYNONYMS
assert "eps" in ont.METRIC_SYNONYMS
assert len(ont.METRIC_SYNONYMS) >= 20, f"Expected 20+ metrics, got {len(ont.METRIC_SYNONYMS)}"
print(f"[OK] FinancialOntology.METRIC_SYNONYMS has {len(ont.METRIC_SYNONYMS)} canonical metrics")

# 2. Ticker aliases
assert "apple" in ont.TICKER_ALIASES
assert ont.TICKER_ALIASES["apple"] == "AAPL"
assert "microsoft" in ont.TICKER_ALIASES
assert ont.TICKER_ALIASES["microsoft"] == "MSFT"
assert "google" in ont.TICKER_ALIASES or "alphabet" in ont.TICKER_ALIASES
print(f"[OK] FinancialOntology.TICKER_ALIASES has {len(ont.TICKER_ALIASES)} entries")

# 3. Metric abbreviations
assert "pe" in ont.METRIC_ABBREVIATIONS
assert ont.METRIC_ABBREVIATIONS["pe"] == "pe_ratio"
assert "roe" in ont.METRIC_ABBREVIATIONS
assert "fcf" in ont.METRIC_ABBREVIATIONS
print(f"[OK] FinancialOntology.METRIC_ABBREVIATIONS has {len(ont.METRIC_ABBREVIATIONS)} entries")

# 4. Concept synonyms
assert "liquidity" in ont.CONCEPT_SYNONYMS
assert "leverage" in ont.CONCEPT_SYNONYMS
assert "momentum" in ont.CONCEPT_SYNONYMS
assert "dividend" in ont.CONCEPT_SYNONYMS
print(f"[OK] FinancialOntology.CONCEPT_SYNONYMS has {len(ont.CONCEPT_SYNONYMS)} concepts")

# 5. reverse mapping: alias -> canonical
# "revenue" should map to "total_revenue"
assert "revenue" in ont.METRIC_SYNONYMS["total_revenue"] or \
       any("revenue" in syns for syns in ont.METRIC_SYNONYMS.values())
print("[OK] Synonym lists contain expected surface forms")

# 6. QueryExpander.expand()
qe = QueryExpander()
result = qe.expand("show me tech stocks with low PE and high FCF yield")
assert isinstance(result, ExpandedQuery), f"Expected ExpandedQuery, got {type(result)}"
assert hasattr(result, 'original')
assert hasattr(result, 'expanded_terms')
assert hasattr(result, 'normalized_query')
assert hasattr(result, 'synonyms_used')
assert hasattr(result, 'detected_tickers')
assert hasattr(result, 'confidence')
assert hasattr(result, 'intent')
assert result.original == "show me tech stocks with low PE and high FCF yield"
print(f"[OK] QueryExpander.expand() returns ExpandedQuery with all required fields")
print(f"     expanded_terms={result.expanded_terms[:4]}, confidence={result.confidence}")

# 7. Ticker detection
result2 = qe.expand("What is the PE ratio of Apple and Microsoft?")
assert isinstance(result2, ExpandedQuery)
# Should detect Apple -> AAPL or Microsoft -> MSFT in detected_tickers
print(f"[OK] expand('Apple and Microsoft'): detected_tickers={result2.detected_tickers}")

# 8. QueryExpander is deterministic
r_a = qe.expand("revenue growth stocks")
r_b = qe.expand("revenue growth stocks")
assert r_a.expanded_terms == r_b.expanded_terms
print("[OK] QueryExpander is deterministic")

print("\n[PASS] dim_056: Query expander financial ontology verified")
PYEOF
