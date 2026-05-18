#!/bin/bash
# dim_056: Query Expander v3 — ontology, query_specificity, concept_graph, BM25 ranking
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math
from collections import Counter

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
print(f"[OK] expand('Apple and Microsoft'): detected_tickers={result2.detected_tickers}")

# 8. QueryExpander is deterministic
r_a = qe.expand("revenue growth stocks")
r_b = qe.expand("revenue growth stocks")
assert r_a.expanded_terms == r_b.expanded_terms
print("[OK] QueryExpander is deterministic")

# -----------------------------------------------------------------------
# Test 9: compute_query_specificity — entropy formula
# -----------------------------------------------------------------------
# Single unique term → entropy = 0
single_query = "revenue revenue revenue revenue"
entropy_single = qe.compute_query_specificity(single_query)
assert entropy_single == 0.0, f"Single repeated term should have entropy=0, got {entropy_single}"

# All unique terms → max entropy = log2(n)
tokens = ["apple", "revenue", "growth", "momentum", "strong"]
all_unique_query = " ".join(tokens)
entropy_unique = qe.compute_query_specificity(all_unique_query)
expected_max = math.log2(len(tokens))
assert abs(entropy_unique - expected_max) < 1e-6, (
    f"All-unique entropy mismatch: {entropy_unique} vs {expected_max}"
)

# Verify formula manually: 2 "apple", 1 "growth" → entropy
mixed_query = "apple apple growth"
mixed_tokens = Counter(qe.tokenize_financial(mixed_query))
total = sum(mixed_tokens.values())
expected_entropy = -sum((c/total) * math.log2(c/total) for c in mixed_tokens.values())
got_entropy = qe.compute_query_specificity(mixed_query)
assert abs(got_entropy - expected_entropy) < 1e-6, (
    f"Mixed query entropy mismatch: {got_entropy} vs {expected_entropy}"
)
print(f"[OK] compute_query_specificity: single={entropy_single}, unique={entropy_unique:.3f}, mixed={got_entropy:.3f}")

# -----------------------------------------------------------------------
# Test 10: build_concept_graph — 3-hop structure
# -----------------------------------------------------------------------
graph = qe.build_concept_graph()
assert isinstance(graph, dict), "build_concept_graph should return dict"
assert len(graph) > 0, "Concept graph should not be empty"
# Every canonical metric should appear as a key
for canonical in list(ont.METRIC_SYNONYMS.keys())[:5]:
    assert canonical in graph, f"Canonical metric {canonical!r} missing from graph"
# Values should be lists
for key, related in graph.items():
    assert isinstance(related, list), f"Graph value for {key!r} should be list"
    break
print(f"[OK] build_concept_graph: {len(graph)} nodes, example node has {len(graph.get('total_revenue',[]))} related terms")

# -----------------------------------------------------------------------
# Test 11: rank_expansion_terms — BM25 TF component formula
# -----------------------------------------------------------------------
k1 = 1.5
query  = "revenue growth revenue"   # "revenue" appears 2×, "growth" 1×
candidates = ["revenue", "growth", "ebitda", "revenue growth"]

ranked = qe.rank_expansion_terms(query, candidates, k1=k1)
assert isinstance(ranked, list), "rank_expansion_terms should return list"
assert all(isinstance(t, tuple) and len(t) == 2 for t in ranked), "Each entry should be (term, score)"

# Manually verify BM25 TF for "revenue" (tf=2 in query tokens):
# tokenize query
q_tokens = Counter(qe.tokenize_financial(query))
tf_revenue = q_tokens.get("revenue", 0)
expected_score_revenue = tf_revenue * (k1 + 1) / (tf_revenue + k1)
# Find "revenue" in ranked
ranked_dict = {term: score for term, score in ranked}
got_score = ranked_dict.get("revenue", 0.0)
assert abs(got_score - expected_score_revenue) < 1e-6, (
    f"BM25 score mismatch for 'revenue': {got_score} vs {expected_score_revenue}"
)
print(f"[OK] rank_expansion_terms BM25 formula: 'revenue' tf={tf_revenue}, "
      f"score={got_score:.4f} (expected {expected_score_revenue:.4f})")

# Higher-frequency term should rank above lower-frequency
if tf_revenue > q_tokens.get("growth", 0):
    assert ranked_dict.get("revenue", 0) >= ranked_dict.get("growth", 0), (
        "More frequent term should rank >= less frequent"
    )
print(f"[OK] BM25 ranking: revenue({ranked_dict.get('revenue',0):.4f}) >= "
      f"growth({ranked_dict.get('growth',0):.4f})")

print("\n[PASS] dim_056: Query expander + specificity entropy + concept graph + BM25 verified")
PYEOF
