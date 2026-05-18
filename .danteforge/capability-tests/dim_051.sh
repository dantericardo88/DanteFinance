#!/bin/bash
# dim_051: Financial RAG v3 — TF-IDF fallback path, chunking, entity extraction
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# ── 1. EmbeddingEngine TF-IDF fallback ──────────────────────────────────────
from sentinel.sai.financial_rag_v3 import EmbeddingEngine, FinancialDocumentChunker, FinancialEntityExtractor, CitationTracker, RetrievedChunk

engine = EmbeddingEngine(dim=384)
assert engine.dim == 384
print("[OK] EmbeddingEngine constructed (dim=384)")

# Embed some texts using the TF-IDF fallback path
texts = [
    "Apple reported revenue of $89 billion in Q4 2024, up 12% year over year.",
    "Microsoft earnings per share exceeded analyst estimates by 15 cents.",
    "The company gross margin improved to 43% driven by software mix.",
]
vecs = engine.embed(texts)
assert vecs.shape == (3, 384), f"Expected (3, 384) got {vecs.shape}"
assert vecs.dtype.name == "float32"
print("[OK] EmbeddingEngine.embed() returns (3, 384) float32 array")

# Embed a single query
q_vec = engine.embed_query("What was Apple revenue?")
assert q_vec.shape == (1, 384)
print("[OK] EmbeddingEngine.embed_query() returns (1, 384)")

# ── 2. FinancialEntityExtractor ───────────────────────────────────────────────
extractor = FinancialEntityExtractor()
text = "Apple Inc. (AAPL) reported $89 billion in revenue for Q4 2024, up 12%."
entities = extractor.extract_entities(text)
assert hasattr(entities, 'tickers'), "Missing tickers attribute"
assert hasattr(entities, 'dollar_amounts'), "Missing dollar_amounts attribute"
assert hasattr(entities, 'percentages'), "Missing percentages attribute"
assert hasattr(entities, 'dates'), "Missing dates attribute"
# AAPL should be extracted as a ticker
assert "AAPL" in entities.tickers, f"AAPL not in tickers: {entities.tickers}"
assert len(entities.dollar_amounts) > 0, "Expected dollar amounts"
assert len(entities.percentages) > 0, f"Expected percentages, got: {entities.percentages}"
print("[OK] FinancialEntityExtractor.extract_entities() extracts tickers, dollar amounts, percentages")

# ── 3. FinancialDocumentChunker ───────────────────────────────────────────────
chunker = FinancialDocumentChunker(chunk_size=50, overlap_sentences=1)

big_text = (
    "Item 1. Business Overview. Apple Inc. designs and sells consumer electronics. "
    "The company reported $89 billion in revenue for fiscal year 2024. "
    "Item 1A. Risk Factors. Competition from other technology companies presents risk. "
    "Currency fluctuations may adversely affect results. Interest rate changes matter. "
    "Item 7. Management Discussion and Analysis. Revenue grew 12 percent year over year. "
    "Operating margins improved significantly driven by higher software licensing revenue."
)
docs = chunker.chunk_10k(big_text, ticker="AAPL", year=2024)
assert len(docs) >= 1, f"Expected at least 1 chunk, got {len(docs)}"
for doc in docs:
    assert hasattr(doc, 'content')
    assert hasattr(doc, 'ticker')
    assert doc.ticker == "AAPL"
print(f"[OK] FinancialDocumentChunker.chunk_10k() produced {len(docs)} chunks")

# ── 4. CitationTracker ────────────────────────────────────────────────────────
tracker = CitationTracker()
# Build a RetrievedChunk to format
chunk = RetrievedChunk(
    doc_id="doc_001",
    content="Apple revenue grew 12%",
    score=0.85,
    ticker="AAPL",
    form_type="10-K",
    year=2024,
    section="Item 7",
)
citation = tracker.format_citation(chunk)
assert "AAPL" in citation
assert "10-K" in citation
print(f"[OK] CitationTracker.format_citation() -> {citation}")

print("\n[PASS] dim_051: Financial RAG v3 — TF-IDF fallback verified")
PYEOF
