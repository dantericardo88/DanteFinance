#!/bin/bash
# dim_057: Earnings RAG v3 — SimpleTFIDF chunking/indexing (no network)
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.earnings_rag_v3 import (
    SimpleTFIDF, LocalVectorStore, Document, RetrievedDocument, EarningsTranscript,
)
import numpy as np

# 1. SimpleTFIDF fit_transform
tfidf = SimpleTFIDF(max_features=500)
texts = [
    "Apple revenue grew 12 percent in fiscal year 2024 iPhone strong demand",
    "Microsoft cloud earnings beat analyst estimates significantly azure growth",
    "Google advertising revenue increased driven by search monetization clicks",
    "Amazon AWS cloud computing revenue accelerated quarter over quarter growth",
]
matrix = tfidf.fit_transform(texts)
assert matrix.shape[0] == 4, f"Expected 4 rows, got {matrix.shape[0]}"
assert matrix.shape[1] > 0, f"Expected >0 features"
assert matrix.ndim == 2
print(f"[OK] SimpleTFIDF.fit_transform() -> shape {matrix.shape}")

# 2. Vocabulary aligned with matrix
assert len(tfidf.vocab) == matrix.shape[1]
assert "apple" in tfidf.vocab or "revenue" in tfidf.vocab
print(f"[OK] SimpleTFIDF vocab size: {len(tfidf.vocab)}")

# 3. Transform query vector
q_vec = tfidf.transform("Apple revenue fiscal year")
assert q_vec.shape[0] == matrix.shape[1]
print(f"[OK] SimpleTFIDF.transform(str) -> vector of length {q_vec.shape[0]}")

# 4. IDF is log-scaled (positive)
assert hasattr(tfidf, 'idf')
assert np.all(tfidf.idf > 0)
print(f"[OK] TF-IDF IDF weights positive: min={tfidf.idf.min():.3f}, max={tfidf.idf.max():.3f}")

# 5. LocalVectorStore - uses len() not size()
store = LocalVectorStore()
docs = [
    Document("d1", "AAPL", "edgar", "10-K", "2024", "AAPL 10-K 2024",
             "Apple revenue grew 89 billion dollars fiscal 2024 driven by iPhone"),
    Document("d2", "MSFT", "edgar", "10-K", "2024", "MSFT 10-K 2024",
             "Microsoft cloud revenue Azure growing 30 percent quarter"),
    Document("d3", "AAPL", "edgar", "8-K", "2024", "AAPL earnings call",
             "Apple CEO Tim Cook iPhone demand services revenue growth"),
]
store.add_documents(docs)
assert len(store) == 3, f"Expected 3 docs, got {len(store)}"
assert len(store.documents) == 3
print(f"[OK] LocalVectorStore: len={len(store)}, documents={len(store.documents)}")

# 6. Search with cosine similarity
results = store.search("Apple revenue iPhone", k=2)
assert len(results) > 0, "Expected search results"
assert all(isinstance(r, RetrievedDocument) for r in results)
assert all(hasattr(r, 'score') for r in results)
assert all(hasattr(r, 'rank') for r in results)
assert results[0].rank == 1
print(f"[OK] LocalVectorStore.search() -> {len(results)} results, top={results[0].document.ticker}, rank={results[0].rank}")

# 7. EarningsTranscript word_count
transcript = EarningsTranscript(
    ticker="AAPL", quarter="Q4", year=2024, date="2024-11-01", source="SEC",
    title="Apple Q4 2024 Earnings Call",
    full_text="Tim Cook Revenue grew 12 percent iPhone demand strong services record",
)
assert transcript.word_count > 0
print(f"[OK] EarningsTranscript.word_count={transcript.word_count}")

# 8. Document chunk metadata
doc = Document("test_doc", "AAPL", "edgar", "10-K", "2024", "Test", "Test content",
               chunk_index=2, chunk_total=7)
assert doc.chunk_index == 2
assert doc.chunk_total == 7
print("[OK] Document dataclass chunk_index/chunk_total fields")

print("\n[PASS] dim_057: Earnings RAG v3 — TF-IDF chunking/indexing verified")
PYEOF
