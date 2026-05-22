#!/bin/bash
# dim_051: Financial RAG v3 — TF-IDF fallback, hybrid search, build_index/query,
#          PDF ingestion, entity-linked retrieval, citation extraction
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# ── 1. EmbeddingEngine TF-IDF fallback ──────────────────────────────────────
from sentinel.sai.financial_rag_v3 import (
    EmbeddingEngine, FinancialDocumentChunker, FinancialEntityExtractor,
    CitationTracker, RetrievedChunk,
    build_index, tfidf_query, TFIDFIndex, Chunk,
    PersistentVectorStore, Document,
    LlamaIndexAdapter,
)

engine = EmbeddingEngine(dim=384)
assert engine.dim == 384
print("[OK] EmbeddingEngine constructed (dim=384)")

texts = [
    "Apple reported revenue of $89 billion in Q4 2024, up 12% year over year.",
    "Microsoft earnings per share exceeded analyst estimates by 15 cents.",
    "The company gross margin improved to 43% driven by software mix.",
]
vecs = engine.embed(texts)
assert vecs.shape == (3, 384), f"Expected (3, 384) got {vecs.shape}"
assert vecs.dtype.name == "float32"
print("[OK] EmbeddingEngine.embed() returns (3, 384) float32 array")

q_vec = engine.embed_query("What was Apple revenue?")
assert q_vec.shape == (1, 384)
print("[OK] EmbeddingEngine.embed_query() returns (1, 384)")

# ── 2. Standalone build_index / query (TF-IDF retrieval) ─────────────────────
corpus = [
    "Apple Inc reported quarterly revenue of $89 billion driven by iPhone and Services.",
    "Microsoft cloud Azure grew 28 percent with strong enterprise demand.",
    "Tesla delivered 439000 vehicles in Q3 and beat production targets.",
    "Google advertising revenue declined 5 percent amid macro headwinds.",
    "Amazon AWS operating income hit $7.2 billion up from $5.1 billion.",
]
idx = build_index(corpus, doc_ids=[f"d{i}" for i in range(len(corpus))])
assert isinstance(idx, TFIDFIndex), "build_index should return TFIDFIndex"
assert len(idx.documents) == 5
assert len(idx.vocab) > 10
print(f"[OK] build_index() created index with vocab size {len(idx.vocab)}")

results = tfidf_query(idx, "Apple revenue quarterly", top_k=3)
assert len(results) > 0, "tfidf_query() returned no results"
assert isinstance(results[0], Chunk), "result items should be Chunk dataclass"
assert results[0].score > 0, "top result score should be > 0"
# The most relevant result should be about Apple revenue
assert "Apple" in results[0].content or "apple" in results[0].content.lower(), \
    f"Expected Apple-related result, got: {results[0].content[:80]}"
print(f"[OK] tfidf_query() returns {len(results)} relevant chunks, top score={results[0].score:.4f}")

# Verify cosine ranking: Apple doc should beat Tesla/Google for Apple query
apple_scores = [r.score for r in results if "Apple" in r.content or "apple" in r.content.lower()]
other_scores = [r.score for r in results if "Tesla" in r.content or "Google" in r.content]
if other_scores:
    assert max(apple_scores) >= max(other_scores), \
        f"Apple doc should score >= others for 'Apple revenue' query"
print("[OK] TF-IDF cosine ranking places Apple doc above Tesla/Google for Apple query")

# ── 3. FinancialEntityExtractor ───────────────────────────────────────────────
extractor = FinancialEntityExtractor()
text = "Apple Inc. (AAPL) reported $89 billion in revenue for Q4 2024, up 12%."
entities = extractor.extract_entities(text)
assert hasattr(entities, 'tickers'), "Missing tickers attribute"
assert hasattr(entities, 'dollar_amounts'), "Missing dollar_amounts attribute"
assert hasattr(entities, 'percentages'), "Missing percentages attribute"
assert hasattr(entities, 'dates'), "Missing dates attribute"
assert "AAPL" in entities.tickers, f"AAPL not in tickers: {entities.tickers}"
assert len(entities.dollar_amounts) > 0, "Expected dollar amounts"
assert len(entities.percentages) > 0, f"Expected percentages"
print("[OK] FinancialEntityExtractor.extract_entities() extracts tickers, dollar amounts, percentages")

# Entity-to-ticker linking
linked = extractor.link_entity_to_ticker("Apple Inc")
assert linked == "AAPL", f"Expected AAPL, got {linked}"
linked_ms = extractor.link_entity_to_ticker("Microsoft")
assert linked_ms == "MSFT", f"Expected MSFT, got {linked_ms}"
print("[OK] FinancialEntityExtractor.link_entity_to_ticker() resolves company names to tickers")

# ── 4. FinancialDocumentChunker ───────────────────────────────────────────────
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
    assert doc.ticker == "AAPL"
print(f"[OK] FinancialDocumentChunker.chunk_10k() produced {len(docs)} chunks")

# ── 5. CitationTracker ────────────────────────────────────────────────────────
tracker = CitationTracker()
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

# generate_answer_with_citations
cited = tracker.generate_answer_with_citations("Revenue grew 12% YoY.", [chunk])
assert cited.citations, "Expected at least one citation"
assert "AAPL" in cited.citations[0]
assert cited.confidence > 0
print(f"[OK] CitationTracker.generate_answer_with_citations() confidence={cited.confidence:.3f}")

# ── 6. PersistentVectorStore — upsert + search + entity-linked retrieval ─────
import tempfile, os as _os
with tempfile.TemporaryDirectory() as tmpdir:
    db_path_override = _os.path.join(tmpdir, "test_vec.db")
    # Monkey-patch the module's DB path for isolation
    import sentinel.sai.financial_rag_v3 as _rag_mod
    orig_path = _rag_mod._SQLITE_VEC_PATH
    import pathlib
    _rag_mod._SQLITE_VEC_PATH = pathlib.Path(db_path_override)

    # Delete any stale ChromaDB collection BEFORE constructing PVS to avoid
    # dimension mismatch errors (collection may have been created with 64-dim).
    try:
        import chromadb
        from chromadb.config import Settings as _ChromaSettings
        from sentinel.sai.financial_rag_v3 import _CHROMA_PATH
        _tmp_client = chromadb.PersistentClient(
            path=str(_CHROMA_PATH),
            settings=_ChromaSettings(anonymized_telemetry=False),
        )
        _tmp_client.delete_collection("test_dim051")
    except Exception:
        pass  # Collection may not exist or ChromaDB not installed — that's fine

    # Use 384-dim (consistent with _EMBEDDING_DIM) to avoid ChromaDB
    # dimension mismatch errors when a persistent collection already exists.
    _store_engine = EmbeddingEngine(dim=384)
    _store_engine._build_vocab_and_idf(corpus)

    import sqlite3, struct, numpy as np, math

    from sentinel.sai.financial_rag_v3 import PersistentVectorStore as PVS

    vs = PVS(collection_name="test_dim051", embedding_engine=_store_engine)
    # Clear any stale data from a previous test run (handles dim mismatch on re-run)
    try:
        vs.delete_collection()
    except Exception:
        pass
    test_docs = [
        Document("d1", corpus[0], "AAPL", "10-K", year=2024, section="Revenue"),
        Document("d2", corpus[1], "MSFT", "10-K", year=2024, section="Cloud"),
        Document("d3", corpus[2], "TSLA", "10-Q", year=2024, section="Deliveries"),
    ]
    vs.add_documents(test_docs)

    results_vs = vs.search("Apple revenue", k=3)
    assert len(results_vs) > 0, "PersistentVectorStore.search() returned no results"
    print(f"[OK] PersistentVectorStore.search() returned {len(results_vs)} results")

    # Entity-linked retrieval
    entity_results = vs.entity_linked_search("What is Apple revenue?", extractor, k=3)
    assert len(entity_results) > 0, "entity_linked_search returned no results"
    # AAPL doc should be top-ranked
    aapl_hit = any(r.ticker == "AAPL" for r in entity_results[:2])
    assert aapl_hit, f"Expected AAPL in top-2, got: {[r.ticker for r in entity_results]}"
    print(f"[OK] PersistentVectorStore.entity_linked_search() -> AAPL doc in top results")

    # Hybrid search
    hybrid_results = vs.hybrid_search("Apple revenue quarterly", k=3)
    assert len(hybrid_results) > 0, "hybrid_search returned no results"
    print(f"[OK] PersistentVectorStore.hybrid_search() returned {len(hybrid_results)} results")

    # Stats
    stats = vs.get_collection_stats()
    assert stats.get("total_chunks", 0) >= 3, f"Expected >= 3 chunks in stats: {stats}"
    print(f"[OK] PersistentVectorStore.get_collection_stats() = {stats}")

    _rag_mod._SQLITE_VEC_PATH = orig_path  # restore

# ── 7. LlamaIndexAdapter PDF fallback uses pdfplumber ──────────────────────────
adapter = LlamaIndexAdapter()
# Create a minimal "PDF" (text file) to exercise the fallback
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tf:
    tf.write("Apple revenue $89 billion Q4 2024 12% growth\n")
    tmpf = tf.name
result_text = adapter._fallback_pdf_read(tmpf)
_os.unlink(tmpf)
# Should return the text (via raw bytes path since it's not a real PDF)
assert isinstance(result_text, str), "Expected string from _fallback_pdf_read"
print(f"[OK] LlamaIndexAdapter._fallback_pdf_read() returned {len(result_text)} chars")

print("\n[PASS] dim_051: Financial RAG v3 — TF-IDF, hybrid search, build_index/query, "
      "PDF ingestion, entity-linked retrieval, citation extraction all verified")
PYEOF
