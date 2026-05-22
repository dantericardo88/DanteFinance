#!/bin/bash
# dim_057: Earnings RAG v3 — TF-IDF, guidance sentences, tone shift, earnings timeline
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.earnings_rag_v3 import (
    SimpleTFIDF, LocalVectorStore, Document, RetrievedDocument, EarningsTranscript,
    EarningsAnalytics,
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

# -----------------------------------------------------------------------
# Test 9: extract_guidance_sentences — guidance keyword + number within 10 words
# -----------------------------------------------------------------------
analytics = EarningsAnalytics()

sample_text = (
    "We expect revenue to reach $45 billion in fiscal 2025. "
    "The weather was nice today and flowers are blooming. "
    "Management anticipates gross margins of approximately 43 percent next quarter. "
    "This is a sentence with no guidance keywords or numbers whatsoever. "
    "We project EPS of $1.80 for the full year fiscal 2025. "
    "The company will invest in new data centers."
)

guidance_sents = analytics.extract_guidance_sentences(sample_text)
assert isinstance(guidance_sents, list), "Should return list"
assert len(guidance_sents) >= 2, (
    f"Expected at least 2 guidance sentences, got {len(guidance_sents)}: {guidance_sents}"
)

# Verify guidance sentences contain keywords AND numbers
import re
_KW_RE = re.compile(r"\b(expect|anticipate|guid|outlook|project|forecast|target)\w*\b", re.I)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?", re.I)
for sent in guidance_sents:
    assert _KW_RE.search(sent), f"Missing guidance keyword in: {sent!r}"
    assert _NUM_RE.search(sent), f"Missing number in: {sent!r}"

print(f"[OK] extract_guidance_sentences: found {len(guidance_sents)} guidance sentences")

# Text with no guidance → empty list
no_guidance = "The sun rose over the mountains. Birds sang in the trees."
assert analytics.extract_guidance_sentences(no_guidance) == []
print("[OK] extract_guidance_sentences: no-guidance text returns []")

# -----------------------------------------------------------------------
# Test 10: compute_tone_shift_score — delta = current - prior
# -----------------------------------------------------------------------
current_text = "strong growth record beat exceeded outperform robust accelerate expand improve"
prior_text   = "decline decrease miss below concern headwind pressure uncertain cautious challenge"

tone = analytics.compute_tone_shift_score(current_text, prior_text)
assert "current_score"  in tone
assert "prior_score"    in tone
assert "delta"          in tone
assert "direction"      in tone

# Current should be positive, prior should be negative
assert tone["current_score"] > 0, f"Positive text should score > 0: {tone['current_score']}"
assert tone["prior_score"]   < 0, f"Negative text should score < 0: {tone['prior_score']}"
assert tone["delta"]         > 0, f"Delta should be positive (improving): {tone['delta']}"
assert tone["direction"]     == "improving", f"Should be 'improving': {tone['direction']}"

# Verify delta arithmetic
expected_delta = round(tone["current_score"] - tone["prior_score"], 4)
assert abs(tone["delta"] - expected_delta) < 1e-9, (
    f"Delta mismatch: {tone['delta']} vs {expected_delta}"
)
print(f"[OK] compute_tone_shift_score: current={tone['current_score']:.4f}, "
      f"prior={tone['prior_score']:.4f}, delta={tone['delta']:.4f}, dir={tone['direction']}")

# Stable when identical text
stable_tone = analytics.compute_tone_shift_score(current_text, current_text)
assert abs(stable_tone["delta"]) < 1e-9, f"Identical text delta should be ~0: {stable_tone['delta']}"
assert stable_tone["direction"] == "stable"
print(f"[OK] compute_tone_shift_score stable: delta={stable_tone['delta']}")

# -----------------------------------------------------------------------
# Test 11: build_earnings_timeline — sorted list with beat/miss/in-line labels
# -----------------------------------------------------------------------
t1 = EarningsTranscript(
    ticker="AAPL", quarter="Q1 2024", year=2024, date="2024-02-01", source="SEC",
    title="Apple Q1 2024",
    full_text=(
        "We expect revenue guidance of $120 billion for next quarter. "
        "Revenue was $119 billion in Q1 2024."
    ),
)
t2 = EarningsTranscript(
    ticker="AAPL", quarter="Q2 2024", year=2024, date="2024-05-01", source="SEC",
    title="Apple Q2 2024",
    full_text=(
        "Management guided for revenue of $100 billion. "
        "Revenue reached $110 billion, exceeding our targets."
    ),
)
t3 = EarningsTranscript(
    ticker="AAPL", quarter="Q3 2024", year=2024, date="2024-08-01", source="SEC",
    title="Apple Q3 2024",
    full_text=(
        "We anticipate revenue guidance of $90 billion for the period. "
        "Revenue was $80 billion for Q3."
    ),
)

timeline = analytics.build_earnings_timeline([t3, t1, t2])  # deliberately unsorted
assert isinstance(timeline, list)
assert len(timeline) == 3

# Verify sorted by date ascending
dates = [e["date"] for e in timeline]
assert dates == sorted(dates), f"Timeline not sorted: {dates}"

# Verify required keys
for entry in timeline:
    for k in ("ticker", "quarter", "date", "label", "revenue_kpi", "guidance_kpi", "word_count"):
        assert k in entry, f"Missing key {k!r} in timeline entry"
    assert entry["label"] in ("beat", "miss", "in-line", "unknown"), (
        f"Invalid label: {entry['label']}"
    )

print(f"[OK] build_earnings_timeline: {len(timeline)} entries, sorted={dates}")
for e in timeline:
    print(f"     {e['quarter']}: label={e['label']}, rev={e['revenue_kpi']}, guide={e['guidance_kpi']}")

print("\n[PASS] dim_057: Earnings RAG + guidance extraction + tone shift + timeline verified")
PYEOF
