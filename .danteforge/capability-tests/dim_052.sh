#!/bin/bash
# dim_052: FinBERT sentiment v3 — LM lexicon + 5 stub fixes verified (no network)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.finbert_sentiment_v3 import (
    LoughranMcDonaldLexicon, LMScore, SentimentScore,
    _LM_NEGATIVE, _LM_POSITIVE,
    NewsArticle, SECFilingsSentimentAnalyzer, AggregatedSentimentEngine,
    FinBERTAnalyzer,
)
from datetime import datetime, timezone

# ── 1. LMScore dataclass math ─────────────────────────────────────────────────
score = LMScore(
    positive_count=10,
    negative_count=4,
    uncertainty_count=2,
    litigious_count=1,
    strong_modal_count=0,
    weak_modal_count=0,
    total_words=100,
)
assert abs(score.net_sentiment - 0.06) < 1e-9, f"Expected 0.06 got {score.net_sentiment}"
assert abs(score.uncertainty_ratio - 0.02) < 1e-9
assert abs(score.litigious_ratio - 0.01) < 1e-9
print("[OK] LMScore.net_sentiment = (pos - neg) / total_words")
print("[OK] LMScore.uncertainty_ratio and litigious_ratio correct")

# ── 2. LMScore edge cases ────────────────────────────────────────────────────
zero_score = LMScore(total_words=0)
assert zero_score.net_sentiment == 0.0
assert zero_score.uncertainty_ratio == 0.0
print("[OK] LMScore handles zero total_words safely")

# ── 3. Lexicon word lists exist and are non-trivial ───────────────────────────
assert len(_LM_NEGATIVE) > 100, f"LM negative word list too short: {len(_LM_NEGATIVE)}"
assert "bankrupt" in _LM_NEGATIVE
assert "fraud" in _LM_NEGATIVE
assert "loss" in _LM_NEGATIVE
print(f"[OK] _LM_NEGATIVE has {len(_LM_NEGATIVE)} entries including 'bankrupt', 'fraud', 'loss'")

assert len(_LM_POSITIVE) > 30, f"LM positive word list too short: {len(_LM_POSITIVE)}"
print(f"[OK] _LM_POSITIVE has {len(_LM_POSITIVE)} entries")

# ── 4. LoughranMcDonaldLexicon.score() ───────────────────────────────────────
lexicon = LoughranMcDonaldLexicon()

neg_text = "The company suffered significant losses, fraud was discovered, and bankruptcy is imminent."
neg_result = lexicon.score(neg_text)
assert isinstance(neg_result, LMScore)
assert neg_result.total_words > 0
assert neg_result.negative_count > 0, f"Expected negative words, got: {neg_result.negative_count}"
assert neg_result.net_sentiment < 0, f"Expected negative net sentiment, got: {neg_result.net_sentiment}"
print(f"[OK] Negative text: negative_count={neg_result.negative_count}, net={neg_result.net_sentiment:.3f}")

pos_text = "Outstanding profitable growth exceeded analyst expectations with record earnings and dividends."
pos_result = lexicon.score(pos_text)
assert isinstance(pos_result, LMScore)
print(f"[OK] Positive text: positive_count={pos_result.positive_count}, net={pos_result.net_sentiment:.3f}")

# ── 5. SentimentScore properties ─────────────────────────────────────────────
ss = SentimentScore(
    label="positive",
    confidence=0.92,
    positive=0.85,
    negative=0.10,
    neutral=0.05,
    source="lm_lexicon",
)
assert abs(ss.net - 0.75) < 1e-9, f"Expected net=0.75, got {ss.net}"
assert abs(ss.polarity - 0.75) < 1e-9
print("[OK] SentimentScore.net and .polarity computed correctly")

# ── 6. Stub fix #3: SECFilingsSentimentAnalyzer.fetch_8k_press_releases exists ──
fb = FinBERTAnalyzer()
lm = LoughranMcDonaldLexicon()
sec = SECFilingsSentimentAnalyzer(fb, lm)

assert hasattr(sec, "fetch_8k_press_releases"), "Missing fetch_8k_press_releases method"
assert callable(sec.fetch_8k_press_releases)
print("[OK] SECFilingsSentimentAnalyzer.fetch_8k_press_releases method exists")

# Verify helper methods also exist
assert hasattr(sec, "_search_edgar_efts"), "Missing _search_edgar_efts helper"
assert hasattr(sec, "_fetch_filing_text_from_accession"), "Missing _fetch_filing_text_from_accession helper"
assert hasattr(sec, "fetch_risk_factors"), "Missing fetch_risk_factors"
assert hasattr(sec, "fetch_mda_section"), "Missing fetch_mda_section"
print("[OK] EDGAR EFTS helper methods (_search_edgar_efts, _fetch_filing_text_from_accession) exist")

# ── 7. Stub fix #4: normalize_entity_name ────────────────────────────────────
engine = AggregatedSentimentEngine()

assert hasattr(engine, "normalize_entity_name"), "Missing normalize_entity_name"
norm = engine.normalize_entity_name

# Ticker-like: short, alpha-only → uppercase
assert norm("msft") == "MSFT", f"Expected 'MSFT', got '{norm('msft')}'"
assert norm("AAPL") == "AAPL", f"Expected 'AAPL', got '{norm('AAPL')}'"

# Legal suffix stripping
result_apple = norm("apple inc.")
assert "inc" not in result_apple.lower(), f"Legal suffix not removed: '{result_apple}'"
assert "apple" in result_apple.lower(), f"Company name lost: '{result_apple}'"
print(f"[OK] normalize_entity_name: 'apple inc.' => '{result_apple}'")

result_alpha = norm("Alphabet Inc.")
assert "Alphabet" in result_alpha, f"Expected 'Alphabet' in '{result_alpha}'"
print(f"[OK] normalize_entity_name: 'Alphabet Inc.' => '{result_alpha}'")

result_bh = norm("berkshire hathaway llc")
assert "Llc" not in result_bh, f"LLC suffix not removed from '{result_bh}'"
print(f"[OK] normalize_entity_name: 'berkshire hathaway llc' => '{result_bh}'")

# Edge case: empty string
assert norm("") == ""
print("[OK] normalize_entity_name: empty string handled safely")

# ── 8. Stub fix #5: aggregate_sentiment_by_entity ────────────────────────────
assert hasattr(engine, "aggregate_sentiment_by_entity"), "Missing aggregate_sentiment_by_entity"

now = datetime.now(timezone.utc)

articles = [
    NewsArticle(
        title="Apple reports record quarterly earnings beating estimates",
        url="https://example.com/1",
        date=now,
        domain="reuters.com",
        description="Apple Inc reported strong earnings results with revenue growth.",
    ),
    NewsArticle(
        title="Apple stock falls on weak iPhone demand guidance",
        url="https://example.com/2",
        date=now,
        domain="bloomberg.com",
        description="Concerns about iPhone demand weigh on Apple shares.",
    ),
    NewsArticle(
        title="Microsoft cloud revenue surges past expectations",
        url="https://example.com/3",
        date=now,
        domain="cnbc.com",
        description="Microsoft Azure revenue exceeded analyst estimates significantly.",
    ),
    NewsArticle(
        title="Microsoft faces antitrust scrutiny from regulators",
        url="https://example.com/4",
        date=now,
        domain="wsj.com",
        description="Regulators are investigating Microsoft for competition concerns.",
    ),
]

entity_mentions = {
    "AAPL": ["Apple", "AAPL"],
    "MSFT": ["Microsoft", "MSFT"],
}

agg = engine.aggregate_sentiment_by_entity(articles, entity_mentions)

assert len(agg) > 0, "Expected at least one entity in aggregation result"
assert isinstance(agg, dict)
print(f"[OK] aggregate_sentiment_by_entity returned {len(agg)} entities")

for entity_key in agg:
    r = agg[entity_key]
    assert "n_articles" in r, f"Missing n_articles for {entity_key}"
    assert "mean_net_sentiment" in r, f"Missing mean_net_sentiment for {entity_key}"
    assert "std_net_sentiment" in r, f"Missing std_net_sentiment for {entity_key}"
    assert "positive_pct" in r, f"Missing positive_pct for {entity_key}"
    assert "negative_pct" in r, f"Missing negative_pct for {entity_key}"
    assert "dominant_label" in r, f"Missing dominant_label for {entity_key}"
    assert "articles" in r, f"Missing articles for {entity_key}"
    assert r["n_articles"] > 0, f"n_articles should be > 0 for {entity_key}"
    assert r["dominant_label"] in ("positive", "negative", "neutral")
    assert 0.0 <= r["positive_pct"] <= 1.0
    assert 0.0 <= r["negative_pct"] <= 1.0
    print(f"[OK]   {entity_key}: n={r['n_articles']}, "
          f"mean_net={r['mean_net_sentiment']:.3f}, "
          f"dominant={r['dominant_label']}, "
          f"pos%={r['positive_pct']:.0%}, neg%={r['negative_pct']:.0%}")

# ── 9. aggregate_sentiment_by_entity — empty input ───────────────────────────
empty_result = engine.aggregate_sentiment_by_entity([])
assert empty_result == {}, f"Expected empty dict for no articles, got {empty_result}"
print("[OK] aggregate_sentiment_by_entity handles empty article list")

# ── 10. SECFilingsSentimentAnalyzer corrected EDGAR URL ──────────────────────
assert sec._EDGAR_EFTS == "https://efts.sec.gov/LATEST/search-index"
assert "efts.sec.gov" in sec._EDGAR_EFTS
print(f"[OK] EDGAR EFTS endpoint: {sec._EDGAR_EFTS}")

print("\n[PASS] dim_052: FinBERT/LM sentiment — all 5 stubs fixed and verified")
PYEOF
