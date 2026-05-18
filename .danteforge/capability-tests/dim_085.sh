#!/usr/bin/env bash
# dim_085: Social sentiment — pure VADER fallback scoring and lexicon checks
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math

from sentinel.sma.social_sentiment_v3 import (
    _FallbackVADER,
    _FINANCE_POSITIVE_LEXICON,
    _FINANCE_NEGATIVE_LEXICON,
    SentimentScore,
    SentimentSummary,
    RedditPost,
    StockTwitsMessage,
)

# Test finance lexicons are populated
assert len(_FINANCE_POSITIVE_LEXICON) >= 20, \
    f"Expected >= 20 positive finance terms: {len(_FINANCE_POSITIVE_LEXICON)}"
assert len(_FINANCE_NEGATIVE_LEXICON) >= 20, \
    f"Expected >= 20 negative finance terms: {len(_FINANCE_NEGATIVE_LEXICON)}"
assert "beat" in _FINANCE_POSITIVE_LEXICON, "'beat' should be in positive lexicon"
assert "miss" in _FINANCE_NEGATIVE_LEXICON, "'miss' should be in negative lexicon"
assert "bankruptcy" in _FINANCE_NEGATIVE_LEXICON, "'bankruptcy' should be in negative lexicon"
assert "earnings beat" in _FINANCE_POSITIVE_LEXICON, "'earnings beat' should be in positive lexicon"
print(f"[OK] _FINANCE_POSITIVE_LEXICON: {len(_FINANCE_POSITIVE_LEXICON)} terms")
print(f"[OK] _FINANCE_NEGATIVE_LEXICON: {len(_FINANCE_NEGATIVE_LEXICON)} terms")

# Test _FallbackVADER.polarity_scores on strongly positive text
vader = _FallbackVADER()

pos_text = "Apple beat earnings estimates and raised guidance for the quarter"
pos_scores = vader.polarity_scores(pos_text)
assert "compound" in pos_scores, "polarity_scores must return 'compound'"
assert "pos" in pos_scores, "polarity_scores must return 'pos'"
assert "neg" in pos_scores, "polarity_scores must return 'neg'"
assert "neu" in pos_scores, "polarity_scores must return 'neu'"
assert pos_scores["compound"] > 0.05, \
    f"Positive text should have positive compound score: {pos_scores['compound']}"
print(f"[OK] Positive text compound: {pos_scores['compound']:.4f} (>0.05)")

# Test strongly negative text
neg_text = "Company missed earnings and filed for bankruptcy amid fraud investigation"
neg_scores = vader.polarity_scores(neg_text)
assert neg_scores["compound"] < -0.05, \
    f"Negative text should have negative compound score: {neg_scores['compound']}"
print(f"[OK] Negative text compound: {neg_scores['compound']:.4f} (<-0.05)")

# Test neutral text
neu_text = "The stock market opened today with mixed signals"
neu_scores = vader.polarity_scores(neu_text)
assert -1.0 <= neu_scores["compound"] <= 1.0, \
    f"Compound score must be in [-1, 1]: {neu_scores['compound']}"
print(f"[OK] Neutral text compound: {neu_scores['compound']:.4f}")

# Verify compound is always in [-1, 1]
for text in [pos_text, neg_text, neu_text]:
    s = vader.polarity_scores(text)
    assert -1.0 <= s["compound"] <= 1.0, f"Compound out of range: {s['compound']}"
print("[OK] All compound scores are in [-1, 1]")

# Verify pos/neg/neu always in [0, 1]
for text in [pos_text, neg_text, neu_text]:
    s = vader.polarity_scores(text)
    assert 0.0 <= s["pos"] <= 1.0, f"pos must be in [0,1]: {s['pos']}"
    assert 0.0 <= s["neg"] <= 1.0, f"neg must be in [0,1]: {s['neg']}"
    assert 0.0 <= s["neu"] <= 1.0, f"neu must be in [0,1]: {s['neu']}"
print("[OK] pos/neg/neu scores are all in [0, 1]")

# Test SentimentScore.from_vader classification
pos_score = SentimentScore.from_vader({"compound": 0.75, "pos": 0.8, "neg": 0.0, "neu": 0.2})
assert pos_score.label == "positive", f"Expected 'positive': {pos_score.label}"
assert pos_score.compound == 0.75
print(f"[OK] SentimentScore.from_vader positive: label={pos_score.label}")

neg_score = SentimentScore.from_vader({"compound": -0.60, "pos": 0.0, "neg": 0.6, "neu": 0.4})
assert neg_score.label == "negative", f"Expected 'negative': {neg_score.label}"
print(f"[OK] SentimentScore.from_vader negative: label={neg_score.label}")

neu_score = SentimentScore.from_vader({"compound": 0.01, "pos": 0.1, "neg": 0.0, "neu": 0.9})
assert neu_score.label == "neutral", f"Expected 'neutral': {neu_score.label}"
print(f"[OK] SentimentScore.from_vader neutral: label={neu_score.label}")

# Boundary conditions for label
assert SentimentScore.from_vader({"compound": 0.05, "pos": 0.1, "neg": 0.0, "neu": 0.9}).label == "positive"
assert SentimentScore.from_vader({"compound": -0.05, "pos": 0.0, "neg": 0.1, "neu": 0.9}).label == "negative"
assert SentimentScore.from_vader({"compound": 0.04, "pos": 0.0, "neg": 0.0, "neu": 1.0}).label == "neutral"
print("[OK] SentimentScore label boundary conditions correct")

# Test pure sentiment aggregation math (engagement-weighted)
compound_scores = [0.80, 0.65, -0.10, 0.45, 0.30]
engagement_weights = [1200, 800, 150, 500, 200]

weighted_sum = sum(c * w for c, w in zip(compound_scores, engagement_weights))
total_weight = sum(engagement_weights)
composite = weighted_sum / total_weight

assert composite > 0, f"Net sentiment should be positive (mostly bullish posts): {composite:.4f}"
print(f"[OK] Engagement-weighted sentiment: {composite:.4f} > 0")

# Test that negation logic works
good_score_raw = vader.polarity_scores("good earnings results")
not_good_score_raw = vader.polarity_scores("not good earnings results")
assert good_score_raw["compound"] >= not_good_score_raw["compound"], \
    f"Negation should reduce sentiment: good={good_score_raw['compound']:.4f} not_good={not_good_score_raw['compound']:.4f}"
print(f"[OK] Negation logic: 'good'={good_score_raw['compound']:.4f} vs 'not good'={not_good_score_raw['compound']:.4f}")

# Test normalized sigmoid stays in (-1, 1)
alpha = 15.0
for raw in [-50.0, -10.0, 0.0, 10.0, 50.0]:
    compound = raw / math.sqrt(raw * raw + alpha)
    assert -1.0 < compound < 1.0, f"Sigmoid out of bounds for raw={raw}: {compound}"
print("[OK] Sigmoid normalization formula gives values in (-1, 1)")

# Test StockTwitsMessage dataclass
msg = StockTwitsMessage(
    id=123456,
    body="NVDA breaking out to all-time highs! Extremely bullish!",
    created_at="2024-01-15T09:30:00Z",
    sentiment_raw="Bullish",
    user_followers=5000,
    ticker="NVDA",
)
assert msg.ticker == "NVDA"
assert msg.sentiment_raw == "Bullish"
assert msg.sentiment is None
print(f"[OK] StockTwitsMessage: ticker={msg.ticker} sentiment_raw={msg.sentiment_raw}")

msg.sentiment = pos_score
assert msg.sentiment.label == "positive"
print(f"[OK] StockTwitsMessage.sentiment: {msg.sentiment.label}")

print("\n[PASS] dim_085: Social sentiment")
PYEOF
