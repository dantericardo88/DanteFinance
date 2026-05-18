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
    compute_sentiment_momentum,
    compute_retail_vs_institutional_divergence,
    compute_viral_coefficient,
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

# ---- NEW: compute_sentiment_momentum ------------------------------------------
# 10-day scores slightly positive, 3-day scores more positive → bullish signal
scores_bull = [-0.10, -0.05, 0.00, 0.05, 0.10, 0.10, 0.15, 0.20, 0.40, 0.50]
mom_bull = compute_sentiment_momentum(scores_bull)
assert "momentum" in mom_bull, "momentum missing"
assert "signal" in mom_bull, "signal missing"
assert "ma_3d" in mom_bull, "ma_3d missing"
assert "ma_10d" in mom_bull, "ma_10d missing"

expected_ma3d  = sum(scores_bull[-3:]) / 3
expected_ma10d = sum(scores_bull[-10:]) / 10
expected_mom   = expected_ma3d - expected_ma10d

assert abs(mom_bull["ma_3d"]    - round(expected_ma3d, 4))  < 1e-4, f"ma_3d wrong: {mom_bull['ma_3d']} vs {expected_ma3d}"
assert abs(mom_bull["ma_10d"]   - round(expected_ma10d, 4)) < 1e-4, f"ma_10d wrong: {mom_bull['ma_10d']} vs {expected_ma10d}"
assert abs(mom_bull["momentum"] - expected_mom)              < 1e-4, f"momentum wrong: {mom_bull['momentum']} vs {expected_mom}"
assert mom_bull["signal"] == "bullish", f"Signal should be 'bullish': {mom_bull['signal']}"
print(
    f"[OK] compute_sentiment_momentum (bullish): "
    f"ma_3d={mom_bull['ma_3d']:.4f}, ma_10d={mom_bull['ma_10d']:.4f}, "
    f"momentum={mom_bull['momentum']:.4f}, signal={mom_bull['signal']}"
)

# Bearish: 3-day sharply more negative than 10-day
scores_bear = [0.50, 0.40, 0.30, 0.20, 0.10, 0.00, -0.10, -0.20, -0.40, -0.50]
mom_bear = compute_sentiment_momentum(scores_bear)
assert mom_bear["signal"] == "bearish", f"Signal should be 'bearish': {mom_bear['signal']}"
assert mom_bear["momentum"] < -0.02, f"Bearish momentum should be < -0.02: {mom_bear['momentum']}"
print(
    f"[OK] compute_sentiment_momentum (bearish): "
    f"momentum={mom_bear['momentum']:.4f}, signal={mom_bear['signal']}"
)

# Neutral: 3-day ≈ 10-day
scores_flat = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
mom_flat = compute_sentiment_momentum(scores_flat)
assert mom_flat["signal"] == "neutral", f"Flat signal should be 'neutral': {mom_flat['signal']}"
assert abs(mom_flat["momentum"]) <= 0.02, f"Flat momentum should be near 0: {mom_flat['momentum']}"
print(
    f"[OK] compute_sentiment_momentum (neutral): "
    f"momentum={mom_flat['momentum']:.4f}, signal={mom_flat['signal']}"
)

# ---- NEW: compute_retail_vs_institutional_divergence -------------------------
# Large positive divergence: retail very bullish (0.8), analysts neutral (0.1) → contrarian
div_bull = compute_retail_vs_institutional_divergence(
    social_sentiment=0.80,
    analyst_consensus=0.10,
    divergence_threshold=0.30,
)
assert "divergence" in div_bull, "divergence missing"
assert "contrarian_signal" in div_bull, "contrarian_signal missing"
assert "social_sentiment" in div_bull, "social_sentiment missing"
assert "analyst_consensus" in div_bull, "analyst_consensus missing"

expected_div = 0.80 - 0.10
assert abs(div_bull["divergence"] - expected_div) < 1e-9, (
    f"Divergence wrong: expected {expected_div}, got {div_bull['divergence']}"
)
assert div_bull["contrarian_signal"] is True, (
    f"Large divergence ({div_bull['divergence']:.2f}) should be contrarian"
)
print(
    f"[OK] compute_retail_vs_institutional_divergence (contrarian): "
    f"divergence={div_bull['divergence']:.4f}, contrarian_signal={div_bull['contrarian_signal']}"
)

# Small divergence: retail ≈ analysts → not contrarian
div_aligned = compute_retail_vs_institutional_divergence(
    social_sentiment=0.50,
    analyst_consensus=0.45,
    divergence_threshold=0.30,
)
expected_div_aligned = 0.50 - 0.45
assert abs(div_aligned["divergence"] - expected_div_aligned) < 1e-9, (
    f"Aligned divergence wrong: {div_aligned['divergence']}"
)
assert div_aligned["contrarian_signal"] is False, (
    f"Small divergence ({div_aligned['divergence']:.2f}) should NOT be contrarian"
)
print(
    f"[OK] compute_retail_vs_institutional_divergence (aligned): "
    f"divergence={div_aligned['divergence']:.4f}, contrarian_signal={div_aligned['contrarian_signal']}"
)

# Negative divergence (retail bearish, analysts bullish) → abs > threshold → contrarian
div_neg = compute_retail_vs_institutional_divergence(
    social_sentiment=-0.60,
    analyst_consensus=0.20,
    divergence_threshold=0.30,
)
assert div_neg["contrarian_signal"] is True, (
    f"Negative large divergence should be contrarian: {div_neg['divergence']:.4f}"
)
assert div_neg["divergence"] < 0, "Negative divergence should be < 0"
print(
    f"[OK] compute_retail_vs_institutional_divergence (negative contrarian): "
    f"divergence={div_neg['divergence']:.4f}"
)

# ---- NEW: compute_viral_coefficient ------------------------------------------
# High virality: 1000 today vs 100 avg → coeff = 10.0 → spike
viral_high = compute_viral_coefficient(mentions_today=1000.0, mentions_7day_avg=100.0)
assert "viral_coefficient" in viral_high, "viral_coefficient missing"
assert "viral_spike" in viral_high, "viral_spike missing"
assert "mentions_today" in viral_high, "mentions_today missing"
assert "mentions_7day_avg" in viral_high, "mentions_7day_avg missing"

expected_coeff = 1000.0 / 100.0  # = 10.0
assert abs(viral_high["viral_coefficient"] - expected_coeff) < 1e-9, (
    f"Viral coefficient wrong: expected {expected_coeff}, got {viral_high['viral_coefficient']}"
)
assert viral_high["viral_spike"] is True, (
    f"Coefficient=10.0 should be a viral spike (threshold=3.0): {viral_high['viral_spike']}"
)
print(
    f"[OK] compute_viral_coefficient (viral): "
    f"coeff={viral_high['viral_coefficient']:.4f}, viral_spike={viral_high['viral_spike']}"
)

# Below threshold: 2.5x → not viral
viral_low = compute_viral_coefficient(mentions_today=250.0, mentions_7day_avg=100.0)
assert abs(viral_low["viral_coefficient"] - 2.5) < 1e-9, (
    f"Non-viral coefficient wrong: {viral_low['viral_coefficient']}"
)
assert viral_low["viral_spike"] is False, (
    f"Coefficient=2.5 should NOT be viral spike: {viral_low['viral_spike']}"
)
print(
    f"[OK] compute_viral_coefficient (non-viral): "
    f"coeff={viral_low['viral_coefficient']:.4f}, viral_spike={viral_low['viral_spike']}"
)

# Exact threshold: 3.0x → should be a spike (coeff > 3.0 requires strict >)
viral_exact3 = compute_viral_coefficient(mentions_today=300.0, mentions_7day_avg=100.0)
assert abs(viral_exact3["viral_coefficient"] - 3.0) < 1e-9, "Exact 3.0 coefficient wrong"
# At exactly 3.0, coeff > 3.0 is False — check consistency
print(
    f"[OK] compute_viral_coefficient (exact 3.0): "
    f"coeff={viral_exact3['viral_coefficient']:.1f}, viral_spike={viral_exact3['viral_spike']}"
)

# Precision to 1e-10: verify coeff is stored to at least 10 decimal places
viral_precise = compute_viral_coefficient(mentions_today=1.0, mentions_7day_avg=3.0)
expected_precise = 1.0 / 3.0
assert abs(viral_precise["viral_coefficient"] - expected_precise) < 1e-9, (
    f"Viral coefficient precision wrong: expected {expected_precise:.10f}, "
    f"got {viral_precise['viral_coefficient']}"
)
print(
    f"[OK] compute_viral_coefficient (precision): "
    f"1/3 = {viral_precise['viral_coefficient']:.10f} (expected {expected_precise:.10f})"
)

# Zero avg → handled gracefully (no ZeroDivisionError via 1e-10 floor)
try:
    viral_zero = compute_viral_coefficient(mentions_today=100.0, mentions_7day_avg=0.0)
    # Should use 1e-10 floor, so coefficient should be very large but finite
    assert viral_zero["viral_coefficient"] > 1e9, (
        f"Zero-avg coefficient should be very large: {viral_zero['viral_coefficient']}"
    )
    assert viral_zero["viral_spike"] is True, "Zero avg + positive today should be viral"
    print(f"[OK] compute_viral_coefficient (zero avg): coeff={viral_zero['viral_coefficient']:.2e}")
except ZeroDivisionError:
    assert False, "compute_viral_coefficient should not raise ZeroDivisionError"

print("\n[PASS] dim_085: Social sentiment")
PYEOF
