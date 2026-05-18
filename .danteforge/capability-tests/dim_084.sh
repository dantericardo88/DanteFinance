#!/usr/bin/env bash
# dim_084: News sentiment pipeline — pure sentiment scoring without network
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math
from datetime import datetime, timedelta, timezone

from sentinel.sma.news_sentiment_pipeline_v3 import (
    NewsArticle, NewsType, EarningsEvent, MAEvent,
    NewsEventClassifier, NewsSentimentPipeline,
    DECAY_LAMBDA, GDELT_TONE_SCALE,
    MarketReactionPredictor,
    detect_narrative_shift_from_series,
    normalize_entity_to_ticker,
    compute_volume_weighted_sentiment,
)

# Test NewsArticle creation and GDELT tone normalization
article = NewsArticle(
    title="Apple beats Q3 earnings expectations by 15%",
    url="https://example.com/aapl-earnings",
    domain="example.com",
    source="test",
    published_at=datetime.now(tz=timezone.utc),
    tickers_mentioned=["AAPL"],
    gdelt_tone=45.0,    # GDELT scale: +45 is moderately positive
)
assert abs(article.sentiment_score - (45.0 / GDELT_TONE_SCALE)) < 1e-9, \
    f"GDELT tone normalization: {article.sentiment_score:.4f} vs {45.0/GDELT_TONE_SCALE:.4f}"
print(f"[OK] NewsArticle: gdelt_tone=45 -> sentiment_score={article.sentiment_score:.4f}")

# Test time_weight (exponential decay)
# Fresh article: age_hours ~ 0, weight ~ 1.0
fresh_weight = article.time_weight(DECAY_LAMBDA)
assert fresh_weight > 0.98, f"Fresh article should have weight ~1: {fresh_weight:.4f}"
print(f"[OK] Fresh article time_weight: {fresh_weight:.4f}")

# Simulate 24-hour old article
old_pub = datetime.now(tz=timezone.utc) - timedelta(hours=24)
old_article = NewsArticle(
    title="Old news", url="https://example.com/old",
    published_at=old_pub, gdelt_tone=-20.0
)
old_weight = old_article.time_weight(DECAY_LAMBDA)
expected_old = math.exp(-DECAY_LAMBDA * 24)
assert abs(old_weight - expected_old) < 0.05, f"24h article weight mismatch: {old_weight:.4f} vs {expected_old:.4f}"
assert old_weight < fresh_weight, "Old article should have lower weight"
print(f"[OK] 24h old article time_weight: {old_weight:.4f} (expected ~{expected_old:.4f})")

# Test NewsEventClassifier extraction methods
classifier = NewsEventClassifier()

# Test earnings event extraction
earnings_article = NewsArticle(
    title="AAPL beats EPS estimate by 15%",
    url="https://example.com/e1",
    body="Apple beat EPS guidance for Q3 2024"
)
earnings = classifier.extract_earnings_event(earnings_article)
print(f"[OK] NewsEventClassifier.extract_earnings_event returned: {type(earnings).__name__}")

# Test analyst action extraction
analyst_article = NewsArticle(
    title="Goldman upgrades MSFT price target $500",
    url="https://example.com/e3",
    body="Analyst upgrade with outperform rating"
)
analyst = classifier.extract_analyst_action(analyst_article)
print(f"[OK] NewsEventClassifier.extract_analyst_action returned: {type(analyst).__name__}")

# Test M&A event extraction
ma_article = NewsArticle(
    title="Company merger deal acquisition",
    url="https://example.com/e2",
    body="The merger deal acquisition is expected to close Q4"
)
ma = classifier.extract_ma_event(ma_article)
print(f"[OK] NewsEventClassifier.extract_ma_event returned: {type(ma).__name__}")

# Test sentiment aggregation logic (pure math)
now = datetime.now(tz=timezone.utc)
articles = [
    NewsArticle(title="AAPL beats earnings", url="u1",
                published_at=now - timedelta(hours=2),
                tickers_mentioned=["AAPL"], gdelt_tone=60.0, sentiment_score=0.6),
    NewsArticle(title="Apple strong guidance", url="u2",
                published_at=now - timedelta(hours=5),
                tickers_mentioned=["AAPL"], gdelt_tone=40.0, sentiment_score=0.4),
    NewsArticle(title="Market uncertainty", url="u3",
                published_at=now - timedelta(hours=10),
                tickers_mentioned=["AAPL"], gdelt_tone=-10.0, sentiment_score=-0.1),
]

# Verify that time-weighted average is positive (articles with +score are more recent)
weighted_sum = sum(a.sentiment_score * a.time_weight() for a in articles)
weight_total = sum(a.time_weight() for a in articles)
avg_sentiment = weighted_sum / weight_total
assert avg_sentiment > 0, f"Net sentiment should be positive: {avg_sentiment:.4f}"
print(f"[OK] Weighted sentiment aggregation: {avg_sentiment:.4f} > 0")

# ---------------------------------------------------------------
# NEW: Market reaction model
# ---------------------------------------------------------------
predictor = MarketReactionPredictor()

# Positive sentiment + decent volume → positive expected move
move_pos = predictor.compute_market_reaction_model(
    sentiment_zscore=2.0, news_volume=20, alpha=0.001
)
move_neg = predictor.compute_market_reaction_model(
    sentiment_zscore=-2.0, news_volume=20, alpha=0.001
)
assert move_pos > 0, f"Positive sentiment should give positive move: {move_pos}"
assert move_neg < 0, f"Negative sentiment should give negative move: {move_neg}"
assert abs(move_pos + move_neg) < 1e-9, "Symmetric sentiment should cancel"
print(f"[OK] Market reaction model: pos={move_pos:.5f} neg={move_neg:.5f}")

# Zero volume → zero move
move_zero = predictor.compute_market_reaction_model(
    sentiment_zscore=3.0, news_volume=0, alpha=0.001
)
assert move_zero == 0.0, f"Zero volume should give zero move: {move_zero}"
print(f"[OK] Market reaction model: zero volume -> zero move")

# ---------------------------------------------------------------
# NEW: Rolling IC (information coefficient)
# ---------------------------------------------------------------
# Construct series where lag-1 IC is clearly positive:
# When sentiment_t is high, return_t+1 is high (1-day lag).
# sent=[−0.2,−0.1,0.1,0.2,0.3], returns=[X,−0.01,−0.005,0.01,0.015,0.02]
# sent[0:5] vs returns[1:6] → ranks align positively.
sentiments_pos = [-0.2, -0.1,  0.1,  0.2,  0.3]
returns_pos    = [ 0.0, -0.01, -0.005, 0.01, 0.015, 0.02]
ic = predictor.compute_rolling_ic(sentiments_pos, returns_pos, window=30)
assert ic > 0, f"IC should be positive (correct direction): {ic:.4f}"
print(f"[OK] Rolling IC: {ic:.4f} > 0 (sentiment predicts returns in correct direction)")

# Negative IC: high sentiment → negative next-day return
returns_neg = [ 0.0, 0.01, 0.005, -0.01, -0.015, -0.02]
ic_neg = predictor.compute_rolling_ic(sentiments_pos, returns_neg, window=30)
assert ic_neg < 0, f"Anti-correlated IC should be negative: {ic_neg:.4f}"
print(f"[OK] Rolling IC anti-corr: {ic_neg:.4f} < 0")

# Short series (< 3) → returns 0.0
ic_short = predictor.compute_rolling_ic([0.1], [0.01], window=30)
assert ic_short == 0.0, f"Short series should return 0.0: {ic_short}"
print(f"[OK] Rolling IC short series: 0.0")

# ---------------------------------------------------------------
# NEW: Narrative shift detection
# ---------------------------------------------------------------
# Stable series → no shift
stable = [0.1] * 30
assert not detect_narrative_shift_from_series(stable), "Stable series should not flag shift"
print(f"[OK] Narrative shift: stable series -> no shift")

# Large sudden delta → shift
shift_series = [0.0] * 29 + [2.5]  # massive jump at the end
assert detect_narrative_shift_from_series(shift_series, window=3, lookback=90), \
    f"Large delta should flag as narrative shift"
print(f"[OK] Narrative shift: large 3-day delta flagged as structural shift")

# Short series → no shift (insufficient data)
assert not detect_narrative_shift_from_series([0.1, 0.2], window=3), \
    "Too short series should not flag shift"
print(f"[OK] Narrative shift: short series -> no shift")

# ---------------------------------------------------------------
# NEW: Entity linking
# ---------------------------------------------------------------
assert normalize_entity_to_ticker("Apple") == "AAPL", \
    f"'Apple' should map to AAPL"
assert normalize_entity_to_ticker("AAPL") == "AAPL", \
    f"'AAPL' should map to AAPL"
assert normalize_entity_to_ticker("Apple Inc") == "AAPL", \
    f"'Apple Inc' should map to AAPL"
assert normalize_entity_to_ticker("apple inc.") == "AAPL", \
    f"'apple inc.' should map to AAPL (case-insensitive)"
assert normalize_entity_to_ticker("Google") == "GOOGL", \
    f"'Google' should map to GOOGL"
assert normalize_entity_to_ticker("UNKNOWN_COMPANY_XYZ") is None, \
    f"Unknown entity should return None"
print(f"[OK] Entity linking: Apple -> AAPL, Apple Inc -> AAPL, Google -> GOOGL, unknown -> None")

# ---------------------------------------------------------------
# NEW: Volume-weighted sentiment
# ---------------------------------------------------------------
# 3 articles at 0.1 + 1 article at 0.5 → weighted avg = (3*0.1+1*0.5)/(3+1) = 0.8/4 = 0.2
sentiments_vw = [0.1, 0.5]
counts_vw = [3, 1]
vw_avg = compute_volume_weighted_sentiment(sentiments_vw, counts_vw)
expected_vw = (3 * 0.1 + 1 * 0.5) / (3 + 1)
assert abs(vw_avg - expected_vw) < 1e-9, \
    f"Volume-weighted avg: {vw_avg:.4f} vs expected {expected_vw:.4f}"
print(f"[OK] Volume-weighted sentiment: {vw_avg:.4f} == {expected_vw:.4f}")

# Equal weights → same as simple average
vw_equal = compute_volume_weighted_sentiment([0.1, 0.3, 0.5], [1, 1, 1])
simple_avg = (0.1 + 0.3 + 0.5) / 3
assert abs(vw_equal - simple_avg) < 1e-9, "Equal weights should equal simple average"
print(f"[OK] Volume-weighted (equal weights) == simple average: {vw_equal:.4f}")

# Zero total count → simple average fallback
vw_zero = compute_volume_weighted_sentiment([0.2, 0.4], [0, 0])
assert abs(vw_zero - 0.3) < 1e-9, f"Zero counts should fall back to simple avg: {vw_zero}"
print(f"[OK] Volume-weighted (zero counts) -> simple average fallback: {vw_zero:.4f}")

print("\n[PASS] dim_084: News sentiment pipeline")
PYEOF
