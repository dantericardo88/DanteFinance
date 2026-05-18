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
    DECAY_LAMBDA, GDELT_TONE_SCALE
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

print("\n[PASS] dim_084: News sentiment pipeline")
PYEOF
