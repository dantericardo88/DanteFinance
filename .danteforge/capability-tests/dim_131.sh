#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
python - <<'PYEOF'
import sys, os, numpy as np, time
sys.path.insert(0, os.getcwd())

from sentinel.sai.web_signals_v3 import (
    WebArticle, SignalPoint, SentimentAggregator, SignalTimeSeries,
    WebSignalGenerator, score_text, recency_weight, aggregate_sentiment, web_signal
)

now = time.time()
day = 3600 * 24

# Create articles
pos_articles = [
    WebArticle("url1", "Strong growth beats estimates", "record revenue growth beat robust momentum", "bloomberg", now - 3600, "AAPL"),
    WebArticle("url2", "Company exceeds expectations", "bullish rally strong beat upgrade positive gain", "reuters", now - 7200, "AAPL"),
    WebArticle("url3", "Surge in sales", "strong surge growth exceed robust record bullish", "seeking_alpha", now - 10800, "AAPL"),
]
neg_articles = [
    WebArticle("url4", "Weak results miss estimates", "decline weak miss bearish downgrade loss risk", "reddit", now - 3600, "AAPL"),
    WebArticle("url5", "Disappointing quarter", "headwind concern drop fall disappoint weak sell decline", "twitter", now - 7200, "AAPL"),
    WebArticle("url6", "Revenue declines", "decline fall loss weak bearish risk concern miss", "reddit", now - 10800, "AAPL"),
]
all_articles = pos_articles + neg_articles

# score_article
agg = SentimentAggregator()
score_pos = agg.score_article(pos_articles[0])
score_neg = agg.score_article(neg_articles[0])
assert score_pos > 0, f"Bloomberg positive should score > 0, got {score_pos:.3f}"
assert score_neg < 0, f"Reddit negative should score < 0, got {score_neg:.3f}"

# aggregate: Bloomberg/Reuters positive outweigh Reddit/Twitter negative
sp = agg.aggregate(all_articles, decay_halflife_hours=24.0)
assert isinstance(sp, SignalPoint)
assert sp.sentiment > 0, f"Weighted aggregate should be positive (Bloomberg > Reddit), got {sp.sentiment:.3f}"
assert 0 <= sp.confidence <= 1
assert sp.n_articles == 6

# by_source
by_src = agg.by_source(all_articles)
assert "bloomberg" in by_src or "reuters" in by_src
assert isinstance(list(by_src.values())[0], float)

# recency_weight
w_recent = recency_weight(now, now, halflife_hours=24.0)
w_old = recency_weight(now - 7 * day, now, halflife_hours=24.0)
assert abs(w_recent - 1.0) < 1e-6, f"Recent weight should be ~1.0, got {w_recent}"
assert w_old < 0.1, f"7-day-old weight should be < 0.1, got {w_old:.4f}"

# SignalTimeSeries
points = []
for i, sent in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
    points.append(SignalPoint(float(i), sent, 0.8, 5, {}))
sts = SignalTimeSeries(points)
arr = sts.sentiment_array()
assert len(arr) == 7
mom = sts.momentum(window=3)
assert len(mom) > 0
slope = sts.trend()
assert slope > 0, f"Upward trend should have positive slope, got {slope:.4f}"
regime = sts.regime()
assert regime in ("bullish", "bearish", "neutral")

# WebSignalGenerator
wsg = WebSignalGenerator()
extreme_pos = np.array([0.8, 0.9, 0.85, 0.88, 0.92])
contrarian = wsg.contrarian_signal(extreme_pos)
assert contrarian < 0, f"Extreme positive -> negative contrarian, got {contrarian}"
mom_sig = wsg.momentum_signal(arr)
assert np.isfinite(mom_sig)
vol_sig = wsg.volume_signal(np.array([10, 12, 9, 11, 50]))  # spike at end
assert np.isfinite(vol_sig)
combined = wsg.combined_signal(points)
assert np.isfinite(combined)

# Module-level functions
st = score_text("strong growth beat record bullish rally")
assert st > 0
st2 = score_text("decline weak miss bearish downgrade loss")
assert st2 < 0
agg_sent = aggregate_sentiment(all_articles)
assert np.isfinite(agg_sent)
ws = web_signal(all_articles)
assert np.isfinite(ws)

print(f"Bloomberg pos score: {score_pos:.3f}, Reddit neg: {score_neg:.3f}")
print(f"Weighted aggregate: {sp.sentiment:.3f} (positive)")
print(f"Recency: recent={w_recent:.3f}, 7d-old={w_old:.4f}")
print(f"Trend slope: {slope:.4f}, Regime: {regime}")
print("[PASS] dim_131: Web signals analytics framework")
PYEOF
