#!/usr/bin/env bash
# dim_134: Sentiment evolution tracking (time-series NLP drift)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ------------------------------------------------------------------ #
# 1. Import from sai layer (legacy test interface)
# ------------------------------------------------------------------ #
try:
    from sentinel.sai.sentiment_evolution_v3 import (
        SentimentEvolution, SentimentDriftDetector, compute_sentiment_momentum
    )
    assert SentimentEvolution is not None, "Missing SentimentEvolution"
    assert SentimentDriftDetector is not None, "Missing SentimentDriftDetector"
    assert compute_sentiment_momentum is not None, "Missing compute_sentiment_momentum"
    print("[OK] SentimentEvolution present")
    print("[OK] SentimentDriftDetector present")
    print("[OK] compute_sentiment_momentum present")
except ImportError as e:
    print(f"[FAIL] Import from sentinel.sai.sentiment_evolution_v3 failed: {e}")
    sys.exit(1)

# ------------------------------------------------------------------ #
# 2. Import full API from sma layer
# ------------------------------------------------------------------ #
from sentinel.sma.sentiment_evolution_v3 import (
    TextSentimentAnalyzer, SentimentTimeSeries, SentimentPoint,
    SentimentEvolutionTracker, score_text, build_sentiment_series,
    sentiment_momentum, sentiment_regime, lead_lag_correlation,
)

# ------------------------------------------------------------------ #
# 3. score_text tests
# ------------------------------------------------------------------ #
pos_score = score_text("strong growth, record results, robust momentum")
assert pos_score > 0.5, f"positive text score={pos_score}, expected > 0.5"
print(f"[OK] score_text positive: {pos_score:.4f}")

neg_score = score_text("declining revenue, significant headwinds, disappointing results")
assert neg_score < -0.3, f"negative text score={neg_score}, expected < -0.3"
print(f"[OK] score_text negative: {neg_score:.4f}")

# ------------------------------------------------------------------ #
# 4. Build 25-text series (10 positive, 10 negative, 5 neutral)
# ------------------------------------------------------------------ #
positive_texts = [
    "strong growth beat expectations record results robust momentum gain",
    "exceeded targets solid performance positive outlook strengthen expand",
    "accelerate momentum bullish recovery raised guidance ahead of forecast",
    "outperform surge rally upbeat better than expected solid gain",
    "record revenue robust growth confident guidance raise increase",
    "strong beat exceeded recovery surge rally bullish solid momentum",
    "growth outperform positive improve accelerate solid raise ahead",
    "record performance confident expand gain strengthen upturn bullish",
    "momentum solid beat exceeded robust better growth raise recovery",
    "outperform strong bullish gain rally accelerate positive record surge",
]

negative_texts = [
    "decline headwind pressure challenging miss below expectations weak",
    "reduce decrease concern risk uncertain volatile loss impair downgrade",
    "disappointing bearish downturn soften compression struggle difficult",
    "deteriorate slowdown cautious below guidance cut disappointing weak",
    "miss pressure headwind volatile uncertain risk concern decline loss",
    "weak bearish deteriorate slowdown difficult struggle compression impair",
    "downgrade cut disappointing cautious reduce concern volatile decline",
    "headwind miss below challenging uncertain loss soften downturn bearish",
    "struggle difficult deteriorate slowdown risk volatile impair decline cut",
    "cautious below pressure challenging disappointing miss weak headwind",
]

neutral_texts = [
    "the company reported quarterly results in line with prior guidance",
    "management held the annual investor day discussing operational metrics",
    "results were reported as part of the scheduled earnings release",
    "the board meeting addressed standard governance and compliance topics",
    "quarterly filings were submitted to the regulator on schedule",
]

all_texts = positive_texts + negative_texts + neutral_texts

ts = build_sentiment_series(all_texts)
scores = ts.scores
assert scores.shape == (25,), f"scores.shape={scores.shape}, expected (25,)"
print(f"[OK] SentimentTimeSeries: scores shape={scores.shape}")

drift = ts.drift()
assert drift.shape == (24,), f"drift.shape={drift.shape}, expected (24,)"
print(f"[OK] drift shape={drift.shape}")

momentum_arr = ts.momentum(k=3)
assert len(momentum_arr) == 22, f"momentum length={len(momentum_arr)}, expected 22"
print(f"[OK] momentum(k=3) length={len(momentum_arr)}")

vol_arr = ts.volatility(window=5)
assert len(vol_arr) > 0, "volatility array should be non-empty"
print(f"[OK] volatility(window=5) length={len(vol_arr)}")

# ------------------------------------------------------------------ #
# 5. Regime: values in {-1, 0, 1}
# ------------------------------------------------------------------ #
regime_arr = ts.regime()
unique_regimes = set(int(r) for r in regime_arr)
assert unique_regimes.issubset({-1, 0, 1}), f"unexpected regime values: {unique_regimes}"
print(f"[OK] regime values in {{-1, 0, 1}}: unique={sorted(unique_regimes)}")

# convenience function
regime_arr2 = sentiment_regime(scores)
assert set(int(r) for r in regime_arr2).issubset({-1, 0, 1})
print("[OK] sentiment_regime convenience function works")

# ------------------------------------------------------------------ #
# 6. SentimentEvolutionTracker
# ------------------------------------------------------------------ #
tracker = SentimentEvolutionTracker(ts)

trend_val = tracker.trend()
assert np.isfinite(trend_val), f"trend not finite: {trend_val}"
print(f"[OK] trend: {trend_val:.6f}")

# ------------------------------------------------------------------ #
# 7. inflection_points: at least 1 (regime changes across pos/neg/neutral)
# ------------------------------------------------------------------ #
inflections = tracker.inflection_points()
assert len(inflections) >= 1, f"Expected >= 1 inflection point, got {len(inflections)}: {inflections}"
print(f"[OK] inflection_points: {len(inflections)} points at indices {inflections}")

# ------------------------------------------------------------------ #
# 8. lead_lag_correlation with synthetic returns
# ------------------------------------------------------------------ #
rng = np.random.default_rng(123)
returns = rng.normal(0, 0.01, 25)
ll = tracker.lead_lag_analysis(returns, max_lag=5)
assert "best_lag" in ll, f"'best_lag' not in lead_lag result: {list(ll.keys())}"
assert "best_correlation" in ll, "'best_correlation' not in result"
assert "lag_profile" in ll, "'lag_profile' not in result"
assert 0 <= ll["best_lag"] <= 5, f"best_lag={ll['best_lag']} not in [0,5]"
assert np.isfinite(ll["best_correlation"]), f"best_correlation not finite"
print(f"[OK] lead_lag_analysis: best_lag={ll['best_lag']}, best_corr={ll['best_correlation']:.4f}")

# convenience function
ll2 = lead_lag_correlation(scores, returns, max_lag=5)
assert "best_lag" in ll2 and 0 <= ll2["best_lag"] <= 5
print("[OK] lead_lag_correlation convenience function works")

# ------------------------------------------------------------------ #
# 9. regime_returns: {'bull', 'bear', 'neutral'} all present
# ------------------------------------------------------------------ #
rr = tracker.regime_returns(returns)
assert "bull" in rr, "'bull' missing from regime_returns"
assert "bear" in rr, "'bear' missing from regime_returns"
assert "neutral" in rr, "'neutral' missing from regime_returns"
for k, v in rr.items():
    assert np.isfinite(v), f"regime_returns['{k}']={v} not finite"
print(f"[OK] regime_returns: bull={rr['bull']:.6f}, bear={rr['bear']:.6f}, neutral={rr['neutral']:.6f}")

# ------------------------------------------------------------------ #
# 10. predict_next: AR(1) extrapolation is finite
# ------------------------------------------------------------------ #
pred = tracker.predict_next()
assert np.isfinite(pred), f"predict_next not finite: {pred}"
print(f"[OK] predict_next: {pred:.4f}")

# ------------------------------------------------------------------ #
# 11. SentimentEvolution (sai layer)
# ------------------------------------------------------------------ #
se = SentimentEvolution(all_texts)
assert se.scores.shape == (25,), f"SentimentEvolution.scores shape mismatch"
assert len(se.drift()) == 24, "SentimentEvolution.drift() length mismatch"
assert set(int(r) for r in se.regime()).issubset({-1, 0, 1})
assert np.isfinite(se.trend()), "SentimentEvolution.trend() not finite"
assert len(se.inflection_points()) >= 1, "SentimentEvolution needs >= 1 inflection"
print(f"[OK] SentimentEvolution: scores={se.scores.shape}, "
      f"inflections={len(se.inflection_points())}, trend={se.trend():.6f}")

# ------------------------------------------------------------------ #
# 12. SentimentDriftDetector (sai layer)
# ------------------------------------------------------------------ #
detector = SentimentDriftDetector(ts)
d_inflections = detector.inflection_points()
assert len(d_inflections) >= 1, "SentimentDriftDetector.inflection_points() needs >= 1"
zero_cross = detector.zero_crossings()  # may be 0
d_trend = detector.trend()
assert np.isfinite(d_trend)
print(f"[OK] SentimentDriftDetector: inflections={len(d_inflections)}, "
      f"zero_crossings={len(zero_cross)}, trend={d_trend:.6f}")

# from_texts constructor
det2 = SentimentDriftDetector.from_texts(all_texts)
assert len(det2.inflection_points()) >= 1
print("[OK] SentimentDriftDetector.from_texts() works")

# ------------------------------------------------------------------ #
# 13. compute_sentiment_momentum (sai convenience function)
# ------------------------------------------------------------------ #
mom = compute_sentiment_momentum(scores, k=3)
assert len(mom) == 22, f"compute_sentiment_momentum length={len(mom)}, expected 22"
assert np.all(np.isfinite(mom)), "momentum contains non-finite values"
print(f"[OK] compute_sentiment_momentum: length={len(mom)}")

# also via sma
mom2 = sentiment_momentum(scores, k=3)
assert np.allclose(mom, mom2), "sma vs sai momentum mismatch"
print("[OK] sma sentiment_momentum matches sai compute_sentiment_momentum")

# ------------------------------------------------------------------ #
# 14. TextSentimentAnalyzer.extract_topics
# ------------------------------------------------------------------ #
analyzer = TextSentimentAnalyzer()
topics = {
    "growth": ["growth", "expand", "accelerate"],
    "risk": ["risk", "uncertain", "volatile"],
}
topic_scores = analyzer.extract_topics(
    "strong growth uncertain risk expand", topics
)
assert "growth" in topic_scores and "risk" in topic_scores
assert topic_scores["growth"] > 0, "growth topic coverage should be > 0"
print(f"[OK] extract_topics: growth={topic_scores['growth']:.4f}, risk={topic_scores['risk']:.4f}")

print("\n[PASS] dim_134: Sentiment evolution tracking -- all checks passed")
PYEOF
