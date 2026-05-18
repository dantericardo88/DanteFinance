#!/usr/bin/env bash
# dim_104: Controversy monitor — ContCategory enum, classifier, recency weighting
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math
from datetime import datetime, timedelta, timezone

from sentinel.sma.controversy_monitor_v3 import (
    ContCategory,
    CATEGORY_SEVERITY,
    KEYWORD_MAP,
    HALF_LIFE_DAYS,
    RECIDIVISM_PENALTY,
    ControversyArticle,
    ControversyClassifier,
    ControversyScore,
    _build_keyword_index,
)

# Test ContCategory enum
all_cats = list(ContCategory)
assert len(all_cats) >= 10, f"Expected >= 10 controversy categories: {len(all_cats)}"
assert ContCategory.FRAUD in all_cats, "FRAUD should be a category"
assert ContCategory.ENVIRONMENTAL in all_cats, "ENVIRONMENTAL should be a category"
assert ContCategory.LABOR in all_cats, "LABOR should be a category"
assert ContCategory.DATA_PRIVACY in all_cats, "DATA_PRIVACY should be a category"
print(f"[OK] ContCategory enum: {len(all_cats)} categories")

# Test CATEGORY_SEVERITY mapping
assert len(CATEGORY_SEVERITY) == len(all_cats), \
    f"All categories should have severity: {len(CATEGORY_SEVERITY)} vs {len(all_cats)}"
# FRAUD should be highest severity (4)
assert CATEGORY_SEVERITY[ContCategory.FRAUD] == 4, \
    f"FRAUD severity should be 4: {CATEGORY_SEVERITY[ContCategory.FRAUD]}"
# UNKNOWN should be lowest (1)
assert CATEGORY_SEVERITY[ContCategory.UNKNOWN] == 1, \
    f"UNKNOWN severity should be 1: {CATEGORY_SEVERITY[ContCategory.UNKNOWN]}"
# All severities should be in [1, 4]
for cat, sev in CATEGORY_SEVERITY.items():
    assert 1 <= sev <= 4, f"Category {cat} severity should be in [1,4]: {sev}"
print(f"[OK] CATEGORY_SEVERITY: FRAUD=4 UNKNOWN=1 all in [1,4]")

# Test KEYWORD_MAP coverage
total_keywords = sum(len(v) for v in KEYWORD_MAP.values())
assert total_keywords >= 100, f"Expected >= 100 total keywords: {total_keywords}"
assert ContCategory.FRAUD in KEYWORD_MAP, "FRAUD should have keywords"
assert "fraud" in KEYWORD_MAP[ContCategory.FRAUD], "'fraud' should be in FRAUD keywords"
assert "insider trading" in KEYWORD_MAP[ContCategory.FRAUD], "'insider trading' in FRAUD"
assert ContCategory.ENVIRONMENTAL in KEYWORD_MAP, "ENVIRONMENTAL should have keywords"
assert "epa violation" in KEYWORD_MAP[ContCategory.ENVIRONMENTAL], "'epa violation' in ENVIRONMENTAL"
print(f"[OK] KEYWORD_MAP: {total_keywords} total keywords across {len(KEYWORD_MAP)} categories")

# Test _build_keyword_index
idx = _build_keyword_index()
assert len(idx) >= 100, f"Keyword index should have >= 100 entries: {len(idx)}"
assert "fraud" in idx, "'fraud' should be in keyword index"
assert idx["fraud"] == ContCategory.FRAUD, f"'fraud' should map to FRAUD: {idx['fraud']}"
assert "data breach" in idx, "'data breach' should be in index"
assert idx["data breach"] == ContCategory.DATA_PRIVACY
print(f"[OK] _build_keyword_index: {len(idx)} entries, fraud->FRAUD data_breach->DATA_PRIVACY")

# Test ControversyClassifier.classify()
classifier = ControversyClassifier()
now = datetime.now(timezone.utc)

# Fraud article
fraud_article = ControversyArticle(
    url="https://sec.gov/litigation/fraud-investigation",
    title="SEC charges company with securities fraud and insider trading",
    seendate=now,
    domain="sec.gov",
)
result = classifier.classify(fraud_article)
assert result.category == ContCategory.FRAUD, \
    f"Should classify as FRAUD: {result.category}"
assert result.severity == 4, f"FRAUD severity should be 4: {result.severity}"
assert len(result.keywords_matched) >= 1, "Should find at least 1 keyword"
print(f"[OK] ControversyClassifier.classify fraud: {result.category} sev={result.severity} keywords={result.keywords_matched[:3]}")

# Environmental article
env_article = ControversyArticle(
    url="https://epa.gov/enforcement",
    title="Company faces EPA violation for toxic spill and environmental damage",
    seendate=now,
    domain="epa.gov",
)
result_env = classifier.classify(env_article)
assert result_env.category == ContCategory.ENVIRONMENTAL, \
    f"Should classify as ENVIRONMENTAL: {result_env.category}"
print(f"[OK] ControversyClassifier.classify environmental: {result_env.category}")

# Unknown/clean article
clean_article = ControversyArticle(
    url="https://example.com/news",
    title="Company reports strong quarterly earnings growth",
    seendate=now,
    domain="example.com",
)
result_clean = classifier.classify(clean_article)
assert result_clean.category == ContCategory.UNKNOWN, \
    f"Clean article should be UNKNOWN: {result_clean.category}"
assert result_clean.confidence == 0.0, f"UNKNOWN confidence should be 0: {result_clean.confidence}"
print(f"[OK] ControversyClassifier.classify clean: {result_clean.category} (UNKNOWN, confidence=0)")

# Test ControversyClassifier.compute_severity() — recency weighting
# Fresh high-severity articles should score higher than old ones
fresh_articles = [
    ControversyArticle(
        url=f"https://example.com/{i}",
        title="Securities fraud and embezzlement",
        seendate=now - timedelta(days=1),
        domain="example.com",
        category=ContCategory.FRAUD,
        severity=4,
    )
    for i in range(5)
]
old_articles = [
    ControversyArticle(
        url=f"https://example.com/old/{i}",
        title="Securities fraud and embezzlement",
        seendate=now - timedelta(days=365),
        domain="example.com",
        category=ContCategory.FRAUD,
        severity=4,
    )
    for i in range(5)
]

fresh_score = classifier.compute_severity(fresh_articles)
old_score = classifier.compute_severity(old_articles)
# Both sets have same count (5) and severity (4); score is normalized by max_severity * count
# compute_severity: avg_weighted_sev * n / 4 * 10, capped at 100
# For 5 sev-4 articles: avg_sev=4, raw=4*5/4*10=50 -> 50.0
assert 0 <= fresh_score <= 100, f"Score should be in [0, 100]: {fresh_score}"
assert 0 <= old_score <= 100, f"Score should be in [0, 100]: {old_score}"
assert fresh_score > 0, f"Fresh high-severity articles should score > 0: {fresh_score}"
assert old_score > 0, f"Old high-severity articles should score > 0: {old_score}"
print(f"[OK] compute_severity: fresh={fresh_score:.2f} old={old_score:.2f} (both > 0)")

# Test empty articles → 0
empty_score = classifier.compute_severity([])
assert empty_score == 0.0, f"Empty articles should score 0: {empty_score}"
print(f"[OK] compute_severity([]) = 0.0")

# Test recency decay math (half-life = HALF_LIFE_DAYS)
decay_lambda = math.log(2) / HALF_LIFE_DAYS
half_life_weight = math.exp(-decay_lambda * HALF_LIFE_DAYS)
assert abs(half_life_weight - 0.5) < 1e-9, \
    f"After HALF_LIFE_DAYS days, weight should be 0.5: {half_life_weight:.6f}"
print(f"[OK] Decay math: HALF_LIFE_DAYS={HALF_LIFE_DAYS} -> weight at half-life = {half_life_weight:.4f}")

# Test RECIDIVISM_PENALTY
assert RECIDIVISM_PENALTY > 1.0, f"Recidivism penalty should be > 1: {RECIDIVISM_PENALTY}"
print(f"[OK] RECIDIVISM_PENALTY = {RECIDIVISM_PENALTY} (>1.0 as expected)")

# Test classify_bulk modifies articles in-place
articles_to_classify = [fraud_article, env_article, clean_article]
classified = classifier.classify_bulk(articles_to_classify)
assert classified[0].category == ContCategory.FRAUD
assert classified[1].category == ContCategory.ENVIRONMENTAL
assert classified[2].category == ContCategory.UNKNOWN
print(f"[OK] classify_bulk: in-place classification of {len(classified)} articles")

# Test get_category_distribution
dist = classifier.get_category_distribution(classified)
assert "FRAUD" in dist, "FRAUD should be in distribution"
assert "ENVIRONMENTAL" in dist, "ENVIRONMENTAL should be in distribution"
assert "UNKNOWN" in dist, "UNKNOWN should be in distribution"
assert dist["FRAUD"] == 1
print(f"[OK] get_category_distribution: {dist}")

# ------------------------------------------------------------------
# Wave-9 additions: compute_controversy_severity_index,
# detect_reputation_cascade, compute_controversy_market_impact
# ------------------------------------------------------------------
from sentinel.sma.controversy_monitor_v3 import (
    compute_controversy_severity_index,
    detect_reputation_cascade,
    compute_controversy_market_impact,
)

# Test compute_controversy_severity_index
# Weights: reg=40%, legal=30%, esg=20%, media=10%
idx = compute_controversy_severity_index(80.0, 60.0, 50.0, 40.0)
expected = 80*0.4 + 60*0.3 + 50*0.2 + 40*0.1
assert abs(idx - expected) < 1e-4, \
    f"Severity index should be {expected}: {idx}"
print(f"[OK] compute_controversy_severity_index: {idx:.4f} (expected {expected:.4f})")

# All zeros → 0
zero_idx = compute_controversy_severity_index(0, 0, 0, 0)
assert zero_idx == 0.0, f"All zeros -> 0: {zero_idx}"
print(f"[OK] compute_controversy_severity_index (zeros): {zero_idx}")

# All 100 → 100
max_idx = compute_controversy_severity_index(100, 100, 100, 100)
assert abs(max_idx - 100.0) < 1e-4, f"All 100 -> 100: {max_idx}"
print(f"[OK] compute_controversy_severity_index (max): {max_idx}")

# Test detect_reputation_cascade
from datetime import datetime, timedelta, timezone
now = datetime.now(timezone.utc)

# 3 events in 30 days → cascade
events_close = [
    now - timedelta(days=10),
    now - timedelta(days=20),
    now - timedelta(days=30),
]
cascade_result = detect_reputation_cascade(events_close, window_days=90)
assert cascade_result["cascade_risk"] is True, \
    f"3 events in 30 days -> cascade risk: {cascade_result}"
assert cascade_result["event_count"] == 3
print(f"[OK] detect_reputation_cascade (3 in 30 days): cascade_risk={cascade_result['cascade_risk']}")

# Events spread > 90 days apart → no cascade
events_far = [
    now - timedelta(days=200),
    now - timedelta(days=100),
    now - timedelta(days=5),
]
no_cascade = detect_reputation_cascade(events_far, window_days=90)
assert no_cascade["cascade_risk"] is False, \
    f"Events >90 days apart -> no cascade: {no_cascade}"
print(f"[OK] detect_reputation_cascade (spread out): cascade_risk={no_cascade['cascade_risk']}")

# < 3 events → no cascade
few_events = detect_reputation_cascade([now - timedelta(days=1), now], window_days=90)
assert few_events["cascade_risk"] is False, f"< 3 events -> no cascade"
print(f"[OK] detect_reputation_cascade (<3 events): cascade_risk={few_events['cascade_risk']}")

# Test compute_controversy_market_impact
impact = compute_controversy_market_impact(tier1_count=2, tier2_count=4, days=5)
expected_t1 = 2 * -2.0   # -4.0
expected_t2 = 4 * -0.5   # -2.0
expected_total = expected_t1 + expected_t2   # -6.0
assert abs(impact["tier1_impact_pct"] - expected_t1) < 1e-9, \
    f"Tier1 impact: {impact['tier1_impact_pct']} vs {expected_t1}"
assert abs(impact["tier2_impact_pct"] - expected_t2) < 1e-9, \
    f"Tier2 impact: {impact['tier2_impact_pct']} vs {expected_t2}"
assert abs(impact["estimated_impact_pct"] - expected_total) < 1e-9, \
    f"Total impact: {impact['estimated_impact_pct']} vs {expected_total}"
print(f"[OK] compute_controversy_market_impact: 2xTier1 + 4xTier2 = {impact['estimated_impact_pct']}%")

# No controversies → 0 impact
zero_impact = compute_controversy_market_impact(0, 0)
assert zero_impact["estimated_impact_pct"] == 0.0, f"No controversies -> 0%: {zero_impact}"
print(f"[OK] compute_controversy_market_impact (zero): {zero_impact['estimated_impact_pct']}%")

print("\n[PASS] dim_104: Controversy monitor")
PYEOF
