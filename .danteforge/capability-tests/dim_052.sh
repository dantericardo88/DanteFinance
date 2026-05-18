#!/bin/bash
# dim_052: FinBERT sentiment v3 — Loughran-McDonald lexicon scoring (no network)
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.finbert_sentiment_v3 import (
    LoughranMcDonaldLexicon, LMScore, SentimentScore,
    _LM_NEGATIVE, _LM_POSITIVE,
)

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

# Negative text should give negative net sentiment
neg_text = "The company suffered significant losses, fraud was discovered, and bankruptcy is imminent."
neg_result = lexicon.score(neg_text)
assert isinstance(neg_result, LMScore)
assert neg_result.total_words > 0
assert neg_result.negative_count > 0, f"Expected negative words, got: {neg_result.negative_count}"
assert neg_result.net_sentiment < 0, f"Expected negative net sentiment, got: {neg_result.net_sentiment}"
print(f"[OK] Negative text: negative_count={neg_result.negative_count}, net={neg_result.net_sentiment:.3f}")

# Positive text
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

print("\n[PASS] dim_052: FinBERT/LM sentiment — pure lexicon scoring verified")
PYEOF
