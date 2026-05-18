#!/bin/bash
# dim_045: fed_speech_nlp — Fed/ECB/BoE speech NLP hawkish/dovish analysis
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.fed_speech_nlp import (
    FED_BASE,
    FED_SPEECHES_JSON,
    HAWKISH_TERMS,
    DOVISH_TERMS,
    THEME_KEYWORDS,
    compute_tone,
    extract_key_passages,
    identify_themes,
    extract_rate_signal,
    SpeechAnalysis,
    FedSpeechNLP,
)
from datetime import date

# --- constants ---
assert "federalreserve.gov" in FED_BASE
assert "federalreserve.gov" in FED_SPEECHES_JSON
print("[OK] FED_BASE and FED_SPEECHES_JSON constants present")

# --- lexicons ---
assert len(HAWKISH_TERMS) > 5
assert len(DOVISH_TERMS) > 5
assert all(isinstance(t, tuple) and len(t) == 2 for t in HAWKISH_TERMS[:3])
assert all(isinstance(t, tuple) and len(t) == 2 for t in DOVISH_TERMS[:3])
print(f"[OK] HAWKISH_TERMS ({len(HAWKISH_TERMS)}) and DOVISH_TERMS ({len(DOVISH_TERMS)}) are term/weight tuples")

# --- THEME_KEYWORDS ---
assert "inflation" in THEME_KEYWORDS
assert "employment" in THEME_KEYWORDS
assert "monetary_policy" in THEME_KEYWORDS
print(f"[OK] THEME_KEYWORDS has {len(THEME_KEYWORDS)} themes")

# --- compute_tone: hawkish text ---
hawkish_text = (
    "Inflation remains elevated and well above our 2 percent target. "
    "We are committed to restrictive policy and higher for longer interest rates. "
    "Price stability is our primary mandate. We must remain vigilant about persistent inflation."
)
h_score, d_score, tone = compute_tone(hawkish_text)
assert tone in ("hawkish", "very_hawkish"), f"Expected hawkish, got {tone}"
assert h_score > d_score, f"Hawkish score should dominate: h={h_score}, d={d_score}"
print(f"[OK] Hawkish text => tone={tone} (h={h_score:.2f}, d={d_score:.2f})")

# --- compute_tone: dovish text ---
dovish_text = (
    "We can afford to be patient and wait for more data. "
    "The labor market has softened and disinflation is underway. "
    "It may be appropriate to ease monetary policy as inflation moderates. "
    "We are ready to cut rates if conditions warrant."
)
h2, d2, tone2 = compute_tone(dovish_text)
assert tone2 in ("dovish", "very_dovish"), f"Expected dovish, got {tone2}"
print(f"[OK] Dovish text => tone={tone2} (h={h2:.2f}, d={d2:.2f})")

# --- compute_tone: empty text -> neutral ---
h3, d3, tone3 = compute_tone("")
assert tone3 == "neutral"
assert h3 == 0.0 and d3 == 0.0
print("[OK] Empty text => neutral, zero scores")

# --- extract_key_passages ---
passages = extract_key_passages(
    hawkish_text,
    keywords=["inflation", "price stability"],
    n_sentences=2,
)
assert len(passages) <= 2
assert any("inflation" in p.lower() or "price stability" in p.lower() for p in passages)
print(f"[OK] extract_key_passages returned {len(passages)} relevant sentences")

# --- identify_themes ---
themed_text = (
    "Inflation expectations are well-anchored. The Federal Reserve is tightening financial conditions. "
    "The labor market remains resilient with low unemployment."
)
themes = identify_themes(themed_text)
assert isinstance(themes, list)
print(f"[OK] identify_themes returned: {themes}")

# --- extract_rate_signal ---
signal = extract_rate_signal(hawkish_text, tone="hawkish")
assert signal in ("hike", "hold", "cut", "data_dependent", "unknown")
print(f"[OK] extract_rate_signal => '{signal}'")

# --- FedSpeechNLP class structure ---
assert hasattr(FedSpeechNLP, "__init__")
assert hasattr(FedSpeechNLP, "analyze_speech") or hasattr(FedSpeechNLP, "analyze")
print("[OK] FedSpeechNLP class structure present")

print("\n[PASS] dim_045: fed_speech_nlp -- all checks passed")
PYEOF
