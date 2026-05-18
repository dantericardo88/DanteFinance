#!/bin/bash
# dim_045: central_bank_nlp_v3 — upgraded hawk/dove NLP with bigrams, negation,
#          intensity modifiers, section context, tone change detector, FOMC parser
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.central_bank_nlp_v3 import (
    HAWKISH_TERMS,
    DOVISH_TERMS,
    INTENSITY_MODIFIERS,
    FOMC_PARTICIPANT_PHRASES,
    _NEGATION_TOKENS,
    HawkishDovishScorer,
    CentralBankDocument,
)
from datetime import date

scorer = HawkishDovishScorer()


# ── 1. Bigram phrases are present in lexicons ─────────────────────────────────
assert "tighten policy" in HAWKISH_TERMS, "Expected bigram 'tighten policy' in HAWKISH_TERMS"
assert "well anchored" in DOVISH_TERMS, "Expected bigram 'well anchored' in DOVISH_TERMS"
assert "remain patient" in DOVISH_TERMS, "Expected bigram 'remain patient' in DOVISH_TERMS"
assert "premature to cut" in HAWKISH_TERMS, "Expected bigram 'premature to cut' in HAWKISH_TERMS"
print(f"[OK] Bigram phrases present in HAWKISH_TERMS ({len(HAWKISH_TERMS)}) "
      f"and DOVISH_TERMS ({len(DOVISH_TERMS)})")


# ── 2. Hawkish text scores more hawkish than dovish text ─────────────────────
hawkish_text = (
    "Rate hike expected soon given above-target inflation. "
    "The committee will tighten policy as inflation remains significantly above target. "
    "We must act decisively to restore price stability. "
    "Ongoing increases in the target range will be appropriate."
)
dovish_text = (
    "A patient approach while monitoring data is warranted. "
    "Inflation expectations remain well anchored, allowing us to remain patient. "
    "We will monitor developments carefully before considering any changes."
)

h_score = scorer.score_sentence(hawkish_text)
d_score = scorer.score_sentence(dovish_text)
assert h_score > d_score, (
    f"Hawkish text should score higher than dovish text. "
    f"hawkish={h_score:.4f}, dovish={d_score:.4f}"
)
assert h_score > 0, f"Hawkish text should produce positive score, got {h_score:.4f}"
print(f"[OK] Hawkish text ({h_score:.4f}) > dovish text ({d_score:.4f})")


# ── 3. Negation inversion: "not considering rate hikes" < "considering rate hikes" ──
hike_text = "The committee is considering rate hikes at the next meeting."
neg_hike_text = "The committee is not considering rate hikes at the next meeting."

pos_score = scorer.score_sentence(hike_text)
neg_score = scorer.score_sentence(neg_hike_text)
assert neg_score < pos_score, (
    f"Negated hike text should score lower than non-negated. "
    f"negated={neg_score:.4f}, affirmed={pos_score:.4f}"
)
print(f"[OK] Negation inversion: 'not considering rate hikes' ({neg_score:.4f}) "
      f"< 'considering rate hikes' ({pos_score:.4f})")


# ── 4. Intensity modifiers scale the score ────────────────────────────────────
base_text = "Inflation is above target."
amplified_text = "Inflation is significantly above target."
dampened_text = "Inflation is modestly above target."

base_s = scorer.score_sentence(base_text)
amp_s = scorer.score_sentence(amplified_text)
damp_s = scorer.score_sentence(dampened_text)

assert amp_s > base_s, (
    f"'significantly' should amplify: amplified={amp_s:.4f} vs base={base_s:.4f}"
)
assert damp_s < base_s, (
    f"'modestly' should dampen: dampened={damp_s:.4f} vs base={base_s:.4f}"
)
print(f"[OK] Intensity modifiers: base={base_s:.4f}, "
      f"significantly={amp_s:.4f} (up), modestly={damp_s:.4f} (down)")

assert "significantly" in INTENSITY_MODIFIERS and INTENSITY_MODIFIERS["significantly"] > 1.0
assert "modestly" in INTENSITY_MODIFIERS and INTENSITY_MODIFIERS["modestly"] < 1.0
print(f"[OK] INTENSITY_MODIFIERS has {len(INTENSITY_MODIFIERS)} entries with correct multipliers")


# ── 5. Section context: policy outlook sentences weighted higher ───────────────
policy_outlook_sent = (
    "The committee expects that ongoing increases in the target range will be "
    "appropriate to return inflation to 2 percent."
)
econ_data_sent = (
    "CPI reading came in at 3.2 percent, above the prior month. "
    "Nonfarm payroll added 200,000 jobs, in line with expectations."
)

# Score as documents so section weighting is applied
doc_policy = CentralBankDocument(
    bank="FED", doc_type="statement",
    text=policy_outlook_sent, date=date.today()
)
doc_econ = CentralBankDocument(
    bank="FED", doc_type="statement",
    text=econ_data_sent, date=date.today()
)

hs_policy = scorer.score_document(doc_policy)
hs_econ = scorer.score_document(doc_econ)
# Policy outlook text should be classified and receive a weight multiplier
section_policy = scorer._classify_section(policy_outlook_sent)
section_econ = scorer._classify_section(econ_data_sent)
assert scorer._section_weight("policy_outlook") > scorer._section_weight("economic_data"), \
    "policy_outlook weight should exceed economic_data weight"
print(f"[OK] Section weighting: policy_outlook={scorer._section_weight('policy_outlook')}, "
      f"economic_data={scorer._section_weight('economic_data')}")
print(f"     policy_outlook sentence classified as: {section_policy}")
print(f"     economic_data sentence classified as: {section_econ}")


# ── 6. Tone change detector: 3-speech sequence going more hawkish → positive delta ──
dovish_doc = CentralBankDocument(
    bank="FED", doc_type="statement",
    text=(
        "The committee will remain patient, monitoring incoming data. "
        "A gradual approach is warranted while inflation remains well anchored. "
        "We remain supportive of maximum employment."
    ),
    date=date(2024, 1, 15),
)
neutral_doc = CentralBankDocument(
    bank="FED", doc_type="statement",
    text=(
        "The committee will continue to assess incoming data. "
        "Economic activity has been expanding at a moderate pace. "
        "The labor market remains strong."
    ),
    date=date(2024, 3, 20),
)
hawkish_doc = CentralBankDocument(
    bank="FED", doc_type="statement",
    text=(
        "Inflation remains elevated and materially above target. "
        "The committee will tighten policy further. "
        "Rate hike expected to restore price stability. "
        "We must act decisively and aggressively to combat persistent inflation."
    ),
    date=date(2024, 5, 1),
)

result = scorer.compute_tone_change_vs_rolling(
    hawkish_doc,
    prior_docs=[dovish_doc, neutral_doc],
    rolling_n=3,
)

assert "current_score" in result
assert "rolling_avg" in result
assert "delta" in result
assert "direction" in result
assert result["delta"] > 0, (
    f"Hawkish speech vs dovish/neutral rolling avg should yield positive delta, "
    f"got delta={result['delta']:.3f}"
)
assert result["direction"] == "hawkish_shift", (
    f"Expected 'hawkish_shift', got '{result['direction']}'"
)
print(f"[OK] Tone change detector: current={result['current_score']:.2f}, "
      f"rolling_avg={result['rolling_avg']:.2f}, delta={result['delta']:.3f}, "
      f"direction={result['direction']}")


# ── 7. FOMC minutes section parser counts participant phrases ─────────────────
minutes_text = (
    "Participants noted that inflation remained above target. "
    "Many participants agreed that further tightening would be appropriate. "
    "Some participants expressed concern about downside risks to growth. "
    "Most participants supported raising the target range at this meeting. "
    "Several participants observed that the labor market remained tight. "
    "Participants observed a need to maintain restrictive policy. "
    "Many members emphasized the importance of price stability. "
    "A few participants preferred a smaller increase at this meeting. "
    "Members noted that ongoing increases would be necessary. "
    "Participants agreed that the committee should remain vigilant about inflation."
)

minutes_result = scorer.parse_fomc_minutes_sections(minutes_text)

assert "participant_phrase_counts" in minutes_result
assert "total_participant_mentions" in minutes_result
assert "many_participants_count" in minutes_result
assert "some_participants_count" in minutes_result
assert "most_participants_count" in minutes_result
assert "consensus_strength" in minutes_result
assert "hawk_dove_from_minutes" in minutes_result

total_mentions = minutes_result["total_participant_mentions"]
assert total_mentions > 0, f"Should count participant mentions, got {total_mentions}"

many_count = minutes_result["many_participants_count"]
assert many_count > 0, f"Should count 'many participants/members', got {many_count}"

print(f"[OK] FOMC minutes parser: total_mentions={total_mentions}, "
      f"many={many_count}, "
      f"some={minutes_result['some_participants_count']}, "
      f"most={minutes_result['most_participants_count']}, "
      f"consensus={minutes_result['consensus_strength']}, "
      f"hawk_dove={minutes_result['hawk_dove_from_minutes']:.3f}")

# Verify specific phrase counts
phrase_counts = minutes_result["participant_phrase_counts"]
assert phrase_counts.get("participants noted", 0) > 0, "Should count 'participants noted'"
assert phrase_counts.get("many participants", 0) > 0, "Should count 'many participants'"
print(f"[OK] 'participants noted' count: {phrase_counts.get('participants noted', 0)}")
print(f"[OK] 'many participants' count:  {phrase_counts.get('many participants', 0)}")


# ── 8. FOMC_PARTICIPANT_PHRASES contains key phrases ─────────────────────────
assert len(FOMC_PARTICIPANT_PHRASES) >= 10
assert "participants noted" in FOMC_PARTICIPANT_PHRASES
assert "many participants" in FOMC_PARTICIPANT_PHRASES
assert "some participants" in FOMC_PARTICIPANT_PHRASES
assert "many members" in FOMC_PARTICIPANT_PHRASES
print(f"[OK] FOMC_PARTICIPANT_PHRASES has {len(FOMC_PARTICIPANT_PHRASES)} entries")


# ── 9. Negation tokens present ────────────────────────────────────────────────
assert "not" in _NEGATION_TOKENS
assert "cannot" in _NEGATION_TOKENS
assert len(_NEGATION_TOKENS) >= 15
print(f"[OK] _NEGATION_TOKENS has {len(_NEGATION_TOKENS)} entries")


print("\n[PASS] dim_045: central_bank_nlp_v3 — all upgraded NLP checks passed")
PYEOF
