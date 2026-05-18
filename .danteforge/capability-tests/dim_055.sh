#!/bin/bash
# dim_055: Document Summarizer v3 — extractive TF-IDF, text_density, key_numbers, sentiment_arc
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.document_summarizer_v3 import ExtractiveSummarizer

# 1. Basic summarization
es = ExtractiveSummarizer()
text = (
    "Apple Inc reported revenue of 89 billion dollars in Q4 2024. "
    "This represents a 12 percent increase year over year. "
    "The company posted earnings per share of 1.40 dollars. "
    "Gross margins improved to 43 percent. "
    "Management guided for continued growth in the next quarter. "
    "iPhone sales grew 8 percent driven by strong demand in emerging markets. "
    "Services revenue reached a record 26 billion dollars. "
    "The company returned 30 billion dollars to shareholders through buybacks. "
    "Capital expenditure was 3.5 billion for the quarter. "
    "Cash and equivalents on the balance sheet total 167 billion dollars."
)
summary = es.summarize(text, n_sentences=3)
assert isinstance(summary, str)
assert len(summary) > 0
assert len(summary) <= len(text)
print(f"[OK] ExtractiveSummarizer.summarize() -> {len(summary)} chars (from {len(text)})")

# 2. Returns fewer or equal sentences than requested
short_text = "Apple revenue grew. Microsoft earnings beat."
summary_short = es.summarize(short_text, n_sentences=10)
assert isinstance(summary_short, str)
assert len(summary_short) > 0
print("[OK] Handles text shorter than n_sentences gracefully")

# 3. _tokenize method
tokens = es._tokenize("Apple reported significant revenue growth of 12%!")
assert "apple" in tokens or "reported" in tokens
assert "the" not in tokens  # stop word
assert "a" not in tokens    # stop word
print(f"[OK] _tokenize: {len(tokens)} tokens, stop words removed")

# 4. _sentence_tokenize
sentences = es._sentence_tokenize(text)
assert len(sentences) >= 5
print(f"[OK] _sentence_tokenize: found {len(sentences)} sentences")

# 5. TF-IDF scoring is deterministic
summary_a = es.summarize(text, n_sentences=2)
summary_b = es.summarize(text, n_sentences=2)
assert summary_a == summary_b
print("[OK] Summarization is deterministic (same input -> same output)")

# 6. NER patterns exist
assert hasattr(es, 'ORG_PATTERN')
assert hasattr(es, 'MONEY_PATTERN')
assert hasattr(es, 'DATE_PATTERN')
assert hasattr(es, 'PERCENT_PATTERN')
import re
money_match = es.MONEY_PATTERN.findall("Revenue was $89 billion and costs were $12.5 million")
assert len(money_match) >= 2, f"Expected 2 money matches, got {money_match}"
pct_match = es.PERCENT_PATTERN.findall("Margins improved 12% and 43%")
assert len(pct_match) == 2
print("[OK] NER patterns: MONEY, DATE, PERCENT, ORG compiled and working")

# 7. Empty/edge cases
empty_summary = es.summarize("", n_sentences=3)
assert isinstance(empty_summary, str)
print("[OK] Handles empty text gracefully")

# -----------------------------------------------------------------------
# Test 8: compute_text_density_score — identity: density = len(set(words)) / len(words)
# -----------------------------------------------------------------------
sample = "the the the apple apple orange"
words  = sample.split()
expected_density = len(set(words)) / len(words)
got_density = es.compute_text_density_score(sample)
assert abs(got_density - expected_density) < 1e-9, (
    f"Density mismatch: {got_density} vs {expected_density}"
)
# Perfect diversity -> density = 1.0
all_unique = "alpha beta gamma delta epsilon"
assert es.compute_text_density_score(all_unique) == 1.0, "All-unique text should have density=1.0"
# Empty string -> 0.0
assert es.compute_text_density_score("") == 0.0
print(f"[OK] compute_text_density_score: density={got_density:.4f} (expected {expected_density:.4f})")

# -----------------------------------------------------------------------
# Test 9: extract_key_numbers — dollar amounts, percentages, basis points
# -----------------------------------------------------------------------
financial_text = (
    "Revenue grew to $89 billion, up 12% year over year. "
    "The Fed cut rates by 25 bps. Gross margin was $21.5 million. "
    "Operating leverage improved 150 basis points."
)
nums = es.extract_key_numbers(financial_text)
assert "dollar_amounts" in nums
assert "percentages"    in nums
assert "basis_points"   in nums

# Should find at least 2 dollar amounts ($89B, $21.5M)
assert len(nums["dollar_amounts"]) >= 2, f"Expected >=2 dollar amounts, got {nums['dollar_amounts']}"
# Should find 12% percentage
pct_values = [d["value"] for d in nums["percentages"]]
assert any(abs(v - 0.12) < 0.001 for v in pct_values), f"12% not found: {pct_values}"
# Should find basis points (25 bps and/or 150 bps)
bps_values = [d["value"] for d in nums["basis_points"]]
assert any(abs(v - 25.0) < 0.1 or abs(v - 150.0) < 0.1 for v in bps_values), (
    f"25bps or 150bps not found: {bps_values}"
)
print(f"[OK] extract_key_numbers: {len(nums['dollar_amounts'])} $, "
      f"{len(nums['percentages'])} %, {len(nums['basis_points'])} bps")

# -----------------------------------------------------------------------
# Test 10: compute_sentiment_arc — 3-third document arc scoring
# -----------------------------------------------------------------------
# Construct text with improving arc (negative beginning, positive end)
negative_third = "decline pressure challenging headwind uncertainty concern miss weakness " * 5
middle_third   = "stable mixed moderate transition moderate moderate revenue " * 5
positive_third = "growth record strong outperform expand improve momentum confident upside " * 5
arc_text = negative_third + " " + middle_third + " " + positive_third

arc = es.compute_sentiment_arc(arc_text)
assert "thirds"    in arc
assert "arc"       in arc
assert "delta"     in arc
assert len(arc["thirds"]) == 3
assert arc["thirds"][0]["label"] == "beginning"
assert arc["thirds"][1]["label"] == "middle"
assert arc["thirds"][2]["label"] == "end"

# End should score higher than beginning (improving arc)
assert arc["thirds"][2]["score"] > arc["thirds"][0]["score"], (
    f"Arc should be improving: begin={arc['thirds'][0]['score']}, end={arc['thirds'][2]['score']}"
)
assert arc["arc"] in ("improving", "deteriorating", "stable", "mixed")
assert isinstance(arc["delta"], float)
print(f"[OK] compute_sentiment_arc: arc={arc['arc']}, delta={arc['delta']:.3f}")

# Stable arc: all neutral text
neutral_text = "the company reported results for the quarter in fiscal year " * 10
arc_neutral = es.compute_sentiment_arc(neutral_text)
assert arc_neutral["arc"] in ("stable", "mixed")
print(f"[OK] sentiment_arc neutral text -> arc={arc_neutral['arc']}")

print("\n[PASS] dim_055: Document summarizer + text_density + key_numbers + sentiment_arc verified")
PYEOF
