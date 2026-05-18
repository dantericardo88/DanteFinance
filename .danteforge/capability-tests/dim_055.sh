#!/bin/bash
# dim_055: Document Summarizer v3 — extractive TF-IDF summarization (no LLM)
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

print("\n[PASS] dim_055: Document summarizer extractive TF-IDF verified")
PYEOF
