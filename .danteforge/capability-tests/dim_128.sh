#!/usr/bin/env bash
# dim_128: Product-line / segment margin analytics with NLP
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.segment_margin_v3 import (
        Segment,
        MarginDecomposition,
        SegmentMarginAnalyzer,
        EarningsCallParser,
        segment_ebitda_margin,
        margin_decomposition,
        earnings_sentiment,
        extract_guidance,
    )
except ImportError as e:
    print(f"[NOT-BUILT] dim_128: import failed -- {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Setup: 3 segments per spec
# Software: rev=60M, ebitda=18M, vc=24M, fc=18M → margin 30%
# Hardware: rev=30M, ebitda=3M,  vc=21M, fc=6M  → margin 10%
# Services: rev=10M, ebitda=2M,  vc=5M,  fc=3M  → margin 20%
# ------------------------------------------------------------------
segments = [
    Segment("Software", 60_000_000.0, 18_000_000.0, 24_000_000.0, 18_000_000.0, year=2024),
    Segment("Hardware", 30_000_000.0,  3_000_000.0, 21_000_000.0,  6_000_000.0, year=2024),
    Segment("Services", 10_000_000.0,  2_000_000.0,  5_000_000.0,  3_000_000.0, year=2024),
]

# ------------------------------------------------------------------
# Test 1: Individual segment EBITDA margins
# ------------------------------------------------------------------
assert abs(segments[0].ebitda_margin - 0.30) < 1e-9, \
    f"Software margin {segments[0].ebitda_margin} != 0.30"
assert abs(segments[1].ebitda_margin - 0.10) < 1e-9, \
    f"Hardware margin {segments[1].ebitda_margin} != 0.10"
assert abs(segments[2].ebitda_margin - 0.20) < 1e-9, \
    f"Services margin {segments[2].ebitda_margin} != 0.20"
print("[OK] Segment EBITDA margins: Software=30%, Hardware=10%, Services=20%")

# Contribution margins
# Software: (60M-24M)/60M = 60%
assert abs(segments[0].contribution_margin - 0.60) < 1e-9, \
    f"Software contribution margin {segments[0].contribution_margin}"
# Hardware: (30M-21M)/30M = 30%
assert abs(segments[1].contribution_margin - 0.30) < 1e-9, \
    f"Hardware contribution margin {segments[1].contribution_margin}"
print("[OK] Contribution margins: Software=60%, Hardware=30%")

# Standalone segment_ebitda_margin
assert abs(segment_ebitda_margin(60_000_000.0, 18_000_000.0) - 0.30) < 1e-9
print("[OK] segment_ebitda_margin() standalone works")

# ------------------------------------------------------------------
# Test 2: company_ebitda_margin = (18+3+2)/(60+30+10) = 23%
# ------------------------------------------------------------------
analyzer = SegmentMarginAnalyzer(segments)
co_margin = analyzer.company_ebitda_margin()
assert 0.20 < co_margin < 0.28, \
    f"company_ebitda_margin {co_margin:.4f} not in (0.20, 0.28)"
assert abs(co_margin - 0.23) < 1e-9, \
    f"company_ebitda_margin {co_margin:.6f} != 0.23"
print(f"[OK] company_ebitda_margin = {co_margin:.2%} (expected 23%)")

# ------------------------------------------------------------------
# Test 3: highest/lowest margin segments
# ------------------------------------------------------------------
highest = analyzer.highest_margin_segment()
assert highest.name == "Software", \
    f"highest_margin_segment expected Software, got {highest.name}"
print(f"[OK] highest_margin_segment = {highest.name} ({highest.ebitda_margin:.0%})")

lowest = analyzer.lowest_margin_segment()
assert lowest.name == "Hardware", \
    f"lowest_margin_segment expected Hardware, got {lowest.name}"
print(f"[OK] lowest_margin_segment = {lowest.name} ({lowest.ebitda_margin:.0%})")

# ------------------------------------------------------------------
# Test 4: segment_quality_scores — all between 0 and 1
# ------------------------------------------------------------------
scores = analyzer.segment_quality_scores()
assert isinstance(scores, dict), "segment_quality_scores must return dict"
assert set(scores.keys()) == {"Software", "Hardware", "Services"}, \
    f"Unexpected keys: {scores.keys()}"
for seg_name, score in scores.items():
    assert 0.0 <= score <= 1.0, \
        f"Quality score for {seg_name} = {score} not in [0, 1]"
print(f"[OK] segment_quality_scores: {scores}")

# ------------------------------------------------------------------
# Test 5: margin_bridge (MarginDecomposition)
# ------------------------------------------------------------------
segments_prior = [
    Segment("Software", 55_000_000.0, 15_400_000.0, 22_000_000.0, 17_600_000.0, year=2023),
    Segment("Hardware", 28_000_000.0,  2_800_000.0, 19_600_000.0,  5_600_000.0, year=2023),
    Segment("Services",  9_000_000.0,  1_620_000.0,  4_500_000.0,  2_880_000.0, year=2023),
]
decomp = analyzer.margin_bridge(segments_prior)
assert isinstance(decomp, MarginDecomposition), "margin_bridge must return MarginDecomposition"
total_change = (60+30+10 - 55-28-9) * 1_000_000.0  # = 8M
assert abs(decomp.total_revenue_change - total_change) < 1.0, \
    f"total_revenue_change {decomp.total_revenue_change} != {total_change}"
print(f"[OK] margin_bridge: total_change=${decomp.total_revenue_change:,.0f}, "
      f"mix_effect=${decomp.mix_effect:,.0f}, "
      f"vol_effect=${decomp.volume_effect:,.0f}")

# ------------------------------------------------------------------
# Test 6: operating_leverage per segment
# ------------------------------------------------------------------
ol = analyzer.operating_leverage(segments_prior)
assert isinstance(ol, dict), "operating_leverage must return dict"
assert "Software" in ol and "Hardware" in ol and "Services" in ol
# Software: rev +9.09%, ebitda +16.88% → OL > 1
import math
sw_ol = ol["Software"]
assert not math.isnan(sw_ol), "Software operating_leverage should not be NaN"
print(f"[OK] operating_leverage: Software={sw_ol:.2f}")

# ------------------------------------------------------------------
# Test 7: EarningsCallParser — sentiment
# ------------------------------------------------------------------
parser = EarningsCallParser()

pos_text = "We had strong growth with record revenue and robust momentum"
pos_score = parser.sentiment_score(pos_text)
assert pos_score > 0, f"Positive text sentiment {pos_score} should be > 0"
print(f"[OK] sentiment_score (positive text) = {pos_score:.3f}")

neg_text = "Results were weak with significant headwinds and declining margins"
neg_score = parser.sentiment_score(neg_text)
assert neg_score < 0, f"Negative text sentiment {neg_score} should be < 0"
print(f"[OK] sentiment_score (negative text) = {neg_score:.3f}")

# ------------------------------------------------------------------
# Test 8: guidance_keywords
# ------------------------------------------------------------------
guidance_text = "We are raising guidance for the full year based on strong demand."
guidance = parser.guidance_keywords(guidance_text)
assert guidance["raised_guidance"] is True, \
    f"raised_guidance should be True, got {guidance}"
assert guidance["lowered_guidance"] is False, \
    f"lowered_guidance should be False for raising text"
print(f"[OK] guidance_keywords: raised={guidance['raised_guidance']}, "
      f"maintained={guidance['maintained_guidance']}, "
      f"lowered={guidance['lowered_guidance']}")

maintain_text = "We are maintaining our guidance for the full year."
g2 = parser.guidance_keywords(maintain_text)
assert g2["maintained_guidance"] is True, f"maintained_guidance should be True, got {g2}"
print(f"[OK] guidance_keywords (maintain): {g2}")

# ------------------------------------------------------------------
# Test 9: extract_keywords returns list
# ------------------------------------------------------------------
kw = parser.extract_keywords(pos_text + " " + neg_text, n=5)
assert isinstance(kw, list), "extract_keywords must return list"
assert len(kw) <= 5, f"extract_keywords returned {len(kw)} > 5"
print(f"[OK] extract_keywords: {kw}")

# ------------------------------------------------------------------
# Test 10: segment_mentions
# ------------------------------------------------------------------
mention_text = "Software revenue grew 20%. Hardware faced headwinds. Software remains our core."
mentions = parser.segment_mentions(mention_text, ["Software", "Hardware", "Services"])
assert mentions["Software"] == 2, f"Software mentions = {mentions['Software']} != 2"
assert mentions["Hardware"] == 1, f"Hardware mentions = {mentions['Hardware']} != 1"
assert mentions["Services"] == 0, f"Services mentions = {mentions['Services']} != 0"
print(f"[OK] segment_mentions: {mentions}")

# ------------------------------------------------------------------
# Test 11: extract_financial_numbers
# ------------------------------------------------------------------
num_text = "Revenue grew 15% to $1.2 billion, with margins at 23 basis points above target."
numbers = parser.extract_financial_numbers(num_text)
assert isinstance(numbers, list), "extract_financial_numbers must return list"
assert len(numbers) > 0, "extract_financial_numbers returned empty list"
units = [d["unit"] for d in numbers]
print(f"[OK] extract_financial_numbers: {len(numbers)} entries, units={units}")

# ------------------------------------------------------------------
# Test 12: Standalone convenience functions
# ------------------------------------------------------------------
sent = earnings_sentiment(pos_text)
assert sent > 0, f"earnings_sentiment() = {sent} should be > 0"
print(f"[OK] earnings_sentiment() standalone = {sent:.3f}")

g3 = extract_guidance("We are raising guidance and raising our outlook for Q4.")
assert g3["raised_guidance"] is True
print(f"[OK] extract_guidance() standalone: {g3}")

# margin_decomposition standalone
decomp2 = margin_decomposition(segments, segments_prior)
assert isinstance(decomp2, MarginDecomposition)
print(f"[OK] margin_decomposition() standalone: total_change=${decomp2.total_revenue_change:,.0f}")

print("\n[PASS] dim_128: Segment margin analytics")
PYEOF
