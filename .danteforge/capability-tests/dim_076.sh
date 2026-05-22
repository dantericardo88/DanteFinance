#!/usr/bin/env bash
# dim_076: NL screener — pure query parsing logic + relative query / sector synonyms / explanation
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.nl_screener_v3 import (
    QueryParser, ScreenerFilter, ScreenerQuery,
    _METRIC_ALIASES, _SECTOR_ALIASES, _SPECIAL_KEYWORDS,
    _parse_number,
)

parser = QueryParser()

# Test _parse_number utility
assert abs(_parse_number("20%") - 0.20) < 1e-6, "_parse_number('20%')"
assert abs(_parse_number("2.5x") - 2.5) < 1e-6, "_parse_number('2.5x')"
assert abs(_parse_number("5B") - 5e9) < 1, "_parse_number('5B')"
assert abs(_parse_number("500M") - 500e6) < 1, "_parse_number('500M')"
assert abs(_parse_number("15") - 15.0) < 1e-6, "_parse_number('15')"
print("[OK] _parse_number handles %, x, B, M, bare numbers")

# Test metric alias lookup
assert _METRIC_ALIASES["p/e"] == "pe_ratio", "p/e alias"
assert _METRIC_ALIASES["roe"] == "roe", "roe alias"
assert _METRIC_ALIASES["gross margin"] == "gross_margin", "gross margin alias"
print(f"[OK] _METRIC_ALIASES has {len(_METRIC_ALIASES)} entries")

# Test sector aliases
assert _SECTOR_ALIASES["tech"] == "Information Technology"
assert _SECTOR_ALIASES["banks"] == "Financials"
assert _SECTOR_ALIASES["pharma"] == "Health Care"
print(f"[OK] _SECTOR_ALIASES has {len(_SECTOR_ALIASES)} entries")

# Test special keywords map
assert "profitable" in _SPECIAL_KEYWORDS
assert "undervalued" in _SPECIAL_KEYWORDS
assert _SPECIAL_KEYWORDS["profitable"].operator == ">"
assert _SPECIAL_KEYWORDS["profitable"].metric == "net_income"
print(f"[OK] _SPECIAL_KEYWORDS has {len(_SPECIAL_KEYWORDS)} entries")

# Test QueryParser.parse
result = parser.parse("profitable tech companies with P/E under 20")
assert isinstance(result, ScreenerQuery), "Should return ScreenerQuery"
assert result.requires_profitable or any(f.metric == "net_income" for f in result.filters), \
    "Should detect profitability requirement"
pe_filters = [f for f in result.filters if f.metric == "pe_ratio"]
assert len(pe_filters) >= 1, f"Should detect P/E filter, got: {result.filters}"
assert pe_filters[0].operator in ("<", "<="), f"P/E should have < or <= op: {pe_filters[0].operator}"
assert pe_filters[0].value == 20.0, f"P/E value should be 20, got {pe_filters[0].value}"
print(f"[OK] parse('profitable tech companies with P/E under 20'): {len(result.filters)} filters, sectors={result.sectors}")

# Test sector detection
result2 = parser.parse("large cap healthcare stocks with dividend yield above 3%")
assert "Health Care" in result2.sectors or "health care" in str(result2.sectors).lower(), \
    f"Healthcare sector not detected: {result2.sectors}"
div_filters = [f for f in result2.filters if f.metric == "dividend_yield"]
assert len(div_filters) >= 1, f"Should detect dividend yield filter: {result2.filters}"
print(f"[OK] parse healthcare+dividend: sectors={result2.sectors}, {len(result2.filters)} filters")

# Test limit extraction
result3 = parser.parse("top 10 value stocks by ROE")
assert result3.limit == 10, f"Should detect limit=10: {result3.limit}"
print(f"[OK] 'top 10' limit detection: {result3.limit}")

# Test ScreenerFilter evaluation
def evaluate_filter(f: ScreenerFilter, value: float) -> bool:
    ops = {"<": lambda a, b: a < b, ">": lambda a, b: a > b,
           "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
           "==": lambda a, b: abs(a - b) < 1e-9}
    return ops[f.operator](value, f.value)

pe_filter = ScreenerFilter(metric="pe_ratio", operator="<", value=20.0)
assert evaluate_filter(pe_filter, 15.0) == True
assert evaluate_filter(pe_filter, 25.0) == False
print("[OK] ScreenerFilter evaluation logic works correctly")

# ── NEW: parse_relative_query ─────────────────────────────────────────────────
# "top 10 by PE" -> sort=pe_ratio, ascending=False, limit=10
r = parser.parse_relative_query("top 10 by PE")
assert r["limit"] == 10, f"Expected limit=10, got {r['limit']}"
assert r["ascending"] is False, f"'top' should be descending (ascending=False)"
assert r["sort_by"] == "pe_ratio", f"Expected sort_by=pe_ratio, got {r['sort_by']}"
print(f"[OK] parse_relative_query('top 10 by PE'): limit={r['limit']} sort={r['sort_by']} asc={r['ascending']}")

# "bottom 5 by dividend yield" -> ascending=True
r2 = parser.parse_relative_query("bottom 5 by dividend yield")
assert r2["limit"] == 5, f"Expected limit=5, got {r2['limit']}"
assert r2["ascending"] is True, f"'bottom' should be ascending"
assert r2["sort_by"] == "dividend_yield", f"Expected dividend_yield, got {r2['sort_by']}"
print(f"[OK] parse_relative_query('bottom 5 by dividend yield'): limit={r2['limit']} asc={r2['ascending']}")

# "top 20 ROE stocks" -> limit=20, sort=roe
r3 = parser.parse_relative_query("top 20 ROE stocks")
assert r3["limit"] == 20, f"Expected limit=20, got {r3['limit']}"
assert r3["sort_by"] == "roe", f"Expected roe, got {r3['sort_by']}"
print(f"[OK] parse_relative_query('top 20 ROE stocks'): limit={r3['limit']} sort={r3['sort_by']}")

# ── NEW: expand_sector_synonyms ───────────────────────────────────────────────
# "tech", "technology", "software" -> all map to GICS sector 45 (Information Technology)
assert parser.expand_sector_synonyms("tech") == "Information Technology", "tech -> IT"
assert parser.expand_sector_synonyms("technology") == "Information Technology", "technology -> IT"
assert parser.expand_sector_synonyms("software") == "Information Technology", "software -> IT"
assert parser.expand_sector_synonyms("banks") == "Financials", "banks -> Financials"
assert parser.expand_sector_synonyms("pharma") == "Health Care", "pharma -> Health Care"
assert parser.expand_sector_synonyms("zzz_unknown") is None, "unknown -> None"
print("[OK] expand_sector_synonyms: tech/technology/software -> Information Technology")
print("[OK] expand_sector_synonyms: banks -> Financials, pharma -> Health Care, unknown -> None")

# ── NEW: generate_screener_explanation ───────────────────────────────────────
sq = ScreenerQuery(
    raw_query="profitable tech companies with PE < 20 and dividend > 2%",
    filters=[
        ScreenerFilter("pe_ratio", "<", 20.0),
        ScreenerFilter("dividend_yield", ">", 0.02),
        ScreenerFilter("net_income", ">", 0),
    ],
    sectors=["Information Technology"],
    limit=15,
    requires_profitable=True,
    requires_dividend=False,
    sort_by=None,
    sort_desc=True,
)
explanation = parser.generate_screener_explanation(sq)
assert isinstance(explanation, str), "Explanation should be a string"
assert len(explanation) > 10, "Explanation should not be empty"
assert "pe ratio" in explanation.lower() or "pe_ratio" in explanation.lower(), \
    f"Explanation should mention PE ratio: {explanation}"
assert "15" in explanation, f"Explanation should mention limit 15: {explanation}"
print(f"[OK] generate_screener_explanation: '{explanation}'")

# Verify it includes the dividend filter
assert "dividend" in explanation.lower(), f"Explanation missing dividend: {explanation}"
print("[OK] generate_screener_explanation: includes PE, dividend, limit")

# ── NEW: recursive-descent grammar parser (nl_grammar) ───────────────────────
from sentinel.sai.nl_grammar import (
    NLGrammarParser, parse_screener_query, is_complex_query,
    Comparison, BoolExpr,
)
gparser = NLGrammarParser()

# Simple comparison + field-alias normalisation
r = gparser.parse("pe < 20")
assert isinstance(r, Comparison), f"Expected Comparison, got {type(r)}"
assert r.field == "price_to_earnings", f"pe should alias -> price_to_earnings, got {r.field}"
assert r.op == "<", f"op should be <, got {r.op}"
assert r.value == 20.0, f"value should be 20.0, got {r.value}"
print(f"[OK] NLGrammarParser.parse('pe < 20'): {r}")

# AND composition with percent-suffix value
r = parse_screener_query("pe < 20 AND roe > 15%")
assert "and" in r, f"AND key missing: {r}"
assert len(r["and"]) == 2, f"AND should have 2 children: {r}"
# 15% should be coerced to 0.15
roe_dict = r["and"][1]
assert "return_on_equity" in roe_dict
assert roe_dict["return_on_equity"][">"] == 0.15, f"15% -> 0.15: {roe_dict}"
print(f"[OK] parse_screener_query('pe < 20 AND roe > 15%'): {r}")

# Nested parens with OR and NOT and dollar/B suffix
r = parse_screener_query("(pe < 20 OR mcap > 1b) AND NOT sector = energy")
assert "and" in r, f"Top-level AND missing: {r}"
assert any("or" in str(c) for c in r["and"]), f"OR branch missing: {r}"
assert any("not" in str(c) for c in r["and"]), f"NOT branch missing: {r}"
print(f"[OK] nested (OR + NOT + suffix): {r}")

# BETWEEN
r = parse_screener_query("pe BETWEEN 10 AND 20")
assert "between" in str(r), f"BETWEEN missing: {r}"
assert r["price_to_earnings"]["between"] == [10.0, 20.0], f"BETWEEN values: {r}"
print(f"[OK] parse_screener_query('pe BETWEEN 10 AND 20'): {r}")

# IN clause
r = parse_screener_query("sector IN (tech, energy, finance)")
assert "in" in str(r), f"IN missing: {r}"
assert len(r["sector"]["in"]) == 3, f"IN list length: {r}"
print(f"[OK] parse_screener_query('sector IN (tech, energy, finance)'): {r}")

# Multi-char operators
r = gparser.parse("market_cap >= 5b")
assert r.op == ">=", f"Expected >=, got {r.op}"
assert r.value == 5e9, f"Expected 5e9, got {r.value}"
print(f"[OK] multi-char op >= with 5b suffix: value={r.value}")

r = gparser.parse("eps != 0")
assert r.op == "!=", f"Expected !=, got {r.op}"
print(f"[OK] != operator: {r}")

# is_complex_query routing predicate
assert is_complex_query("pe < 20 AND roe > 15%"), "AND query should be complex"
assert is_complex_query("(pe < 20)"), "parenthesised should be complex"
assert is_complex_query("pe BETWEEN 10 AND 20"), "BETWEEN should be complex"
assert not is_complex_query("profitable tech companies"), \
    "plain English should NOT be complex"
print("[OK] is_complex_query routing predicate works")

# End-to-end: complex query routes through grammar in QueryParser.parse
sq = parser.parse("pe < 20 AND roe > 15%")
assert isinstance(sq, ScreenerQuery), "Complex query still returns ScreenerQuery"
metrics_seen = {f.metric for f in sq.filters}
assert "pe_ratio" in metrics_seen, f"pe_ratio missing from complex parse: {metrics_seen}"
assert "roe" in metrics_seen, f"roe missing from complex parse: {metrics_seen}"
print(f"[OK] QueryParser.parse routes complex query through grammar: {sq.filters}")

# Error handling
try:
    gparser.parse("pe <")
    raise AssertionError("Should have raised SyntaxError on incomplete query")
except SyntaxError:
    print("[OK] SyntaxError raised on malformed query")

try:
    gparser.parse("(pe < 20")
    raise AssertionError("Should have raised SyntaxError on unmatched paren")
except SyntaxError:
    print("[OK] SyntaxError raised on unmatched paren")

print("\n[PASS] dim_076: NL screener")
PYEOF
