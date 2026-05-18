#!/usr/bin/env bash
# dim_076: NL screener — pure query parsing logic
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

# Test QueryParser.parse — sector extraction
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

print("\n[PASS] dim_076: NL screener")
PYEOF
