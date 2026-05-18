#!/bin/bash
# dim_053: NL Screener v3 — QueryParser rule-based NL parsing (no network)
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.nl_screener_v3 import (
    QueryParser, ScreenerQuery, ScreenerFilter, _parse_number, _METRIC_ALIASES,
)

# 1. _parse_number helper
assert _parse_number("20") == 20.0
assert _parse_number("20%") == 0.20
assert abs(_parse_number("2B") - 2e9) < 1e-3
assert abs(_parse_number("500M") - 500e6) < 1e-3
print("[OK] _parse_number handles integers, percentages, billions, millions")

# 2. _METRIC_ALIASES coverage
assert "pe" in _METRIC_ALIASES
assert _METRIC_ALIASES["pe"] == "pe_ratio"
assert "roe" in _METRIC_ALIASES
assert "dividend yield" in _METRIC_ALIASES
print(f"[OK] _METRIC_ALIASES has {len(_METRIC_ALIASES)} entries")

# 3. QueryParser.parse() - profitable tech, PE filter
qp = QueryParser()
result = qp.parse("profitable tech companies with PE under 20")
assert isinstance(result, ScreenerQuery)
assert result.requires_profitable or any(f.metric == "net_income" for f in result.filters), \
    f"Expected profitable filter, got: {result.filters}"
assert "Information Technology" in result.sectors, f"sectors={result.sectors}"
print(f"[OK] 'profitable tech PE<20': sectors={result.sectors}, profitable={result.requires_profitable}")

# 4. Dividend filter
result2 = qp.parse("large cap dividend stocks with yield above 3%")
assert result2.requires_dividend or any(f.metric == "dividend_yield" for f in result2.filters), \
    f"filters={result2.filters}"
print(f"[OK] 'dividend yield>3%' parsed correctly")

# 5. Special keywords
result3 = qp.parse("undervalued value stocks no debt")
assert len(result3.filters) > 0, "Expected at least one filter"
print(f"[OK] 'undervalued no debt': {len(result3.filters)} filters")

# 6. ScreenerQuery defaults
empty = ScreenerQuery(raw_query="test")
assert empty.limit == 25
assert empty.sort_desc == True
assert empty.filters == []
print("[OK] ScreenerQuery defaults correct (limit=25)")

print("\n[PASS] dim_053: NL screener QueryParser verified")
PYEOF
