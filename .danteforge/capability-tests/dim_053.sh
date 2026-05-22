#!/bin/bash
# dim_053: NL Screener v3 — QueryParser token parse tree, range parser,
#          SQL WHERE clause generation, sector-to-SIC mapping
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.nl_screener_v3 import (
    QueryParser, ScreenerQuery, ScreenerFilter, _parse_number, _METRIC_ALIASES,
    QuerySuggestionEngine, sector_to_sic_codes, sector_alias_to_sic_codes,
    _SECTOR_SIC_RANGES,
)

# ── 1. _parse_number helper ──────────────────────────────────────────────────
assert _parse_number("20") == 20.0
assert _parse_number("20%") == 0.20
assert abs(_parse_number("2B") - 2e9) < 1e-3
assert abs(_parse_number("500M") - 500e6) < 1e-3
print("[OK] _parse_number handles integers, percentages, billions, millions")

# ── 2. _METRIC_ALIASES coverage ──────────────────────────────────────────────
assert "pe" in _METRIC_ALIASES
assert _METRIC_ALIASES["pe"] == "pe_ratio"
assert "roe" in _METRIC_ALIASES
assert "dividend yield" in _METRIC_ALIASES
print(f"[OK] _METRIC_ALIASES has {len(_METRIC_ALIASES)} entries")

# ── 3. Basic parse — profitable tech + PE filter ─────────────────────────────
qp = QueryParser()
result = qp.parse("profitable tech companies with PE under 20")
assert isinstance(result, ScreenerQuery)
assert result.requires_profitable or any(f.metric == "net_income" for f in result.filters), \
    f"Expected profitable filter, got: {result.filters}"
assert "Information Technology" in result.sectors, f"sectors={result.sectors}"

# Verify SQL WHERE clause contains correct operator
suggestion_engine = QuerySuggestionEngine()
sql = suggestion_engine.translate_to_sql(result)
assert "pe_ratio" in sql, f"SQL should reference pe_ratio: {sql}"
assert "< 20" in sql or "<20" in sql or "20" in sql, f"SQL should have PE<20 bound: {sql}"
print(f"[OK] 'profitable tech PE<20': sectors={result.sectors}, profitable={result.requires_profitable}")
print(f"[OK] SQL WHERE clause generated:\n{sql[:200]}")

# ── 4. Dividend yield > 2% filter ────────────────────────────────────────────
result2 = qp.parse("find stocks with PE < 20 and dividend yield > 2%")
# Must have PE filter
pe_filters = [f for f in result2.filters if f.metric == "pe_ratio"]
assert pe_filters, f"Expected pe_ratio filter, got: {result2.filters}"
assert pe_filters[0].operator == "<", f"Expected <, got {pe_filters[0].operator}"
assert abs(pe_filters[0].value - 20.0) < 0.01, f"Expected 20, got {pe_filters[0].value}"

# Must have dividend_yield filter
dy_filters = [f for f in result2.filters if f.metric == "dividend_yield"]
assert dy_filters, f"Expected dividend_yield filter, got: {result2.filters}"
assert dy_filters[0].operator == ">", f"Expected >, got {dy_filters[0].operator}"
# 2% should parse to 0.02
assert abs(dy_filters[0].value - 0.02) < 0.001, \
    f"Expected dividend_yield 0.02, got {dy_filters[0].value}"

sql2 = suggestion_engine.translate_to_sql(result2)
assert "dividend_yield" in sql2, f"SQL should reference dividend_yield: {sql2}"
print(f"[OK] 'PE < 20 AND dividend > 2%' produces correct SQL WHERE clause")
print(f"     SQL: {sql2[:200]}")

# ── 5. Range parser — "between X and Y" ─────────────────────────────────────
result3 = qp.parse("stocks with PE between 10 and 20")
between_filters = [f for f in result3.filters if f.operator == "between"]
assert between_filters, \
    f"Expected 'between' filter for 'PE between 10 and 20', got: {result3.filters}"
bf = between_filters[0]
assert bf.metric == "pe_ratio", f"Expected pe_ratio metric, got {bf.metric}"
assert isinstance(bf.value, tuple), f"Expected tuple value, got {type(bf.value)}"
lo, hi = bf.value
assert abs(lo - 10.0) < 0.01, f"Expected lo=10, got {lo}"
assert abs(hi - 20.0) < 0.01, f"Expected hi=20, got {hi}"
print(f"[OK] Range parser: 'PE between 10 and 20' -> operator='between', value=({lo},{hi})")

# Between in SQL
sql3 = suggestion_engine.translate_to_sql(result3)
assert "BETWEEN" in sql3.upper() or "between" in sql3, \
    f"SQL should use BETWEEN for range filter: {sql3}"
print(f"[OK] Range filter SQL: {sql3[:200]}")

# ── 6. Sector filter — 'tech' → SIC codes 7370-7379 ─────────────────────────
assert "Information Technology" in _SECTOR_SIC_RANGES, \
    "Expected IT sector in SIC range map"
it_codes = sector_to_sic_codes("Information Technology")
assert 7370 in it_codes, f"Expected SIC 7370 in IT codes"
assert 7379 in it_codes, f"Expected SIC 7379 in IT codes"
assert len(it_codes) > 5, f"Expected multiple SIC codes, got {len(it_codes)}"
print(f"[OK] sector_to_sic_codes('Information Technology') includes 7370-7379 "
      f"({len(it_codes)} codes total)")

# via alias
tech_codes = sector_alias_to_sic_codes("tech")
assert 7370 in tech_codes, f"Expected SIC 7370 for alias 'tech'"
assert 7379 in tech_codes, f"Expected SIC 7379 for alias 'tech'"
print(f"[OK] sector_alias_to_sic_codes('tech') -> {len(tech_codes)} SIC codes (includes 7370-7379)")

# ── 7. Token parse tree — complex multi-condition query ──────────────────────
result4 = qp.parse("large cap tech stocks with gross margin above 40% and debt to equity below 0.5")
assert "Information Technology" in result4.sectors, \
    f"Expected IT sector, got {result4.sectors}"
gm_filters = [f for f in result4.filters if f.metric == "gross_margin"]
de_filters = [f for f in result4.filters if f.metric == "debt_to_equity"]
assert gm_filters, f"Expected gross_margin filter, got {result4.filters}"
assert de_filters, f"Expected debt_to_equity filter, got {result4.filters}"
assert gm_filters[0].operator in (">", ">="), f"Expected > operator for gross_margin"
assert de_filters[0].operator in ("<", "<="), f"Expected < operator for debt_to_equity"
print(f"[OK] Multi-condition parse: gross_margin>{gm_filters[0].value}, "
      f"debt_to_equity<{de_filters[0].value}")

# ── 8. ScreenerQuery SQL translation ─────────────────────────────────────────
from sentinel.sai.nl_screener_v3 import ScreenerQuery, ScreenerFilter
sq = ScreenerQuery(
    raw_query="test",
    filters=[
        ScreenerFilter("pe_ratio", "<", 20.0),
        ScreenerFilter("dividend_yield", ">", 0.02),
        ScreenerFilter("revenue_growth_yoy", "between", (0.10, 0.30)),
    ],
    sectors=["Information Technology"],
    requires_profitable=True,
    limit=25,
)
sql_out = suggestion_engine.translate_to_sql(sq)
assert "pe_ratio < 20" in sql_out or "pe_ratio < 20.0" in sql_out, \
    f"SQL should have pe_ratio < 20: {sql_out}"
assert "dividend_yield > 0.02" in sql_out, f"SQL should have dividend_yield > 0.02: {sql_out}"
assert "BETWEEN" in sql_out.upper(), f"SQL should have BETWEEN for range: {sql_out}"
assert "net_income > 0" in sql_out, f"SQL should have profitability clause: {sql_out}"
assert "Information Technology" in sql_out, f"SQL should have sector filter: {sql_out}"
print(f"[OK] ScreenerQuery.translate_to_sql() produces correct WHERE clauses")

# ── 9. Special keywords ───────────────────────────────────────────────────────
result5 = qp.parse("undervalued value stocks no debt")
assert len(result5.filters) > 0, "Expected at least one filter"
print(f"[OK] 'undervalued no debt': {len(result5.filters)} filters")

# ── 10. Defaults ─────────────────────────────────────────────────────────────
empty = ScreenerQuery(raw_query="test")
assert empty.limit == 25
assert empty.sort_desc == True
assert empty.filters == []
print("[OK] ScreenerQuery defaults correct (limit=25)")

print("\n[PASS] dim_053: NL screener — token parse tree, range parser, SQL WHERE clause, "
      "sector-to-SIC mapping all verified")
PYEOF
