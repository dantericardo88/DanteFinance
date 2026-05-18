#!/usr/bin/env bash
# dim_098: VC/PE tracker — fund universe, check size tables, fund type logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.vcpe_tracker_enhanced import (
    _CHECK_SIZES,
    _BENCHMARK_RETURNS,
    _TVPI_BY_VINTAGE,
    _HOT_SICS,
    VCPEFundUniverse,
)

# Test _CHECK_SIZES structure
assert len(_CHECK_SIZES) >= 6, f"Expected >= 6 fund types: {len(_CHECK_SIZES)}"
for fund_type, sizes in _CHECK_SIZES.items():
    assert "min" in sizes, f"Fund type '{fund_type}' missing 'min'"
    assert "max" in sizes, f"Fund type '{fund_type}' missing 'max'"
    assert "typical" in sizes, f"Fund type '{fund_type}' missing 'typical'"
    assert sizes["min"] <= sizes["typical"] <= sizes["max"], \
        f"Check size ordering wrong for {fund_type}: min={sizes['min']} typical={sizes['typical']} max={sizes['max']}"
print(f"[OK] _CHECK_SIZES: {len(_CHECK_SIZES)} fund types, all have valid min/typical/max")

# Verify specific check size ranges
assert _CHECK_SIZES["Seed"]["typical"] <= 2_000_000, \
    f"Seed typical check should be <= $2M: {_CHECK_SIZES['Seed']['typical']:,}"
assert _CHECK_SIZES["Buyout"]["typical"] >= 100_000_000, \
    f"Buyout typical check should be >= $100M: {_CHECK_SIZES['Buyout']['typical']:,}"
assert _CHECK_SIZES["Buyout"]["min"] > _CHECK_SIZES["Growth-VC"]["min"], \
    "Buyout min should be larger than Growth-VC min"
print(f"[OK] Check sizes: Seed typical=${_CHECK_SIZES['Seed']['typical']:,} Buyout typical=${_CHECK_SIZES['Buyout']['typical']:,}")

# Verify Accelerator < Seed < Early-VC < Growth-VC < Buyout ordering
fund_order = ["Accelerator", "Seed", "Early-VC", "Growth-VC", "Buyout"]
for i in range(len(fund_order) - 1):
    a, b = fund_order[i], fund_order[i+1]
    assert _CHECK_SIZES[a]["typical"] < _CHECK_SIZES[b]["typical"], \
        f"Check size ordering: {a} typical should be < {b} typical"
print(f"[OK] Check size ordering: Accelerator < Seed < Early-VC < Growth-VC < Buyout")

# Test _BENCHMARK_RETURNS
assert len(_BENCHMARK_RETURNS) >= 5, f"Expected >= 5 benchmark returns: {len(_BENCHMARK_RETURNS)}"
assert "US_VC_10Y" in _BENCHMARK_RETURNS, "Should have US_VC_10Y benchmark"
assert "US_PE_10Y" in _BENCHMARK_RETURNS, "Should have US_PE_10Y benchmark"

for bench_name, bench_data in _BENCHMARK_RETURNS.items():
    assert "metric" in bench_data, f"{bench_name} missing 'metric'"
    assert "value" in bench_data, f"{bench_name} missing 'value'"
    assert "source" in bench_data, f"{bench_name} missing 'source'"
    assert bench_data["value"] > 0, f"{bench_name} return value should be positive: {bench_data['value']}"

vc_10y = _BENCHMARK_RETURNS["US_VC_10Y"]
pe_10y = _BENCHMARK_RETURNS["US_PE_10Y"]
assert vc_10y["value"] > pe_10y["value"], \
    f"VC 10Y IRR should exceed PE 10Y (higher risk): VC={vc_10y['value']} PE={pe_10y['value']}"
print(f"[OK] _BENCHMARK_RETURNS: {len(_BENCHMARK_RETURNS)} benchmarks, US_VC_10Y={vc_10y['value']}% > US_PE_10Y={pe_10y['value']}%")

# Test _TVPI_BY_VINTAGE
assert len(_TVPI_BY_VINTAGE) >= 5, f"Expected >= 5 vintage years: {len(_TVPI_BY_VINTAGE)}"
# Earlier vintages should have higher TVPI (more mature)
vintages = sorted(_TVPI_BY_VINTAGE.items())
# 2015 (oldest) should have higher TVPI than 2022 (newest)
assert _TVPI_BY_VINTAGE["2015"] > _TVPI_BY_VINTAGE["2022"], \
    f"2015 vintage should have higher TVPI than 2022: {_TVPI_BY_VINTAGE['2015']} vs {_TVPI_BY_VINTAGE['2022']}"
for year, tvpi in _TVPI_BY_VINTAGE.items():
    assert tvpi >= 1.0, f"TVPI should be >= 1.0 (positive returns): {year}={tvpi}"
print(f"[OK] _TVPI_BY_VINTAGE: {len(_TVPI_BY_VINTAGE)} vintages, 2015={_TVPI_BY_VINTAGE['2015']} 2022={_TVPI_BY_VINTAGE['2022']}")

# Test _HOT_SICS
assert len(_HOT_SICS) >= 5, f"Expected >= 5 hot SICs: {len(_HOT_SICS)}"
assert "7372" in _HOT_SICS, "Software SIC 7372 should be hot"
assert _HOT_SICS["7372"] == "Software", f"7372 should be Software: {_HOT_SICS['7372']}"
assert "3674" in _HOT_SICS, "Semiconductors SIC 3674 should be hot"
print(f"[OK] _HOT_SICS: {len(_HOT_SICS)} hot VC sectors")

# Test VCPEFundUniverse
universe = VCPEFundUniverse()
funds = universe.KNOWN_VC_PE_FUNDS
assert len(funds) >= 50, f"Expected >= 50 known VC/PE funds: {len(funds)}"
print(f"[OK] VCPEFundUniverse: {len(funds)} known funds")

# Verify key fund names exist
for expected in ["Y Combinator", "Sequoia Capital", "Andreessen Horowitz"]:
    assert expected in funds, f"'{expected}' should be in known funds"
print(f"[OK] Key funds present: Y Combinator, Sequoia Capital, Andreessen Horowitz")

# Verify fund structure
for fund_name, fund_data in list(funds.items())[:10]:  # check first 10
    assert "cik" in fund_data, f"Fund '{fund_name}' missing CIK"
    assert "fund_type" in fund_data, f"Fund '{fund_name}' missing fund_type"
    # CIK should be 10 digits (EDGAR format)
    if fund_data["cik"]:
        cik = str(fund_data["cik"])
        assert len(cik) == 10, f"Fund '{fund_name}' CIK should be 10 chars: '{cik}'"
        assert cik.startswith("000"), f"Fund '{fund_name}' CIK should start with '000': '{cik}'"
print(f"[OK] Fund CIK format: 10-digit strings starting with '000'")

# Verify fund types are valid
valid_fund_types = {"Seed", "Early-VC", "Growth-VC", "Growth-PE", "Buyout",
                    "Accelerator", "Crossover", "Mezzanine", "Hedge-Fund",
                    "Crypto-VC", "Impact-VC", "Debt-VC"}
for fund_name, fund_data in funds.items():
    ft = fund_data.get("fund_type", "")
    assert ft in valid_fund_types, \
        f"Fund '{fund_name}' has invalid fund_type: '{ft}' (valid: {valid_fund_types})"
print(f"[OK] All fund types are valid")

# Test check size lookup by fund type
def get_check_size(fund_type: str) -> dict:
    return _CHECK_SIZES.get(fund_type, _CHECK_SIZES["Early-VC"])

yc_fund_type = funds["Y Combinator"]["fund_type"]
yc_check = get_check_size(yc_fund_type)
assert yc_check["max"] <= 1_000_000, \
    f"Y Combinator checks should be small (accelerator): max=${yc_check['max']:,}"
print(f"[OK] Y Combinator check size (type={yc_fund_type}): max=${yc_check['max']:,}")

# Test fund type distribution
type_counts = {}
for fund_name, fund_data in funds.items():
    ft = fund_data.get("fund_type", "Unknown")
    type_counts[ft] = type_counts.get(ft, 0) + 1
assert len(type_counts) >= 4, f"Expected >= 4 fund type varieties: {list(type_counts.keys())}"
print(f"[OK] Fund type distribution: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

# Test hq_state for top VC states
top_states = set()
for fund_name, fund_data in funds.items():
    state = fund_data.get("hq_state", "")
    if state:
        top_states.add(state)
assert "CA" in top_states, "CA should be a fund HQ state"
assert "NY" in top_states, "NY should be a fund HQ state"
print(f"[OK] Fund HQ states: {sorted(top_states)}")

print("\n[PASS] dim_098: VC/PE tracker")
PYEOF
