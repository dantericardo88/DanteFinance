#!/bin/bash
# dim_037: msrb_emma_adapter — MSRB EMMA municipal bond adapter (enhanced)
# Tests: constants, helpers, Pydantic models, analytics (pure math, no network)
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from datetime import date

from sentinel.sds.adapters.msrb_emma_adapter import (
    EMMA_BASE,
    SENTINEL_UA,
    MuniBond,
    MuniTrade,
    MuniYieldPoint,
    MuniScreenResult,
    MSRBEmmaAdapter,
    _parse_date,
    _float_or_none,
    _years_to_maturity,
    tax_equivalent_yield,
    after_tax_corporate_yield,
    muni_premium_to_treasury,
    call_adjusted_ytm,
    assign_credit_tier,
    get_universe_issuers,
    MUNI_UNIVERSE_200,
)

# -----------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------
assert "emma.msrb.org" in EMMA_BASE
assert "SENTINEL" in SENTINEL_UA
print("[OK] EMMA_BASE and SENTINEL_UA constants present")

# -----------------------------------------------------------------------
# 2. _parse_date
# -----------------------------------------------------------------------
d = _parse_date("2024-03-15T00:00:00")
assert d == date(2024, 3, 15), f"Got {d}"
d2 = _parse_date("2024-03-15")
assert d2 == date(2024, 3, 15)
assert _parse_date(None) is None
assert _parse_date("") is None
print("[OK] _parse_date: ISO datetime, ISO date, None, empty string")

# -----------------------------------------------------------------------
# 3. _float_or_none
# -----------------------------------------------------------------------
assert _float_or_none("4.5") == 4.5
assert _float_or_none(3.0) == 3.0
assert _float_or_none(None) is None
assert _float_or_none("bad") is None
print("[OK] _float_or_none handles valid, invalid, None")

# -----------------------------------------------------------------------
# 4. _years_to_maturity
# -----------------------------------------------------------------------
future = date.today().replace(year=date.today().year + 5)
ytm_years = _years_to_maturity(future)
assert ytm_years is not None and 4.0 < ytm_years < 6.0, f"Expected ~5yr, got {ytm_years}"
assert _years_to_maturity(None) is None
print(f"[OK] _years_to_maturity: future date = {ytm_years:.2f}yr, None -> None")

# -----------------------------------------------------------------------
# 5. MuniBond Pydantic model
# -----------------------------------------------------------------------
bond = MuniBond(
    cusip="13063C5M5",
    issuer_name="California General Obligation",
    description="CA GO 5.0% 2030",
    state="CA",
    security_type="General Obligation",
    maturity_date=date(2030, 11, 1),
    coupon=5.0,
    interest_payment_frequency="Semiannual",
    outstanding_principal=1_500_000_000.0,
    tax_status="Federal Tax-Exempt",
)
assert bond.cusip == "13063C5M5"
assert bond.state == "CA"
assert bond.coupon == 5.0
print("[OK] MuniBond Pydantic model created")

# -----------------------------------------------------------------------
# 6. MuniTrade Pydantic model
# -----------------------------------------------------------------------
trade = MuniTrade(
    trade_date=date(2024, 3, 15),
    settlement_date=date(2024, 3, 18),
    price=104.5,
    yield_pct=4.35,
    par_value=1_000_000.0,
    trade_type="Customer Buy",
)
assert trade.price == 104.5
assert trade.trade_type == "Customer Buy"
print("[OK] MuniTrade Pydantic model created")

# -----------------------------------------------------------------------
# 7. MuniYieldPoint Pydantic model
# -----------------------------------------------------------------------
yp = MuniYieldPoint(
    maturity_years=10.0,
    yield_pct=3.75,
    cusip="13063C5M5",
    issuer_name="California GO",
)
assert yp.maturity_years == 10.0
assert yp.yield_pct == 3.75
print("[OK] MuniYieldPoint Pydantic model created")

# -----------------------------------------------------------------------
# 8. MuniScreenResult Pydantic model
# -----------------------------------------------------------------------
sr = MuniScreenResult(
    bonds=[bond],
    total_found=1,
    query={"state": "CA", "maturity_max_years": 10},
)
assert sr.total_found == 1
assert len(sr.bonds) == 1
print("[OK] MuniScreenResult Pydantic model created")

# -----------------------------------------------------------------------
# 9. MSRBEmmaAdapter class structure
# -----------------------------------------------------------------------
adapter = MSRBEmmaAdapter()
assert hasattr(adapter, "search_bonds")
assert hasattr(adapter, "_get")
assert adapter._timeout == 30.0
print("[OK] MSRBEmmaAdapter instantiates with correct structure")

# -----------------------------------------------------------------------
# 10. Tax-equivalent yield (TEY) — the core formula
# -----------------------------------------------------------------------
# 4.0% muni at 37% bracket → TEY = 4.0 / (1 - 0.37) = 4.0 / 0.63 ≈ 6.3492%
tey = tax_equivalent_yield(4.0, 0.37)
assert abs(tey - 4.0 / 0.63) < 1e-9, f"TEY mismatch: {tey}"
assert abs(tey - 6.349206349) < 1e-6, f"TEY not ~6.35%: {tey}"
print(f"[OK] TEY: 4.0% muni @ 37% bracket = {tey:.6f}% (expected ~6.3492%)")

# 3.5% muni at 22% bracket
tey22 = tax_equivalent_yield(3.5, 0.22)
assert abs(tey22 - 3.5 / 0.78) < 1e-9
print(f"[OK] TEY: 3.5% muni @ 22% bracket = {tey22:.4f}%")

# Edge: zero yield
tey_zero = tax_equivalent_yield(0.0, 0.37)
assert tey_zero == 0.0
print("[OK] TEY: 0% muni -> 0% TEY")

# -----------------------------------------------------------------------
# 11. After-tax corporate yield
# -----------------------------------------------------------------------
# 6% corporate at 37% -> after-tax = 6 * (1 - 0.37) = 3.78%
atc = after_tax_corporate_yield(6.0, 0.37)
assert abs(atc - 6.0 * 0.63) < 1e-9, f"After-tax corp mismatch: {atc}"
assert abs(atc - 3.78) < 1e-9, f"Expected 3.78, got {atc}"
assert isinstance(atc, float)
print(f"[OK] After-tax corporate: 6.0% corp @ 37% = {atc:.4f}% (expected 3.78%)")

# -----------------------------------------------------------------------
# 12. Muni premium to Treasury (TEY - treasury)
# -----------------------------------------------------------------------
# 4.0% muni, 4.5% treasury, 37% bracket
# TEY = 4.0/0.63 ≈ 6.3492%, premium = 6.3492 - 4.5 = 1.8492%
premium = muni_premium_to_treasury(4.0, 4.5, 0.37)
expected_premium = tax_equivalent_yield(4.0, 0.37) - 4.5
assert abs(premium - expected_premium) < 1e-9, f"Premium mismatch: {premium}"
assert premium > 0, "Cheap munis should have positive premium"
assert isinstance(premium, float)
print(f"[OK] Muni premium: 4.0% muni, 4.5% treasury @ 37% = {premium:.4f}% (munis cheap)")

# Rich munis: muni TEY below treasury
premium_rich = muni_premium_to_treasury(2.5, 4.5, 0.37)
assert premium_rich < 0, "Rich munis should have negative premium"
print(f"[OK] Muni premium: 2.5% muni, 4.5% treasury @ 37% = {premium_rich:.4f}% (munis rich)")

# -----------------------------------------------------------------------
# 13. Call-adjusted YTM (yield-to-worst)
# -----------------------------------------------------------------------
# Premium bond: 5% coupon, price=105, 10yr maturity, callable in 3yr at 100
# For a premium bond, YTC should be lower than YTM (yield-to-worst < YTM)
result = call_adjusted_ytm(
    coupon_rate=5.0,
    years_to_maturity=10.0,
    price=105.0,
    call_price=100.0,
    years_to_call=3.0,
)
assert "ytm" in result
assert "ytc" in result
assert "ytw" in result
assert result["is_premium"] is True, f"Price 105 should be premium, got {result}"
assert result["ytc"] is not None
# For premium bond: YTC <= YTM (call is bad for investor)
assert result["ytc"] <= result["ytm"], (
    f"Premium bond: YTC {result['ytc']:.4f} should be <= YTM {result['ytm']:.4f}"
)
# YTW = min(YTM, YTC) for premium bond
assert abs(result["ytw"] - min(result["ytm"], result["ytc"])) < 1e-9
print(
    f"[OK] Call-adjusted YTM: premium bond price=105, coupon=5%, "
    f"YTM={result['ytm']:.4f}%, YTC={result['ytc']:.4f}%, YTW={result['ytw']:.4f}%"
)

# Discount bond: 3% coupon, price=95, 10yr, callable in 3yr at 100
# YTC > YTM since call brings investor to par (above market price) early
result_disc = call_adjusted_ytm(
    coupon_rate=3.0,
    years_to_maturity=10.0,
    price=95.0,
    call_price=100.0,
    years_to_call=3.0,
)
assert result_disc["is_premium"] is False
# YTW = min(YTM, YTC)
assert result_disc["ytw"] == min(result_disc["ytm"], result_disc["ytc"])
print(
    f"[OK] Call-adjusted YTM: discount bond price=95, coupon=3%, "
    f"YTM={result_disc['ytm']:.4f}%, YTC={result_disc['ytc']:.4f}%"
)

# Non-callable bond
result_nc = call_adjusted_ytm(5.0, 10.0, 100.0, years_to_call=None)
assert result_nc["ytc"] is None
assert result_nc["ytw"] == result_nc["ytm"]
print(f"[OK] Non-callable bond: YTW={result_nc['ytw']:.4f}% = YTM")

# -----------------------------------------------------------------------
# 14. Credit tier assignment
# -----------------------------------------------------------------------
# High fiscal score state -> AAA
tier_aaa = assign_credit_tier("TN", bond_type="GO")
assert tier_aaa["credit_tier"] == "AAA", f"TN (score=83) should be AAA, got {tier_aaa}"
assert tier_aaa["source"] == "fiscal_score"
print(f"[OK] Credit tier: TN GO -> {tier_aaa['credit_tier']} (fiscal score={tier_aaa['fiscal_score']})")

# Explicit S&P rating overrides fiscal score
tier_sp = assign_credit_tier("IL", bond_type="GO", rating_sp="AA+")
assert tier_sp["credit_tier"] == "AA"
assert tier_sp["source"] == "sp_rating"
assert tier_sp["implied_rating"] == "AA+"
print(f"[OK] Credit tier: IL GO with rating AA+ -> {tier_sp['credit_tier']} (sp_rating source)")

# Distressed state -> BBB or below
tier_il = assign_credit_tier("IL", bond_type="GO")  # fiscal score=28 -> BBB
assert tier_il["credit_tier"] in ("BBB", "BB"), f"IL (score=28) should be BBB/BB, got {tier_il}"
print(f"[OK] Credit tier: IL GO (no rating) -> {tier_il['credit_tier']} (fiscal score={tier_il['fiscal_score']})")

# PR -> BB or below
tier_pr = assign_credit_tier("PR", bond_type="GO")
assert tier_pr["credit_tier"] in ("BB", "CCC", "D", "BBB"), f"PR should be junk, got {tier_pr}"
print(f"[OK] Credit tier: PR GO -> {tier_pr['credit_tier']} (fiscal score={tier_pr['fiscal_score']})")

# Revenue bond gets non-zero default risk premium
tier_rev = assign_credit_tier("CA", bond_type="Revenue")
assert tier_rev["default_risk_premium_bps"] > 0, "Revenue bonds should have positive risk premium"
print(f"[OK] Credit tier: CA Revenue -> {tier_rev['credit_tier']}, risk premium={tier_rev['default_risk_premium_bps']}bps")

# fiscal_score override
tier_override = assign_credit_tier("XX", bond_type="GO", fiscal_score=90)
assert tier_override["credit_tier"] == "AAA", f"Score 90 should be AAA, got {tier_override}"
print(f"[OK] Credit tier: fiscal_score=90 override -> {tier_override['credit_tier']}")

# -----------------------------------------------------------------------
# 15. Dynamic universe loader (static fallback, no network)
# -----------------------------------------------------------------------
universe = get_universe_issuers(try_live=False)
assert len(universe) >= 100, f"Expected 100+ issuers, got {len(universe)}"
assert len(MUNI_UNIVERSE_200) >= 100, f"MUNI_UNIVERSE_200 too small: {len(MUNI_UNIVERSE_200)}"

# All entries have required keys
required_keys = {"cusip", "issuer", "state", "type", "coupon", "maturity"}
for entry in universe[:10]:
    missing = required_keys - set(entry.keys())
    assert not missing, f"Missing keys {missing} in entry {entry.get('cusip')}"

# Verify state coverage: at least 40 unique states represented
states = {e["state"] for e in universe if e.get("state")}
assert len(states) >= 40, f"Expected 40+ states, got {len(states)}: {sorted(states)}"

# Both GO and Revenue types present
types = {e["type"] for e in universe}
assert "GO" in types, "Universe missing GO bonds"
assert "Revenue" in types, "Universe missing Revenue bonds"

print(
    f"[OK] Dynamic universe: {len(universe)} issuers, "
    f"{len(states)} states, types={sorted(types)}"
)

# -----------------------------------------------------------------------
# 16. Muni vs corporate after-tax comparison produces float
# -----------------------------------------------------------------------
muni_yield_pct = 4.0
corp_yield_pct = 6.5
tax_rate = 0.37

tey_val = tax_equivalent_yield(muni_yield_pct, tax_rate)
atc_val = after_tax_corporate_yield(corp_yield_pct, tax_rate)
advantage_bps = (tey_val - corp_yield_pct) * 100  # muni TEY vs gross corp

assert isinstance(tey_val, float)
assert isinstance(atc_val, float)
assert isinstance(advantage_bps, float)
print(
    f"[OK] Muni vs corp comparison: muni TEY={tey_val:.4f}%, "
    f"after-tax corp={atc_val:.4f}%, TEY-corp spread={advantage_bps:.1f}bps"
)

# -----------------------------------------------------------------------
# 17. AMT flag propagation
# -----------------------------------------------------------------------
bond_amt = MuniBond(
    cusip="TESTAMT01",
    issuer_name="Test AMT Bond",
    description="Private Activity AMT Bond",
    state="TX",
    security_type="Revenue",
    interest_payment_frequency="Semiannual",
    is_amt=True,
    tax_status="Federal AMT",
)
assert bond_amt.is_amt is True
assert bond_amt.tax_status == "Federal AMT"
print("[OK] AMT flag: is_amt=True propagated correctly on MuniBond")

bond_no_amt = MuniBond(
    cusip="TESTNOAMT",
    issuer_name="Test Non-AMT Bond",
    description="GO Bond",
    state="CA",
    security_type="General Obligation",
    interest_payment_frequency="Semiannual",
    is_amt=False,
)
assert bond_no_amt.is_amt is False
print("[OK] AMT flag: is_amt=False (default) on non-AMT bond")

# -----------------------------------------------------------------------
# 18. GO vs Revenue bond_type field
# -----------------------------------------------------------------------
bond_go = MuniBond(
    cusip="TESTGO001",
    issuer_name="California GO",
    description="CA GO",
    state="CA",
    security_type="General Obligation",
    interest_payment_frequency="Semiannual",
    bond_type="GO",
)
bond_rev = MuniBond(
    cusip="TESTREV01",
    issuer_name="CA Water Rev",
    description="CA Water Revenue",
    state="CA",
    security_type="Revenue",
    interest_payment_frequency="Semiannual",
    bond_type="Revenue",
)
assert bond_go.bond_type == "GO"
assert bond_rev.bond_type == "Revenue"
print("[OK] GO vs Revenue bond_type distinction works")

print("\n[PASS] dim_037: msrb_emma_adapter -- all checks passed")
PYEOF
