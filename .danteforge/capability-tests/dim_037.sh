#!/bin/bash
# dim_037: msrb_emma_adapter — MSRB EMMA municipal bond adapter
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
)

# --- constants ---
assert "emma.msrb.org" in EMMA_BASE
assert "SENTINEL" in SENTINEL_UA
print("[OK] EMMA_BASE and SENTINEL_UA constants present")

# --- _parse_date ---
d = _parse_date("2024-03-15T00:00:00")
assert d == date(2024, 3, 15), f"Got {d}"
d2 = _parse_date("2024-03-15")
assert d2 == date(2024, 3, 15)
assert _parse_date(None) is None
assert _parse_date("") is None
print("[OK] _parse_date: ISO datetime, ISO date, None, empty string")

# --- _float_or_none ---
assert _float_or_none("4.5") == 4.5
assert _float_or_none(3.0) == 3.0
assert _float_or_none(None) is None
assert _float_or_none("bad") is None
print("[OK] _float_or_none handles valid, invalid, None")

# --- _years_to_maturity ---
future = date.today().replace(year=date.today().year + 5)
ytm = _years_to_maturity(future)
assert ytm is not None and 4.0 < ytm < 6.0, f"Expected ~5yr, got {ytm}"
assert _years_to_maturity(None) is None
print(f"[OK] _years_to_maturity: future date = {ytm:.2f}yr, None -> None")

# --- MuniBond Pydantic model ---
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

# --- MuniTrade Pydantic model ---
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

# --- MuniYieldPoint Pydantic model ---
yp = MuniYieldPoint(
    maturity_years=10.0,
    yield_pct=3.75,
    cusip="13063C5M5",
    issuer_name="California GO",
)
assert yp.maturity_years == 10.0
assert yp.yield_pct == 3.75
print("[OK] MuniYieldPoint Pydantic model created")

# --- MuniScreenResult Pydantic model ---
sr = MuniScreenResult(
    bonds=[bond],
    total_found=1,
    query={"state": "CA", "maturity_max_years": 10},
)
assert sr.total_found == 1
assert len(sr.bonds) == 1
print("[OK] MuniScreenResult Pydantic model created")

# --- MSRBEmmaAdapter class structure ---
adapter = MSRBEmmaAdapter()
assert hasattr(adapter, "search_bonds")
assert hasattr(adapter, "_get")
assert adapter._timeout == 30.0
print("[OK] MSRBEmmaAdapter instantiates with correct structure")

print("\n[PASS] dim_037: msrb_emma_adapter -- all checks passed")
PYEOF
