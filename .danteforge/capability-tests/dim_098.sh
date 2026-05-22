#!/usr/bin/env bash
# dim_098: VC/PE tracker — FundUniverse, FormDParser, DealFlowVelocity,
# FundLifecycle, plus IRR / DPI / TVPI / RVPI fund-performance metrics.
# All assertions are pure computation — no network calls.
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, sqlite3, tempfile, xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

sys.path.insert(0, os.getcwd())

from sentinel.sfe.vcpe_tracker_v3 import (
    FundUniverse,
    FormDParser,
    DealFlowVelocity,
    FundLifecycle,
    VCPEDatabase,
    _UNIVERSE_SEED,
    DiscoveredFund,
    compute_fund_irr,
    compute_fund_dpi,
    compute_fund_tvpi,
    compute_fund_rvpi,
    compute_fund_metrics,
    record_fund_cashflow,
    set_fund_metadata,
    discover_new_funds_via_iapd,
)

# ── 1. FundUniverse: fund_count > 0 using in-memory DB ───────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "test_vcpe.db"
    db = VCPEDatabase(db_path=db_path)
    universe = FundUniverse(db=db)

    count = universe.fund_count
    assert count > 0, f"FundUniverse.fund_count should be > 0 after seed load, got {count}"
    assert count == len(set(cik for _, cik in _UNIVERSE_SEED)), \
        f"fund_count should equal unique seed CIKs: expected ~{len(_UNIVERSE_SEED)}, got {count}"
    print(f"[OK] FundUniverse.fund_count = {count} (>0, seed loaded)")

    # Verify get_all_funds returns DiscoveredFund objects
    funds = universe.get_all_funds()
    assert len(funds) == count, f"get_all_funds() count mismatch: {len(funds)} vs {count}"
    assert all(isinstance(f, DiscoveredFund) for f in funds), \
        "get_all_funds() should return DiscoveredFund instances"
    # Check key funds are present
    names = {f.fund_name for f in funds}
    for expected in ("Sequoia Capital", "Andreessen Horowitz", "Blackstone", "KKR"):
        assert expected in names, f"'{expected}' should be in FundUniverse seed"
    print(f"[OK] FundUniverse.get_all_funds: {len(funds)} funds, key names present")

    # update_fund_amount works
    seed_cik = funds[0].cik
    universe.update_fund_amount(seed_cik, amount_raised=5_000_000.0,
                                exempt_type="Rule 506(b)", state="CA")
    updated = [f for f in universe.get_all_funds() if f.cik == seed_cik][0]
    assert updated.amount_raised == 5_000_000.0, \
        f"update_fund_amount failed: got {updated.amount_raised}"
    assert updated.exempt_offering_type == "Rule 506(b)", \
        f"exempt_offering_type not updated: {updated.exempt_offering_type}"
    print(f"[OK] FundUniverse.update_fund_amount: CIK {seed_cik} amount=${updated.amount_raised:,.0f}")

    db.close()

# ── 2. FormDParser: parse a sample XML string (no network) ───────────────────
SAMPLE_FORM_D_XML = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="urn:us:gov:sec:formd">
  <schemaVersion>X0206</schemaVersion>
  <submissionType>D</submissionType>
  <isAmendment>false</isAmendment>
  <primaryIssuer>
    <issuerName>Acme Venture Partners LLC</issuerName>
    <issuerStateOrCountry>CA</issuerStateOrCountry>
  </primaryIssuer>
  <offeringData>
    <industryGroup>
      <industryGroupType>Pooled Investment Fund</industryGroupType>
    </industryGroup>
    <typeOfFiling>
      <isAmendment>false</isAmendment>
    </typeOfFiling>
    <offeringInformation>
      <totalOfferingAmount>50000000</totalOfferingAmount>
      <totalAmountSold>25000000</totalAmountSold>
      <totalNumberAlreadyInvested>12</totalNumberAlreadyInvested>
    </offeringInformation>
    <salesCompensationList/>
    <offeringSalesAmounts>
      <totalOfferingAmount>50000000</totalOfferingAmount>
      <totalAmountSold>25000000</totalAmountSold>
      <totalRemaining>25000000</totalRemaining>
    </offeringSalesAmounts>
    <investors>
      <totalNumberAlreadyInvested>12</totalNumberAlreadyInvested>
    </investors>
    <salesCommissionsFindersFees>
      <salesCommissions>
        <isSalesCommission>false</isSalesCommission>
      </salesCommissions>
    </salesCommissionsFindersFees>
    <useOfProceeds>
      <useOfProceedsIsSpecified>false</useOfProceedsIsSpecified>
    </useOfProceeds>
    <exemptionsAndExclusions>Rule 506(b)</exemptionsAndExclusions>
    <dateOfFirstSale>2024-03-15</dateOfFirstSale>
  </offeringData>
</edgarSubmission>
"""

parsed = FormDParser.parse_xml_string(SAMPLE_FORM_D_XML)
assert parsed, "FormDParser.parse_xml_string should return a non-empty dict"
assert parsed["issuer_name"] == "Acme Venture Partners LLC", \
    f"issuer_name wrong: {parsed['issuer_name']!r}"
assert parsed["state_of_incorporation"] == "CA", \
    f"state_of_incorporation wrong: {parsed['state_of_incorporation']!r}"
assert parsed["total_offering_amount"] == 50_000_000.0, \
    f"total_offering_amount wrong: {parsed['total_offering_amount']}"
assert parsed["total_amount_sold"] == 25_000_000.0, \
    f"total_amount_sold wrong: {parsed['total_amount_sold']}"
assert parsed["investor_count"] == 12, \
    f"investor_count wrong: {parsed['investor_count']}"
assert parsed["exemption_type"] == "Rule 506(b)", \
    f"exemption_type wrong: {parsed['exemption_type']!r}"
assert parsed["is_amendment"] is False, \
    f"is_amendment wrong: {parsed['is_amendment']}"
assert parsed["date_of_first_sale"] == "2024-03-15", \
    f"date_of_first_sale wrong: {parsed['date_of_first_sale']!r}"
print(f"[OK] FormDParser.parse_xml_string: issuer={parsed['issuer_name']!r} "
      f"amount_sold=${parsed['total_amount_sold']:,.0f} investors={parsed['investor_count']}")

# Bad XML returns empty dict, not an exception
bad_result = FormDParser.parse_xml_string("<not valid xml><<<")
assert isinstance(bad_result, dict), "Bad XML should return empty dict"
print(f"[OK] FormDParser handles malformed XML gracefully")

# ── 3. DealFlowVelocity: pure math verification ───────────────────────────────

# Basic cases
v = DealFlowVelocity.compute_velocity([10, 10, 10, 15])
assert abs(v - 0.5) < 1e-9, f"Expected 0.5, got {v}"

v = DealFlowVelocity.compute_velocity([10, 20])
assert abs(v - 1.0) < 1e-9, f"Expected 1.0 (100% more), got {v}"

v = DealFlowVelocity.compute_velocity([20, 10])
assert abs(v - (-0.5)) < 1e-9, f"Expected -0.5 (50% fewer), got {v}"

v = DealFlowVelocity.compute_velocity([0, 0])
assert v == 0.0, f"Zero prior avg should yield 0.0 velocity, got {v}"

# Multiple prior quarters
v = DealFlowVelocity.compute_velocity([8, 10, 12, 15])
prior_avg = (8 + 10 + 12) / 3   # = 10.0
expected = (15 - prior_avg) / prior_avg   # = 0.5
assert abs(v - expected) < 1e-9, f"Multi-quarter velocity wrong: expected {expected:.4f}, got {v:.4f}"

print(f"[OK] DealFlowVelocity.compute_velocity: [10,10,10,15]=0.5, [10,20]=1.0, "
      f"[20,10]=-0.5, [0,0]=0.0, [8,10,12,15]={expected:.3f}")

# quarter_key
qk = DealFlowVelocity.quarter_key(date(2024, 1, 15))
assert qk == "2024Q1", f"Q1 wrong: {qk}"
qk = DealFlowVelocity.quarter_key(date(2024, 4, 1))
assert qk == "2024Q2", f"Q2 wrong: {qk}"
qk = DealFlowVelocity.quarter_key(date(2024, 7, 31))
assert qk == "2024Q3", f"Q3 wrong: {qk}"
qk = DealFlowVelocity.quarter_key(date(2024, 12, 31))
assert qk == "2024Q4", f"Q4 wrong: {qk}"
print(f"[OK] DealFlowVelocity.quarter_key: 2024-01-15=2024Q1, 2024-04-01=2024Q2, "
      f"2024-07-31=2024Q3, 2024-12-31=2024Q4")

# DB-backed record + get_velocity
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "test_vel.db"
    db = VCPEDatabase(db_path=db_path)
    dfv = DealFlowVelocity(db=db)

    test_cik = "9999999"
    # Simulate 4 quarters of filings: Q1=10, Q2=10, Q3=10, Q4=15
    for _ in range(10):
        dfv.record_filing(test_cik, "2024-01-15")   # Q1
    for _ in range(10):
        dfv.record_filing(test_cik, "2024-04-15")   # Q2
    for _ in range(10):
        dfv.record_filing(test_cik, "2024-07-15")   # Q3
    for _ in range(15):
        dfv.record_filing(test_cik, "2024-10-15")   # Q4

    vel = dfv.get_velocity(test_cik, num_prior_quarters=3)
    assert vel is not None, "get_velocity should return a float, got None"
    assert abs(vel - 0.5) < 1e-9, f"DB-backed velocity wrong: expected 0.5, got {vel}"
    print(f"[OK] DealFlowVelocity DB-backed: 4 quarters [10,10,10,15] => velocity={vel:.2f}")
    db.close()

# ValueError on too-short list
try:
    DealFlowVelocity.compute_velocity([5])
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print(f"[OK] DealFlowVelocity raises ValueError for < 2 quarters")

# ── 4. FundLifecycle: age computation and stage transitions ───────────────────

ref = date(2025, 1, 1)

# Age computation
age = FundLifecycle.fund_age_years("2020-01-01", as_of=ref)
assert abs(age - 5.0) < 0.01, f"5-year-old fund: expected ~5.0, got {age}"

age = FundLifecycle.fund_age_years("2024-01-01", as_of=ref)
assert abs(age - 1.0) < 0.01, f"1-year-old fund: expected ~1.0, got {age}"

age = FundLifecycle.fund_age_years("", as_of=ref)
assert age == 0.0, f"Empty date should return 0.0, got {age}"

age = FundLifecycle.fund_age_years("bad-date", as_of=ref)
assert age == 0.0, f"Bad date should return 0.0, got {age}"
print(f"[OK] FundLifecycle.fund_age_years: 2020-01-01=>{FundLifecycle.fund_age_years('2020-01-01', as_of=ref):.2f}yr, "
      f"empty/bad=>0.0")

# Stage transitions
assert FundLifecycle.lifecycle_stage(0.5)  == "formation",   f"0.5yr => formation"
assert FundLifecycle.lifecycle_stage(2.0)  == "fundraising", f"2yr => fundraising"
assert FundLifecycle.lifecycle_stage(5.0)  == "investing",   f"5yr => investing"
assert FundLifecycle.lifecycle_stage(9.0)  == "harvesting",  f"9yr => harvesting"
assert FundLifecycle.lifecycle_stage(15.0) == "mature",      f"15yr => mature"
print(f"[OK] FundLifecycle.lifecycle_stage: 0.5=>formation, 2=>fundraising, "
      f"5=>investing, 9=>harvesting, 15=>mature")

# is_mature_harvesting threshold = 10 years
assert not FundLifecycle.is_mature_harvesting("2020-01-01", as_of=ref), \
    "5-year-old fund should NOT be mature/harvesting"
assert FundLifecycle.is_mature_harvesting("2010-01-01", as_of=ref), \
    "15-year-old fund SHOULD be mature/harvesting"
assert not FundLifecycle.is_mature_harvesting("2016-06-01", as_of=ref), \
    "8.5-year-old fund should NOT be mature/harvesting"
print(f"[OK] FundLifecycle.is_mature_harvesting: 5yr=>False, 15yr=>True, 8.5yr=>False")

# Full profile
profile = FundLifecycle.profile(
    formation_date="2018-03-01",
    amendment_dates=["2018-09-15", "2019-03-01"],
    as_of=ref,
)
assert profile["stage"] == "investing",   f"Stage wrong: {profile['stage']}"
assert abs(profile["age_years"] - 6.84) < 0.1, f"age_years wrong: {profile['age_years']}"
assert profile["is_mature_harvesting"] is False, \
    f"6.8yr fund should not be mature: {profile['is_mature_harvesting']}"
assert profile["amendment_count"] == 2, f"amendment_count wrong: {profile['amendment_count']}"
assert profile["last_amendment_date"] == "2019-03-01", \
    f"last_amendment_date wrong: {profile['last_amendment_date']}"
assert profile["final_close_estimated"] is True, \
    f"No amendments in 5+ years should set final_close_estimated=True: {profile['final_close_estimated']}"
print(f"[OK] FundLifecycle.profile: 2018-03-01 fund => stage={profile['stage']!r}, "
      f"age={profile['age_years']}yr, final_close={profile['final_close_estimated']}")

# A very new fund
new_profile = FundLifecycle.profile("2024-06-01", as_of=ref)
assert new_profile["stage"] == "formation", f"6mo fund => formation, got {new_profile['stage']}"
assert new_profile["final_close_estimated"] is False, \
    "New fund should not have final_close_estimated"
print(f"[OK] FundLifecycle.profile new fund (2024-06-01): stage={new_profile['stage']!r}")

# ── 5. Fund performance metrics: IRR / DPI / TVPI / RVPI ─────────────────────

# IRR: $-100 today, $200 in 5 years → IRR ~14.87%
irr = compute_fund_irr([(date(2020, 1, 1), -100.0), (date(2025, 1, 1), 200.0)])
assert irr is not None, "compute_fund_irr returned None for a valid 2-flow stream"
assert 0.14 < irr < 0.16, f"IRR should be ~0.1487 for 2x in 5y, got {irr}"
print(f"[OK] compute_fund_irr: 2x in 5y => IRR={irr:.4f} (~14.87%)")

# IRR with terminal NAV: $-100 today, $0 today + 5y, terminal NAV = $200 at year 5
irr_nav = compute_fund_irr(
    [(date(2020, 1, 1), -100.0), (date(2025, 1, 1), 0.0)],
    terminal_nav=200.0,
)
assert irr_nav is not None and 0.14 < irr_nav < 0.16, \
    f"IRR with terminal_nav=200 should match: got {irr_nav}"
print(f"[OK] compute_fund_irr with terminal_nav=200 => IRR={irr_nav:.4f}")

# IRR degenerate cases
assert compute_fund_irr([]) is None, "Empty cash flows should return None"
assert compute_fund_irr([(date(2020, 1, 1), -100.0)]) is None, \
    "Single flow should return None"
assert compute_fund_irr([(date(2020, 1, 1), -100.0),
                        (date(2021, 1, 1), -50.0)]) is None, \
    "All-negative flows should return None"
print(f"[OK] compute_fund_irr handles degenerate inputs (empty / single / all-neg)")

# DPI: $100 distributed on $50 called = 2.0
assert compute_fund_dpi(100, 50) == 2.0, \
    f"DPI(100,50) should be 2.0, got {compute_fund_dpi(100, 50)}"
assert compute_fund_dpi(0, 50) == 0.0, "DPI(0,50) should be 0.0"
assert compute_fund_dpi(100, 0) == 0.0, "DPI with capital_called=0 should be 0.0"
print(f"[OK] compute_fund_dpi: (100,50)=2.0, (0,50)=0.0, (100,0)=0.0")

# TVPI: ($100 distributed + $30 NAV) / $50 called = 2.6
assert compute_fund_tvpi(100, 30, 50) == 2.6, \
    f"TVPI(100,30,50) should be 2.6, got {compute_fund_tvpi(100, 30, 50)}"
assert compute_fund_tvpi(0, 0, 50) == 0.0, "TVPI with no value should be 0.0"
assert compute_fund_tvpi(100, 30, 0) == 0.0, "TVPI with capital_called=0 should be 0.0"
print(f"[OK] compute_fund_tvpi: (100,30,50)=2.6, edge cases handled")

# RVPI: $30 NAV / $50 called = 0.6
assert compute_fund_rvpi(30, 50) == 0.6, \
    f"RVPI(30,50) should be 0.6, got {compute_fund_rvpi(30, 50)}"
assert compute_fund_rvpi(0, 50) == 0.0, "RVPI(0,50) should be 0.0"
assert compute_fund_rvpi(30, 0) == 0.0, "RVPI with capital_called=0 should be 0.0"
print(f"[OK] compute_fund_rvpi: (30,50)=0.6, edge cases handled")

# Identity: TVPI == DPI + RVPI (within float epsilon)
d, n, c = 100.0, 30.0, 50.0
assert abs(compute_fund_tvpi(d, n, c) - (compute_fund_dpi(d, c) + compute_fund_rvpi(n, c))) < 1e-12, \
    "TVPI must equal DPI + RVPI"
print(f"[OK] Identity: TVPI = DPI + RVPI verified")

# ── 6. compute_fund_metrics: end-to-end DB-backed wrapper ────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "test_metrics.db"
    db = VCPEDatabase(db_path=db_path)

    # Seed the universe so we get a real fund name back
    universe = FundUniverse(db=db)
    universe._load_seed()

    test_cik = "1056831"   # Sequoia Capital (in seed)

    # Record a synthetic 5-year fund cash-flow stream:
    #   Year 0: $-100M called
    #   Year 5: $200M distributed
    record_fund_cashflow(test_cik, date(2020, 1, 1), -100_000_000.0, "call", db=db)
    record_fund_cashflow(test_cik, date(2025, 1, 1),  200_000_000.0, "distribution", db=db)
    set_fund_metadata(test_cik, vintage_year=2020, strategy="Venture", current_nav=0.0, db=db)

    metrics = compute_fund_metrics(test_cik, db=db)
    assert metrics["fund_name"] == "Sequoia Capital", \
        f"fund_name lookup failed: {metrics['fund_name']!r}"
    assert metrics["vintage_year"] == 2020, f"vintage_year wrong: {metrics['vintage_year']}"
    assert metrics["strategy"] == "Venture", f"strategy wrong: {metrics['strategy']!r}"
    assert metrics["irr"] is not None and 0.14 < metrics["irr"] < 0.16, \
        f"compute_fund_metrics IRR wrong: {metrics['irr']}"
    assert metrics["dpi"] == 2.0, f"compute_fund_metrics DPI wrong: {metrics['dpi']}"
    assert metrics["tvpi"] == 2.0, f"compute_fund_metrics TVPI wrong: {metrics['tvpi']}"
    assert metrics["rvpi"] == 0.0, f"compute_fund_metrics RVPI wrong: {metrics['rvpi']}"
    print(f"[OK] compute_fund_metrics: {metrics['fund_name']} "
          f"IRR={metrics['irr']:.4f} DPI={metrics['dpi']} "
          f"TVPI={metrics['tvpi']} RVPI={metrics['rvpi']}")

    # Add interim NAV so RVPI > 0
    set_fund_metadata(test_cik, current_nav=30_000_000.0, db=db)
    metrics2 = compute_fund_metrics(test_cik, db=db)
    assert metrics2["rvpi"] == 0.3, f"RVPI with NAV=30M wrong: {metrics2['rvpi']}"
    # TVPI = (200M + 30M) / 100M = 2.3
    assert abs(metrics2["tvpi"] - 2.3) < 1e-9, f"TVPI wrong: {metrics2['tvpi']}"
    print(f"[OK] compute_fund_metrics with NAV=30M => RVPI={metrics2['rvpi']} TVPI={metrics2['tvpi']}")

    # Unknown CIK still returns the dict (with None fields), doesn't crash
    empty = compute_fund_metrics("0000000", db=db)
    assert empty["fund_name"] is None
    assert empty["irr"] is None
    assert empty["dpi"] is None
    print(f"[OK] compute_fund_metrics for unknown CIK returns None fields gracefully")

    db.close()

# ── 7. discover_new_funds_via_iapd is importable and signature-compatible ────
assert callable(discover_new_funds_via_iapd), \
    "discover_new_funds_via_iapd must be a callable"
# We don't actually hit the network in CI — just verify it's wired up correctly.
import inspect
sig = inspect.signature(discover_new_funds_via_iapd)
assert "query" in sig.parameters, "discover_new_funds_via_iapd must accept query="
print(f"[OK] discover_new_funds_via_iapd is importable with signature {sig}")

print("\n[PASS] dim_098: VC/PE tracker - FundUniverse, FormDParser, DealFlowVelocity, "
      "FundLifecycle, IRR/DPI/TVPI/RVPI metrics")
PYEOF
