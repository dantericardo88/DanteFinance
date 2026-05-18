#!/usr/bin/env bash
# dim_098: VC/PE tracker — FundUniverse, FormDParser, DealFlowVelocity, FundLifecycle
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

print("\n[PASS] dim_098: VC/PE tracker - FundUniverse, FormDParser, DealFlowVelocity, FundLifecycle")
PYEOF
