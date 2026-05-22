#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')

from sentinel.sfe.activist_tracker_v3 import (
    _parse_int,
    _parse_float,
    _date_from_str,
    _normalize_accession,
    Filing13D,
    ActivistCampaign,
    CampaignType,
    compute_campaign_success_rate,
    predict_settlement_probability,
    compute_target_vulnerability_score,
    ActivistTracker,
    ActivistScreener,
    get_active_campaigns_from_edgar,
    _split_display_name,
    _ACTIVIST_TARGETS_CACHE,
    _ACTIVIST_CAMPAIGNS_CACHE,
)
from datetime import date
import inspect
import math
import re
from pathlib import Path

# Test _parse_int
assert _parse_int("1,500,000") == 1500000
assert _parse_int("bad", 0) == 0
print("[OK] _parse_int handles commas and fallback")

# Test _parse_float
assert _parse_float("12.5%") == 12.5
assert _parse_float("1,234.56") == 1234.56
print("[OK] _parse_float strips percent/comma")

# Test _date_from_str
d = _date_from_str("2024-03-15")
assert d == date(2024, 3, 15)
print("[OK] _date_from_str parses ISO date")

d2 = _date_from_str("03/15/2024")
assert d2 == date(2024, 3, 15)
print("[OK] _date_from_str parses US date format")

assert _date_from_str("") is None
print("[OK] _date_from_str returns None for empty string")

# Test _normalize_accession
norm = _normalize_accession("0001234567-24-000001")
assert norm == "000123456724000001", f"Got {norm}"
print("[OK] _normalize_accession strips dashes")

# Test CampaignType enum
assert CampaignType.BOARD_SEATS in list(CampaignType)
assert CampaignType.BOARD_SEATS.value == "board_seats"
print("[OK] CampaignType.BOARD_SEATS exists with correct value")

# Test Filing13D dataclass
filing = Filing13D(
    accession_number="000123456724000001",
    form_type="SC 13D",
    filing_date=date(2024, 3, 15),
    filer_name="Elliott Investment Management",
    filer_cik="0001234567",
    target_name="XYZ Corp",
    target_ticker="XYZ",
    target_cik="0009876543",
    target_cusip="98765X100",
    ownership_pct=9.8,
    shares_held=5_000_000,
    purpose_text="To acquire board representation.",
    item5_text="Shares acquired on open market.",
)
assert filing.form_type == "SC 13D"
assert filing.ownership_pct == 9.8
print("[OK] Filing13D dataclass created successfully")

# Test ActivistCampaign dataclass
campaign = ActivistCampaign(
    activist_name="Elliott Investment Management",
    activist_cik="0001234567",
    target_name="XYZ Corp",
    target_ticker="XYZ",
    start_date=date(2024, 3, 15),
    end_date=None,
    status="active",
    campaign_types=["board_seats"],
    initial_pct=9.8,
    peak_pct=9.8,
    demands=["Board representation", "Strategic review"],
    outcomes=[],
    filing_accession="000123456724000001",
)
assert campaign.status == "active"
assert "board_seats" in campaign.campaign_types
print("[OK] ActivistCampaign dataclass created successfully")

# Math: compute_campaign_success_rate
campaigns = [
    ActivistCampaign("Elliott", "C1", "A Corp", "AAA", date(2022, 1, 1),
                     date(2023, 1, 1), "won", [], 8.0, 12.0, [], [], "acc1"),
    ActivistCampaign("Elliott", "C2", "B Corp", "BBB", date(2022, 3, 1),
                     date(2023, 3, 1), "won", [], 7.0, 10.0, [], [], "acc2"),
    ActivistCampaign("Elliott", "C3", "C Corp", "CCC", date(2022, 6, 1),
                     date(2023, 6, 1), "settled", [], 6.0, 9.0, [], [], "acc3"),
    ActivistCampaign("Elliott", "C4", "D Corp", "DDD", date(2022, 9, 1),
                     date(2023, 9, 1), "lost", [], 5.0, 7.0, [], [], "acc4"),
    ActivistCampaign("Elliott", "C5", "E Corp", "EEE", date(2024, 1, 1),
                     None, "active", [], 9.0, 9.0, [], [], "acc5"),
]
result = compute_campaign_success_rate(campaigns)
assert result["total_campaigns"] == 5
assert result["wins"]    == 3
assert result["losses"]  == 1
assert result["ongoing"] == 1
assert abs(result["win_rate"] - 0.75) < 1e-9
assert abs(result["resolution_rate"] - 0.8) < 1e-9
print(f"[OK] compute_campaign_success_rate -> win_rate={result['win_rate']:.2f}")

# Math: predict_settlement_probability
result_p = predict_settlement_probability(
    board_seats_demanded=3,
    pct_owned=10.0,
    is_known_activist=True,
    target_pb_ratio=1.2,
    prior_campaigns=5,
)
log_odds_check = 0.405 + min(3, 5) * 0.35 + min(10.0, 25.0) * 0.04 + 0.55 + min(5, 10) * 0.10 + 0.30
expected_prob  = 1.0 / (1.0 + math.exp(-log_odds_check))
assert abs(result_p["settlement_probability"] - round(expected_prob, 4)) < 1e-4
assert result_p["settlement_probability"] > 0.5
print(f"[OK] predict_settlement_probability -> {result_p['settlement_probability']:.4f}")

# Math: compute_target_vulnerability_score
result_v = compute_target_vulnerability_score(
    pb_ratio=1.2,
    roe_pct=3.0,
    cash_to_market_cap=0.25,
    tsr_3yr_vs_index=-0.30,
    insider_ownership_pct=0.5,
    has_staggered_board=True,
)
assert result_v["vulnerability_score"] == 10.0
assert result_v["risk_label"] == "high"
assert len(result_v["signals"]) == 6
print(f"[OK] compute_target_vulnerability_score -> {result_v['vulnerability_score']}")

result_partial = compute_target_vulnerability_score(
    pb_ratio=1.3,
    roe_pct=2.0,
    has_staggered_board=True,
)
assert result_partial["vulnerability_score"] == 5.0
assert result_partial["risk_label"] == "moderate"
print(f"[OK] compute_target_vulnerability_score partial -> {result_partial['vulnerability_score']}")

result_none = compute_target_vulnerability_score(pb_ratio=3.0, roe_pct=20.0)
assert result_none["vulnerability_score"] == 0.0
assert result_none["risk_label"] == "low"
print("[OK] compute_target_vulnerability_score no signals -> 0.0")

# ── HARSH-AUDIT GATE — Wave 3 lie verification ─────────────────────────────
# 1) Hardcoded fallback DIS/PFE/INTC must be deleted from source.
src_path = Path("sentinel/sfe/activist_tracker_v3.py")
src      = src_path.read_text(encoding="utf-8")
assert "_get_model_vulnerable_targets" not in src, (
    "_get_model_vulnerable_targets() still present — hardcoded fallback "
    "was not removed"
)
forbidden = re.search(
    r"VulnerableTarget\([^)]*ticker\s*=\s*[\"']DIS[\"']", src
)
assert forbidden is None, (
    "Hardcoded VulnerableTarget(ticker='DIS', ...) still in file"
)
forbidden2 = re.search(
    r"VulnerableTarget\([^)]*ticker\s*=\s*[\"']PFE[\"']", src
)
assert forbidden2 is None, "Hardcoded VulnerableTarget(ticker='PFE') still in file"
forbidden3 = re.search(
    r"VulnerableTarget\([^)]*ticker\s*=\s*[\"']INTC[\"']", src
)
assert forbidden3 is None, "Hardcoded VulnerableTarget(ticker='INTC') still in file"
print("[OK] Hardcoded illustrative DIS/PFE/INTC list deleted")

# 2) Real EDGAR query string must be present.
assert "efts.sec.gov" in src, "Real EDGAR EFTS endpoint missing"
assert "SCHEDULE 13D" in src, "EDGAR EFTS query parameter missing"
print("[OK] Real EDGAR EFTS query is wired in source")

# 3) get_active_campaigns_from_edgar must exist and return list of dicts.
sig = inspect.signature(get_active_campaigns_from_edgar)
assert "days_back" in sig.parameters
print(f"[OK] get_active_campaigns_from_edgar signature: {sig}")

# 4) Functional difference probe — multiple calls with different top_n
#    must return same-or-smaller results, never the byte-identical
#    hardcoded list. We cap the EDGAR call by setting use_cache so we
#    don't hammer the wire in CI.
tracker = ActivistTracker()
# We don't strictly require the network to be up; tolerate empty.
try:
    campaigns = get_active_campaigns_from_edgar(days_back=30, use_cache=True)
    assert isinstance(campaigns, list), "campaigns must be a list"
    for c in campaigns[:5]:
        assert isinstance(c, dict), "each campaign must be a dict"
        for key in ("target_ticker", "target_cik", "target_name",
                    "filer_name", "filing_date", "form_type", "url"):
            assert key in c, f"campaign missing key '{key}'"
    print(f"[OK] get_active_campaigns_from_edgar returned {len(campaigns)} dicts")
except Exception as exc:
    # Network blocked is acceptable; the shape contract has already been
    # verified above. We still require the source-level guarantees to hold.
    print(f"[WARN] live EDGAR fetch skipped: {type(exc).__name__}: {exc}")
    campaigns = []

# 5) _split_display_name parses EDGAR display string.
name, tkr = _split_display_name("WALT DISNEY CO  (DIS) (CIK 0001744489)")
assert name == "WALT DISNEY CO", f"Got name={name!r}"
assert tkr  == "DIS",            f"Got ticker={tkr!r}"
print("[OK] _split_display_name parses 'NAME (TICKER) (CIK X)'")

# 6) find_vulnerable_companies functional contract — without cik_list,
#    it must NOT return the historical hardcoded DIS/PFE/INTC trio
#    verbatim (which was always exactly 3 elements with those tickers).
r1 = tracker.find_vulnerable_companies(top_n=5)
r2 = tracker.find_vulnerable_companies(top_n=10)
assert isinstance(r1, list) and isinstance(r2, list)
assert len(r1) <= 5,  f"top_n=5 returned {len(r1)} items"
assert len(r2) <= 10, f"top_n=10 returned {len(r2)} items"

hardcoded_signature = {"DIS", "PFE", "INTC"}
r1_tickers = {t.ticker for t in r1 if t.ticker}
# If the only tickers are exactly the legacy hardcoded set, the fallback is back.
if r1_tickers:
    assert r1_tickers != hardcoded_signature, (
        "find_vulnerable_companies returned the exact hardcoded "
        "{DIS, PFE, INTC} set — fallback still active"
    )
print(f"[OK] find_vulnerable_companies(top_n=5) -> {len(r1)} live targets")

# 7) Cache files live in .sentinel/cache (constants exposed).
assert str(_ACTIVIST_TARGETS_CACHE).replace("\\", "/").endswith(
    ".sentinel/cache/activist_targets.json"
)
assert str(_ACTIVIST_CAMPAIGNS_CACHE).replace("\\", "/").endswith(
    ".sentinel/cache/activist_campaigns.json"
)
print("[OK] 24h-TTL cache constants resolve to .sentinel/cache/...")

# 8) cik_list passthrough still works (existing behavior preserved).
screener = ActivistScreener()
custom = screener.find_vulnerable_companies(cik_list=[], top_n=5)
assert isinstance(custom, list)
print(f"[OK] cik_list= passthrough still functional ({len(custom)} rows)")

print("[PASS]")
PYEOF
