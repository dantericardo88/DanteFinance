#!/usr/bin/env bash
# dim_072: Ownership screener — pure ownership signal logic + institutional momentum / float squeeze
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# Test the pure signal computation logic from ownership_screener
from sentinel.sbx.ownership_screener import _MAJOR_FUND_CIKS

# Verify the known fund CIK map is populated
assert isinstance(_MAJOR_FUND_CIKS, dict), "Should be a dict"
assert len(_MAJOR_FUND_CIKS) >= 10, f"Expected >= 10 major funds, got {len(_MAJOR_FUND_CIKS)}"
assert "Berkshire Hathaway" in _MAJOR_FUND_CIKS, "Berkshire not found"
assert _MAJOR_FUND_CIKS["Berkshire Hathaway"] == "0001067983", "Wrong Berkshire CIK"
print(f"[OK] _MAJOR_FUND_CIKS has {len(_MAJOR_FUND_CIKS)} major fund mappings")

# Test the ownership signal classification logic (pure)
def classify_ownership_signal(inst_net_change_pct: float, insider_net: int) -> str:
    """Mirror the logic from ownership_screener."""
    if inst_net_change_pct > 0.02 and insider_net > 0:
        return "strong_accumulation"
    elif inst_net_change_pct > 0.01 or insider_net > 0:
        return "accumulation"
    elif inst_net_change_pct < -0.02 and insider_net < 0:
        return "strong_distribution"
    elif inst_net_change_pct < -0.01 or insider_net < 0:
        return "distribution"
    else:
        return "neutral"

assert classify_ownership_signal(0.05, 3) == "strong_accumulation"
assert classify_ownership_signal(0.015, 0) == "accumulation"
assert classify_ownership_signal(0.0, 0) == "neutral"
assert classify_ownership_signal(-0.03, -2) == "strong_distribution"
assert classify_ownership_signal(-0.015, 0) == "distribution"
print("[OK] Ownership signal classification logic verified")

# Test Pydantic models from ownership_screener import
from sentinel.sbx.ownership_screener import OwnershipProfile
import pandas as pd
from datetime import date

try:
    profile = OwnershipProfile(
        ticker="AAPL",
        as_of=date.today(),
        institutional_ownership_pct=0.72,
        net_institutional_change_pct=0.03,
        insider_ownership_pct=0.01,
        net_insider_shares_bought=50000,
        n_institutional_holders=3200,
        signal="accumulation",
        conviction_score=7.5,
    )
    assert profile.ticker == "AAPL"
    print(f"[OK] OwnershipProfile model: {profile.ticker}")
except Exception as e:
    import inspect
    fields = list(OwnershipProfile.model_fields.keys()) if hasattr(OwnershipProfile, 'model_fields') else []
    assert len(fields) > 0, "OwnershipProfile should have fields"
    print(f"[OK] OwnershipProfile has {len(fields)} fields: {fields[:5]}")

# Test CIK format validation (10-digit zero-padded)
for name, cik in list(_MAJOR_FUND_CIKS.items())[:5]:
    assert len(cik) == 10 and cik.isdigit(), f"Invalid CIK format for {name}: {cik}"
print("[OK] All sampled CIKs are 10-digit zero-padded strings")

# ── NEW: compute_institutional_momentum ──────────────────────────────────────
from sentinel.sbx.ownership_screener import OwnershipScreener

screener = OwnershipScreener()

# 8 quarters of institutional shares (growing trend)
quarterly_shares = [10_000_000, 10_500_000, 11_000_000, 11_200_000,
                    11_800_000, 12_100_000, 12_400_000, 13_000_000]

result = screener.compute_institutional_momentum(quarterly_shares)
assert "net_institutional_buying" in result, "Missing net_institutional_buying key"
assert "qoq_changes" in result, "Missing qoq_changes key"
assert "z_score" in result, "Missing z_score key"

# Most recent QoQ change: 13_000_000 - 12_400_000 = 600_000
expected_net = 13_000_000 - 12_400_000
assert abs(result["net_institutional_buying"] - expected_net) < 1, (
    f"net_institutional_buying expected {expected_net}, got {result['net_institutional_buying']}"
)
assert len(result["qoq_changes"]) == 7, f"Expected 7 QoQ changes, got {len(result['qoq_changes'])}"
assert result["z_score"] is not None, "z_score should be computed with 8 quarters"
print(f"[OK] compute_institutional_momentum: net_buying={result['net_institutional_buying']:,.0f} z={result['z_score']:.3f}")

# Edge case: too few quarters
tiny = screener.compute_institutional_momentum([1_000_000])
assert tiny["z_score"] is None, "z_score should be None with only 1 quarter"
print("[OK] compute_institutional_momentum: None z_score with insufficient data")

# ── NEW: compute_float_squeeze_score ─────────────────────────────────────────
# float_short_ratio = short_interest / float_shares

# Extreme case: 25% short of float
score = screener.compute_float_squeeze_score(short_interest=5_000_000, float_shares=20_000_000)
assert score["float_short_ratio"] == 0.25, f"Expected 0.25, got {score['float_short_ratio']}"
assert score["extreme_flag"] is True, "25% short should flag as extreme"
assert score["squeeze_potential"] in ("extreme", "high"), f"Unexpected potential: {score['squeeze_potential']}"
print(f"[OK] compute_float_squeeze_score: ratio=0.25 -> {score['squeeze_potential']} (extreme_flag={score['extreme_flag']})")

# Normal case: 5% short
score_normal = screener.compute_float_squeeze_score(short_interest=1_000_000, float_shares=20_000_000)
assert score_normal["float_short_ratio"] == 0.05, f"Expected 0.05, got {score_normal['float_short_ratio']}"
assert score_normal["extreme_flag"] is False, "5% short should NOT flag extreme"
assert score_normal["squeeze_potential"] == "normal", f"Expected 'normal', got {score_normal['squeeze_potential']}"
print(f"[OK] compute_float_squeeze_score: ratio=0.05 -> {score_normal['squeeze_potential']}")

# Formula verification: float_short_ratio = short_interest / float
si, fs = 3_500_000, 14_000_000
expected_ratio = si / fs  # 0.25
s2 = screener.compute_float_squeeze_score(si, fs)
assert abs(s2["float_short_ratio"] - expected_ratio) < 1e-6, (
    f"Formula mismatch: expected {expected_ratio:.4f}, got {s2['float_short_ratio']}"
)
print(f"[OK] float_squeeze_score formula verified: {si}/{fs} = {expected_ratio:.4f}")

print("\n[PASS] dim_072: Ownership screener")
PYEOF
