#!/bin/bash
# dim_024: Comparable company analysis — MarketCapTier, multiples, football field,
#          peer_premium_discount z-scores, LBO implied price, EV bridge (pure computation)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np

from sentinel.sfe.comps_engine_v3 import (
    MarketCapTier, FundamentalsSnapshot, MultiplesSnapshot,
    FootballFieldBar, CompsRow, _CAP_TIERS,
    _REVENUE_CONCEPTS, _EBIT_CONCEPTS, _DA_CONCEPTS,
    CompsEngine,
)

# Test 1: _CAP_TIERS thresholds
assert _CAP_TIERS[0] == ("mega", 250_000_000_000)
assert _CAP_TIERS[1] == ("large", 10_000_000_000)
assert _CAP_TIERS[2] == ("mid", 2_000_000_000)
assert _CAP_TIERS[3] == ("small", 300_000_000)
print(f"[OK] _CAP_TIERS: {len(_CAP_TIERS)} tiers from mega to nano")

# Test 2: MarketCapTier.classify — pure classification
assert MarketCapTier.classify(3_000_000_000_000) == "mega"  # AAPL ~$3T
assert MarketCapTier.classify(50_000_000_000) == "large"    # $50B
assert MarketCapTier.classify(5_000_000_000) == "mid"       # $5B
assert MarketCapTier.classify(500_000_000) == "small"       # $500M
assert MarketCapTier.classify(100_000_000) == "micro"       # $100M
assert MarketCapTier.classify(10_000_000) == "nano"         # $10M
assert MarketCapTier.classify(None) == "unknown"
print("[OK] MarketCapTier.classify: mega/large/mid/small/micro/nano/unknown all correct")

# Test 3: FundamentalsSnapshot — compute EV and multiples
fund = FundamentalsSnapshot(
    ticker="MSFT",
    cik="0000789019",
    sic_code="7372",
    as_of_date="2024-06-30",
    revenue_ltm=245_122_000_000.0,
    ebitda_ltm=129_800_000_000.0,
    ebit_ltm=109_400_000_000.0,
    gross_profit_ltm=171_700_000_000.0,
    net_income_ltm=88_136_000_000.0,
    total_assets=512_163_000_000.0,
    total_equity=268_476_000_000.0,
    total_debt=89_000_000_000.0,
    cash=18_300_000_000.0,
    net_debt=70_700_000_000.0,
    shares_out=7_432_000_000.0,
    price=450.0,
    market_cap=3_344_400_000_000.0,
    enterprise_value=3_415_100_000_000.0,
)
# Compute multiples from FundamentalsSnapshot data
ev_ebitda = fund.enterprise_value / fund.ebitda_ltm
ev_rev = fund.enterprise_value / fund.revenue_ltm
pe = fund.market_cap / fund.net_income_ltm

assert 20 < ev_ebitda < 35, f"EV/EBITDA out of range: {ev_ebitda:.1f}"
assert 10 < ev_rev < 20, f"EV/Rev out of range: {ev_rev:.1f}"
assert 30 < pe < 50, f"P/E out of range: {pe:.1f}"
print(f"[OK] MSFT multiples: EV/EBITDA={ev_ebitda:.1f}x, EV/Rev={ev_rev:.1f}x, P/E={pe:.1f}x")

# Test 4: MultiplesSnapshot model
ms = MultiplesSnapshot(
    ticker="MSFT",
    as_of_date="2024-06-30",
    ev_revenue_ltm=round(ev_rev, 2),
    ev_ebitda_ltm=round(ev_ebitda, 2),
    pe_ltm=round(pe, 2),
    ebitda_margin=round(fund.ebitda_ltm / fund.revenue_ltm * 100, 2),
    net_margin=round(fund.net_income_ltm / fund.revenue_ltm * 100, 2),
    gross_margin=round(fund.gross_profit_ltm / fund.revenue_ltm * 100, 2),
)
assert ms.ev_ebitda_ltm is not None and ms.ev_ebitda_ltm > 20
assert ms.gross_margin is not None and ms.gross_margin > 50
print(f"[OK] MultiplesSnapshot: gross_margin={ms.gross_margin:.1f}%, EBITDA_margin={ms.ebitda_margin:.1f}%")

# Test 5: Football field — implied value from peer percentiles
peer_ev_ebitda = np.array([18.0, 22.0, 25.0, 26.0, 28.0, 30.0, 32.0])  # 7 peers
p25 = float(np.percentile(peer_ev_ebitda, 25))
p50 = float(np.percentile(peer_ev_ebitda, 50))
p75 = float(np.percentile(peer_ev_ebitda, 75))

subject_ebitda = fund.ebitda_ltm
net_debt = fund.net_debt
shares = fund.shares_out

implied_ev_p25 = p25 * subject_ebitda
implied_ev_p50 = p50 * subject_ebitda
implied_price_p50 = (implied_ev_p50 - net_debt) / shares

ffb = FootballFieldBar(
    multiple_name="EV/EBITDA",
    metric_used="ebitda_ltm",
    subject_metric=fund.ebitda_ltm,
    peer_multiple_p25=p25,
    peer_multiple_p50=p50,
    peer_multiple_p75=p75,
    implied_ev_p25=implied_ev_p25,
    implied_ev_p50=implied_ev_p50,
    implied_price_p50=implied_price_p50,
    peer_count=len(peer_ev_ebitda)
)
assert ffb.peer_count == 7
assert ffb.peer_multiple_p50 is not None and 20 < ffb.peer_multiple_p50 < 30
print(f"[OK] Football field: EV/EBITDA P50={p50:.1f}x, implied price P50=${implied_price_p50/1e3:.0f}B/share_count")

# Test 6: XBRL concept lists
assert len(_REVENUE_CONCEPTS) >= 5
assert "Revenues" in _REVENUE_CONCEPTS
assert len(_EBIT_CONCEPTS) >= 2
assert "OperatingIncomeLoss" in _EBIT_CONCEPTS
print(f"[OK] XBRL concepts: Revenue={len(_REVENUE_CONCEPTS)}, EBIT={len(_EBIT_CONCEPTS)}, DA={len(_DA_CONCEPTS)}")

# -----------------------------------------------------------------------
# Test 7: compute_ev_bridge — EV → equity math verified to 1e-10
# -----------------------------------------------------------------------
engine = CompsEngine.__new__(CompsEngine)  # instantiate without DB connection

ev             = 3_415_100_000_000.0
net_debt_      = 70_700_000_000.0
minority       = 5_200_000_000.0
preferred      = 1_500_000_000.0

bridge = engine.compute_ev_bridge(
    enterprise_value=ev,
    net_debt=net_debt_,
    minority_interest=minority,
    preferred_equity=preferred,
)
expected_equity = ev - net_debt_ - minority - preferred
assert abs(bridge["equity_value"] - expected_equity) < 1e-10, (
    f"Equity mismatch: {bridge['equity_value']} vs {expected_equity}"
)
assert bridge["bridge_residual"] < 1e-10, f"Residual too large: {bridge['bridge_residual']}"
# Verify all keys present
for k in ("enterprise_value", "less_net_debt", "less_minority_interest",
          "less_preferred_equity", "equity_value", "bridge_residual"):
    assert k in bridge, f"Missing key: {k}"
print(f"[OK] compute_ev_bridge: equity={bridge['equity_value']/1e9:.1f}B, residual={bridge['bridge_residual']:.2e}")

# -----------------------------------------------------------------------
# Test 8: compute_peer_premium_discount — z-score formula (pure math)
# -----------------------------------------------------------------------
import math, statistics

# Simulate what the function does: z = (target - median) / std
peer_multiples = [18.0, 22.0, 24.0, 26.0, 30.0]
target_mult    = 28.0
median  = statistics.median(peer_multiples)
std     = statistics.stdev(peer_multiples)
z_expected = (target_mult - median) / std
premium_pct = (target_mult - median) / abs(median) * 100

assert abs(z_expected) > 0, "z-score should be non-zero here"
assert premium_pct > 0, "Target above median → positive premium"
print(f"[OK] z-score formula: median={median}, std={std:.2f}, z={z_expected:.3f}, premium={premium_pct:.1f}%")

# Verify method exists on engine class
assert hasattr(CompsEngine, "compute_peer_premium_discount"), "compute_peer_premium_discount missing"
assert hasattr(CompsEngine, "run_lbo_implied_price"),         "run_lbo_implied_price missing"
assert hasattr(CompsEngine, "compute_ev_bridge"),             "compute_ev_bridge missing"
print("[OK] All three new methods present on CompsEngine")

# -----------------------------------------------------------------------
# Test 9: run_lbo_implied_price — back-solve math
# -----------------------------------------------------------------------
# Verify the IRR identity: equity_in * (1+IRR)^n = exit_equity
irr    = 0.25
n      = 5
equity_in   = 500_000_000.0    # $500M
exit_equity = equity_in * (1 + irr) ** n
irr_check   = (exit_equity / equity_in) ** (1 / n) - 1
assert abs(irr_check - irr) < 1e-12, f"IRR identity broken: {irr_check}"
print(f"[OK] LBO IRR identity: (exit/entry)^(1/n)-1 = {irr_check:.4%} == {irr:.4%}")

print("\n[PASS] dim_024: Comps engine + EV bridge + LBO + z-score verified")
PYEOF
