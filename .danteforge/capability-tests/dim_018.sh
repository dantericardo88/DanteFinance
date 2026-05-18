#!/bin/bash
# dim_018: Analyst estimates consensus — model interfaces and helper functions
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sfe.analyst_estimates import (
    RecommendationGrade, QuarterlyEstimate, AnalystEstimates,
    AnalystTickerSummary, AnalystSentimentScreen,
    _consensus_label, _classify_action,
)

# Test 1: _consensus_label pure computation
assert _consensus_label(1.2) == "Strong Buy"
assert _consensus_label(1.8) == "Buy"
assert _consensus_label(2.8) == "Hold"
assert _consensus_label(3.8) == "Sell"
assert _consensus_label(4.8) == "Strong Sell"
assert _consensus_label(None) == "N/A"
print("[OK] _consensus_label: 1.2->Strong Buy, 2.8->Hold, 4.8->Strong Sell, None->N/A")

# Test 2: _classify_action pure computation
assert _classify_action("Upgraded", "Buy", "Hold") == "upgrade"
assert _classify_action("Downgraded", "Hold", "Buy") == "downgrade"
assert _classify_action("Initiates Coverage", "Buy", "") == "initiated"
assert _classify_action("Reiterated", "Buy", "Buy") == "reiterated"
# Grade-based inference
assert _classify_action("maintains", "Strong Buy", "Sell") == "upgrade"
assert _classify_action("maintains", "Underweight", "Buy") == "downgrade"
print("[OK] _classify_action: upgrade/downgrade/initiated/reiterated all work")

# Test 3: QuarterlyEstimate model
qe = QuarterlyEstimate(
    period="0q",
    avg_estimate=2.85,
    low_estimate=2.50,
    high_estimate=3.20,
    number_of_analysts=28
)
assert qe.period == "0q"
assert qe.number_of_analysts == 28
print(f"[OK] QuarterlyEstimate: period={qe.period}, avg_est={qe.avg_estimate}, n_analysts={qe.number_of_analysts}")

# Test 4: RecommendationGrade model
rg = RecommendationGrade(
    firm="Goldman Sachs",
    to_grade="Buy",
    from_grade="Hold",
    action="upgrade",
    date="2024-03-15"
)
assert rg.action == "upgrade"
print(f"[OK] RecommendationGrade: {rg.firm} {rg.action} to {rg.to_grade}")

# Test 5: AnalystEstimates model — consensus label from rec_mean
consensus = _consensus_label(2.3)
ae = AnalystEstimates(
    ticker="AAPL",
    current_price=185.0,
    target_mean=210.0,
    target_median=208.0,
    target_high=250.0,
    target_low=160.0,
    price_upside_pct=(210.0 - 185.0) / 185.0 * 100,
    num_analysts=42,
    recommendation_mean=2.3,
    consensus_label=consensus,
    analyst_dispersion=(250.0 - 160.0) / 210.0 * 100,
    recent_grades=[rg],
    upgrades_90d=5,
    downgrades_90d=1,
    eps_estimates=[qe],
    revenue_estimates=[],
    as_of="2024-03-15",
    warnings=[]
)
assert ae.consensus_label == "Buy"
assert abs(ae.price_upside_pct - (210-185)/185*100) < 1e-3
assert ae.upgrades_90d == 5
print(f"[OK] AnalystEstimates: {ae.ticker} consensus={ae.consensus_label}, upside={ae.price_upside_pct:.1f}%")

# Test 6: AnalystTickerSummary
ats = AnalystTickerSummary(
    ticker="MSFT",
    upside_pct=18.5,
    consensus="Buy",
    num_analysts=45,
    target_mean=450.0,
    current_price=380.0
)
assert ats.upside_pct == 18.5
print(f"[OK] AnalystTickerSummary: {ats.ticker} upside={ats.upside_pct}%")

print("[PASS]")
PYEOF
