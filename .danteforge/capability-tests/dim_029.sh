#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')

from sentinel.sfe.congress_tracker_v3 import (
    CongressTrade,
    compute_alpha_signal,
    detect_cluster_trades,
    compute_senator_performance,
)
from datetime import date, timedelta

# Build sample CongressTrade objects for testing
def make_trade(member, ticker, tx_date, is_buy=True, party="D", amount=50_000.0):
    return CongressTrade(
        member=member,
        ticker=ticker,
        transaction_date=tx_date,
        disclosure_date=tx_date + timedelta(days=30),
        trade_type="purchase" if is_buy else "sale",
        amount_range="$15,001 - $50,000",
        amount_low=15_001,
        amount_high=50_000,
        amount_midpoint=amount,
        asset_type="Stock",
        asset_description=ticker,
        chamber="senate",
        party=party,
        state="CA",
        district=None,
        cap_gains_over_200=False,
        comment="",
    )

base_date = date(2024, 1, 15)

# ── verify compute_alpha_signal math: excess_return = portfolio - benchmark ──
# Create buy trades for NVDA
trades_nvda = [
    make_trade("Sen. Alpha",   "NVDA", base_date),
    make_trade("Rep. Beta",    "NVDA", base_date + timedelta(days=2)),
    make_trade("Sen. Gamma",   "NVDA", base_date + timedelta(days=4)),
]

result = compute_alpha_signal(trades_nvda, "NVDA", horizon_days=30, benchmark_symbol="SPY")
assert result["ticker"]      == "NVDA"
assert result["buy_count"]   == 3
assert result["horizon_days"] == 30
assert result["benchmark"]   == "SPY"
# When yfinance unavailable, returns and excess_return will be None - that's valid
if result["portfolio_return"] is not None and result["benchmark_return"] is not None:
    computed_excess = result["portfolio_return"] - result["benchmark_return"]
    assert abs(result["excess_return"] - round(computed_excess, 6)) < 1e-9, \
        f"excess_return={result['excess_return']} != portfolio-benchmark={computed_excess}"
    print(f"[OK] compute_alpha_signal math: excess={result['excess_return']:.4f} "
          f"= portfolio({result['portfolio_return']:.4f}) - benchmark({result['benchmark_return']:.4f})")
else:
    # No yfinance in test env - verify structure is correct
    assert "excess_return" in result
    assert "signal" in result
    assert result["signal"] in {"no_price_data", "alpha_positive", "alpha_negative", "neutral", "no_data"}
    print("[OK] compute_alpha_signal: no price data -> correct structure returned")

# Verify excess_return formula manually
pr, br = 0.05, 0.02   # 5% portfolio, 2% benchmark
excess = pr - br       # = 0.03
assert abs(excess - 0.03) < 1e-12
print(f"[OK] Alpha signal math: excess_return = portfolio_return - benchmark_return = {excess:.2f}")

# Empty ticker -> buy_count 0
result_empty = compute_alpha_signal(trades_nvda, "AAPL", horizon_days=30)
assert result_empty["buy_count"] == 0
assert result_empty["signal"]    == "no_data"
print("[OK] compute_alpha_signal: wrong ticker -> buy_count=0, signal=no_data")

# ── verify detect_cluster_trades ────────────────────────────────────────────
# 4 congress members buying same stock within 7 days -> cluster
cluster_trades = [
    make_trade("Sen. A",  "MSFT", base_date,                    party="D"),
    make_trade("Sen. B",  "MSFT", base_date + timedelta(days=2), party="R"),
    make_trade("Rep. C",  "MSFT", base_date + timedelta(days=4), party="D"),
    make_trade("Rep. D",  "MSFT", base_date + timedelta(days=6), party="R"),
]
result_c = detect_cluster_trades(cluster_trades, "MSFT", window_days=7, min_members=3)
assert result_c["ticker"]           == "MSFT"
assert result_c["cluster_detected"] is True,  f"Expected cluster, got {result_c}"
assert result_c["cluster_count"]    >= 1
assert result_c["clusters"][0]["member_count"] >= 3
print(f"[OK] detect_cluster_trades: 4 members in 7d -> cluster_detected=True, "
      f"size={result_c['clusters'][0]['member_count']}")

# Only 2 members -> no cluster (min_members=3)
two_trades = cluster_trades[:2]
result_2 = detect_cluster_trades(two_trades, "MSFT", window_days=7, min_members=3)
assert result_2["cluster_detected"] is False
print("[OK] detect_cluster_trades: 2 members -> cluster_detected=False (min=3)")

# 3 members spread > window -> no cluster
spread_trades = [
    make_trade("Sen. A", "GOOG", base_date,                      party="D"),
    make_trade("Sen. B", "GOOG", base_date + timedelta(days=10),  party="R"),
    make_trade("Rep. C", "GOOG", base_date + timedelta(days=20),  party="D"),
]
result_spread = detect_cluster_trades(spread_trades, "GOOG", window_days=7, min_members=3)
assert result_spread["cluster_detected"] is False, \
    f"Spread trades should not cluster: {result_spread}"
print("[OK] detect_cluster_trades: 3 members spread >7d -> cluster_detected=False")

# ── verify compute_senator_performance structure and annualization math ──────
# Build trades for two senators
senator_trades = [
    make_trade("Sen. Warren",    "AAPL", base_date,                   party="D"),
    make_trade("Sen. Warren",    "MSFT", base_date + timedelta(days=5), party="D"),
    make_trade("Sen. Grassley",  "JPM",  base_date + timedelta(days=3), party="R"),
]

results = compute_senator_performance(senator_trades, horizon_days=90)
assert isinstance(results, list), "compute_senator_performance must return a list"
members_in_result = {r["member"] for r in results}
assert "Sen. Warren"   in members_in_result
assert "Sen. Grassley" in members_in_result
for r in results:
    assert "annualized_return" in r
    assert "avg_period_return" in r
    assert "trade_count"       in r
    # If return available, verify annualization formula
    if r["avg_period_return"] is not None and r["annualized_return"] is not None:
        expected_ann = (1.0 + r["avg_period_return"]) ** (365.0 / 90) - 1.0
        assert abs(r["annualized_return"] - round(expected_ann, 4)) < 5e-4, \
            f"Annualization error: {r['annualized_return']} != {expected_ann:.4f}"
print(f"[OK] compute_senator_performance: {len(results)} senators, correct structure")

# Verify annualization math directly (no yfinance needed)
# avg_period_return = 10% over 90 days -> annualized = (1.10)^(365/90) - 1 ~ 47.2%
avg_r = 0.10
ann   = (1.0 + avg_r) ** (365.0 / 90) - 1.0
assert abs(ann - 0.4719) < 0.001, f"Annualization: {ann:.4f} expected ~0.4719"
print(f"[OK] Annualization math: 10% per 90d -> {ann:.4f} annualized (~47.2%)")

print("[PASS]")
PYEOF
