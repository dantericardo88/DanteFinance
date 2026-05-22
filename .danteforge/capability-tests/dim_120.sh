#!/usr/bin/env bash
# dim_120: Transaction cost analysis (TCA / implementation shortfall)
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

try:
    from sentinel.sma.tca_v3 import (
        Order, ISDecomposition, MarketImpactEstimate,
        ImplementationShortfallAnalyzer, BenchmarkAnalyzer,
        MarketImpactModel, BrokerAnalytics, TCAReport,
        compute_is, vwap, twap, market_impact,
    )
except ImportError as e:
    print(f"[NOT-BUILT] dim_120: module not yet implemented -- {e}")
    print("[SCORE=0] Crusade target: implement sentinel.sma.tca_v3")
    sys.exit(0)

# ── 1. Existence checks ───────────────────────────────────────────────────────
assert Order is not None, "Missing Order"
assert ISDecomposition is not None, "Missing ISDecomposition"
assert MarketImpactEstimate is not None, "Missing MarketImpactEstimate"
assert ImplementationShortfallAnalyzer is not None, "Missing ImplementationShortfallAnalyzer"
assert BenchmarkAnalyzer is not None, "Missing BenchmarkAnalyzer"
assert MarketImpactModel is not None, "Missing MarketImpactModel"
assert BrokerAnalytics is not None, "Missing BrokerAnalytics"
assert TCAReport is not None, "Missing TCAReport"
assert compute_is is not None, "Missing compute_is"
assert vwap is not None, "Missing vwap"
assert twap is not None, "Missing twap"
assert market_impact is not None, "Missing market_impact"
print("[OK] All classes and convenience functions present")

# ── 2. Build the reference order ─────────────────────────────────────────────
# decision=100, arrival=100.10, fill=100.25, close=101.00
# target=1000, filled=900, direction=+1, commission=0.01
order = Order(
    order_id="TEST001",
    direction=1,
    target_shares=1000.0,
    decision_price=100.00,
    arrival_price=100.10,
    avg_fill_price=100.25,
    shares_filled=900.0,
    close_price=101.00,
    commission_per_share=0.01,
    broker="broker_a",
    timestamp=36000.0,  # 10:00 AM in seconds
)
print("[OK] Order created successfully")

# ── 3. IS decomposition ───────────────────────────────────────────────────────
analyzer = ImplementationShortfallAnalyzer()
decomp = analyzer.decompose(order)

assert isinstance(decomp, ISDecomposition), "decompose() must return ISDecomposition"

# total IS must be positive (bought above decision price)
assert decomp.total_is_bps > 0, (
    f"total_is_bps should be > 0 for a buy filled above decision price, got {decomp.total_is_bps:.4f}"
)
print(f"[OK] total_is_bps = {decomp.total_is_bps:.4f} (> 0 as expected)")

# delay cost: arrival (100.10) > decision (100.00) → positive
assert decomp.delay_cost_bps > 0, (
    f"delay_cost_bps should be > 0 (arrival > decision for buy), got {decomp.delay_cost_bps:.4f}"
)
print(f"[OK] delay_cost_bps = {decomp.delay_cost_bps:.4f} (> 0)")

# execution cost: fill (100.25) > arrival (100.10) → positive
assert decomp.execution_cost_bps > 0, (
    f"execution_cost_bps should be > 0 (fill > arrival for buy), got {decomp.execution_cost_bps:.4f}"
)
print(f"[OK] execution_cost_bps = {decomp.execution_cost_bps:.4f} (> 0)")

# fill rate = 900/1000 = 0.9
assert abs(decomp.fill_rate - 0.9) < 1e-9, (
    f"fill_rate should be 0.9, got {decomp.fill_rate}"
)
print(f"[OK] fill_rate = {decomp.fill_rate:.4f} (== 0.9)")

# commission always positive
assert decomp.commission_bps > 0, f"commission_bps should be > 0, got {decomp.commission_bps}"
print(f"[OK] commission_bps = {decomp.commission_bps:.4f} (> 0)")

# opportunity cost: close (101.00) > decision (100.00), direction +1, unfilled=100 → positive
assert decomp.opportunity_cost_bps > 0, (
    f"opportunity_cost_bps should be > 0 for unfilled buy with rising close, got {decomp.opportunity_cost_bps:.4f}"
)
print(f"[OK] opportunity_cost_bps = {decomp.opportunity_cost_bps:.4f} (> 0)")

# Verify component arithmetic: components sum to total (approximately)
implied_total = (
    decomp.delay_cost_bps
    + decomp.execution_cost_bps
    + decomp.opportunity_cost_bps
    + decomp.commission_bps
)
assert abs(implied_total - decomp.total_is_bps) < 1e-6, (
    f"IS components don't sum to total: {implied_total:.6f} vs {decomp.total_is_bps:.6f}"
)
print(f"[OK] IS components sum correctly to {decomp.total_is_bps:.4f} bps")

# ── 4. is_efficient ───────────────────────────────────────────────────────────
assert not decomp.is_efficient(threshold_bps=5.0), (
    "Order should NOT be efficient at 5 bps threshold for this high-IS order"
)
assert decomp.is_efficient(threshold_bps=10000.0), "is_efficient(10000) must be True"
print("[OK] is_efficient() works correctly")

# ── 5. VWAP benchmark ────────────────────────────────────────────────────────
market_prices = np.array([100.05, 100.10, 100.20, 100.30])
market_volumes = np.array([300.0, 400.0, 200.0, 100.0])

# Verify standalone vwap function
# Expected: (100.05*300 + 100.10*400 + 100.20*200 + 100.30*100) / 1000
expected_vwap = (100.05 * 300 + 100.10 * 400 + 100.20 * 200 + 100.30 * 100) / 1000.0
computed_vwap = vwap(market_prices, market_volumes)
assert abs(computed_vwap - expected_vwap) < 1e-6, (
    f"vwap() = {computed_vwap:.6f}, expected {expected_vwap:.6f}"
)
assert abs(computed_vwap - 100.125) < 1e-3, (
    f"VWAP should be ~100.125, got {computed_vwap:.6f}"
)
print(f"[OK] vwap() = {computed_vwap:.6f} ~100.125")

# BenchmarkAnalyzer VWAP slippage
bench = BenchmarkAnalyzer()
vwap_slip = bench.vwap_slippage(order, market_prices, market_volumes)
# fill (100.25) > VWAP (100.135) for buy → positive slippage
assert vwap_slip > 0, f"vwap_slippage should be > 0 for buy above VWAP, got {vwap_slip:.4f}"
print(f"[OK] VWAP slippage = {vwap_slip:.4f} bps (> 0)")

# ── 6. TWAP benchmark ────────────────────────────────────────────────────────
computed_twap = twap(market_prices)
expected_twap = float(np.mean(market_prices))
assert abs(computed_twap - expected_twap) < 1e-9, (
    f"twap() mismatch: {computed_twap:.6f} vs {expected_twap:.6f}"
)
print(f"[OK] twap() = {computed_twap:.6f}")

twap_slip = bench.twap_slippage(order, market_prices)
assert isinstance(twap_slip, float), "twap_slippage must return float"
print(f"[OK] TWAP slippage = {twap_slip:.4f} bps")

# ── 7. Arrival slippage ───────────────────────────────────────────────────────
arr_slip = bench.arrival_slippage(order)
# fill (100.25) > arrival (100.10) for buy → positive
assert arr_slip > 0, f"arrival_slippage should be > 0, got {arr_slip:.4f}"
print(f"[OK] arrival_slippage = {arr_slip:.4f} bps (> 0)")

# ── 8. Market impact model ────────────────────────────────────────────────────
# order=50000, adv=1e6, sigma=0.02, spread=0.05%, T=1
mi = market_impact(
    order_size=50000.0,
    adv=1_000_000.0,
    sigma=0.02,
    spread_pct=0.0005,
    T_days=1.0,
)
assert isinstance(mi, MarketImpactEstimate), "market_impact() must return MarketImpactEstimate"
assert mi.total_bps > 0, f"total_bps should be > 0, got {mi.total_bps:.4f}"
assert mi.permanent_bps > 0, f"permanent_bps should be > 0, got {mi.permanent_bps:.4f}"
assert mi.temporary_bps > 0, f"temporary_bps should be > 0, got {mi.temporary_bps:.4f}"
assert mi.spread_bps > 0, f"spread_bps should be > 0, got {mi.spread_bps:.4f}"
assert mi.timing_risk_bps > 0, f"timing_risk_bps should be > 0, got {mi.timing_risk_bps:.4f}"
print(f"[OK] market_impact: total={mi.total_bps:.4f} bps (perm={mi.permanent_bps:.2f}, "
      f"tmp={mi.temporary_bps:.2f}, spread={mi.spread_bps:.2f}, timing_risk={mi.timing_risk_bps:.2f})")

# ── 9. MarketImpactModel.optimal_participation_rate ───────────────────────────
model = MarketImpactModel(gamma=0.5, eta=0.1)
h_star = model.optimal_participation_rate(sigma=0.02, lam_risk=0.001)
assert h_star > 0, f"optimal_participation_rate should be > 0, got {h_star}"
print(f"[OK] optimal_participation_rate = {h_star:.6f}")

# ── 10. Batch decompose + aggregate_stats ────────────────────────────────────
order2 = Order(
    order_id="TEST002",
    direction=-1,
    target_shares=500.0,
    decision_price=100.00,
    arrival_price=99.95,
    avg_fill_price=99.80,
    shares_filled=500.0,
    close_price=99.70,
    commission_per_share=0.01,
    broker="broker_b",
    timestamp=39600.0,  # 11:00 AM
)
order3 = Order(
    order_id="TEST003",
    direction=1,
    target_shares=2000.0,
    decision_price=50.00,
    arrival_price=50.05,
    avg_fill_price=50.10,
    shares_filled=2000.0,
    close_price=50.15,
    commission_per_share=0.005,
    broker="broker_a",
    timestamp=43200.0,  # 12:00 PM
)
all_orders = [order, order2, order3]
batch = analyzer.batch_decompose(all_orders)
assert len(batch) == 3, f"batch_decompose should return 3 items, got {len(batch)}"
stats = analyzer.aggregate_stats(batch)
assert stats["count"] == 3, f"aggregate count mismatch: {stats['count']}"
assert "avg_total_is_bps" in stats, "aggregate_stats missing avg_total_is_bps"
assert "avg_fill_rate" in stats, "aggregate_stats missing avg_fill_rate"
assert 0.0 <= stats["avg_fill_rate"] <= 1.0, f"fill_rate out of range: {stats['avg_fill_rate']}"
print(f"[OK] batch_decompose: {stats['count']} orders, avg_is={stats['avg_total_is_bps']:.4f} bps")

# ── 11. BrokerAnalytics ───────────────────────────────────────────────────────
ba = BrokerAnalytics(all_orders)
by_broker = ba.by_broker()
assert len(by_broker) == 2, f"Expected 2 brokers, got {len(by_broker)}"
assert "broker_a" in by_broker, "Missing broker_a"
assert "broker_b" in by_broker, "Missing broker_b"

ranked = ba.rank_brokers(metric="total_is_bps")
assert isinstance(ranked, list), "rank_brokers must return list"
assert len(ranked) == 2, f"Expected 2 ranked entries, got {len(ranked)}"
assert all(isinstance(r, tuple) and len(r) == 2 for r in ranked), \
    "rank_brokers must return list of (str, float) tuples"
print(f"[OK] rank_brokers: {ranked}")

best = ba.best_broker()
assert isinstance(best, str) and len(best) > 0, "best_broker must return non-empty string"
print(f"[OK] best_broker = '{best}'")

# ── 12. TCAReport ────────────────────────────────────────────────────────────
report = TCAReport(all_orders, adv=1_000_000.0, sigma=0.02)
summary = report.summary()
assert "count" in summary, "TCAReport.summary() missing 'count'"
assert summary["count"] == 3, f"TCAReport count mismatch: {summary['count']}"
print(f"[OK] TCAReport.summary(): {summary['count']} orders")

by_dir = report.by_direction()
assert "buy" in by_dir, "by_direction() missing 'buy'"
assert "sell" in by_dir, "by_direction() missing 'sell'"
buy_stats = by_dir["buy"]
sell_stats = by_dir["sell"]
assert buy_stats.get("count", 0) >= 1, "Expected at least 1 buy order"
assert sell_stats.get("count", 0) >= 1, "Expected at least 1 sell order"
print(f"[OK] by_direction(): buy_count={buy_stats.get('count')}, sell_count={sell_stats.get('count')}")

tod = report.time_of_day_analysis()
assert isinstance(tod, dict), "time_of_day_analysis() must return dict"
assert len(tod) >= 1, "time_of_day_analysis() must return at least one bucket"
print(f"[OK] time_of_day_analysis(): {len(tod)} hour buckets")

# ── 13. Standalone compute_is matches analyzer ─────────────────────────────
d1 = compute_is(order)
d2 = analyzer.decompose(order)
assert abs(d1.total_is_bps - d2.total_is_bps) < 1e-9, \
    f"compute_is() mismatch with analyzer.decompose(): {d1.total_is_bps} vs {d2.total_is_bps}"
print("[OK] compute_is() matches ImplementationShortfallAnalyzer.decompose()")

print("\n[PASS] dim_120: Transaction cost analysis (TCA / implementation shortfall) -- all checks passed")
PYEOF
