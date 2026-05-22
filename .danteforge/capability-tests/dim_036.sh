#!/bin/bash
# dim_036: trace_bond_v3 — FINRA TRACE live feed + bond price consolidator
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
PYTHONIOENCODING=utf-8 timeout 30 python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# ── existing imports ──────────────────────────────────────────────────────────
from sentinel.sfe.trace_bond_v3 import (
    FRED_OAS_SERIES,
    FRED_TREASURY_SERIES,
    FRED_CSV,
    BondMath,
    TreasuryOASService,
    BondPricer,
    FINRA_TRACE_URL,
    TRACELiveFeed,
    SpreadAnalytics,
    BondPriceConsolidator,
    _TSY_FALLBACK,
)
from datetime import datetime, timedelta

# ── 1. constants ─────────────────────────────────────────────────────────────
assert "finra.org" in FINRA_TRACE_URL, "FINRA_TRACE_URL missing"
assert "fred.stlouisfed.org" in FRED_CSV
assert "10Y" in FRED_TREASURY_SERIES
assert "AAA" in FRED_OAS_SERIES
print("[OK] FINRA_TRACE_URL, FRED_CSV, FRED_OAS_SERIES constants present")

# ── 2. TRACELiveFeed.vwap — pure math, no network ────────────────────────────
trades_a = [
    {"date": "2026-05-18", "cusip": "037833AK5", "quantity": 100, "price": 98.5, "yield_": 4.2},
    {"date": "2026-05-18", "cusip": "037833AK5", "quantity": 200, "price": 98.2, "yield_": 4.25},
]
vwap = TRACELiveFeed.vwap(trades_a)
expected_vwap = (100 * 98.5 + 200 * 98.2) / (100 + 200)   # = 98.30
assert vwap is not None, "VWAP should not be None"
assert abs(vwap - expected_vwap) < 0.0001, f"VWAP expected {expected_vwap:.4f}, got {vwap:.4f}"
print(f"[OK] TRACELiveFeed.vwap: {vwap:.4f} == {expected_vwap:.4f}")

# edge: empty trades → None
assert TRACELiveFeed.vwap([]) is None
# edge: zero quantity → None
assert TRACELiveFeed.vwap([{"quantity": 0, "price": 99.0}]) is None
print("[OK] TRACELiveFeed.vwap edge cases: empty→None, zero qty→None")

# ── 3. TRACELiveFeed.is_recent ────────────────────────────────────────────────
today_str = datetime.utcnow().strftime("%Y-%m-%d")
recent_trades  = [{"date": today_str, "cusip": "X", "quantity": 100, "price": 99.0, "yield_": 4.0}]
old_trades     = [{"date": "2020-01-01", "cusip": "X", "quantity": 100, "price": 99.0, "yield_": 4.0}]

feed = TRACELiveFeed()
assert feed.is_recent(recent_trades, max_hours=24.0) is True,  "Today's trade should be recent"
assert feed.is_recent(old_trades,    max_hours=24.0) is False, "2020 trade should not be recent"
assert feed.is_recent([],            max_hours=24.0) is False, "Empty trades should not be recent"
print("[OK] TRACELiveFeed.is_recent: today=True, old=False, empty=False")

# ── 4. BondPriceConsolidator — TRACE price takes priority within 24h ─────────
class _MockFeed:
    """Returns pre-canned recent trades for consolidator unit test."""
    def get_recent_trades(self, cusip, n=20):
        return [
            {"date": today_str, "cusip": cusip, "quantity": 100, "price": 98.5, "yield_": 4.3},
            {"date": today_str, "cusip": cusip, "quantity": 200, "price": 98.2, "yield_": 4.35},
        ]
    def is_recent(self, trades, max_hours=24.0):
        return True   # always recent in mock

consolidator = BondPriceConsolidator(trace_feed=_MockFeed())
bond = {"cusip": "037833AK5", "coupon": 3.2, "maturity": "2025-05-13", "callable": True}
result = consolidator.consolidated_price("AAPL", bond)

assert result["price_source"] == "TRACE", f"Expected TRACE source, got {result['price_source']}"
assert result["trace_vwap"] is not None
assert result["consolidated_price"] == round(expected_vwap, 4), (
    f"Consolidated price should be TRACE VWAP {expected_vwap:.4f}, got {result['consolidated_price']}"
)
print(f"[OK] BondPriceConsolidator: TRACE price takes priority, source={result['price_source']}, "
      f"price={result['consolidated_price']:.4f}")

# No-feed → falls back to model
class _EmptyFeed:
    def get_recent_trades(self, cusip, n=20): return []
    def is_recent(self, trades, max_hours=24.0): return False

consolidator_model = BondPriceConsolidator(trace_feed=_EmptyFeed())
result_model = consolidator_model.consolidated_price("AAPL", bond)
assert result_model["price_source"] == "MODEL", f"Expected MODEL, got {result_model['price_source']}"
assert result_model["consolidated_price"] == result_model["model_price"]
print(f"[OK] BondPriceConsolidator: no recent TRACE → falls back to MODEL price")

# ── 5. SpreadAnalytics.i_spread ───────────────────────────────────────────────
# YTM 5.5% - swap rate 4.5% = 100 bps
i_spr = SpreadAnalytics.i_spread(ytm_pct=5.5, swap_rate_pct=4.5)
assert abs(i_spr - 100.0) < 0.001, f"I-spread expected 100bps, got {i_spr}"
print(f"[OK] SpreadAnalytics.i_spread: 5.5% YTM - 4.5% swap = {i_spr:.2f} bps")

# YTM 4.8% - swap 4.3% = 50 bps
i_spr2 = SpreadAnalytics.i_spread(ytm_pct=4.8, swap_rate_pct=4.3)
assert abs(i_spr2 - 50.0) < 0.001, f"I-spread expected 50bps, got {i_spr2}"
print(f"[OK] SpreadAnalytics.i_spread: 4.8% - 4.3% = {i_spr2:.2f} bps")

# ── 6. SpreadAnalytics.asset_swap_spread ──────────────────────────────────────
# At-par bond: coupon 5%, par swap 4.5% → ASW = 50 bps
asw_par = SpreadAnalytics.asset_swap_spread(coupon_pct=5.0, par_swap_rate_pct=4.5, price=100.0)
assert abs(asw_par - 50.0) < 0.001, f"ASW at par expected 50bps, got {asw_par}"
print(f"[OK] SpreadAnalytics.asset_swap_spread at-par: coupon5%−swap4.5% = {asw_par:.2f} bps")

# Off-par bond (price=105) gets a price adjustment
asw_offpar = SpreadAnalytics.asset_swap_spread(coupon_pct=5.0, par_swap_rate_pct=4.5, price=105.0)
assert asw_offpar < asw_par, "Off-par premium bond should have lower ASW than at-par"
print(f"[OK] SpreadAnalytics.asset_swap_spread off-par (price=105): {asw_offpar:.2f} bps < {asw_par:.2f}")

# ── 7. SpreadAnalytics.running_dv01 ──────────────────────────────────────────
# DV01=$500/1M, face=$1000, 1000 bonds → notional=$1M → running DV01=$500
rdv01 = SpreadAnalytics.running_dv01(dv01_per_1m=500.0, face_value=1000.0, position_size=1000)
assert abs(rdv01 - 500.0) < 0.001, f"Running DV01 expected $500, got {rdv01}"
print(f"[OK] SpreadAnalytics.running_dv01: 500/1M × $1M = ${rdv01:.4f}")

# DV01=$250/1M, face=$1000, 500 bonds → notional=$500k → running DV01=$125
rdv01b = SpreadAnalytics.running_dv01(dv01_per_1m=250.0, face_value=1000.0, position_size=500)
assert abs(rdv01b - 125.0) < 0.001, f"Running DV01 expected $125, got {rdv01b}"
print(f"[OK] SpreadAnalytics.running_dv01: 250/1M × $500k = ${rdv01b:.4f}")

# ── 8. BondMath.i_spread (legacy path still works) ───────────────────────────
assert hasattr(BondMath, "i_spread")
i_spr_bm = BondMath.i_spread(ytm_pct=5.5, swap_rate_pct=4.5)
assert abs(i_spr_bm - 100.0) < 0.001
print(f"[OK] BondMath.i_spread (static): 5.5% - 4.5% = {i_spr_bm:.2f} bps")

# ── 9. BondPricer still works ─────────────────────────────────────────────────
pricer = BondPricer()
bond2  = {"cusip": "594918BQ6", "coupon": 3.125, "maturity": "2028-11-03", "callable": True}
result2 = pricer.price_bond("MSFT", bond2)
assert "model_price" in result2
assert result2["model_price"] > 0
print(f"[OK] BondPricer still works: MSFT 3.125% 2028 → model_price={result2['model_price']:.4f}")

print("\n[PASS] dim_036: trace_bond_v3 -- all checks passed")
PYEOF
