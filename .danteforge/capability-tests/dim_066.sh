#!/bin/bash
# dim_066: Paper trading v3 — Portfolio, PaperBroker, PerformanceAnalytics pure math
# Enhanced: bid/ask spread, market impact, order lifecycle, P&L attribution
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, date

from sentinel.sbx.paper_trading_v3 import (
    OrderType, OrderSide, OrderStatus, SlippageModel,
    Order, Trade, Position, Portfolio, Tearsheet, RiskCheck,
    NYSE_HOLIDAYS, PaperBroker, RiskManager, PerformanceAnalytics,
    PaperTradingSession, MarketSimulator, PnLAttribution,
)

# ── 1. Enums ──────────────────────────────────────────────────────────────────
assert OrderType.MARKET == "MARKET"
assert OrderType.LIMIT == "LIMIT"
assert OrderSide.BUY == "BUY"
assert OrderSide.SELL == "SELL"
assert OrderStatus.PENDING == "PENDING"
assert OrderStatus.SUBMITTED == "SUBMITTED"
assert OrderStatus.PARTIALLY_FILLED == "PARTIALLY_FILLED"
assert OrderStatus.FILLED == "FILLED"
assert SlippageModel.NONE == "none"
assert SlippageModel.MARKET_IMPACT == "market_impact"
print("[OK] Enums: all values correct including SUBMITTED and PARTIALLY_FILLED")

# ── 2. NYSE_HOLIDAYS ─────────────────────────────────────────────────────────
assert isinstance(NYSE_HOLIDAYS, set)
assert len(NYSE_HOLIDAYS) >= 20
assert date(2024, 1, 1) in NYSE_HOLIDAYS
assert date(2024, 12, 25) in NYSE_HOLIDAYS
assert date(2025, 1, 1) in NYSE_HOLIDAYS
print(f"[OK] NYSE_HOLIDAYS: {len(NYSE_HOLIDAYS)} dates verified")

# ── 3. Bid/ask spread — high vol → wider spread than low vol ─────────────────
ms = MarketSimulator(slippage_model=SlippageModel.PROPORTIONAL)
ms._price_cache["HIGH_VOL"] = 100.0
ms._price_cache["LOW_VOL"]  = 100.0

# High-vol stock: realized vol = 0.05 (5%) → spread_bps = max(5, 5.0) = 5
# Low-vol stock:  realized vol = 0.01 (1%) → spread_bps = max(5, 1.0) = 5 (floored)
# Use vol=0.10 vs vol=0.005 to see differentiation clearly
ms.cache_realized_vol("HIGH_VOL", 0.10)   # 10% daily vol → 10 bps spread
ms.cache_realized_vol("LOW_VOL",  0.005)  # 0.5% daily vol → 5 bps (floor)

bid_hi, ask_hi = ms.get_bid_ask("HIGH_VOL", realized_vol=0.10)
bid_lo, ask_lo = ms.get_bid_ask("LOW_VOL",  realized_vol=0.005)

spread_hi_bps = (ask_hi - bid_hi) / 100.0 * 10_000
spread_lo_bps = (ask_lo - bid_lo) / 100.0 * 10_000

assert spread_hi_bps > spread_lo_bps, (
    f"High-vol spread ({spread_hi_bps:.2f} bps) must exceed low-vol ({spread_lo_bps:.2f} bps)"
)
# Verify formula: spread_bps = max(5, vol * 100); at vol=0.10 → 10 bps
assert abs(spread_hi_bps - 10.0) < 0.01, f"Expected 10 bps, got {spread_hi_bps:.4f}"
assert abs(spread_lo_bps - 5.0)  < 0.01, f"Expected 5 bps (floor), got {spread_lo_bps:.4f}"
print(f"[OK] Bid/ask spread: HIGH_VOL={spread_hi_bps:.1f} bps > LOW_VOL={spread_lo_bps:.1f} bps (floor=5)")

# ── 4. Market impact — larger order has bigger impact (monotone) ──────────────
ms2 = MarketSimulator(slippage_model=SlippageModel.MARKET_IMPACT)
ms2._price_cache["AAPL"] = 200.0
ms2._adv_cache["AAPL"]   = 1_000_000  # 1M shares avg daily volume

# impact = 0.1 * sqrt(order_size / avg_volume) — price fraction
small_order = Order(ticker="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1_000)
large_order = Order(ticker="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100_000)

fill_small = ms2.simulate_slippage(small_order, 200.0)
fill_large = ms2.simulate_slippage(large_order, 200.0)

impact_small = fill_small - 200.0
impact_large = fill_large - 200.0

assert impact_large > impact_small > 0, (
    f"Large order impact ({impact_large:.4f}) must exceed small ({impact_small:.4f}), both positive"
)
# Verify formula: 0.1 * sqrt(1000/1000000) * 200 = 0.1 * 0.03162 * 200 = 0.6325
expected_small = 0.1 * math.sqrt(1_000 / 1_000_000) * 200.0
assert abs(impact_small - expected_small) < 0.001, (
    f"Small order impact mismatch: {impact_small:.6f} vs {expected_small:.6f}"
)
# Verify monotonicity: 0.1 * sqrt(100000/1000000) * 200 = 0.1 * 0.31623 * 200 = 6.3246
expected_large = 0.1 * math.sqrt(100_000 / 1_000_000) * 200.0
assert abs(impact_large - expected_large) < 0.001, (
    f"Large order impact mismatch: {impact_large:.6f} vs {expected_large:.6f}"
)
print(f"[OK] Market impact (Almgren-Chriss sqrt): small={impact_small:.4f}, large={impact_large:.4f} (monotone)")

# ── 5. Order lifecycle transitions ──────────────────────────────────────────
order = Order(ticker="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
assert order.status == OrderStatus.PENDING
assert order.submitted_at is None
assert order.partially_filled_at is None
assert order.filled_at is None

order.transition(OrderStatus.SUBMITTED)
assert order.status == OrderStatus.SUBMITTED
assert order.submitted_at is not None
submitted_ts = order.submitted_at

order.transition(OrderStatus.PARTIALLY_FILLED)
assert order.status == OrderStatus.PARTIALLY_FILLED
assert order.partially_filled_at is not None

order.transition(OrderStatus.FILLED)
assert order.status == OrderStatus.FILLED
assert order.filled_at is not None

# Timestamps are monotone (submitted <= partially_filled <= filled)
assert order.submitted_at <= order.partially_filled_at, "submitted_at must be <= partially_filled_at"
assert order.partially_filled_at <= order.filled_at, "partially_filled_at must be <= filled_at"

# Idempotency: calling transition again does NOT overwrite existing timestamps
order.transition(OrderStatus.SUBMITTED)
assert order.submitted_at == submitted_ts, "submitted_at must be idempotent"

print("[OK] Order lifecycle: PENDING->SUBMITTED->PARTIALLY_FILLED->FILLED with timestamps, idempotent")

# ── 6. P&L attribution: realized + unrealized + costs = total P&L ───────────
port = Portfolio(initial_cash=100_000.0)
# Simulate a position: bought 100 AAPL at $150, now at $160 → $1000 unrealized
port.positions["AAPL"] = Position("AAPL", quantity=100.0, avg_cost=150.0, current_price=160.0)
port.cash = 85_000.0  # 100_000 - 100*150 = 85_000

# Create trades representing the buy and a prior closed position
trades = [
    Trade(
        trade_id="t1", order_id="o1", ticker="AAPL",
        side=OrderSide.BUY, quantity=100, price=150.0,
        commission=1.0, timestamp=datetime.now(timezone.utc),
        realized_pnl=0.0, slippage_cost=5.0, market_impact_cost=3.0,
    ),
    Trade(
        trade_id="t2", order_id="o2", ticker="MSFT",
        side=OrderSide.SELL, quantity=50, price=300.0,
        commission=1.0, timestamp=datetime.now(timezone.utc),
        realized_pnl=500.0, slippage_cost=2.0, market_impact_cost=1.0,
    ),
]

attribution = PerformanceAnalytics.compute_pnl_attribution(trades, port)

# AAPL: realized=0, unrealized=1000, commission=1, slippage=5, impact=3
aapl = attribution["AAPL"]
assert isinstance(aapl, PnLAttribution)
assert abs(aapl.realized_pnl - 0.0) < 1e-9
assert abs(aapl.unrealized_pnl - 1000.0) < 1e-9  # 100 * (160 - 150)
assert abs(aapl.commission_total - 1.0) < 1e-9
assert abs(aapl.slippage_total - 5.0) < 1e-9
assert abs(aapl.market_impact_total - 3.0) < 1e-9

# gross_pnl = realized + unrealized
assert abs(aapl.gross_pnl - 1000.0) < 1e-9, f"gross_pnl: {aapl.gross_pnl}"

# total_costs = commission + slippage + impact
assert abs(aapl.total_costs - 9.0) < 1e-9, f"total_costs: {aapl.total_costs}"

# net_pnl = gross_pnl - total_costs
assert abs(aapl.net_pnl - (1000.0 - 9.0)) < 1e-9, f"net_pnl: {aapl.net_pnl}"

# MSFT: realized=500, unrealized=0 (no open position), commission=1, slip=2, impact=1
msft = attribution["MSFT"]
assert abs(msft.realized_pnl - 500.0) < 1e-9
assert abs(msft.unrealized_pnl - 0.0) < 1e-9
assert abs(msft.net_pnl - (500.0 - 4.0)) < 1e-9, f"MSFT net_pnl: {msft.net_pnl}"

print(f"[OK] P&L attribution AAPL: gross={aapl.gross_pnl:.0f}, costs={aapl.total_costs:.0f}, net={aapl.net_pnl:.0f}")
print(f"[OK] P&L attribution MSFT: realized={msft.realized_pnl:.0f}, net={msft.net_pnl:.0f}")

# ── 7. Position pure math (regression) ───────────────────────────────────────
pos = Position(ticker="AAPL", quantity=100.0, avg_cost=180.0, current_price=185.0)
assert abs(pos.market_value - 18500.0) < 0.01
assert abs(pos.unrealized_pnl - 500.0) < 0.01
pos.add_shares(50, 190.0)
expected_avg = (100*180 + 50*190) / 150
assert abs(pos.avg_cost - expected_avg) < 1e-6
pos.current_price = 200.0
realized = pos.reduce_shares(50.0)
expected_realized = (200.0 - expected_avg) * 50.0
assert abs(realized - expected_realized) < 0.01
print(f"[OK] Position pure math: avg_cost={expected_avg:.4f}, realized={realized:.4f}")

# ── 8. PerformanceAnalytics tearsheet (regression) ───────────────────────────
dates = pd.date_range("2023-01-01", periods=252, freq="B")
rets = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
equity = pd.Series(100_000 * (1 + rets).cumprod(), index=dates)
tearsheet = PerformanceAnalytics.compute_tearsheet(equity, [])
assert isinstance(tearsheet, Tearsheet)
assert tearsheet.max_drawdown <= 0.0
assert isinstance(tearsheet.avg_hold_days, float)
print(f"[OK] Tearsheet: sharpe={tearsheet.sharpe:.4f}, max_dd={tearsheet.max_drawdown:.4f}, avg_hold_days={tearsheet.avg_hold_days:.2f}")

print("\n[PASS] dim_066: Paper trading v3 — bid/ask spread, market impact, order lifecycle, P&L attribution verified")
PYEOF
