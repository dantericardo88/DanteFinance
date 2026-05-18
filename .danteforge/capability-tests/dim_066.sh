#!/bin/bash
# dim_066: Paper trading v3 — Portfolio, PaperBroker, PerformanceAnalytics pure math
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime, timezone, date

from sentinel.sbx.paper_trading_v3 import (
    OrderType,
    OrderSide,
    OrderStatus,
    SlippageModel,
    Order,
    Trade,
    Position,
    Portfolio,
    Tearsheet,
    RiskCheck,
    NYSE_HOLIDAYS,
    PaperBroker,
    RiskManager,
    PerformanceAnalytics,
    PaperTradingSession,
)

# 1. Enums
assert OrderType.MARKET == "MARKET"
assert OrderType.LIMIT == "LIMIT"
assert OrderSide.BUY == "BUY"
assert OrderSide.SELL == "SELL"
assert OrderStatus.PENDING == "PENDING"
assert OrderStatus.FILLED == "FILLED"
assert SlippageModel.NONE == "none"
assert SlippageModel.MARKET_IMPACT == "market_impact"
print(f"[OK] Enums: OrderType, OrderSide, OrderStatus, SlippageModel all correct")

# 2. NYSE_HOLIDAYS set
assert isinstance(NYSE_HOLIDAYS, set)
assert len(NYSE_HOLIDAYS) >= 20
assert date(2024, 1, 1) in NYSE_HOLIDAYS  # New Year's 2024
assert date(2024, 12, 25) in NYSE_HOLIDAYS  # Christmas 2024
assert date(2025, 1, 1) in NYSE_HOLIDAYS  # New Year's 2025
print(f"[OK] NYSE_HOLIDAYS: {len(NYSE_HOLIDAYS)} dates, 2024/2025 holidays verified")

# 3. Position dataclass + pure computed properties
pos = Position(ticker="AAPL", quantity=100.0, avg_cost=180.0, current_price=185.0)
assert abs(pos.market_value - 18500.0) < 0.01
assert abs(pos.unrealized_pnl - 500.0) < 0.01
assert abs(pos.pnl_pct - (500.0 / 18000.0)) < 1e-6
print(f"[OK] Position: market_value={pos.market_value:.2f}, unrealized_pnl={pos.unrealized_pnl:.2f}, pnl_pct={pos.pnl_pct:.4f}")

# add_shares (averaging)
pos.add_shares(50, 190.0)
assert abs(pos.quantity - 150.0) < 1e-9
expected_avg = (100*180 + 50*190) / 150
assert abs(pos.avg_cost - expected_avg) < 1e-6
print(f"[OK] Position.add_shares(): avg_cost after average-up={pos.avg_cost:.4f}")

# reduce_shares (realized PnL)
pos.current_price = 200.0
realized = pos.reduce_shares(50.0)
assert abs(pos.quantity - 100.0) < 1e-9
expected_realized = (200.0 - expected_avg) * 50.0
assert abs(realized - expected_realized) < 0.01
print(f"[OK] Position.reduce_shares(): realized_pnl={realized:.4f}")

# 4. Portfolio pure computation
port = Portfolio(initial_cash=100_000.0)
assert port.cash == 100_000.0
assert port.get_equity() == 100_000.0
assert len(port.get_all_positions()) == 0

# Add mock positions via internal dict
port.positions["AAPL"] = Position("AAPL", 100.0, 180.0, 185.0)
port.positions["MSFT"] = Position("MSFT", 50.0, 340.0, 350.0)
port.cash = 50_000.0

equity = port.get_equity()
expected = 50_000.0 + 100*185.0 + 50*350.0
assert abs(equity - expected) < 0.01
print(f"[OK] Portfolio.get_equity() = {equity:,.2f}")

weights = port.get_portfolio_weights()
assert "AAPL" in weights
assert "MSFT" in weights
# weights are fractions of equity (cash excluded) but sum can be less than 1
for w in weights.values():
    assert 0.0 <= w <= 1.0
print(f"[OK] Portfolio.get_portfolio_weights(): {weights}")

# 5. Portfolio compute_drawdown (returns 0-1 drawdown as positive fraction)
port2 = Portfolio(initial_cash=100_000.0)
port2._equity_history = [100_000, 105_000, 102_000, 98_000, 101_000]
port2._peak_equity = 105_000.0
port2.cash = 98_000.0  # current equity = 98k
dd = port2.compute_drawdown()
assert 0.0 <= dd <= 1.0, f"Drawdown must be in [0,1], got {dd}"
print(f"[OK] Portfolio.compute_drawdown() = {dd:.4f} (positive fraction)")

# 6. PerformanceAnalytics.compute_tearsheet
dates = pd.date_range("2023-01-01", periods=252, freq="B")
rets = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
equity = pd.Series(100_000 * (1 + rets).cumprod(), index=dates)

tearsheet = PerformanceAnalytics.compute_tearsheet(equity, [])
assert isinstance(tearsheet, Tearsheet)
assert isinstance(tearsheet.sharpe, float)
assert isinstance(tearsheet.max_drawdown, float)
assert tearsheet.max_drawdown <= 0.0
assert isinstance(tearsheet.cagr, float)
assert isinstance(tearsheet.volatility_annual, float)
print(f"[OK] PerformanceAnalytics.compute_tearsheet(): sharpe={tearsheet.sharpe:.4f}, max_dd={tearsheet.max_drawdown:.4f}, cagr={tearsheet.cagr:.4f}")

# 7. RiskManager structure (uses drawdown_halt not max_drawdown_halt)
rm = RiskManager(max_position_pct=0.20, max_sector_pct=0.40, drawdown_halt=0.15)
assert rm.max_position_pct == 0.20
assert rm.drawdown_halt == 0.15
assert hasattr(rm, 'check_position_limits')
assert hasattr(rm, 'check_drawdown_stop')
print(f"[OK] RiskManager: max_pos={rm.max_position_pct}, drawdown_halt={rm.drawdown_halt}")

# 8. PaperBroker structure (requires portfolio + market_sim)
from sentinel.sbx.paper_trading_v3 import MarketSimulator
ms = MarketSimulator(slippage_model=SlippageModel.NONE)
p = Portfolio(initial_cash=50_000.0)
broker = PaperBroker(portfolio=p, market_sim=ms, commission_per_share=0.005)
assert hasattr(broker, 'submit_order')
assert hasattr(broker, 'process_bar')
assert hasattr(broker, 'get_open_orders')
assert broker.portfolio is p
print(f"[OK] PaperBroker: has submit_order, process_bar, get_open_orders, portfolio attached")

# 9. PaperTradingSession structure (requires strategy_fn + tickers)
def dummy_strategy(prices, portfolio): return {}
session = PaperTradingSession(
    strategy_fn=dummy_strategy,
    tickers=["AAPL", "MSFT"],
    initial_cash=100_000.0,
)
assert hasattr(session, 'portfolio')
assert hasattr(session, 'broker')
assert hasattr(session, 'run_backtest')
assert session.initial_cash == 100_000.0
assert session.tickers == ["AAPL", "MSFT"]
print(f"[OK] PaperTradingSession: initial_cash={session.initial_cash:,.0f}, tickers={session.tickers}, has run_backtest()")

print("\n[PASS] dim_066: Paper trading v3 -- Portfolio, analytics, pure math verified")
PYEOF
