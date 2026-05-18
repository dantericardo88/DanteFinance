#!/bin/bash
# dim_062: Event-driven backtest v3 — EventBus, dataclasses, BacktestEngine structure
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime

from sentinel.sbx.event_driven_backtest_v3 import (
    EventBus,
    BarEvent,
    TradeEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PortfolioUpdateEvent,
    Position,
    Trade,
    BacktestResult,
    OptimizationResult,
    MovingAverageCrossStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    BacktestEngine,
    BacktestAnalytics,
    RiskManager,
)

# 1. EventBus basic operations
bus = EventBus(replay_mode=True)
ts = datetime(2024, 1, 15, 9, 30)

bar = BarEvent(timestamp=ts, symbol="AAPL", open=180.0, high=182.0, low=179.0, close=181.5, volume=1_000_000)
bar._priority = 0
bus.publish(bar)
assert bus._event_count == 1
print(f"[OK] EventBus.publish() BarEvent -> event_count={bus._event_count}")

signal = SignalEvent(timestamp=ts, symbol="AAPL", signal_type="MA_CROSS", direction="LONG", strength=1.0)
signal._priority = 1
bus.publish(signal)
assert bus._event_count == 2
print(f"[OK] EventBus.publish() SignalEvent -> event_count={bus._event_count}")

# 2. EventBus subscribe and dispatch
received_events = []
bus.subscribe(BarEvent, lambda e: received_events.append(e))
bus.subscribe(SignalEvent, lambda e: received_events.append(e))

processed = bus.process_next()
assert processed is True
assert len(received_events) == 1
assert isinstance(received_events[0], BarEvent)
print(f"[OK] EventBus priority dispatch: BarEvent (priority=0) processed first")

processed2 = bus.process_next()
assert processed2 is True
assert len(received_events) == 2
assert isinstance(received_events[1], SignalEvent)
print(f"[OK] EventBus priority dispatch: SignalEvent (priority=1) processed second")

processed3 = bus.process_next()
assert processed3 is False
print(f"[OK] EventBus.process_next() returns False when empty")

# 3. Event dataclass fields
order = OrderEvent(timestamp=ts, symbol="MSFT", order_type="MARKET", direction="BUY", quantity=100.0, price=340.0)
assert order.symbol == "MSFT"
assert order.order_type == "MARKET"
assert order.quantity == 100.0
assert len(order.order_id) > 0
print(f"[OK] OrderEvent dataclass: symbol={order.symbol}, qty={order.quantity}, id={order.order_id[:8]}...")

fill = FillEvent(timestamp=ts, symbol="MSFT", direction="BUY", quantity=100.0, fill_price=340.05, commission=1.0, order_id=order.order_id)
assert fill.fill_price == 340.05
assert fill.commission == 1.0
print(f"[OK] FillEvent dataclass: fill_price={fill.fill_price}, commission={fill.commission}")

# 4. Position dataclass
pos = Position(symbol="AAPL", quantity=100.0, avg_cost=180.0, last_price=181.5)
assert pos.direction == "LONG"
assert abs(pos.market_value - 18150.0) < 0.01
print(f"[OK] Position: direction={pos.direction}, market_value={pos.market_value:.2f}")

pos_short = Position(symbol="SPY", quantity=-50.0, avg_cost=450.0, last_price=448.0)
assert pos_short.direction == "SHORT"
print(f"[OK] Position: SHORT direction detected correctly")

pos_flat = Position(symbol="SPY", quantity=0.0, avg_cost=450.0, last_price=448.0)
assert pos_flat.direction == "FLAT"
print(f"[OK] Position: FLAT direction detected correctly")

# 5. MovingAverageCrossStrategy
mac = MovingAverageCrossStrategy(fast_period=5, slow_period=20)
assert mac.fast_period == 5
assert mac.slow_period == 20
assert mac.name == "MovingAverageCross"
print(f"[OK] MovingAverageCrossStrategy: fast={mac.fast_period}, slow={mac.slow_period}")

# 6. MeanReversionStrategy (uses period/std_dev/exit_threshold)
mrs = MeanReversionStrategy(period=20, std_dev=2.0, exit_threshold=0.5)
assert mrs.period == 20
assert mrs.name == "MeanReversion"
print(f"[OK] MeanReversionStrategy: period={mrs.period}")

# 7. MomentumStrategy (uses lookback_days/skip_days)
mom = MomentumStrategy(lookback_days=126, skip_days=21)
assert mom.lookback_days == 126
assert mom.name == "Momentum"
print(f"[OK] MomentumStrategy: lookback_days={mom.lookback_days}")

# 8. BacktestEngine initialization
strategy = MovingAverageCrossStrategy(fast_period=10, slow_period=30)
engine = BacktestEngine(
    strategy=strategy,
    start="2022-01-01",
    end="2023-01-01",
    symbols=["AAPL", "MSFT"],
    initial_cash=100_000.0,
)
assert engine.initial_cash == 100_000.0
assert engine.symbols == ["AAPL", "MSFT"]
assert engine.start == "2022-01-01"
print(f"[OK] BacktestEngine: cash={engine.initial_cash:,.0f}, symbols={engine.symbols}")

# 9. RiskManager initialization (uses max_position_pct/max_drawdown_halt)
rm = RiskManager(max_position_pct=0.10, max_drawdown_halt=0.20, vol_target=0.10)
assert rm.max_position_pct == 0.10
assert rm.max_drawdown_halt == 0.20
print(f"[OK] RiskManager: max_pos_pct={rm.max_position_pct}, max_dd_halt={rm.max_drawdown_halt}")

# 10. BacktestAnalytics has expected methods
assert hasattr(BacktestAnalytics, 'compute_metrics')
assert hasattr(BacktestAnalytics, 'generate_tearsheet')
assert callable(BacktestAnalytics.generate_tearsheet)
print(f"[OK] BacktestAnalytics has compute_metrics and generate_tearsheet")

print("\n[PASS] dim_062: Event-driven backtest v3 -- EventBus, strategies, analytics verified")
PYEOF
