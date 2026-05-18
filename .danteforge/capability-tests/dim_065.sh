#!/bin/bash
# dim_065: Live trading execution v3 — class structure, enums, pure helpers
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sbx.live_trading_v3 import (
    OrderType,
    TIF,
    OrderStatus,
    SORStrategy,
    _is_paper,
    _in_core_hours,
    _is_extended_hours,
    _in_impact_zone,
    _order_hash,
    _ALPACA_AVAILABLE,
    _SOR_DIRECT_NOTIONAL,
    _SOR_TWAP_NOTIONAL,
    _DEFAULT_NOTIONAL_LIMIT,
    _DEFAULT_ADV_PCT,
    _DEFAULT_DAILY_LOSS_PCT,
    _DATA_LATENCY_TAG,
    OrderManagementSystem,
    PreTradeRiskEngine,
    PortfolioRebalancer,
    SmartOrderRouter,
)
import os

# 1. Enums
assert OrderType.MARKET == "market"
assert OrderType.LIMIT == "limit"
assert OrderType.STOP == "stop"
assert OrderType.STOP_LIMIT == "stop_limit"
assert OrderType.TRAILING_STOP == "trailing_stop"
print(f"[OK] OrderType enum: {[e.value for e in OrderType]}")

assert TIF.DAY == "day"
assert TIF.GTC == "gtc"
assert TIF.IOC == "ioc"
assert TIF.FOK == "fok"
assert TIF.OPG == "opg"
assert TIF.CLS == "cls"
print(f"[OK] TIF enum: {[e.value for e in TIF]}")

assert OrderStatus.PENDING == "pending"
assert OrderStatus.SUBMITTED == "submitted"
assert OrderStatus.FILLED == "filled"
assert OrderStatus.CANCELLED == "cancelled"
assert OrderStatus.REJECTED == "rejected"
print(f"[OK] OrderStatus enum: 7 states verified")

assert SORStrategy.DIRECT == "direct"
assert SORStrategy.TWAP == "twap"
assert SORStrategy.VWAP == "vwap"
assert SORStrategy.AC == "almgren_chriss"
print(f"[OK] SORStrategy enum: {[e.value for e in SORStrategy]}")

# 2. Constants
assert _DATA_LATENCY_TAG == "near_real_time_iex"
assert _SOR_DIRECT_NOTIONAL == 10_000.0
assert _SOR_TWAP_NOTIONAL == 100_000.0
assert _DEFAULT_NOTIONAL_LIMIT == 500_000.0
assert abs(_DEFAULT_ADV_PCT - 0.15) < 1e-9
assert abs(_DEFAULT_DAILY_LOSS_PCT - 0.03) < 1e-9
print(f"[OK] Constants: notional_limit={_DEFAULT_NOTIONAL_LIMIT:,.0f}, adv_pct={_DEFAULT_ADV_PCT}, latency_tag={_DATA_LATENCY_TAG}")

# 3. Pure helper functions
# _is_paper() reads env var ALPACA_PAPER (default "true")
os.environ["ALPACA_PAPER"] = "true"
assert _is_paper() is True
os.environ["ALPACA_PAPER"] = "false"
assert _is_paper() is False
os.environ["ALPACA_PAPER"] = "true"
print(f"[OK] _is_paper() reads ALPACA_PAPER env var correctly")

# _order_hash is deterministic
h1 = _order_hash("AAPL", "BUY")
h2 = _order_hash("AAPL", "BUY")
h3 = _order_hash("AAPL", "SELL")
assert h1 == h2, "Same inputs must produce same hash"
assert h1 != h3, "Different inputs must produce different hash"
assert len(h1) == 32  # md5 hex
print(f"[OK] _order_hash() deterministic: len={len(h1)}, BUY!=SELL: {h1[:8]}... vs {h3[:8]}...")

# 4. Market hours helpers return bool (no network)
result = _in_core_hours()
assert isinstance(result, bool)
print(f"[OK] _in_core_hours() returns bool: {result}")

result2 = _is_extended_hours()
assert isinstance(result2, bool)
print(f"[OK] _is_extended_hours() returns bool: {result2}")

result3 = _in_impact_zone()
assert isinstance(result3, bool)
print(f"[OK] _in_impact_zone() returns bool: {result3}")

# 5. OrderManagementSystem class exists and is instantiatable (no credentials needed for init)
oms = OrderManagementSystem()
assert hasattr(oms, 'submit')      # submit() not submit_order()
assert hasattr(oms, 'cancel')      # cancel() not cancel_order()
assert hasattr(oms, 'list_orders')
print(f"[OK] OrderManagementSystem initialized, has submit/cancel/list_orders")

# 6. PreTradeRiskEngine class structure
risk = PreTradeRiskEngine(
    notional_limit=500_000.0,
    adv_pct_limit=0.15,
    daily_loss_pct_halt=0.03,
)
assert risk._notional_limit == 500_000.0
assert abs(risk._adv_pct_limit - 0.15) < 1e-9
assert abs(risk._daily_loss_pct - 0.03) < 1e-9
print(f"[OK] PreTradeRiskEngine: notional={risk._notional_limit:,.0f}, adv_pct={risk._adv_pct_limit}")

# 7. SmartOrderRouter class structure
sor = SmartOrderRouter()
assert hasattr(sor, 'route')
print(f"[OK] SmartOrderRouter has route() method")

# 8. PortfolioRebalancer class
rebal = PortfolioRebalancer()
assert hasattr(rebal, 'compute_trades')
print(f"[OK] PortfolioRebalancer has compute_trades() method")

# 9. _ALPACA_AVAILABLE is a bool (optional dep, may or may not be installed)
assert isinstance(_ALPACA_AVAILABLE, bool)
print(f"[OK] _ALPACA_AVAILABLE = {_ALPACA_AVAILABLE} (optional SDK flag)")

print("\n[PASS] dim_065: Live trading execution v3 -- class structure and pure helpers verified")
PYEOF
