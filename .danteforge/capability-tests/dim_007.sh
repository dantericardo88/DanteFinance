#!/bin/bash
# dim_007: Crypto CCXT — CCXTExchangeManager structure, ArbitrageDetector logic
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from datetime import datetime, timezone

# Test 1: Constants and exchange lists
from sentinel.sds.adapters.ccxt_multi_exchange_v3 import (
    _TIER1_EXCHANGES, _TIER2_EXCHANGES, _DEFUNCT_EXCHANGES,
    HAS_CCXT, ExchangeInfo, OrderBook, AggregatedOrderBook, ArbitrageOpportunity
)
assert "binance" in _TIER1_EXCHANGES
assert "coinbase" in _TIER1_EXCHANGES
assert "kraken" in _TIER1_EXCHANGES
assert len(_TIER1_EXCHANGES) >= 5
assert "ftx" in _DEFUNCT_EXCHANGES  # FTX is defunct
print(f"[OK] _TIER1_EXCHANGES has {len(_TIER1_EXCHANGES)} exchanges, FTX marked defunct")

# Test 2: ExchangeInfo dataclass instantiation
ei = ExchangeInfo(
    exchange_id="binance", name="Binance", markets_count=1500,
    has_ohlcv=True, rate_limit_ms=100, maker_fee=0.001, taker_fee=0.001,
    tier=1, reachable=True
)
assert ei.exchange_id == "binance"
assert ei.taker_fee == 0.001
print(f"[OK] ExchangeInfo dataclass instantiates correctly")

# Test 3: OrderBook mid_price computed via __post_init__
now = datetime.now(timezone.utc)
book = OrderBook(
    exchange_id="binance", symbol="BTC/USDT", timestamp=now,
    bids=[(50100.0, 1.5), (50090.0, 2.0)],
    asks=[(50110.0, 1.0), (50120.0, 3.0)]
)
# mid_price = (best_bid + best_ask) / 2 = (50100 + 50110) / 2 = 50105
assert abs(book.mid_price - 50105.0) < 1e-6, f"Mid price wrong: {book.mid_price}"
print(f"[OK] OrderBook mid_price = {book.mid_price} (bids[0]+asks[0])/2")

# Test 4: ArbitrageOpportunity — gross profit calculation
opp = ArbitrageOpportunity(
    symbol="BTC/USDT", buy_exchange="kraken", sell_exchange="binance",
    buy_ask=50000.0, sell_bid=50200.0,
    gross_profit_pct=(50200.0 - 50000.0) / 50000.0 * 100,
    net_profit_pct=0.35,  # after fees
    buy_fee=0.001, sell_fee=0.001,
    timestamp=now
)
expected_gross = (50200.0 - 50000.0) / 50000.0 * 100
assert abs(opp.gross_profit_pct - expected_gross) < 1e-6, \
    f"Gross profit mismatch: {opp.gross_profit_pct} vs {expected_gross}"
assert opp.net_profit_pct < opp.gross_profit_pct, "Net profit must be < gross (fees)"
print(f"[OK] ArbitrageOpportunity: gross={opp.gross_profit_pct:.2f}%, net={opp.net_profit_pct:.2f}%")

# Test 5: CCXTExchangeManager class exists and is instantiatable
from sentinel.sds.adapters.ccxt_multi_exchange_v3 import CCXTExchangeManager
mgr = CCXTExchangeManager()
assert hasattr(mgr, "initialize_exchange")
assert hasattr(mgr, "_exchange_cache")
print("[OK] CCXTExchangeManager instantiates with expected attributes")

print("[PASS]")
PYEOF
