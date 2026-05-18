#!/usr/bin/env bash
# dim_106: CCXT multi-exchange — registry, models, cache, arbitrage math
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import time

from sentinel.sfe.ccxt_multi_exchange import (
    CCXTExchangeRegistry,
    OHLCVBar,
    ExchangeInfo,
    ArbitrageOpportunity,
    ConsolidatedOrderBook,
    OrderBookLevel,
    _TF_MAP_BINANCE,
    _CACHE_TTL,
    _BINANCE_REST,
    _COINBASE_REST,
    _KRAKEN_REST,
    _cache_get,
    _cache_set,
    _mem_cache,
    UnifiedOHLCVCollector,
    OrderBookAggregator,
)
import pandas as pd
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Test CCXTExchangeRegistry
# ---------------------------------------------------------------------------
registry = CCXTExchangeRegistry()

exchanges = registry.list_exchanges()
assert len(exchanges) >= 20, f"Expected >= 20 exchanges: {len(exchanges)}"
print(f"[OK] CCXTExchangeRegistry: {len(exchanges)} exchanges")

for name in ["binance", "coinbase", "kraken", "bybit", "okx", "gemini", "bitstamp"]:
    assert name in exchanges, f"Exchange {name!r} should be registered"
print(f"[OK] Required exchanges present: binance/coinbase/kraken/bybit/okx/gemini/bitstamp")

required_keys = {"has_spot", "has_futures", "maker_fee", "taker_fee",
                 "min_order_size_usd", "supported_fiats", "countries_blocked", "rest_url"}
for exch_id, data in CCXTExchangeRegistry.SUPPORTED_EXCHANGES.items():
    for k in required_keys:
        assert k in data, f"Exchange {exch_id!r} missing key {k!r}"
    assert isinstance(data["maker_fee"], float), f"{exch_id} maker_fee should be float"
    assert isinstance(data["taker_fee"], float), f"{exch_id} taker_fee should be float"
    assert data["min_order_size_usd"] >= 0, f"{exch_id} min_order_size_usd should be >= 0"
    assert isinstance(data["supported_fiats"], list), f"{exch_id} supported_fiats should be list"
    assert isinstance(data["rest_url"], str) and data["rest_url"].startswith("http"), \
        f"{exch_id} rest_url should be valid URL: {data['rest_url']}"
print(f"[OK] All {len(exchanges)} exchanges have required keys with valid types")

# Test get_info() returns ExchangeInfo model
info = registry.get_info("binance")
assert isinstance(info, ExchangeInfo), f"get_info should return ExchangeInfo: {type(info)}"
assert info.exchange_id == "binance"
assert info.has_spot is True
assert info.has_futures is True
assert info.maker_fee == 0.001
assert info.taker_fee == 0.001
assert "USD" in info.supported_fiats
print(f"[OK] get_info('binance'): id={info.exchange_id} spot={info.has_spot} futures={info.has_futures}")

info_cb = registry.get_info("coinbase")
assert info_cb.has_spot is True
assert info_cb.has_futures is False
print(f"[OK] get_info('coinbase'): spot={info_cb.has_spot} futures={info_cb.has_futures}")

info_db = registry.get_info("deribit")
assert info_db.has_options is True
assert info_db.has_spot is False
print(f"[OK] get_info('deribit'): options={info_db.has_options} spot={info_db.has_spot}")

try:
    registry.get_info("nonexistent_exchange_xyz")
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print(f"[OK] get_info(unknown) raises ValueError")

futures_exchanges = registry.filter_by_capability(has_futures=True)
assert "binance" in futures_exchanges, "Binance should be in futures exchanges"
assert "coinbase" not in futures_exchanges, "Coinbase should not be in futures exchanges"
print(f"[OK] filter_by_capability(has_futures=True): {len(futures_exchanges)} exchanges")

# ---------------------------------------------------------------------------
# Test URL constants
# ---------------------------------------------------------------------------
assert _BINANCE_REST.startswith("https://api.binance.com"), f"Binance URL: {_BINANCE_REST}"
assert _COINBASE_REST.startswith("https://"), f"Coinbase URL: {_COINBASE_REST}"
assert _KRAKEN_REST.startswith("https://api.kraken.com"), f"Kraken URL: {_KRAKEN_REST}"
print(f"[OK] REST URL constants valid")

# ---------------------------------------------------------------------------
# Test _TF_MAP_BINANCE
# ---------------------------------------------------------------------------
assert len(_TF_MAP_BINANCE) >= 8, f"Expected >= 8 timeframes: {len(_TF_MAP_BINANCE)}"
for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
    assert tf in _TF_MAP_BINANCE, f"Timeframe {tf!r} should be in _TF_MAP_BINANCE"
assert _TF_MAP_BINANCE["1m"] == "1m"
assert _TF_MAP_BINANCE["1d"] == "1d"
print(f"[OK] _TF_MAP_BINANCE: {len(_TF_MAP_BINANCE)} timeframes")

# ---------------------------------------------------------------------------
# Test _CACHE_TTL
# ---------------------------------------------------------------------------
assert _CACHE_TTL > 0, f"Cache TTL should be positive: {_CACHE_TTL}"
assert _CACHE_TTL <= 300, f"Cache TTL should be <= 300s: {_CACHE_TTL}"
print(f"[OK] _CACHE_TTL = {_CACHE_TTL}s")

# ---------------------------------------------------------------------------
# Test _cache_get / _cache_set
# ---------------------------------------------------------------------------
_cache_set("test_key_106", {"price": 65000.0, "exchange": "binance"})
result = _cache_get("test_key_106")
assert result is not None, "Cached value should be retrievable"
assert result["price"] == 65000.0
assert result["exchange"] == "binance"
print(f"[OK] _cache_set/_cache_get round-trip: price={result['price']}")

missing = _cache_get("nonexistent_key_xyz_106")
assert missing is None, f"Missing cache key should return None: {missing}"
print(f"[OK] _cache_get(missing) = None")

_cache_set("key_a_106", 100)
_cache_set("key_b_106", 200)
assert _cache_get("key_a_106") == 100
assert _cache_get("key_b_106") == 200
print(f"[OK] Multiple cache keys are independent")

# ---------------------------------------------------------------------------
# Test OHLCVBar Pydantic model
# ---------------------------------------------------------------------------
bar = OHLCVBar(
    ts=1700000000000,
    open=65000.0,
    high=65500.0,
    low=64800.0,
    close=65200.0,
    volume=1250.5,
    exchange="binance",
)
assert bar.ts == 1700000000000
assert bar.high >= bar.open
assert bar.high >= bar.close
assert bar.low <= bar.open
assert bar.low <= bar.close
assert bar.volume > 0
bar_no_exch = OHLCVBar(ts=1700000001000, open=100.0, high=105.0, low=99.0, close=103.0, volume=500.0)
assert bar_no_exch.exchange == ""
print(f"[OK] OHLCVBar: O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume} exch={bar.exchange}")

# ---------------------------------------------------------------------------
# Test ExchangeInfo Pydantic model
# ---------------------------------------------------------------------------
exch_info = ExchangeInfo(
    exchange_id="test_exchange",
    has_spot=True,
    has_futures=True,
    has_options=False,
    has_margin=True,
    maker_fee=0.001,
    taker_fee=0.002,
    min_order_size_usd=5.0,
    supported_fiats=["USD", "EUR"],
    countries_blocked=["US"],
    rest_url="https://api.test.com",
)
assert exch_info.taker_fee > exch_info.maker_fee
assert "USD" in exch_info.supported_fiats
assert "US" in exch_info.countries_blocked
print(f"[OK] ExchangeInfo model: id={exch_info.exchange_id} maker={exch_info.maker_fee} taker={exch_info.taker_fee}")

# ---------------------------------------------------------------------------
# Test ArbitrageOpportunity model and gross_pct formula
# ---------------------------------------------------------------------------
buy_price = 65000.0
sell_price = 65500.0
gross_pct = (sell_price - buy_price) / buy_price * 100  # 0.7692...
fee_pct = (0.001 + 0.001) * 100  # 0.2%
net_pct = gross_pct - fee_pct

arb = ArbitrageOpportunity(
    symbol="BTC/USDT",
    buy_exchange="coinbase",
    sell_exchange="kraken",
    buy_price=buy_price,
    sell_price=sell_price,
    gross_pct=round(gross_pct, 4),
    fee_pct=round(fee_pct, 4),
    net_pct=round(net_pct, 4),
    transfer_time_min=30,
    actionable=net_pct > 0.1,
    detected_at=datetime.now(timezone.utc).isoformat(),
)
assert arb.sell_price > arb.buy_price
assert abs(arb.gross_pct - 0.7692) < 0.001, f"gross_pct ~0.769: {arb.gross_pct}"
assert arb.net_pct < arb.gross_pct
assert arb.actionable is True
assert arb.transfer_time_min == 30
print(f"[OK] ArbitrageOpportunity: gross={arb.gross_pct:.4f}% net={arb.net_pct:.4f}% actionable={arb.actionable}")

# Non-actionable arb
tiny_arb = ArbitrageOpportunity(
    symbol="ETH/USDT",
    buy_exchange="binance",
    sell_exchange="coinbase",
    buy_price=3000.0,
    sell_price=3002.0,
    gross_pct=round((3002-3000)/3000*100, 4),
    fee_pct=0.2,
    net_pct=round((3002-3000)/3000*100 - 0.2, 4),
    transfer_time_min=5,
    actionable=False,
    detected_at=datetime.now(timezone.utc).isoformat(),
)
assert tiny_arb.actionable is False
print(f"[OK] Non-actionable arb: gross={tiny_arb.gross_pct:.4f}% net={tiny_arb.net_pct:.4f}%")

# ---------------------------------------------------------------------------
# Test ConsolidatedOrderBook model
# ---------------------------------------------------------------------------
book = ConsolidatedOrderBook(
    symbol="BTC/USDT",
    timestamp=datetime.now(timezone.utc).isoformat(),
    best_bid=64990.0,
    best_ask=65010.0,
    spread=20.0,
    spread_pct=0.031,
    imbalance=0.05,
    bids=[
        OrderBookLevel(price=64990.0, size=0.5),
        OrderBookLevel(price=64980.0, size=1.2),
    ],
    asks=[
        OrderBookLevel(price=65010.0, size=0.3),
        OrderBookLevel(price=65020.0, size=0.8),
    ],
    exchange_spreads={"binance": 15.0, "coinbase": 25.0},
)
assert book.best_ask > book.best_bid
assert book.spread == book.best_ask - book.best_bid
assert -1.0 <= book.imbalance <= 1.0
assert len(book.bids) == 2
assert len(book.asks) == 2
assert book.bids[0].price > book.bids[1].price, "Bids should be descending"
assert book.asks[0].price < book.asks[1].price, "Asks should be ascending"
print(f"[OK] ConsolidatedOrderBook: bid={book.best_bid} ask={book.best_ask} spread={book.spread}")

# ---------------------------------------------------------------------------
# Test OrderBookAggregator._aggregate_levels (static, no network)
# ---------------------------------------------------------------------------
levels = [(100.0, 1.0), (100.0, 0.5), (99.5, 2.0), (99.0, 1.0)]
aggregated = OrderBookAggregator._aggregate_levels(levels)
agg_dict = dict(aggregated)
assert 100.0 in agg_dict
assert abs(agg_dict[100.0] - 1.5) < 1e-9, f"100.0 size should be 1.5 (summed): {agg_dict[100.0]}"
assert abs(agg_dict[99.5] - 2.0) < 1e-9
print(f"[OK] OrderBookAggregator._aggregate_levels: 100.0 -> {agg_dict[100.0]} (1.5 expected)")

# ---------------------------------------------------------------------------
# Test UnifiedOHLCVCollector._bars_to_df (static, no network)
# ---------------------------------------------------------------------------
raw_bars = [
    [1700000000000, 65000.0, 65500.0, 64800.0, 65200.0, 1250.5],
    [1700003600000, 65200.0, 65800.0, 65100.0, 65600.0, 980.0],
    [1700007200000, 65600.0, 66000.0, 65400.0, 65800.0, 1100.0],
]
df = UnifiedOHLCVCollector._bars_to_df(raw_bars, "binance")
assert len(df) == 3, f"Should have 3 rows: {len(df)}"
assert list(df.columns)[:6] == ["ts", "open", "high", "low", "close", "volume"]
assert df["exchange"].iloc[0] == "binance"
assert df["ts"].is_monotonic_increasing, "Bars should be sorted ascending"
print(f"[OK] _bars_to_df: {len(df)} rows sorted, exchange='binance'")

empty_df = UnifiedOHLCVCollector._bars_to_df([], "binance")
assert empty_df.empty
assert "ts" in empty_df.columns
print(f"[OK] _bars_to_df(empty) returns empty DataFrame with columns")

# ---------------------------------------------------------------------------
# Test VWAP aggregation logic (pure pandas)
# ---------------------------------------------------------------------------
collector = UnifiedOHLCVCollector()
ts1, ts2 = 1700000000000, 1700003600000
df_binance = pd.DataFrame({
    "ts": [ts1, ts2],
    "open": [65000.0, 65200.0], "high": [65500.0, 65800.0],
    "low": [64800.0, 65100.0], "close": [65200.0, 65600.0],
    "volume": [1000.0, 800.0], "exchange": ["binance", "binance"],
})
df_coinbase = pd.DataFrame({
    "ts": [ts1, ts2],
    "open": [65010.0, 65210.0], "high": [65510.0, 65810.0],
    "low": [64810.0, 65110.0], "close": [65210.0, 65610.0],
    "volume": [500.0, 300.0], "exchange": ["coinbase", "coinbase"],
})
combined = pd.concat([df_binance, df_coinbase], ignore_index=True)
vwap_df = collector._compute_vwap_aggregation(combined)
assert len(vwap_df) == 2, f"VWAP should have 2 rows: {len(vwap_df)}"
assert abs(vwap_df.loc[vwap_df["ts"]==ts1, "volume"].iloc[0] - 1500.0) < 1e-6
expected_vwap = (65200.0 * 1000.0 + 65210.0 * 500.0) / 1500.0
assert abs(vwap_df.loc[vwap_df["ts"]==ts1, "close"].iloc[0] - expected_vwap) < 0.01, \
    f"VWAP close should be {expected_vwap:.2f}"
assert vwap_df.loc[vwap_df["ts"]==ts1, "n_exchanges"].iloc[0] == 2
print(f"[OK] VWAP aggregation: vol=1500 vwap_close={expected_vwap:.2f} n_exchanges=2")

empty_result = collector._compute_vwap_aggregation(pd.DataFrame())
assert empty_result.empty
print(f"[OK] _compute_vwap_aggregation(empty) returns empty")

# ---------------------------------------------------------------------------
# Test VALID_TIMEFRAMES
# ---------------------------------------------------------------------------
valid_tfs = UnifiedOHLCVCollector.VALID_TIMEFRAMES
for tf in ["1m", "1h", "1d", "1w"]:
    assert tf in valid_tfs
assert len(valid_tfs) >= 6
print(f"[OK] VALID_TIMEFRAMES: {sorted(valid_tfs)}")

try:
    UnifiedOHLCVCollector().fetch_ohlcv("BTC/USDT", "binance", timeframe="bad_tf")
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print(f"[OK] Invalid timeframe raises ValueError")

print("\n[PASS] dim_106: CCXT multi-exchange")
PYEOF
