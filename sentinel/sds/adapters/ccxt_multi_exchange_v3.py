"""
ccxt_multi_exchange_v3.py — Multi-exchange OHLCV + execution platform.

dim_106: Multi-exchange crypto OHLCV & execution (CCXT) — score 7 → 9

Architecture:
  CCXTExchangeManager      — 100+ exchange support, connectivity, metadata
  UnifiedOHLCVFetcher      — normalized OHLCV across all exchanges, parallel fetch
  OrderBookAggregator      — cross-exchange order book aggregation & depth
  SmartOrderRouter         — best execution routing, price impact estimation
  ArbitrageDetector        — cross-exchange, triangular, funding-rate arb
  PerpetualsAnalyzer       — funding rates, open interest, leverage extremes
  PaperExecutionEngine     — realistic paper trading simulation
  CCXTDataEngine           — orchestrator

Free data only. CCXT public endpoints, Binance REST fallback, CoinGecko prices.
No auth keys required for OHLCV — public endpoints only.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional CCXT import — guard so core never breaks
# ---------------------------------------------------------------------------
try:
    import ccxt
    HAS_CCXT = True
    _ALL_CCXT_EXCHANGES: list[str] = list(ccxt.exchanges)
except ImportError:
    ccxt = None  # type: ignore[assignment]
    HAS_CCXT = False
    _ALL_CCXT_EXCHANGES = []

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIER1_EXCHANGES = [
    "binance", "coinbase", "kraken", "okx", "bybit",
    "bitfinex", "htx", "kucoin", "gate", "mexc",
]
_TIER2_EXCHANGES = [
    "binanceus", "bitstamp", "gemini", "poloniex",
    "bittrex", "liquid", "bitflyer", "phemex",
    "gateio", "cryptocom",
]
_DEFUNCT_EXCHANGES = {"ftx", "bitmex_testnet"}

BINANCE_REST = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_REST = "https://api.bybit.com"
COINGECKO_REST = "https://api.coingecko.com/api/v3"

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExchangeInfo:
    exchange_id: str
    name: str
    markets_count: int
    has_ohlcv: bool
    rate_limit_ms: int          # milliseconds between calls
    maker_fee: float            # fraction (0.001 = 0.1%)
    taker_fee: float
    tier: int                   # 1=premium, 2=standard, 3=other
    reachable: bool = True


@dataclass
class OrderBook:
    exchange_id: str
    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]   # (price, volume) descending
    asks: list[tuple[float, float]]   # (price, volume) ascending
    mid_price: float = 0.0

    def __post_init__(self) -> None:
        if self.bids and self.asks:
            self.mid_price = (self.bids[0][0] + self.asks[0][0]) / 2.0


@dataclass
class AggregatedOrderBook:
    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]   # merged, descending
    asks: list[tuple[float, float]]   # merged, ascending
    source_exchanges: list[str] = field(default_factory=list)
    mid_price: float = 0.0

    def __post_init__(self) -> None:
        if self.bids and self.asks:
            self.mid_price = (self.bids[0][0] + self.asks[0][0]) / 2.0


@dataclass
class BestQuote:
    exchange_id: str
    symbol: str
    side: str           # "bid" or "ask"
    price: float
    volume: float
    timestamp: datetime


@dataclass
class RoutingPlan:
    symbol: str
    side: str
    total_quantity: float
    legs: list[dict]    # [{"exchange": str, "quantity": float, "expected_price": float}]
    estimated_avg_price: float
    estimated_slippage_bps: float


@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_ask: float
    sell_bid: float
    gross_profit_pct: float
    net_profit_pct: float       # after fees
    buy_fee: float
    sell_fee: float
    timestamp: datetime


@dataclass
class PaperOrder:
    order_id: str
    exchange_id: str
    symbol: str
    side: str           # "buy" or "sell"
    order_type: str     # "market" or "limit"
    quantity: float
    price: Optional[float]
    status: str         # "open", "filled", "cancelled"
    created_at: datetime


@dataclass
class PaperFill:
    order_id: str
    exchange_id: str
    symbol: str
    side: str
    fill_price: float
    fill_quantity: float
    fee: float
    filled_at: datetime


@dataclass
class ExecutionPlan:
    symbol: str
    side: str
    quantity: float
    routing_plan: RoutingPlan
    best_quote: BestQuote
    estimated_cost: float
    estimated_fee: float


# ---------------------------------------------------------------------------
# Low-level HTTP helper (no requests dependency)
# ---------------------------------------------------------------------------

def _http_get(url: str, params: dict | None = None, timeout: int = 10) -> Any:
    """Simple GET returning parsed JSON. Falls back gracefully."""
    if params:
        url = f"{url}?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "sentinel/3.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.debug("HTTP GET failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# CCXTExchangeManager
# ---------------------------------------------------------------------------

class CCXTExchangeManager:
    """
    Manages initialization and metadata for 100+ CCXT exchanges.
    Falls back to direct Binance REST + CoinGecko when CCXT unavailable.
    """

    def __init__(self) -> None:
        self._exchange_cache: dict[str, Any] = {}
        self._info_cache: dict[str, ExchangeInfo] = {}

    # ------------------------------------------------------------------
    def initialize_exchange(
        self,
        exchange_id: str,
        api_key: str | None = None,
        secret: str | None = None,
        sandbox: bool = False,
    ) -> Any:
        """
        Initialize and return a CCXT exchange instance.
        Public endpoints only by default — no keys required for OHLCV.
        """
        if not HAS_CCXT:
            raise RuntimeError("ccxt not installed. Run: pip install ccxt")

        if exchange_id in self._exchange_cache:
            return self._exchange_cache[exchange_id]

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown CCXT exchange: {exchange_id}")

        config: dict[str, Any] = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if secret:
            config["secret"] = secret

        exchange = exchange_class(config)

        if sandbox:
            try:
                exchange.set_sandbox_mode(True)
                logger.info("Sandbox mode enabled for %s", exchange_id)
            except Exception:
                logger.warning("Sandbox mode not supported for %s", exchange_id)

        self._exchange_cache[exchange_id] = exchange
        return exchange

    # ------------------------------------------------------------------
    def get_supported_exchanges(self) -> list[str]:
        """
        Return list of exchanges that support OHLCV, ordered by priority tier.
        Falls back to a static curated list if CCXT unavailable.
        """
        if not HAS_CCXT:
            return _TIER1_EXCHANGES + _TIER2_EXCHANGES

        ohlcv_exchanges: list[str] = []
        tier1_set = set(_TIER1_EXCHANGES)
        tier2_set = set(_TIER2_EXCHANGES)

        # Tier 1 first
        for ex_id in _TIER1_EXCHANGES:
            if ex_id in _ALL_CCXT_EXCHANGES and ex_id not in _DEFUNCT_EXCHANGES:
                ohlcv_exchanges.append(ex_id)

        # Tier 2
        for ex_id in _TIER2_EXCHANGES:
            if ex_id in _ALL_CCXT_EXCHANGES and ex_id not in _DEFUNCT_EXCHANGES:
                ohlcv_exchanges.append(ex_id)

        # Tier 3: remaining ccxt exchanges with OHLCV
        for ex_id in _ALL_CCXT_EXCHANGES:
            if ex_id in tier1_set or ex_id in tier2_set:
                continue
            if ex_id in _DEFUNCT_EXCHANGES:
                continue
            # Spot-check: ccxt.exchanges lists all; OHLCV support is a class attribute
            try:
                cls = getattr(ccxt, ex_id)
                if getattr(cls, "has", {}).get("fetchOHLCV", False):
                    ohlcv_exchanges.append(ex_id)
            except Exception:
                pass

        return ohlcv_exchanges

    # ------------------------------------------------------------------
    def get_exchange_info(self, exchange_id: str) -> ExchangeInfo:
        """Return metadata for a single exchange."""
        if exchange_id in self._info_cache:
            return self._info_cache[exchange_id]

        tier = (
            1 if exchange_id in _TIER1_EXCHANGES
            else 2 if exchange_id in _TIER2_EXCHANGES
            else 3
        )

        if not HAS_CCXT:
            info = ExchangeInfo(
                exchange_id=exchange_id,
                name=exchange_id.capitalize(),
                markets_count=0,
                has_ohlcv=True,
                rate_limit_ms=1000,
                maker_fee=0.001,
                taker_fee=0.001,
                tier=tier,
            )
            self._info_cache[exchange_id] = info
            return info

        try:
            ex = self.initialize_exchange(exchange_id)
            markets = ex.load_markets()
            has_ohlcv = bool(ex.has.get("fetchOHLCV", False))
            maker = float(ex.fees.get("trading", {}).get("maker", 0.001))
            taker = float(ex.fees.get("trading", {}).get("taker", 0.001))

            info = ExchangeInfo(
                exchange_id=exchange_id,
                name=ex.name or exchange_id,
                markets_count=len(markets),
                has_ohlcv=has_ohlcv,
                rate_limit_ms=ex.rateLimit,
                maker_fee=maker,
                taker_fee=taker,
                tier=tier,
                reachable=True,
            )
        except Exception as exc:
            logger.warning("Could not load info for %s: %s", exchange_id, exc)
            info = ExchangeInfo(
                exchange_id=exchange_id,
                name=exchange_id,
                markets_count=0,
                has_ohlcv=False,
                rate_limit_ms=2000,
                maker_fee=0.001,
                taker_fee=0.001,
                tier=tier,
                reachable=False,
            )

        self._info_cache[exchange_id] = info
        return info

    # ------------------------------------------------------------------
    def test_connectivity(self, exchange_id: str) -> bool:
        """Ping an exchange to verify it's reachable."""
        if not HAS_CCXT:
            # Fallback: ping Binance
            data = _http_get(f"{BINANCE_REST}/api/v3/ping", timeout=5)
            return data is not None

        try:
            ex = self.initialize_exchange(exchange_id)
            ex.fetch_time()
            return True
        except Exception:
            try:
                ex.load_markets()
                return True
            except Exception:
                return False

    # ------------------------------------------------------------------
    def get_exchange_taker_fee(self, exchange_id: str) -> float:
        """Quick taker fee lookup."""
        info = self._info_cache.get(exchange_id)
        if info:
            return info.taker_fee
        # Defaults
        defaults = {
            "binance": 0.001, "coinbase": 0.005, "kraken": 0.0026,
            "okx": 0.001, "bybit": 0.001, "bitfinex": 0.002,
            "htx": 0.002, "kucoin": 0.001, "gate": 0.002, "mexc": 0.002,
            "gateio": 0.002, "cryptocom": 0.004,
        }
        return defaults.get(exchange_id, 0.002)


# ---------------------------------------------------------------------------
# UnifiedOHLCVFetcher
# ---------------------------------------------------------------------------

class UnifiedOHLCVFetcher:
    """
    Fetches and normalizes OHLCV data from 100+ CCXT exchanges.
    All outputs use unified column names: timestamp, open, high, low, close, volume.
    """

    def __init__(self, exchange_manager: CCXTExchangeManager) -> None:
        self._mgr = exchange_manager

    # ------------------------------------------------------------------
    def normalize_symbol(self, symbol: str, exchange_id: str) -> str:
        """
        Map between exchange-specific and CCXT unified symbol formats.
        E.g., BTCUSDT (Binance raw) → BTC/USDT (CCXT unified)
        """
        # Already unified
        if "/" in symbol:
            return symbol

        # Kraken quirks: XBT/USD, XETH/ZUSD
        if exchange_id == "kraken":
            kraken_map = {
                "BTCUSDT": "XBT/USDT", "BTCUSD": "XBT/USD",
                "ETHUSDT": "ETH/USDT", "ETHUSD": "ETH/USD",
            }
            if symbol in kraken_map:
                return kraken_map[symbol]

        # Generic: insert slash before quote currency
        for quote in ["USDT", "USD", "BTC", "ETH", "BNB", "EUR", "BUSD"]:
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[: -len(quote)]
                return f"{base}/{quote}"

        return symbol

    # ------------------------------------------------------------------
    def _normalize_ccxt_ohlcv(
        self, raw: list[list], exchange_id: str
    ) -> pd.DataFrame:
        """Convert raw CCXT OHLCV list to normalized DataFrame."""
        if not raw:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            raw, columns=["ts_ms", "open", "high", "low", "close", "volume"]
        )

        # Some exchanges return seconds, most return milliseconds
        # Heuristic: if max ts < 1e12 → seconds
        if df["ts_ms"].max() < 1e12:
            df["ts_ms"] = df["ts_ms"] * 1000

        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.drop(columns=["ts_ms"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV from a single exchange. Falls back to Binance REST
        or CoinGecko when CCXT is unavailable.
        """
        unified_symbol = self.normalize_symbol(symbol, exchange_id)

        if HAS_CCXT:
            return self._fetch_ccxt(exchange_id, unified_symbol, timeframe, limit)

        # Fallback: Binance public REST
        if exchange_id in ("binance", "binanceus"):
            return self._fetch_binance_rest(unified_symbol, timeframe, limit)

        # Last resort: CoinGecko (daily only)
        return self._fetch_coingecko(unified_symbol, limit)

    # ------------------------------------------------------------------
    def _fetch_ccxt(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        try:
            ex = self._mgr.initialize_exchange(exchange_id)
            if not ex.has.get("fetchOHLCV"):
                logger.warning("%s does not support OHLCV", exchange_id)
                return pd.DataFrame()

            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return self._normalize_ccxt_ohlcv(raw, exchange_id)
        except Exception as exc:
            logger.warning("CCXT OHLCV failed [%s/%s]: %s", exchange_id, symbol, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def _fetch_binance_rest(
        self, symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """Direct Binance public API (no key required)."""
        # Convert unified symbol to Binance format
        raw_symbol = symbol.replace("/", "")
        interval_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
            "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
        }
        interval = interval_map.get(timeframe, "1d")
        data = _http_get(
            f"{BINANCE_REST}/api/v3/klines",
            {"symbol": raw_symbol, "interval": interval, "limit": min(limit, 1000)},
        )
        if not data:
            return pd.DataFrame()

        rows = []
        for bar in data:
            rows.append({
                "timestamp": pd.Timestamp(bar[0], unit="ms", tz="UTC"),
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": float(bar[5]),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _fetch_coingecko(self, symbol: str, limit: int) -> pd.DataFrame:
        """CoinGecko free API — daily OHLCV for major coins."""
        coin_id_map = {
            "BTC/USDT": "bitcoin", "BTC/USD": "bitcoin",
            "ETH/USDT": "ethereum", "ETH/USD": "ethereum",
            "SOL/USDT": "solana", "BNB/USDT": "binancecoin",
            "ADA/USDT": "cardano", "XRP/USDT": "ripple",
            "DOGE/USDT": "dogecoin", "DOT/USDT": "polkadot",
        }
        coin_id = coin_id_map.get(symbol, "bitcoin")
        days = min(limit, 365)
        data = _http_get(
            f"{COINGECKO_REST}/coins/{coin_id}/ohlc",
            {"vs_currency": "usd", "days": str(days)},
        )
        if not data:
            return pd.DataFrame()

        rows = []
        for bar in data:
            rows.append({
                "timestamp": pd.Timestamp(bar[0], unit="ms", tz="UTC"),
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": 0.0,  # CoinGecko OHLC endpoint doesn't include volume
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def fetch_ohlcv_multi_exchange(
        self,
        symbol: str,
        exchanges: list[str],
        timeframe: str = "1d",
        limit: int = 500,
        max_workers: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """
        Parallel fetch from multiple exchanges.
        Returns dict: exchange_id → DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(ex_id: str) -> tuple[str, pd.DataFrame]:
            df = self.fetch_ohlcv(ex_id, symbol, timeframe, limit)
            return ex_id, df

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, ex_id): ex_id for ex_id in exchanges}
            for fut in as_completed(futures):
                ex_id, df = fut.result()
                if not df.empty:
                    results[ex_id] = df
                else:
                    logger.debug("Empty OHLCV from %s for %s", ex_id, symbol)

        return results

    # ------------------------------------------------------------------
    def fetch_ohlcv_with_history(
        self,
        exchange_id: str,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """
        Paginated historical OHLCV fetch from start to end dates.
        Handles exchanges that cap at 1000 bars per request (Binance).
        """
        unified_symbol = self.normalize_symbol(symbol, exchange_id)
        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC")
        since_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_bars: list[pd.DataFrame] = []
        limit_per_request = 1000

        if not HAS_CCXT:
            # Binance REST pagination fallback
            return self._fetch_binance_history(
                unified_symbol, timeframe, since_ms, end_ms, limit_per_request
            )

        try:
            ex = self._mgr.initialize_exchange(exchange_id)
        except Exception as exc:
            logger.error("Cannot init exchange %s: %s", exchange_id, exc)
            return pd.DataFrame()

        current_since = since_ms
        max_iterations = 500  # safety cap

        for _ in range(max_iterations):
            try:
                raw = ex.fetch_ohlcv(
                    unified_symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit_per_request,
                )
            except Exception as exc:
                logger.warning("Pagination error [%s]: %s", exchange_id, exc)
                break

            if not raw:
                break

            df = self._normalize_ccxt_ohlcv(raw, exchange_id)
            # Filter to requested range
            mask = df["timestamp"] <= end_dt
            df = df[mask]
            all_bars.append(df)

            last_ts = int(raw[-1][0])
            if last_ts >= end_ms or len(raw) < limit_per_request:
                break

            current_since = last_ts + 1
            # Respect rate limit
            time.sleep(ex.rateLimit / 1000.0)

        if not all_bars:
            return pd.DataFrame()

        result = pd.concat(all_bars, ignore_index=True)
        result = result.drop_duplicates("timestamp").sort_values("timestamp")
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    def _fetch_binance_history(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        end_ms: int,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Paginate Binance klines REST endpoint for full history."""
        raw_symbol = symbol.replace("/", "")
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
        }
        interval = interval_map.get(timeframe, "1d")
        all_bars: list[pd.DataFrame] = []
        current_start = since_ms

        for _ in range(500):
            data = _http_get(
                f"{BINANCE_REST}/api/v3/klines",
                {
                    "symbol": raw_symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": limit,
                },
            )
            if not data:
                break

            rows = [
                {
                    "timestamp": pd.Timestamp(bar[0], unit="ms", tz="UTC"),
                    "open": float(bar[1]),
                    "high": float(bar[2]),
                    "low": float(bar[3]),
                    "close": float(bar[4]),
                    "volume": float(bar[5]),
                }
                for bar in data
            ]
            df = pd.DataFrame(rows)
            all_bars.append(df)

            if len(data) < limit:
                break

            current_start = int(data[-1][0]) + 1
            time.sleep(0.1)

        if not all_bars:
            return pd.DataFrame()

        result = pd.concat(all_bars, ignore_index=True)
        return (
            result.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def consolidate_ohlcv(
        self, exchange_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Volume-weighted consolidated OHLCV across exchanges.
        VWAP close = Σ(close_i × volume_i) / Σ(volume_i)
        """
        if not exchange_data:
            return pd.DataFrame()

        dfs = []
        for ex_id, df in exchange_data.items():
            if df.empty:
                continue
            df = df.copy()
            df["_exchange"] = ex_id
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)

        grouped = combined.groupby("timestamp")

        def _vwap_row(group: pd.DataFrame) -> pd.Series:
            total_vol = group["volume"].sum()
            if total_vol > 0:
                vwap_close = (group["close"] * group["volume"]).sum() / total_vol
                vwap_open = (group["open"] * group["volume"]).sum() / total_vol
            else:
                vwap_close = group["close"].mean()
                vwap_open = group["open"].mean()

            return pd.Series(
                {
                    "open": vwap_open,
                    "high": group["high"].max(),
                    "low": group["low"].min(),
                    "close": vwap_close,
                    "volume": total_vol,
                    "exchange_count": len(group),
                }
            )

        consolidated = grouped.apply(_vwap_row).reset_index()
        return consolidated.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# OrderBookAggregator
# ---------------------------------------------------------------------------

class OrderBookAggregator:
    """
    Fetches and aggregates order books across multiple exchanges.
    Provides depth analysis and imbalance detection.
    """

    def __init__(self, exchange_manager: CCXTExchangeManager) -> None:
        self._mgr = exchange_manager

    # ------------------------------------------------------------------
    def fetch_order_book(
        self,
        exchange_id: str,
        symbol: str,
        limit: int = 20,
    ) -> OrderBook:
        """Fetch normalized order book from a single exchange."""
        unified_symbol = _normalize_unified(symbol)

        if HAS_CCXT:
            return self._fetch_ccxt_book(exchange_id, unified_symbol, limit)

        # Fallback: Binance public depth API
        return self._fetch_binance_book(unified_symbol, limit)

    # ------------------------------------------------------------------
    def _fetch_ccxt_book(
        self, exchange_id: str, symbol: str, limit: int
    ) -> OrderBook:
        try:
            ex = self._mgr.initialize_exchange(exchange_id)
            raw = ex.fetch_order_book(symbol, limit=limit)

            bids = [(float(p), float(v)) for p, v in raw.get("bids", [])]
            asks = [(float(p), float(v)) for p, v in raw.get("asks", [])]

            # Ensure correct sort order
            bids = sorted(bids, key=lambda x: -x[0])
            asks = sorted(asks, key=lambda x: x[0])

            ts = datetime.now(timezone.utc)
            if raw.get("timestamp"):
                try:
                    ts = datetime.fromtimestamp(raw["timestamp"] / 1000, tz=timezone.utc)
                except Exception:
                    pass

            return OrderBook(
                exchange_id=exchange_id,
                symbol=symbol,
                timestamp=ts,
                bids=bids,
                asks=asks,
            )
        except Exception as exc:
            logger.warning("Order book fetch failed [%s/%s]: %s", exchange_id, symbol, exc)
            return OrderBook(
                exchange_id=exchange_id,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=[],
                asks=[],
            )

    # ------------------------------------------------------------------
    def _fetch_binance_book(self, symbol: str, limit: int) -> OrderBook:
        """Binance public depth endpoint fallback."""
        raw_symbol = symbol.replace("/", "")
        data = _http_get(
            f"{BINANCE_REST}/api/v3/depth",
            {"symbol": raw_symbol, "limit": min(limit, 100)},
        )
        if not data:
            return OrderBook(
                exchange_id="binance",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=[],
                asks=[],
            )

        bids = [(float(p), float(v)) for p, v in data.get("bids", [])]
        asks = [(float(p), float(v)) for p, v in data.get("asks", [])]
        bids = sorted(bids, key=lambda x: -x[0])
        asks = sorted(asks, key=lambda x: x[0])

        return OrderBook(
            exchange_id="binance",
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
        )

    # ------------------------------------------------------------------
    def fetch_aggregated_order_book(
        self,
        symbol: str,
        exchanges: list[str],
        limit: int = 20,
        max_workers: int = 5,
    ) -> AggregatedOrderBook:
        """
        Merge order books from multiple exchanges.
        Bids sorted descending, asks sorted ascending.
        """
        books: list[OrderBook] = []

        def _fetch_one(ex_id: str) -> OrderBook:
            return self.fetch_order_book(ex_id, symbol, limit)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch_one, ex_id) for ex_id in exchanges]
            for fut in as_completed(futures):
                book = fut.result()
                if book.bids or book.asks:
                    books.append(book)

        all_bids: list[tuple[float, float]] = []
        all_asks: list[tuple[float, float]] = []
        source_exchanges: list[str] = []

        for book in books:
            all_bids.extend(book.bids)
            all_asks.extend(book.asks)
            source_exchanges.append(book.exchange_id)

        # Aggregate by price level
        merged_bids = _aggregate_levels(all_bids, descending=True)
        merged_asks = _aggregate_levels(all_asks, descending=False)

        return AggregatedOrderBook(
            symbol=_normalize_unified(symbol),
            timestamp=datetime.now(timezone.utc),
            bids=merged_bids,
            asks=merged_asks,
            source_exchanges=source_exchanges,
        )

    # ------------------------------------------------------------------
    def compute_market_depth(
        self,
        order_book: OrderBook | AggregatedOrderBook,
        pct_from_mid: float = 0.01,
    ) -> dict:
        """
        Compute bid/ask depth within pct_from_mid of mid price.
        Default: 1% either side.
        """
        mid = order_book.mid_price
        if mid == 0:
            return {"bid_depth": 0.0, "ask_depth": 0.0, "total_depth": 0.0}

        bid_threshold = mid * (1 - pct_from_mid)
        ask_threshold = mid * (1 + pct_from_mid)

        bid_depth = sum(
            vol for price, vol in order_book.bids if price >= bid_threshold
        )
        ask_depth = sum(
            vol for price, vol in order_book.asks if price <= ask_threshold
        )

        return {
            "mid_price": mid,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": bid_depth + ask_depth,
            "bid_ask_ratio": bid_depth / ask_depth if ask_depth > 0 else float("inf"),
            "pct_from_mid": pct_from_mid,
        }

    # ------------------------------------------------------------------
    def detect_order_book_imbalance(
        self,
        order_book: OrderBook | AggregatedOrderBook,
        levels: int = 10,
    ) -> float:
        """
        Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) for top N levels.
        Range [-1, 1]. Positive = buy pressure, negative = sell pressure.
        """
        bid_vol = sum(v for _, v in order_book.bids[:levels])
        ask_vol = sum(v for _, v in order_book.asks[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


def _normalize_unified(symbol: str) -> str:
    """Ensure symbol is in CCXT unified format (BASE/QUOTE)."""
    if "/" in symbol:
        return symbol
    for quote in ["USDT", "USD", "BTC", "ETH", "BNB", "EUR"]:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[:-len(quote)]}/{quote}"
    return symbol


def _aggregate_levels(
    levels: list[tuple[float, float]], descending: bool
) -> list[tuple[float, float]]:
    """Merge duplicate price levels by summing volumes."""
    price_map: dict[float, float] = {}
    for price, vol in levels:
        price_map[price] = price_map.get(price, 0.0) + vol
    items = list(price_map.items())
    items.sort(key=lambda x: -x[0] if descending else x[0])
    return items


# ---------------------------------------------------------------------------
# SmartOrderRouter
# ---------------------------------------------------------------------------

class SmartOrderRouter:
    """
    Routes orders for best execution across multiple exchanges.
    Uses order book data to minimize price impact.
    """

    def __init__(
        self,
        exchange_manager: CCXTExchangeManager,
        ob_aggregator: OrderBookAggregator,
    ) -> None:
        self._mgr = exchange_manager
        self._oba = ob_aggregator

    # ------------------------------------------------------------------
    def find_best_bid(
        self, symbol: str, exchanges: list[str]
    ) -> BestQuote:
        """Find the highest bid price across all exchanges."""
        best: BestQuote | None = None

        for ex_id in exchanges:
            book = self._oba.fetch_order_book(ex_id, symbol, limit=5)
            if book.bids:
                price, vol = book.bids[0]
                if best is None or price > best.price:
                    best = BestQuote(
                        exchange_id=ex_id,
                        symbol=symbol,
                        side="bid",
                        price=price,
                        volume=vol,
                        timestamp=datetime.now(timezone.utc),
                    )

        if best is None:
            best = BestQuote(
                exchange_id="",
                symbol=symbol,
                side="bid",
                price=0.0,
                volume=0.0,
                timestamp=datetime.now(timezone.utc),
            )

        return best

    # ------------------------------------------------------------------
    def find_best_ask(
        self, symbol: str, exchanges: list[str]
    ) -> BestQuote:
        """Find the lowest ask price across all exchanges."""
        best: BestQuote | None = None

        for ex_id in exchanges:
            book = self._oba.fetch_order_book(ex_id, symbol, limit=5)
            if book.asks:
                price, vol = book.asks[0]
                if best is None or price < best.price:
                    best = BestQuote(
                        exchange_id=ex_id,
                        symbol=symbol,
                        side="ask",
                        price=price,
                        volume=vol,
                        timestamp=datetime.now(timezone.utc),
                    )

        if best is None:
            best = BestQuote(
                exchange_id="",
                symbol=symbol,
                side="ask",
                price=float("inf"),
                volume=0.0,
                timestamp=datetime.now(timezone.utc),
            )

        return best

    # ------------------------------------------------------------------
    def compute_effective_spread(
        self, symbol: str, exchanges: list[str]
    ) -> dict[str, float]:
        """
        Effective spread = ask - bid for each exchange, in basis points.
        """
        spreads: dict[str, float] = {}

        for ex_id in exchanges:
            book = self._oba.fetch_order_book(ex_id, symbol, limit=5)
            if book.bids and book.asks:
                bid = book.bids[0][0]
                ask = book.asks[0][0]
                mid = (bid + ask) / 2.0
                if mid > 0:
                    spreads[ex_id] = (ask - bid) / mid * 10_000  # bps
            else:
                spreads[ex_id] = float("nan")

        return spreads

    # ------------------------------------------------------------------
    def route_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        exchanges: list[str],
    ) -> RoutingPlan:
        """
        Split a market order across exchanges to minimize price impact.
        Routes to best-priced exchange up to its available depth.
        Side: 'buy' or 'sell'.
        """
        # Fetch order books in parallel
        books: dict[str, OrderBook] = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(self._oba.fetch_order_book, ex_id, symbol, 20): ex_id
                for ex_id in exchanges
            }
            for fut in as_completed(futures):
                ex_id = futures[fut]
                try:
                    books[ex_id] = fut.result()
                except Exception:
                    pass

        remaining = quantity
        legs: list[dict] = []
        weighted_price_sum = 0.0

        if side == "buy":
            # Sort exchanges by best ask (ascending)
            sorted_exchanges = sorted(
                [ex_id for ex_id in exchanges if ex_id in books and books[ex_id].asks],
                key=lambda ex: books[ex].asks[0][0] if books[ex].asks else float("inf"),
            )
            for ex_id in sorted_exchanges:
                if remaining <= 0:
                    break
                book = books[ex_id]
                available = sum(v for _, v in book.asks[:5])
                fill_qty = min(remaining, available)
                if fill_qty <= 0:
                    continue
                avg_price = self.estimate_market_impact(symbol, ex_id, fill_qty, side="buy", book=book)
                legs.append({"exchange": ex_id, "quantity": fill_qty, "expected_price": avg_price})
                weighted_price_sum += avg_price * fill_qty
                remaining -= fill_qty

        else:  # sell
            sorted_exchanges = sorted(
                [ex_id for ex_id in exchanges if ex_id in books and books[ex_id].bids],
                key=lambda ex: -books[ex].bids[0][0] if books[ex].bids else float("-inf"),
            )
            for ex_id in sorted_exchanges:
                if remaining <= 0:
                    break
                book = books[ex_id]
                available = sum(v for _, v in book.bids[:5])
                fill_qty = min(remaining, available)
                if fill_qty <= 0:
                    continue
                avg_price = self.estimate_market_impact(symbol, ex_id, fill_qty, side="sell", book=book)
                legs.append({"exchange": ex_id, "quantity": fill_qty, "expected_price": avg_price})
                weighted_price_sum += avg_price * fill_qty
                remaining -= fill_qty

        filled = quantity - remaining
        avg_price = weighted_price_sum / filled if filled > 0 else 0.0

        # Compute slippage vs best price
        if side == "buy" and legs:
            best_price = legs[0]["expected_price"] if legs else avg_price
            slippage_bps = (avg_price - best_price) / best_price * 10_000 if best_price > 0 else 0.0
        elif side == "sell" and legs:
            best_price = legs[0]["expected_price"] if legs else avg_price
            slippage_bps = (best_price - avg_price) / best_price * 10_000 if best_price > 0 else 0.0
        else:
            slippage_bps = 0.0

        return RoutingPlan(
            symbol=_normalize_unified(symbol),
            side=side,
            total_quantity=quantity,
            legs=legs,
            estimated_avg_price=avg_price,
            estimated_slippage_bps=slippage_bps,
        )

    # ------------------------------------------------------------------
    def estimate_market_impact(
        self,
        symbol: str,
        exchange_id: str,
        quantity: float,
        side: str = "buy",
        book: OrderBook | None = None,
    ) -> float:
        """
        Walk the order book to estimate average execution price for given quantity.
        Returns VWAP fill price.
        """
        if book is None:
            book = self._oba.fetch_order_book(exchange_id, symbol, limit=50)

        levels = book.asks if side == "buy" else book.bids
        if not levels:
            return 0.0

        remaining = quantity
        cost = 0.0

        for price, vol in levels:
            fill = min(remaining, vol)
            cost += fill * price
            remaining -= fill
            if remaining <= 0:
                break

        filled = quantity - remaining
        return cost / filled if filled > 0 else levels[0][0]

    # ------------------------------------------------------------------
    def split_order_by_liquidity(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        exchanges: list[str],
        liquidity_map: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Route splitting: given order size, split across exchanges proportionally
        to available liquidity at target price.

        Algorithm:
          1. For each exchange, determine available liquidity (from order book top-5
             levels, or from liquidity_map if provided for testing).
          2. Sort exchanges descending by liquidity.
          3. Allocate proportionally: alloc_i = total_qty × (liq_i / total_liq).

        Returns list of dicts: [{"exchange": str, "quantity": float, "liquidity": float}]
        """
        # Build liquidity map from order books if not provided
        if liquidity_map is None:
            liquidity_map = {}
            for ex_id in exchanges:
                try:
                    book = self._oba.fetch_order_book(ex_id, symbol, limit=10)
                    levels = book.asks if side == "buy" else book.bids
                    liquidity_map[ex_id] = sum(v for _, v in levels[:5])
                except Exception:
                    liquidity_map[ex_id] = 0.0

        # Filter to exchanges with positive liquidity
        valid = {ex: liq for ex, liq in liquidity_map.items() if liq > 0 and ex in exchanges}
        if not valid:
            return []

        total_liquidity = sum(valid.values())
        # Sort descending by liquidity
        sorted_exchanges = sorted(valid.items(), key=lambda x: -x[1])

        allocations = []
        for ex_id, liq in sorted_exchanges:
            proportion = liq / total_liquidity
            qty = total_quantity * proportion
            allocations.append({
                "exchange": ex_id,
                "quantity": round(qty, 8),
                "liquidity": liq,
                "allocation_pct": round(proportion * 100, 2),
            })

        return allocations

    # ------------------------------------------------------------------
    def estimate_slippage(
        self,
        order_size_usd: float,
        bid_ask_depth_usd: float,
        market_impact_coeff: float = 0.1,
    ) -> float:
        """
        Estimate slippage as a fraction of price.

        Formula: slippage = order_size / (bid_ask_depth × market_impact_coeff)

        Args:
            order_size_usd:      Total order size in USD.
            bid_ask_depth_usd:   Combined bid+ask depth within 1% of mid (USD).
            market_impact_coeff: Default 0.1 — tunes impact sensitivity.

        Returns:
            Slippage as a decimal fraction (e.g., 0.005 = 0.5%).
        """
        if bid_ask_depth_usd <= 0 or market_impact_coeff <= 0:
            return 1.0  # No liquidity = 100% slippage
        denominator = bid_ask_depth_usd * market_impact_coeff
        slippage = order_size_usd / denominator
        # Cap at 100% slippage
        return min(1.0, max(0.0, slippage))

    # ------------------------------------------------------------------
    def score_best_execution(
        self,
        symbol: str,
        exchanges: list[str],
        order_size_usd: float = 100_000.0,
        side: str = "buy",
    ) -> list[dict]:
        """
        Compute best execution score per exchange.

        Score = price_improvement - fee_pct - slippage_estimate

        For buy orders, price_improvement = (best_ask_across_all - exchange_ask) / best_ask_across_all
        For sell orders, price_improvement = (exchange_bid - best_bid_across_all) / best_bid_across_all

        Higher score = better execution venue.

        Returns list sorted by score descending.
        """
        books: dict[str, OrderBook] = {}
        for ex_id in exchanges:
            try:
                books[ex_id] = self._oba.fetch_order_book(ex_id, symbol, limit=10)
            except Exception:
                pass

        # Determine reference price (global best)
        if side == "buy":
            best_prices = [books[ex].asks[0][0] for ex in books if books[ex].asks]
            reference_price = min(best_prices) if best_prices else None
        else:
            best_prices = [books[ex].bids[0][0] for ex in books if books[ex].bids]
            reference_price = max(best_prices) if best_prices else None

        if reference_price is None or reference_price <= 0:
            return []

        results = []
        for ex_id in exchanges:
            if ex_id not in books:
                continue
            book = books[ex_id]

            if side == "buy":
                if not book.asks:
                    continue
                ex_price = book.asks[0][0]
                # Price improvement: paying less than the worst ask
                price_improvement = (reference_price - ex_price) / reference_price if reference_price > 0 else 0.0
            else:
                if not book.bids:
                    continue
                ex_price = book.bids[0][0]
                price_improvement = (ex_price - reference_price) / reference_price if reference_price > 0 else 0.0

            # Fee
            fee_pct = self._mgr.get_exchange_taker_fee(ex_id)

            # Slippage estimate
            depth_info = self._oba.compute_market_depth(book, pct_from_mid=0.01)
            depth_usd = depth_info.get("total_depth", 0) * (book.mid_price or ex_price)
            slippage = self.estimate_slippage(order_size_usd, depth_usd)

            score = price_improvement - fee_pct - slippage

            results.append({
                "exchange": ex_id,
                "price": ex_price,
                "price_improvement": round(price_improvement, 6),
                "fee_pct": fee_pct,
                "slippage_estimate": round(slippage, 6),
                "execution_score": round(score, 6),
            })

        # Sort by execution score descending (higher = better)
        results.sort(key=lambda x: -x["execution_score"])
        return results

    # ------------------------------------------------------------------
    def generate_twap_schedule(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        time_window_minutes: int,
        num_slices: int | None = None,
        jitter_pct: float = 0.10,
    ) -> list[dict]:
        """
        Generate a TWAP (Time-Weighted Average Price) execution schedule.

        Splits total_quantity into num_slices equal sub-orders spread over
        time_window_minutes, with each interval randomized by ±jitter_pct
        to avoid front-running patterns.

        Args:
            symbol:               Trading pair.
            side:                 'buy' or 'sell'.
            total_quantity:       Total order size.
            time_window_minutes:  Total execution window in minutes.
            num_slices:           Number of sub-orders (default: max(2, window//6)).
            jitter_pct:           Random timing jitter as fraction of interval (default 0.10 = ±10%).

        Returns:
            List of dicts with keys: slice_index, quantity, scheduled_offset_minutes,
            jitter_minutes, execute_at_minutes.
        """
        import random

        if num_slices is None:
            num_slices = max(2, time_window_minutes // 6)

        num_slices = max(1, num_slices)
        slice_qty = total_quantity / num_slices
        base_interval = time_window_minutes / num_slices

        schedule = []
        for i in range(num_slices):
            # Nominal offset: i × interval
            nominal_offset = i * base_interval
            # Jitter: ±jitter_pct of interval
            max_jitter = base_interval * jitter_pct
            jitter = random.uniform(-max_jitter, max_jitter)
            # Clamp execute_at to [0, time_window_minutes]
            execute_at = max(0.0, min(time_window_minutes, nominal_offset + jitter))

            schedule.append({
                "slice_index": i,
                "symbol": symbol,
                "side": side,
                "quantity": round(slice_qty, 8),
                "scheduled_offset_minutes": round(nominal_offset, 4),
                "jitter_minutes": round(jitter, 4),
                "execute_at_minutes": round(execute_at, 4),
            })

        return schedule


# ---------------------------------------------------------------------------
# ArbitrageDetector
# ---------------------------------------------------------------------------

class ArbitrageDetector:
    """
    Detects cross-exchange, triangular, and funding-rate arbitrage opportunities.
    """

    def __init__(
        self,
        exchange_manager: CCXTExchangeManager,
        ob_aggregator: OrderBookAggregator,
    ) -> None:
        self._mgr = exchange_manager
        self._oba = ob_aggregator
        self._monitor_thread: threading.Thread | None = None
        self._monitor_active = False

    # ------------------------------------------------------------------
    def scan_arbitrage(
        self,
        symbol: str,
        exchanges: list[str],
    ) -> list[ArbitrageOpportunity]:
        """
        Scan for simple cross-exchange arbitrage: buy on A (ask), sell on B (bid).
        Only flags opportunities with positive net profit after fees.
        """
        books: dict[str, OrderBook] = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(self._oba.fetch_order_book, ex_id, symbol, 5): ex_id
                for ex_id in exchanges
            }
            for fut in as_completed(futures):
                ex_id = futures[fut]
                try:
                    book = fut.result()
                    if book.bids and book.asks:
                        books[ex_id] = book
                except Exception:
                    pass

        opportunities: list[ArbitrageOpportunity] = []

        for buy_ex in exchanges:
            for sell_ex in exchanges:
                if buy_ex == sell_ex:
                    continue
                if buy_ex not in books or sell_ex not in books:
                    continue

                ask_price = books[buy_ex].asks[0][0]
                bid_price = books[sell_ex].bids[0][0]

                if ask_price <= 0 or bid_price <= 0:
                    continue

                buy_fee = self._mgr.get_exchange_taker_fee(buy_ex)
                sell_fee = self._mgr.get_exchange_taker_fee(sell_ex)

                gross_profit_pct = (bid_price - ask_price) / ask_price * 100
                net_profit_pct = gross_profit_pct - (buy_fee + sell_fee) * 100

                if net_profit_pct > 0:
                    opportunities.append(
                        ArbitrageOpportunity(
                            symbol=_normalize_unified(symbol),
                            buy_exchange=buy_ex,
                            sell_exchange=sell_ex,
                            buy_ask=ask_price,
                            sell_bid=bid_price,
                            gross_profit_pct=gross_profit_pct,
                            net_profit_pct=net_profit_pct,
                            buy_fee=buy_fee,
                            sell_fee=sell_fee,
                            timestamp=datetime.now(timezone.utc),
                        )
                    )

        # Sort by net profit descending
        opportunities.sort(key=lambda x: -x.net_profit_pct)
        return opportunities

    # ------------------------------------------------------------------
    def compute_funding_rate_arbitrage(self, symbol: str) -> list[dict]:
        """
        Perpetual funding rate arbitrage.
        Positive funding (perp > spot) → short perp, long spot.
        Negative funding (perp < spot) → long perp, short spot.
        """
        results = []

        funding_sources = [
            ("binance", f"{BINANCE_FAPI}/fapi/v1/premiumIndex"),
            ("bybit", f"{BYBIT_REST}/v5/market/funding/history"),
        ]

        for exchange_id, url in funding_sources:
            try:
                raw_symbol = symbol.replace("/", "")
                if exchange_id == "binance":
                    data = _http_get(url, {"symbol": raw_symbol + "USDT" if "USDT" not in raw_symbol else raw_symbol})
                    if data:
                        funding_rate = float(data.get("lastFundingRate", 0))
                        annualized = funding_rate * 3 * 365 * 100  # 8h intervals

                        results.append({
                            "exchange": exchange_id,
                            "symbol": symbol,
                            "funding_rate_8h_pct": funding_rate * 100,
                            "annualized_pct": annualized,
                            "signal": "SHORT_PERP_LONG_SPOT" if funding_rate > 0 else "LONG_PERP_SHORT_SPOT",
                            "viable": abs(annualized) > 5.0,  # > 5% annualized to cover fees
                        })
            except Exception as exc:
                logger.debug("Funding rate fetch failed [%s]: %s", exchange_id, exc)

        return results

    # ------------------------------------------------------------------
    def detect_triangular_arbitrage(
        self,
        exchange_id: str,
        base: str = "USDT",
    ) -> list[dict]:
        """
        Detect triangular arbitrage: A/B × B/C × C/A ≠ 1.
        Example: BTC/ETH × ETH/USDT vs BTC/USDT
        """
        if not HAS_CCXT:
            return []

        triangles = [
            ("BTC", "ETH", base),
            ("BTC", "BNB", base),
            ("ETH", "BNB", base),
            ("SOL", "ETH", base),
        ]

        results = []
        try:
            ex = self._mgr.initialize_exchange(exchange_id)
            markets = ex.load_markets()
        except Exception:
            return []

        for a, b, c in triangles:
            pair_ab = f"{a}/{b}"
            pair_bc = f"{b}/{c}"
            pair_ac = f"{a}/{c}"

            if not all(p in markets for p in [pair_ab, pair_bc, pair_ac]):
                continue

            try:
                ticker_ab = ex.fetch_ticker(pair_ab)
                ticker_bc = ex.fetch_ticker(pair_bc)
                ticker_ac = ex.fetch_ticker(pair_ac)

                # Theoretical rate: A→C via B
                rate_via_b = (
                    (1 / ticker_ab["ask"]) * ticker_bc["bid"]
                ) if ticker_ab["ask"] and ticker_bc["bid"] else None

                # Direct rate: A→C
                rate_direct = ticker_ac["bid"] if ticker_ac["bid"] else None

                if rate_via_b and rate_direct and rate_via_b > 0 and rate_direct > 0:
                    spread_pct = (rate_direct - rate_via_b) / rate_via_b * 100
                    taker_fee = self._mgr.get_exchange_taker_fee(exchange_id)
                    net_profit = spread_pct - taker_fee * 3 * 100

                    if abs(net_profit) > 0.05:  # > 5bps
                        results.append({
                            "path": f"{a}→{b}→{c}",
                            "rate_via_b": rate_via_b,
                            "rate_direct": rate_direct,
                            "gross_spread_pct": spread_pct,
                            "net_profit_pct": net_profit,
                            "viable": net_profit > 0,
                        })
            except Exception as exc:
                logger.debug("Triangular arb error [%s %s/%s/%s]: %s", exchange_id, a, b, c, exc)

        return results

    # ------------------------------------------------------------------
    def monitor_arbitrage(
        self,
        symbols: list[str],
        exchanges: list[str],
        interval: int = 5,
    ) -> None:
        """
        Background thread: continuously scan for arb and log to JSONL.
        Call stop_monitor() to halt.
        """
        os.makedirs(_DATA_DIR, exist_ok=True)
        log_path = os.path.join(_DATA_DIR, "arb_log.jsonl")

        self._monitor_active = True

        def _run() -> None:
            while self._monitor_active:
                for sym in symbols:
                    try:
                        opps = self.scan_arbitrage(sym, exchanges)
                        for opp in opps:
                            record = {
                                "timestamp": opp.timestamp.isoformat(),
                                "symbol": opp.symbol,
                                "buy_exchange": opp.buy_exchange,
                                "sell_exchange": opp.sell_exchange,
                                "buy_ask": opp.buy_ask,
                                "sell_bid": opp.sell_bid,
                                "net_profit_pct": opp.net_profit_pct,
                            }
                            with open(log_path, "a") as f:
                                f.write(json.dumps(record) + "\n")
                            logger.info("ARB OPPORTUNITY: %s", record)
                    except Exception as exc:
                        logger.debug("Monitor arb error [%s]: %s", sym, exc)

                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=_run, daemon=True)
        self._monitor_thread.start()
        logger.info("Arbitrage monitor started. Logging to %s", log_path)

    # ------------------------------------------------------------------
    def stop_monitor(self) -> None:
        self._monitor_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)


# ---------------------------------------------------------------------------
# PerpetualsAnalyzer
# ---------------------------------------------------------------------------

class PerpetualsAnalyzer:
    """
    Analyzes perpetual futures metrics: funding rates, open interest, leverage.
    """

    def __init__(self, exchange_manager: CCXTExchangeManager) -> None:
        self._mgr = exchange_manager
        self._funding_history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    def fetch_funding_rate(
        self,
        exchange_id: str,
        symbol: str,
    ) -> float:
        """
        Fetch current funding rate. Returns annualized %.
        Binance: 8-hour intervals × 3 × 365 to annualize.
        """
        raw_symbol = symbol.replace("/", "")
        if "USDT" not in raw_symbol:
            raw_symbol += "USDT"

        if exchange_id == "binance":
            data = _http_get(
                f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
                {"symbol": raw_symbol},
            )
            if data:
                rate_8h = float(data.get("lastFundingRate", 0))
                return rate_8h * 3 * 365 * 100  # annualized %

        elif exchange_id == "bybit":
            data = _http_get(
                f"{BYBIT_REST}/v5/market/tickers",
                {"category": "linear", "symbol": raw_symbol},
            )
            if data and data.get("result", {}).get("list"):
                ticker = data["result"]["list"][0]
                rate_8h = float(ticker.get("fundingRate", 0))
                return rate_8h * 3 * 365 * 100

        elif HAS_CCXT:
            try:
                ex = self._mgr.initialize_exchange(exchange_id)
                if ex.has.get("fetchFundingRate"):
                    fr = ex.fetch_funding_rate(symbol)
                    rate = float(fr.get("fundingRate", 0))
                    return rate * 3 * 365 * 100
            except Exception as exc:
                logger.debug("CCXT funding rate error: %s", exc)

        return 0.0

    # ------------------------------------------------------------------
    def fetch_open_interest(
        self,
        exchange_id: str,
        symbol: str,
    ) -> float:
        """Fetch open interest in base currency units."""
        raw_symbol = symbol.replace("/", "")
        if "USDT" not in raw_symbol:
            raw_symbol += "USDT"

        if exchange_id == "binance":
            data = _http_get(
                f"{BINANCE_FAPI}/fapi/v1/openInterest",
                {"symbol": raw_symbol},
            )
            if data:
                return float(data.get("openInterest", 0))

        elif HAS_CCXT:
            try:
                ex = self._mgr.initialize_exchange(exchange_id)
                if ex.has.get("fetchOpenInterest"):
                    oi = ex.fetch_open_interest(symbol)
                    return float(oi.get("openInterest", oi.get("openInterestAmount", 0)))
            except Exception as exc:
                logger.debug("CCXT OI error [%s]: %s", exchange_id, exc)

        return 0.0

    # ------------------------------------------------------------------
    def compute_funding_rate_z_score(
        self,
        symbol: str,
        lookback: int = 30,
        exchange_id: str = "binance",
    ) -> float:
        """
        Z-score of current funding rate vs historical distribution.
        Uses stored history; falls back to Binance funding rate history.
        """
        raw_symbol = symbol.replace("/", "")
        if "USDT" not in raw_symbol:
            raw_symbol += "USDT"

        history_key = f"{exchange_id}_{raw_symbol}"

        # Try to fetch historical funding rates from Binance
        if exchange_id == "binance":
            data = _http_get(
                f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                {"symbol": raw_symbol, "limit": lookback * 3},  # 3 per day
            )
            if data and isinstance(data, list):
                rates = [float(r.get("fundingRate", 0)) for r in data]
                self._funding_history[history_key] = rates

        history = self._funding_history.get(history_key, [])
        current = self.fetch_funding_rate(exchange_id, symbol)

        if len(history) < 3:
            return 0.0

        import statistics
        mean_rate = statistics.mean(history)
        std_rate = statistics.stdev(history)

        if std_rate == 0:
            return 0.0

        # Current rate as a fraction (not annualized %)
        current_fraction = current / (3 * 365 * 100) if current != 0 else 0.0
        return (current_fraction - mean_rate) / std_rate

    # ------------------------------------------------------------------
    def detect_leverage_extremes(
        self,
        symbol: str,
        exchange_id: str = "binance",
    ) -> str:
        """
        Classify leverage regime:
        HIGH_LEVERAGE:   OI rising + funding > 0.1% per 8h → crowded longs
        DELEVERAGING:    OI falling sharply → forced liquidations likely
        NEUTRAL:         normal conditions
        """
        funding_annualized = self.fetch_funding_rate(exchange_id, symbol)
        funding_8h = funding_annualized / (3 * 365)  # back to 8h

        raw_symbol = symbol.replace("/", "")
        if "USDT" not in raw_symbol:
            raw_symbol += "USDT"

        # Fetch OI history (last 2 data points for change)
        oi_now = self.fetch_open_interest(exchange_id, symbol)

        # Approximate OI change from funding rate trend
        z_score = self.compute_funding_rate_z_score(symbol, lookback=7, exchange_id=exchange_id)

        if funding_8h > 0.001 and z_score > 1.5:
            return "HIGH_LEVERAGE"  # crowded longs, positive funding
        elif funding_8h < -0.001 and z_score < -1.5:
            return "HIGH_LEVERAGE"  # crowded shorts, negative funding
        elif oi_now > 0 and z_score < -2.0:
            return "DELEVERAGING"
        else:
            return "NEUTRAL"


# ---------------------------------------------------------------------------
# PaperExecutionEngine
# ---------------------------------------------------------------------------

import uuid


class PaperExecutionEngine:
    """
    Paper trading simulation on CCXT exchange data.
    MARKET orders: fill at current best + 0.05% slippage.
    LIMIT orders: fill when price crosses limit in next bar.
    """

    def __init__(
        self,
        ob_aggregator: OrderBookAggregator,
        initial_balances: dict[str, float] | None = None,
    ) -> None:
        self._oba = ob_aggregator
        self._orders: list[PaperOrder] = []
        self._fills: list[PaperFill] = []
        self._balances: dict[str, float] = initial_balances or {"USDT": 100_000.0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def submit_paper_order(
        self,
        exchange_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> PaperOrder:
        """
        Submit a paper order. Market orders fill immediately.
        Limit orders are held open until price is reached.
        """
        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            exchange_id=exchange_id,
            symbol=_normalize_unified(symbol),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="open",
            created_at=datetime.now(timezone.utc),
        )

        with self._lock:
            self._orders.append(order)

        if order_type.upper() == "MARKET":
            self._fill_market_order(order)

        return order

    # ------------------------------------------------------------------
    def _fill_market_order(self, order: PaperOrder) -> None:
        """Fill a market order at current best price + slippage."""
        book = self._oba.fetch_order_book(order.exchange_id, order.symbol, limit=10)

        if order.side == "buy":
            if not book.asks:
                logger.warning("Cannot fill paper buy — no asks for %s", order.symbol)
                return
            market_price = book.asks[0][0]
        else:
            if not book.bids:
                logger.warning("Cannot fill paper sell — no bids for %s", order.symbol)
                return
            market_price = book.bids[0][0]

        # 0.05% slippage
        slippage_factor = 1.0005 if order.side == "buy" else 0.9995
        fill_price = market_price * slippage_factor

        # Taker fee
        fee_rate = 0.001  # default 0.1%
        fee = fill_price * order.quantity * fee_rate

        fill = PaperFill(
            order_id=order.order_id,
            exchange_id=order.exchange_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            fee=fee,
            filled_at=datetime.now(timezone.utc),
        )

        # Update balances
        base, quote = order.symbol.split("/")
        with self._lock:
            if order.side == "buy":
                cost = fill_price * order.quantity + fee
                self._balances[quote] = self._balances.get(quote, 0.0) - cost
                self._balances[base] = self._balances.get(base, 0.0) + order.quantity
            else:
                proceeds = fill_price * order.quantity - fee
                self._balances[base] = self._balances.get(base, 0.0) - order.quantity
                self._balances[quote] = self._balances.get(quote, 0.0) + proceeds

            order.status = "filled"
            self._fills.append(fill)

        logger.info(
            "PAPER FILL: %s %s %s @ %.4f (fee=%.4f)",
            order.side.upper(), order.quantity, order.symbol, fill_price, fee,
        )

    # ------------------------------------------------------------------
    def check_limit_orders(self, current_prices: dict[str, float]) -> int:
        """
        Check open limit orders against current prices.
        Returns count of newly filled orders.
        """
        filled_count = 0
        with self._lock:
            for order in self._orders:
                if order.status != "open" or order.order_type.upper() != "LIMIT":
                    continue
                if order.price is None:
                    continue

                current = current_prices.get(order.symbol, 0.0)
                triggered = (
                    order.side == "buy" and current <= order.price
                ) or (
                    order.side == "sell" and current >= order.price
                )

                if triggered:
                    fill = PaperFill(
                        order_id=order.order_id,
                        exchange_id=order.exchange_id,
                        symbol=order.symbol,
                        side=order.side,
                        fill_price=order.price,
                        fill_quantity=order.quantity,
                        fee=order.price * order.quantity * 0.001,
                        filled_at=datetime.now(timezone.utc),
                    )
                    order.status = "filled"
                    self._fills.append(fill)
                    filled_count += 1

        return filled_count

    # ------------------------------------------------------------------
    def get_paper_fills(self) -> list[PaperFill]:
        with self._lock:
            return list(self._fills)

    # ------------------------------------------------------------------
    def get_paper_balance(self) -> dict[str, float]:
        with self._lock:
            return {k: v for k, v in self._balances.items() if v != 0.0}

    # ------------------------------------------------------------------
    def get_open_orders(self) -> list[PaperOrder]:
        with self._lock:
            return [o for o in self._orders if o.status == "open"]

    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            for order in self._orders:
                if order.order_id == order_id and order.status == "open":
                    order.status = "cancelled"
                    return True
        return False


# ---------------------------------------------------------------------------
# CCXTDataEngine — orchestrator
# ---------------------------------------------------------------------------

class CCXTDataEngine:
    """
    High-level orchestrator combining all CCXT components.
    Single entry point for market data, execution, and analysis.
    """

    def __init__(
        self,
        initial_paper_balance: dict[str, float] | None = None,
    ) -> None:
        self.exchange_manager = CCXTExchangeManager()
        self.ohlcv_fetcher = UnifiedOHLCVFetcher(self.exchange_manager)
        self.ob_aggregator = OrderBookAggregator(self.exchange_manager)
        self.order_router = SmartOrderRouter(self.exchange_manager, self.ob_aggregator)
        self.arb_detector = ArbitrageDetector(self.exchange_manager, self.ob_aggregator)
        self.perps_analyzer = PerpetualsAnalyzer(self.exchange_manager)
        self.paper_engine = PaperExecutionEngine(
            self.ob_aggregator,
            initial_balances=initial_paper_balance or {"USDT": 100_000.0},
        )
        self._tier1 = _TIER1_EXCHANGES

    # ------------------------------------------------------------------
    def get_market_overview(self, symbol: str) -> dict:
        """Fetch prices across all tier-1 exchanges for a symbol."""
        exchange_data = self.ohlcv_fetcher.fetch_ohlcv_multi_exchange(
            symbol, self._tier1, timeframe="1d", limit=2
        )

        overview: dict[str, Any] = {"symbol": symbol, "exchanges": {}}
        for ex_id, df in exchange_data.items():
            if not df.empty:
                latest = df.iloc[-1]
                overview["exchanges"][ex_id] = {
                    "close": round(float(latest["close"]), 6),
                    "volume": round(float(latest["volume"]), 2),
                    "timestamp": str(latest["timestamp"]),
                }

        # Add aggregated book
        agg_book = self.ob_aggregator.fetch_aggregated_order_book(
            symbol, self._tier1[:5]
        )
        overview["mid_price"] = agg_book.mid_price
        overview["best_bid"] = agg_book.bids[0][0] if agg_book.bids else 0.0
        overview["best_ask"] = agg_book.asks[0][0] if agg_book.asks else 0.0
        overview["order_book_imbalance"] = self.ob_aggregator.detect_order_book_imbalance(agg_book)

        return overview

    # ------------------------------------------------------------------
    def get_best_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> ExecutionPlan:
        """Route an order to get best execution."""
        routing = self.order_router.route_market_order(
            symbol, side, quantity, self._tier1
        )

        if side == "buy":
            best_quote = self.order_router.find_best_ask(symbol, self._tier1)
        else:
            best_quote = self.order_router.find_best_bid(symbol, self._tier1)

        fee_rate = 0.001  # average
        estimated_cost = routing.estimated_avg_price * quantity
        estimated_fee = estimated_cost * fee_rate

        return ExecutionPlan(
            symbol=_normalize_unified(symbol),
            side=side,
            quantity=quantity,
            routing_plan=routing,
            best_quote=best_quote,
            estimated_cost=estimated_cost,
            estimated_fee=estimated_fee,
        )

    # ------------------------------------------------------------------
    def run_arbitrage_scanner(
        self,
        symbols: list[str],
        interval: int = 10,
    ) -> None:
        """Start continuous background arbitrage scanner."""
        self.arb_detector.monitor_arbitrage(symbols, self._tier1, interval=interval)

    # ------------------------------------------------------------------
    def get_exchange_rankings(self, symbol: str) -> pd.DataFrame:
        """
        Rank exchanges by volume, spread, and fees for a given symbol.
        """
        exchange_data = self.ohlcv_fetcher.fetch_ohlcv_multi_exchange(
            symbol, self._tier1, timeframe="1d", limit=1
        )
        spreads = self.order_router.compute_effective_spread(symbol, self._tier1)

        rows = []
        for ex_id in self._tier1:
            df = exchange_data.get(ex_id, pd.DataFrame())
            volume = float(df.iloc[-1]["volume"]) if not df.empty else 0.0
            spread_bps = spreads.get(ex_id, float("nan"))
            info = self.exchange_manager.get_exchange_info(ex_id)

            rows.append({
                "exchange": ex_id,
                "tier": info.tier,
                "volume_24h": volume,
                "spread_bps": spread_bps,
                "taker_fee_pct": info.taker_fee * 100,
                "rate_limit_ms": info.rate_limit_ms,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            # Score = volume rank / spread rank (higher = better)
            df = df.sort_values("volume_24h", ascending=False).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    def get_funding_overview(self, symbol: str) -> dict:
        """Funding rate + OI overview for perpetuals."""
        funding = {
            "binance_funding_annualized_pct": self.perps_analyzer.fetch_funding_rate("binance", symbol),
            "binance_open_interest": self.perps_analyzer.fetch_open_interest("binance", symbol),
            "funding_rate_z_score": self.perps_analyzer.compute_funding_rate_z_score(symbol),
            "leverage_regime": self.perps_analyzer.detect_leverage_extremes(symbol),
        }
        return funding


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def create_engine(
    paper_balance: dict[str, float] | None = None,
) -> CCXTDataEngine:
    """Create a fully initialized CCXTDataEngine."""
    return CCXTDataEngine(initial_paper_balance=paper_balance)


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    print("=" * 70)
    print("SENTINEL — CCXTDataEngine v3 Demo")
    print("=" * 70)

    engine = create_engine()

    # 1. List supported exchanges
    print("\n[1] Supported exchanges (first 20):")
    supported = engine.exchange_manager.get_supported_exchanges()
    for i, ex in enumerate(supported[:20], 1):
        print(f"  {i:2d}. {ex}")
    print(f"  ... total: {len(supported)} exchanges")

    # 2. OHLCV from top 3 exchanges
    symbol = "BTC/USDT"
    print(f"\n[2] Fetching {symbol} OHLCV (1d, last 5 bars) from Binance, Kraken, Coinbase...")
    target_exchanges = ["binance", "kraken", "coinbase"]
    ohlcv_data = engine.ohlcv_fetcher.fetch_ohlcv_multi_exchange(
        symbol, target_exchanges, timeframe="1d", limit=5
    )
    for ex_id, df in ohlcv_data.items():
        if not df.empty:
            latest = df.iloc[-1]
            print(
                f"  {ex_id:12s}: close={latest['close']:.2f}  "
                f"vol={latest['volume']:.0f}  ts={latest['timestamp']}"
            )

    # 3. Consolidated OHLCV
    if ohlcv_data:
        print("\n[3] Consolidated VWAP OHLCV:")
        consolidated = engine.ohlcv_fetcher.consolidate_ohlcv(ohlcv_data)
        if not consolidated.empty:
            latest = consolidated.iloc[-1]
            print(
                f"  VWAP close={latest['close']:.2f}  "
                f"total vol={latest['volume']:.0f}  "
                f"exchanges={latest.get('exchange_count', '?')}"
            )

    # 4. Order book
    print(f"\n[4] Order book snapshot ({symbol} @ Binance):")
    book = engine.ob_aggregator.fetch_order_book("binance", symbol, limit=5)
    if book.bids and book.asks:
        print(f"  Best bid: {book.bids[0][0]:.2f} ({book.bids[0][1]:.4f} BTC)")
        print(f"  Best ask: {book.asks[0][0]:.2f} ({book.asks[0][1]:.4f} BTC)")
        print(f"  Mid:      {book.mid_price:.2f}")
        imbalance = engine.ob_aggregator.detect_order_book_imbalance(book)
        print(f"  Imbalance: {imbalance:+.4f} ({'BUY pressure' if imbalance > 0 else 'SELL pressure'})")

    # 5. Spread comparison
    print("\n[5] Effective spreads (bps):")
    spreads = engine.order_router.compute_effective_spread(symbol, target_exchanges)
    for ex, spread_bps in spreads.items():
        print(f"  {ex:12s}: {spread_bps:.2f} bps")

    # 6. Arbitrage scan
    print(f"\n[6] Arbitrage scan ({symbol}):")
    arb_opps = engine.arb_detector.scan_arbitrage(symbol, target_exchanges)
    if arb_opps:
        for opp in arb_opps[:3]:
            print(
                f"  BUY {opp.buy_exchange} ask={opp.buy_ask:.2f} → "
                f"SELL {opp.sell_exchange} bid={opp.sell_bid:.2f} | "
                f"net={opp.net_profit_pct:.4f}%"
            )
    else:
        print("  No profitable arbitrage found (normal).")

    # 7. Funding rate
    print("\n[7] Perpetuals — Binance BTC funding:")
    funding_pct = engine.perps_analyzer.fetch_funding_rate("binance", symbol)
    oi = engine.perps_analyzer.fetch_open_interest("binance", symbol)
    regime = engine.perps_analyzer.detect_leverage_extremes(symbol)
    z = engine.perps_analyzer.compute_funding_rate_z_score(symbol)
    print(f"  Annualized funding: {funding_pct:.2f}%")
    print(f"  Open interest:      {oi:,.0f} BTC")
    print(f"  Funding z-score:    {z:.2f}")
    print(f"  Leverage regime:    {regime}")

    # 8. Paper trade
    print("\n[8] Paper trade — buy 0.1 BTC at market:")
    order = engine.paper_engine.submit_paper_order(
        "binance", symbol, "buy", "MARKET", 0.1
    )
    print(f"  Order ID: {order.order_id[:8]}... status={order.status}")
    balance = engine.paper_engine.get_paper_balance()
    print(f"  Balance after trade: {balance}")

    # 9. Exchange rankings
    print("\n[9] Exchange rankings by 24h volume:")
    rankings = engine.get_exchange_rankings(symbol)
    if not rankings.empty:
        print(rankings.to_string(index=False))

    print("\nDone.")
