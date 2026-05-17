"""
sentinel/sds/adapters/onchain_metrics_v3.py
dim_108: On-Chain Metrics — Deep Intelligence (score 8 → 9)

Comprehensive Bitcoin on-chain analytics:
  MVRV Z-Score, SOPR, Puell Multiple, Reserve Risk proxy,
  Coin Days Destroyed, Active Addresses, NUPL, NVT Signal,
  Thermocap, Long/Short Holder Supply, Composite Signal

Free Data Sources (no API keys required)
-----------------------------------------
  https://blockchain.info/stats?format=json     — network stats
  https://api.blockchain.info/charts/{metric}   — historical charts
  https://api.coingecko.com/api/v3             — price/market data
  https://api.alternative.me/fng/              — Fear & Greed Index

Public API
----------
  engine = OnChainMetricsEngine()
  dashboard = engine.get_full_dashboard()
  report    = engine.run_daily_update()

  mvrv_calc = RealizedCapCalculator()
  z_score   = mvrv_calc.compute_mvrv_z_score(...)

  composite  = OnChainCompositeSignal()
  score      = composite.compute_composite_bull_bear_score()
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore[assignment]

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCKCHAIN_INFO_STATS = "https://blockchain.info/stats?format=json"
BLOCKCHAIN_INFO_CHARTS = "https://api.blockchain.info/charts/{metric}"
BLOCKCHAIN_INFO_TICKER = "https://blockchain.info/ticker"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL = "https://api.alternative.me/fng/"

# SQLite cache path
CACHE_DB_PATH = Path(__file__).parent.parent.parent / "data" / "onchain_cache.db"

# Rate limiting
RATE_LIMIT_DELAY = 1.2  # seconds between requests
CACHE_TTL_SECONDS = 3600  # 1-hour cache

# Historical MVRV regime thresholds (Bitcoin-specific, well-documented)
MVRV_TOP_THRESHOLD = 3.0     # > 3.0: historically near tops
MVRV_BOTTOM_THRESHOLD = 1.0  # < 1.0: historically near bottoms
MVRV_Z_TOP = 7.0             # Z > 7: extreme overvaluation
MVRV_Z_BOTTOM = 0.0          # Z < 0: undervaluation

# NVT thresholds
NVT_UNDERVALUED = 65.0
NVT_OVERVALUED = 150.0

# Puell Multiple thresholds
PUELL_SELL_ZONE = 4.0
PUELL_BUY_ZONE = 0.5

# NUPL zones
NUPL_CAPITULATION = 0.0
NUPL_HOPE = 0.25
NUPL_OPTIMISM = 0.50
NUPL_BELIEF = 0.75

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NetworkStats:
    """Bitcoin network statistics snapshot."""
    timestamp: datetime
    hashrate_ehash: float = 0.0
    difficulty: float = 0.0
    block_height: int = 0
    mempool_tx_count: int = 0
    mempool_size_mb: float = 0.0
    avg_block_time_minutes: float = 10.0
    total_btc: float = 0.0
    miners_revenue_usd: float = 0.0
    tx_fees_usd: float = 0.0
    n_transactions: int = 0
    trade_volume_usd: float = 0.0


@dataclass
class MVRVData:
    """MVRV and Realized Cap data snapshot."""
    timestamp: datetime
    market_cap: float = 0.0
    realized_cap: float = 0.0
    mvrv_ratio: float = 0.0
    mvrv_z_score: float = 0.0
    realized_price: float = 0.0
    nupl: float = 0.0
    nupl_zone: str = "UNKNOWN"


@dataclass
class OnChainScore:
    """Composite on-chain bull/bear score."""
    timestamp: datetime
    composite_score: float = 50.0
    sentiment: str = "NEUTRAL"
    mvrv_z_score: float = 0.0
    mvrv_component: float = 50.0
    puell_multiple: float = 1.0
    puell_component: float = 50.0
    sopr_value: float = 1.0
    sopr_component: float = 50.0
    nvt_signal: float = 100.0
    nvt_component: float = 50.0
    address_momentum: float = 0.0
    address_component: float = 50.0
    cycle_position: str = "UNKNOWN"
    nupl: float = 0.0
    fear_greed_index: int = 50

    def __post_init__(self) -> None:
        if self.composite_score > 70:
            self.sentiment = "EXTREME_GREED"
        elif self.composite_score > 55:
            self.sentiment = "GREED"
        elif self.composite_score >= 45:
            self.sentiment = "NEUTRAL"
        elif self.composite_score >= 30:
            self.sentiment = "FEAR"
        else:
            self.sentiment = "EXTREME_FEAR"


# ---------------------------------------------------------------------------
# SQLite Cache
# ---------------------------------------------------------------------------

class _SqliteCache:
    """
    Persistent SQLite cache with TTL for on-chain data.
    Avoids hammering free APIs on repeated calls.
    """

    def __init__(self, db_path: Path = CACHE_DB_PATH, ttl: int = CACHE_TTL_SECONDS) -> None:
        self._db_path = db_path
        self._ttl = ttl
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS onchain_cache (
                    key      TEXT PRIMARY KEY,
                    value    TEXT NOT NULL,
                    ts       REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON onchain_cache(ts)")
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            try:
                with sqlite3.connect(str(self._db_path)) as conn:
                    row = conn.execute(
                        "SELECT value, ts FROM onchain_cache WHERE key=?", (key,)
                    ).fetchone()
                    if row is None:
                        return None
                    value_str, ts = row
                    if time.time() - ts > self._ttl:
                        conn.execute("DELETE FROM onchain_cache WHERE key=?", (key,))
                        return None
                    return json.loads(value_str)
            except Exception as e:
                logger.debug("Cache get error: %s", e)
                return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            try:
                with sqlite3.connect(str(self._db_path)) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO onchain_cache(key, value, ts) VALUES(?,?,?)",
                        (key, json.dumps(value, default=str), time.time()),
                    )
                    conn.commit()
            except Exception as e:
                logger.debug("Cache set error: %s", e)

    def purge_expired(self) -> None:
        """Remove all expired cache entries."""
        with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "DELETE FROM onchain_cache WHERE ? - ts > ?",
                    (time.time(), self._ttl)
                )
                conn.commit()


_CACHE = _SqliteCache()
_LAST_REQUEST: float = 0.0
_REQUEST_LOCK = threading.Lock()


def _rate_limited_get(
    url: str,
    params: Optional[Dict] = None,
    timeout: int = 25,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
) -> Optional[Any]:
    """
    Rate-limited HTTP GET with SQLite caching and retry logic.
    Respects free-tier constraints.
    """
    global _LAST_REQUEST

    cache_key = url + "|" + str(sorted((params or {}).items()))
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    with _REQUEST_LOCK:
        elapsed = time.time() - _LAST_REQUEST
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        _LAST_REQUEST = time.time()

    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "SENTINEL/3.0 OnChainMetrics"},
            )
            resp.raise_for_status()

            # Handle both JSON and text responses
            try:
                data = resp.json()
            except Exception:
                data = resp.text

            if use_cache:
                _CACHE.set(cache_key, data)
            return data

        except requests.exceptions.HTTPError as e:
            status = getattr(resp, "status_code", 0)
            if status == 429:
                wait = 15 * (attempt + 1)
                logger.warning("Rate limited by %s, waiting %ds", url, wait)
                time.sleep(wait)
            elif status in (403, 404):
                logger.debug("HTTP %d for %s", status, url)
                return None
            else:
                logger.warning("HTTP %d error fetching %s: %s", status, url, e)
                return None
        except Exception as e:
            logger.warning("Fetch attempt %d failed for %s: %s", attempt + 1, url, e)
            if attempt < 2:
                time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# BlockchainInfoClient
# ---------------------------------------------------------------------------

class BlockchainInfoClient:
    """
    Wrapper for Blockchain.info public APIs.
    Provides network stats, historical chart data, and supply info.
    All endpoints are free, no API key required.
    """

    # Available Blockchain.info chart metrics
    VALID_METRICS = {
        "hash-rate", "difficulty", "n-transactions", "n-unique-addresses",
        "market-cap", "trade-volume", "total-bitcoins", "miners-revenue",
        "transaction-fees", "mempool-size", "mempool-count", "utxo-count",
        "blocks-size", "avg-block-size", "n-orphaned-blocks", "median-transaction-fee",
        "cost-per-transaction", "estimated-transaction-volume-usd",
    }

    def get_network_stats(self) -> NetworkStats:
        """
        Fetch current Bitcoin network statistics.

        Returns:
            NetworkStats dataclass with hashrate, difficulty, mempool, etc.
        """
        data = _rate_limited_get(BLOCKCHAIN_INFO_STATS)
        if not data or not isinstance(data, dict):
            logger.warning("Network stats unavailable")
            return NetworkStats(timestamp=datetime.now(timezone.utc))

        hashrate_raw = float(data.get("hash_rate") or 0)
        # Blockchain.info returns GH/s, convert to EH/s
        hashrate_eh = hashrate_raw / 1e9

        miners_rev_btc = float(data.get("miners_revenue_btc") or 0)
        miners_rev_usd = float(data.get("miners_revenue_usd") or 0)

        return NetworkStats(
            timestamp=datetime.now(timezone.utc),
            hashrate_ehash=round(hashrate_eh, 2),
            difficulty=float(data.get("difficulty") or 0),
            block_height=int(data.get("n_blocks_total") or 0),
            mempool_tx_count=int(data.get("unconfirmed_count") or 0),
            mempool_size_mb=float(data.get("mempool_size") or 0) / 1e6,
            avg_block_time_minutes=float(data.get("minutes_between_blocks") or 10.0),
            total_btc=float(data.get("totalbc") or 0) / 1e8,
            miners_revenue_usd=miners_rev_usd,
            n_transactions=int(data.get("n_tx") or 0),
            trade_volume_usd=float(data.get("trade_volume_usd") or 0),
        )

    def get_chart_data(
        self,
        metric: str,
        timespan: str = "1years",
        sampled: bool = True,
    ) -> "pd.Series":
        """
        Fetch historical chart data from Blockchain.info.

        Args:
            metric:   One of VALID_METRICS (e.g. 'hash-rate', 'n-transactions')
            timespan: '1years', '2years', '5years', 'all', '1weeks', etc.
            sampled:  Use sampled data (True recommended for long timespans)

        Returns:
            pd.Series indexed by datetime UTC
        """
        if metric not in self.VALID_METRICS:
            logger.warning("Unknown metric '%s'. Known: %s", metric, self.VALID_METRICS)

        url = BLOCKCHAIN_INFO_CHARTS.format(metric=metric)
        params: Dict[str, Any] = {
            "timespan": timespan,
            "format": "json",
            "sampled": "true" if sampled else "false",
        }

        data = _rate_limited_get(url, params=params)
        if not data or not HAS_PANDAS:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return pd.Series(dtype=float)

        values_raw = data.get("values", [])
        timestamps, values = [], []
        for entry in values_raw:
            if isinstance(entry, dict):
                ts = entry.get("x")
                val = entry.get("y")
                if ts is not None and val is not None:
                    timestamps.append(datetime.fromtimestamp(float(ts), tz=timezone.utc))
                    values.append(float(val))

        if not timestamps:
            return pd.Series(dtype=float)

        series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        series = series.sort_index()
        series.name = metric
        return series

    def get_total_supply(self) -> float:
        """
        Total BTC in circulation (satoshis → BTC).

        Returns:
            Float BTC amount (currently ~19.7M BTC)
        """
        data = _rate_limited_get(BLOCKCHAIN_INFO_STATS)
        if data and isinstance(data, dict):
            total_satoshis = float(data.get("totalbc") or 0)
            return total_satoshis / 1e8
        return 19_700_000.0  # Approximate fallback

    def get_mempool_stats(self) -> Dict[str, Any]:
        """
        Current mempool statistics.

        Returns:
            dict with pending tx count, mempool size, fee estimates
        """
        data = _rate_limited_get(BLOCKCHAIN_INFO_STATS)
        if not data or not isinstance(data, dict):
            return {}
        return {
            "unconfirmed_count": int(data.get("unconfirmed_count") or 0),
            "mempool_size_bytes": int(data.get("mempool_size") or 0),
            "mempool_size_mb": round(float(data.get("mempool_size") or 0) / 1e6, 2),
            "minutes_between_blocks": float(data.get("minutes_between_blocks") or 10.0),
        }

    def get_price_history(
        self, timespan: str = "1years"
    ) -> Tuple["pd.Series", "pd.Series"]:
        """
        Get BTC price and volume history from CoinGecko.

        Returns:
            (price_series, volume_series) both indexed by datetime
        """
        if not HAS_PANDAS:
            return pd.Series(), pd.Series()  # type: ignore[return-value]

        # Map timespan to CoinGecko days parameter
        days_map = {
            "1weeks": "7", "2weeks": "14", "1months": "30",
            "3months": "90", "6months": "180", "1years": "365",
            "2years": "730", "all": "max",
        }
        days = days_map.get(timespan, "365")

        url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        data = _rate_limited_get(url, params)

        if not data or not isinstance(data, dict):
            # Fallback: use market-cap chart (USD) from Blockchain.info
            mcap = self.get_chart_data("market-cap", timespan=timespan)
            supply_series = self.get_chart_data("total-bitcoins", timespan=timespan)
            if not mcap.empty and not supply_series.empty:
                price = mcap / (supply_series / 1e8)
                volume = self.get_chart_data("trade-volume", timespan=timespan)
                return price, volume
            return pd.Series(dtype=float), pd.Series(dtype=float)

        price_data = data.get("prices", [])
        vol_data = data.get("total_volumes", [])

        prices, vols = {}, {}
        for ts_ms, price in price_data:
            dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
            prices[dt] = float(price)
        for ts_ms, vol in vol_data:
            dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
            vols[dt] = float(vol)

        price_series = pd.Series(prices).sort_index()
        vol_series = pd.Series(vols).sort_index()
        price_series.name = "btc_price_usd"
        vol_series.name = "btc_volume_usd"
        return price_series, vol_series

    def get_current_price(self) -> float:
        """Fetch current BTC price in USD."""
        data = _rate_limited_get(BLOCKCHAIN_INFO_TICKER)
        if data and isinstance(data, dict):
            usd_data = data.get("USD") or data.get("usd") or {}
            if isinstance(usd_data, dict):
                return float(usd_data.get("last") or usd_data.get("15m") or 0)
        # Fallback to CoinGecko
        cg = _rate_limited_get(
            f"{COINGECKO_BASE}/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"},
        )
        if cg and isinstance(cg, dict):
            return float((cg.get("bitcoin") or {}).get("usd") or 0)
        return 0.0

    def get_miners_revenue_history(self, timespan: str = "1years") -> "pd.Series":
        """Historical daily miner revenue in USD."""
        return self.get_chart_data("miners-revenue", timespan=timespan)

    def get_transaction_volume_history(self, timespan: str = "1years") -> "pd.Series":
        """Historical estimated transaction volume in USD."""
        return self.get_chart_data("estimated-transaction-volume-usd", timespan=timespan)

    def get_active_addresses_history(self, timespan: str = "1years") -> "pd.Series":
        """Historical unique active address count."""
        return self.get_chart_data("n-unique-addresses", timespan=timespan)


# ---------------------------------------------------------------------------
# RealizedCapCalculator
# ---------------------------------------------------------------------------

class RealizedCapCalculator:
    """
    Compute Bitcoin Realized Cap and MVRV-related metrics.

    True Realized Cap requires full UTXO set traversal (not public).
    We use two approximation methods:
      1. VWAP(365d) × circulating_supply  ← primary
      2. Blockchain.info market-cap + correction factor
    """

    def __init__(self) -> None:
        self._client = BlockchainInfoClient()

    def compute_realized_cap_proxy(
        self,
        price_history: "pd.Series",
        volume_history: "pd.Series",
        window: int = 365,
    ) -> "pd.Series":
        """
        Approximate Realized Cap as VWAP(window days) × circulating supply.

        This proxies the aggregate cost basis because:
          - Coins traded recently at near-current prices
          - Coins held long-term contribute their old (lower) price to VWAP

        Formula:
          VWAP(t, W) = Σ(price × volume, t-W:t) / Σ(volume, t-W:t)
          RealizedCap(t) ≈ supply × VWAP(t, W)

        Args:
            price_history:  Daily BTC price series
            volume_history: Daily USD volume series
            window:         Rolling window in days (default 365)

        Returns:
            pd.Series of Realized Cap estimates (USD)
        """
        if not HAS_PANDAS or price_history.empty or volume_history.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        # Align on common index
        df = pd.DataFrame({"price": price_history, "volume": volume_history}).dropna()
        if df.empty:
            return pd.Series(dtype=float)

        # Compute rolling VWAP
        pv = df["price"] * df["volume"]
        rolling_pv = pv.rolling(window=min(window, len(df)), min_periods=max(1, window // 4)).sum()
        rolling_vol = df["volume"].rolling(window=min(window, len(df)), min_periods=max(1, window // 4)).sum()
        vwap = rolling_pv / rolling_vol.replace(0, float("nan"))

        # Current supply: ~19.7M BTC (increases ~0.9% per year)
        # Approximate time-varying supply
        supply = 19_000_000 + (df.index - df.index[0]).days / 365 * 170_000
        supply_series = pd.Series(supply.values if hasattr(supply, "values") else supply, index=df.index)

        realized_cap = vwap * supply_series
        realized_cap.name = "realized_cap_proxy"
        return realized_cap

    def compute_realized_price(self, realized_cap: float, supply: float) -> float:
        """
        Realized price = Realized Cap / circulating supply.
        This is the average cost basis per BTC.

        Args:
            realized_cap: Realized Cap in USD
            supply:       BTC in circulation

        Returns:
            USD price per BTC
        """
        if supply <= 0:
            return 0.0
        return realized_cap / supply

    def compute_mvrv_ratio(self, market_cap: float, realized_cap: float) -> float:
        """
        MVRV = Market Cap / Realized Cap.

        Interpretation:
          > 3.0: historically signals market tops (2013: 5.8, 2017: 4.7, 2021: 3.5)
          1.0-3.0: normal bull market range
          < 1.0: historically signals bottoms (buy zone)

        Args:
            market_cap:   Current market cap in USD
            realized_cap: Realized Cap in USD (cost basis aggregate)

        Returns:
            MVRV ratio (dimensionless)
        """
        if realized_cap <= 0:
            return 1.0
        return market_cap / realized_cap

    def compute_mvrv_z_score(
        self,
        market_cap: float,
        realized_cap: float,
        mvrv_history: "pd.Series",
    ) -> float:
        """
        MVRV Z-Score = (MVRV - mean(MVRV_history)) / std(MVRV_history)

        More robust than raw MVRV because it normalizes for different bull cycles.
        Historical extremes:
          Z > 7:  Extreme overvaluation (Jan 2021 top: 7.2, Dec 2017: 8.8)
          Z < 0:  Undervaluation / accumulation zone
          Z = 0-2: Early/mid bull market

        Args:
            market_cap:   Current market cap in USD
            realized_cap: Current realized cap in USD
            mvrv_history: Historical MVRV ratio series

        Returns:
            Z-score float
        """
        current_mvrv = self.compute_mvrv_ratio(market_cap, realized_cap)

        if not HAS_PANDAS or mvrv_history.empty:
            # Fallback: use known approximate mean/std from Bitcoin history
            hist_mean = 1.75
            hist_std = 1.20
        else:
            hist_mean = float(mvrv_history.mean())
            hist_std = float(mvrv_history.std())
            if hist_std < 0.01:
                hist_std = 0.01

        z_score = (current_mvrv - hist_mean) / hist_std
        return round(z_score, 3)

    def compute_full_mvrv_data(self, timespan: str = "1years") -> MVRVData:
        """
        Compute complete MVRV dataset including Z-Score and NUPL.

        Args:
            timespan: Historical lookback ('1years', '2years', 'all')

        Returns:
            MVRVData dataclass
        """
        client = self._client
        price_history, volume_history = client.get_price_history(timespan=timespan)
        supply = client.get_total_supply()
        current_price = client.get_current_price()

        if current_price <= 0:
            current_price = float(price_history.iloc[-1]) if not price_history.empty else 0.0

        market_cap = current_price * supply

        # Realized Cap proxy
        realized_cap_series = self.compute_realized_cap_proxy(price_history, volume_history)
        realized_cap = float(realized_cap_series.iloc[-1]) if not realized_cap_series.empty else market_cap * 0.6

        # MVRV ratio history
        if HAS_PANDAS and not price_history.empty:
            # Compute MVRV history
            price_aligned = price_history.reindex(realized_cap_series.index, method="ffill")
            mvrv_history = (price_aligned * supply) / realized_cap_series.replace(0, float("nan"))
            mvrv_history = mvrv_history.dropna()
        else:
            mvrv_history = pd.Series(dtype=float) if HAS_PANDAS else {}

        mvrv_ratio = self.compute_mvrv_ratio(market_cap, realized_cap)
        z_score = self.compute_mvrv_z_score(market_cap, realized_cap, mvrv_history)
        realized_price = self.compute_realized_price(realized_cap, supply)

        # NUPL
        nupl = (market_cap - realized_cap) / market_cap if market_cap > 0 else 0
        nupl_zone = NUPLAnalyzer.classify_nupl_static(nupl)

        return MVRVData(
            timestamp=datetime.now(timezone.utc),
            market_cap=market_cap,
            realized_cap=realized_cap,
            mvrv_ratio=round(mvrv_ratio, 3),
            mvrv_z_score=round(z_score, 3),
            realized_price=round(realized_price, 2),
            nupl=round(nupl, 4),
            nupl_zone=nupl_zone,
        )


# ---------------------------------------------------------------------------
# NVTAnalyzer
# ---------------------------------------------------------------------------

class NVTAnalyzer:
    """
    Network Value to Transactions ratio analysis.
    NVT is like a P/E ratio for Bitcoin: price / network utility.
    """

    def __init__(self) -> None:
        self._client = BlockchainInfoClient()

    def compute_nvt_ratio(self, market_cap: float, daily_tx_volume: float) -> float:
        """
        NVT = Market Cap / Daily Transaction Volume (USD).

        Analogy: P/E ratio for Bitcoin.
        Low NVT = undervalued relative to network usage.
        High NVT = overvalued relative to network usage.

        Args:
            market_cap:        Current BTC market cap in USD
            daily_tx_volume:   Daily on-chain transaction volume in USD

        Returns:
            NVT ratio (dimensionless)
        """
        if daily_tx_volume <= 0:
            return float("inf")
        return market_cap / daily_tx_volume

    def compute_nvt_signal(
        self, nvt_history: "pd.Series", smoothing: int = 90
    ) -> "pd.Series":
        """
        NVT Signal (Willy Woo's version) = NVT smoothed with 90-day moving average.

        The 90-day smoothing of the denominator (transaction volume) removes
        short-term transaction spikes and reveals the underlying trend.

        Args:
            nvt_history: Daily NVT ratio time series
            smoothing:   Window for moving average (default 90 days)

        Returns:
            Smoothed NVT Signal series
        """
        if not HAS_PANDAS or nvt_history.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        nvt_signal = nvt_history.rolling(window=smoothing, min_periods=smoothing // 4).mean()
        nvt_signal.name = "nvt_signal"
        return nvt_signal

    def compute_nvt_golden_cross(
        self, nvt_signal: "pd.Series", short_window: int = 28, long_window: int = 90
    ) -> "pd.Series":
        """
        NVT Golden Cross: short MA vs long MA crossover signals.

        Bullish: short MA crosses above long MA → coins accumulating value
        Bearish: short MA crosses below long MA → value destruction signal

        Args:
            nvt_signal:   NVT Signal series
            short_window: Short MA window (default 28 days)
            long_window:  Long MA window (default 90 days)

        Returns:
            Series of signals: +1 (bullish cross), -1 (bearish cross), 0 (neutral)
        """
        if not HAS_PANDAS or nvt_signal.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        short_ma = nvt_signal.rolling(window=short_window, min_periods=short_window // 4).mean()
        long_ma = nvt_signal.rolling(window=long_window, min_periods=long_window // 4).mean()

        # Crossover detection
        cross = pd.Series(0.0, index=nvt_signal.index)
        prev_diff = (short_ma - long_ma).shift(1)
        curr_diff = short_ma - long_ma

        cross[(prev_diff < 0) & (curr_diff > 0)] = 1.0   # Bullish cross
        cross[(prev_diff > 0) & (curr_diff < 0)] = -1.0  # Bearish cross
        cross.name = "nvt_golden_cross"
        return cross

    def classify_nvt_regime(self, nvt: float) -> str:
        """
        Classify NVT into regime categories.

        Historical BTC NVT ranges:
          UNDERVALUED:  NVT < 65  (accumulation, network growing faster than price)
          FAIR_VALUE:   65-150    (healthy market)
          OVERVALUED:   NVT > 150 (price running ahead of on-chain activity)

        Args:
            nvt: Current NVT ratio

        Returns:
            Regime string
        """
        if nvt < NVT_UNDERVALUED:
            return "UNDERVALUED"
        elif nvt < NVT_OVERVALUED:
            return "FAIR_VALUE"
        else:
            return "OVERVALUED"

    def compute_nvt_full_history(
        self, timespan: str = "1years"
    ) -> Tuple["pd.Series", "pd.Series", str]:
        """
        Compute full NVT and NVT Signal history.

        Returns:
            (nvt_series, nvt_signal_series, current_regime)
        """
        client = self._client
        price_history, _ = client.get_price_history(timespan=timespan)
        tx_volume = client.get_transaction_volume_history(timespan=timespan)
        supply = client.get_total_supply()

        if not HAS_PANDAS or price_history.empty or tx_volume.empty:
            empty = pd.Series(dtype=float) if HAS_PANDAS else {}
            return empty, empty, "UNKNOWN"  # type: ignore[return-value]

        # Align series
        df = pd.DataFrame({
            "price": price_history,
            "tx_volume": tx_volume,
        }).dropna()

        if df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), "UNKNOWN"

        market_cap = df["price"] * supply
        nvt = market_cap / df["tx_volume"].replace(0, float("nan"))
        nvt.name = "nvt_ratio"

        nvt_signal = self.compute_nvt_signal(nvt, smoothing=90)
        current_nvt = float(nvt.iloc[-1]) if not nvt.empty else 100.0
        regime = self.classify_nvt_regime(current_nvt)

        return nvt, nvt_signal, regime


# ---------------------------------------------------------------------------
# SOPRAnalyzer
# ---------------------------------------------------------------------------

class SOPRAnalyzer:
    """
    Spent Output Profit Ratio (SOPR) analysis.

    True SOPR requires full UTXO set data (not available for free).
    We compute a proxy from price momentum and volume-weighted average price.

    SOPR > 1: Coins being sold at profit (bullish in bull, bearish signal at extremes)
    SOPR < 1: Coins being sold at loss (bearish, but capitulation = bottom signal)
    SOPR = 1: Break-even; historically acts as support/resistance
    """

    def compute_sopr_proxy(
        self,
        price_history: "pd.Series",
        volume_history: "pd.Series",
        lookback: int = 30,
    ) -> "pd.Series":
        """
        Approximate SOPR using 30-day VWAP comparison.

        Methodology:
          - 30-day VWAP approximates average cost basis of recently-moved coins
          - SOPR proxy = current_price / VWAP(30d)
          - > 1: spending at profit, < 1: spending at loss

        This captures the same behavioral signal as true SOPR but with
        only publicly available price and volume data.

        Args:
            price_history:  Daily BTC price
            volume_history: Daily USD volume
            lookback:       VWAP window in days (default 30)

        Returns:
            SOPR proxy series
        """
        if not HAS_PANDAS or price_history.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        df = pd.DataFrame({"price": price_history, "volume": volume_history}).dropna()
        if df.empty:
            return pd.Series(dtype=float)

        # Rolling VWAP
        pv = df["price"] * df["volume"]
        window = min(lookback, len(df))
        vwap = pv.rolling(window=window, min_periods=window // 4).sum() / \
               df["volume"].rolling(window=window, min_periods=window // 4).sum().replace(0, float("nan"))

        sopr_proxy = df["price"] / vwap.replace(0, float("nan"))
        sopr_proxy.name = "sopr_proxy"
        return sopr_proxy

    def compute_adjusted_sopr(
        self, sopr: "pd.Series", smoothing: int = 14
    ) -> "pd.Series":
        """
        Adjusted SOPR: smooth to filter noise from short-term transactions.

        True aSOPR excludes same-day transactions (short-term holder noise).
        Our proxy uses 14-day smoothing to remove intraday volatility.

        Args:
            sopr:      Raw SOPR proxy series
            smoothing: Smoothing window in days (default 14)

        Returns:
            Adjusted SOPR series
        """
        if not HAS_PANDAS or sopr.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        adj_sopr = sopr.rolling(window=smoothing, min_periods=smoothing // 4).mean()
        adj_sopr.name = "adjusted_sopr"
        return adj_sopr

    def interpret_sopr(self, sopr: float, sopr_trend: Optional[float] = None) -> str:
        """
        Interpret current SOPR value with optional trend context.

        Market dynamics:
          - Bull market + SOPR > 1: normal profit-taking, healthy
          - Bear market + SOPR < 1: spending at loss, potential capitulation
          - SOPR = 1 + bounce: breakeven acts as support (accumulation)
          - SOPR = 1 + rejection: breakeven acts as resistance (distribution)

        Args:
            sopr:       Current SOPR value
            sopr_trend: Optional trend direction (+1 rising, -1 falling)

        Returns:
            Interpretation string
        """
        if sopr > 1.05:
            if sopr_trend is not None and sopr_trend > 0:
                return "BULL_MARKET_PROFIT_TAKING"
            return "SPENDING_AT_PROFIT"
        elif sopr > 1.0:
            return "SLIGHT_PROFIT_LOCKING"
        elif sopr > 0.97:
            if sopr_trend is not None and sopr_trend > 0:
                return "BREAKEVEN_SUPPORT_BOUNCE"
            elif sopr_trend is not None and sopr_trend < 0:
                return "BREAKEVEN_RESISTANCE_REJECTION"
            return "NEAR_BREAKEVEN"
        elif sopr > 0.90:
            return "SPENDING_AT_LOSS_MODERATE"
        else:
            return "CAPITULATION_SIGNAL"

    def compute_sopr_history(
        self, timespan: str = "1years"
    ) -> Tuple["pd.Series", "pd.Series", str]:
        """
        Compute full SOPR and adjusted SOPR history.

        Returns:
            (sopr_series, adj_sopr_series, current_interpretation)
        """
        client = BlockchainInfoClient()
        price, volume = client.get_price_history(timespan=timespan)

        sopr = self.compute_sopr_proxy(price, volume, lookback=30)
        adj_sopr = self.compute_adjusted_sopr(sopr, smoothing=14)

        current_sopr = float(sopr.iloc[-1]) if not sopr.empty else 1.0
        prev_sopr = float(sopr.iloc[-7]) if len(sopr) >= 7 else current_sopr
        trend = 1.0 if current_sopr > prev_sopr else (-1.0 if current_sopr < prev_sopr else 0.0)

        interpretation = self.interpret_sopr(current_sopr, trend)
        return sopr, adj_sopr, interpretation


# ---------------------------------------------------------------------------
# PuellMultiple
# ---------------------------------------------------------------------------

class PuellMultiple:
    """
    Puell Multiple: miner revenue relative to 365-day average.

    High Puell → miners over-earning → likely to sell → sell pressure → top signal.
    Low Puell  → miners under-earning → capitulation → bottom signal.
    """

    def __init__(self) -> None:
        self._client = BlockchainInfoClient()

    def compute_puell_multiple(
        self, daily_miner_revenue: float, ma_365: float
    ) -> float:
        """
        Puell Multiple = daily_miner_revenue / 365d_MA(daily_miner_revenue)

        Args:
            daily_miner_revenue: Today's miner revenue in USD
            ma_365:              365-day moving average of daily miner revenue

        Returns:
            Puell Multiple (>4 = sell zone, <0.5 = buy zone)
        """
        if ma_365 <= 0:
            return 1.0
        return daily_miner_revenue / ma_365

    def get_puell_history(self, timespan: str = "2years") -> "pd.Series":
        """
        Compute Puell Multiple history from Blockchain.info miner revenue data.

        Args:
            timespan: Historical window

        Returns:
            pd.Series of daily Puell Multiple values
        """
        revenue = self._client.get_miners_revenue_history(timespan=timespan)

        if not HAS_PANDAS or revenue.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        # 365-day rolling average
        window = min(365, len(revenue))
        ma_365 = revenue.rolling(window=window, min_periods=window // 4).mean()
        puell = revenue / ma_365.replace(0, float("nan"))
        puell.name = "puell_multiple"
        return puell.dropna()

    def get_current_puell(self) -> Tuple[float, str]:
        """
        Get current Puell Multiple and zone classification.

        Returns:
            (puell_value, zone_string)
        """
        puell_history = self.get_puell_history(timespan="2years")
        if puell_history.empty:
            # Fallback: compute from current stats
            stats_data = _rate_limited_get(BLOCKCHAIN_INFO_STATS)
            if stats_data and isinstance(stats_data, dict):
                rev_usd = float(stats_data.get("miners_revenue_usd") or 0)
                # Approximate 365d MA from market data
                ma_approx = rev_usd * 0.85  # rough placeholder
                puell = self.compute_puell_multiple(rev_usd, ma_approx)
            else:
                puell = 1.0
        else:
            puell = float(puell_history.iloc[-1])

        zone = self.classify_puell_zone(puell)
        return round(puell, 3), zone

    def classify_puell_zone(self, puell: float) -> str:
        """
        Classify Puell Multiple into market zones.

        Args:
            puell: Current Puell Multiple value

        Returns:
            Zone string: SELL_ZONE | CAUTION | NEUTRAL | BUY_ZONE
        """
        if puell > PUELL_SELL_ZONE:
            return "SELL_ZONE"
        elif puell > 2.0:
            return "CAUTION"
        elif puell > PUELL_BUY_ZONE:
            return "NEUTRAL"
        else:
            return "BUY_ZONE"


# ---------------------------------------------------------------------------
# CoinDaysDestroyed
# ---------------------------------------------------------------------------

class CoinDaysDestroyed:
    """
    Coin Days Destroyed (CDD): measure of old coins moving.

    CDD = Σ(coins_moved × coin_age_days)
    High CDD → long-term holders selling → potentially bearish signal near tops.
    Low CDD  → HODLers holding → accumulation phase.
    """

    def __init__(self) -> None:
        self._client = BlockchainInfoClient()

    def compute_cdd(
        self, tx_volume_btc: float, avg_coin_age_days: float
    ) -> float:
        """
        Compute approximate CDD for a single day.

        True CDD requires individual UTXO ages. We proxy using:
          CDD ≈ tx_volume_btc × avg_coin_age_estimate

        Args:
            tx_volume_btc:    BTC volume moved on-chain today
            avg_coin_age_days: Estimated average age of moved coins

        Returns:
            Coin Days Destroyed estimate
        """
        return tx_volume_btc * avg_coin_age_days

    def compute_binary_cdd(
        self, cdd_series: "pd.Series", window: int = 365
    ) -> "pd.Series":
        """
        Binary CDD (bCDD) = CDD / 365d_MA(CDD)

        Normalizes CDD by its long-term average, making it comparable
        across different market cap levels and Bitcoin supply amounts.

        Args:
            cdd_series: Daily CDD time series
            window:     Moving average window (default 365 days)

        Returns:
            bCDD series (>1 = above average old coin activity)
        """
        if not HAS_PANDAS or cdd_series.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        ma = cdd_series.rolling(window=min(window, len(cdd_series)), min_periods=window // 4).mean()
        bcdd = cdd_series / ma.replace(0, float("nan"))
        bcdd.name = "binary_cdd"
        return bcdd.dropna()

    def compute_cdd_proxy_history(
        self, timespan: str = "1years"
    ) -> "pd.Series":
        """
        Build CDD proxy from on-chain transaction data.

        Methodology:
          - Transaction volume (BTC) × estimated coin age
          - Coin age estimation: based on UTXO count and average output age
          - Proxy: use n-transactions and market cap as inputs

        Returns:
            pd.Series of estimated daily CDD
        """
        client = self._client
        tx_count = client.get_chart_data("n-transactions", timespan=timespan)
        trade_vol = client.get_chart_data("trade-volume", timespan=timespan)  # USD

        if not HAS_PANDAS or tx_count.empty:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        price, _ = client.get_price_history(timespan=timespan)
        if price.empty:
            return pd.Series(dtype=float)

        # Estimate BTC moved: USD volume / price
        df = pd.DataFrame({"price": price, "usd_volume": trade_vol}).dropna()
        if df.empty:
            return pd.Series(dtype=float)

        btc_moved = df["usd_volume"] / df["price"].replace(0, float("nan"))

        # Estimate average coin age from UTXO-to-transaction ratio proxy
        # More UTXOs per transaction → older coins being spent
        # Default to 90-day average age estimate (BTC empirical average)
        avg_coin_age = 90.0

        cdd_proxy = btc_moved * avg_coin_age
        cdd_proxy.name = "cdd_proxy"
        return cdd_proxy.dropna()

    def compute_cdd_trend(self, cdd_history: "pd.Series") -> str:
        """
        Classify CDD trend direction.

        Rising CDD: experienced investors (HODLers) becoming active → often near tops.
        Falling CDD: HODLers sitting tight → accumulation or early bull phase.

        Args:
            cdd_history: Historical CDD series

        Returns:
            Trend string: RISING | STABLE | FALLING
        """
        if not HAS_PANDAS or len(cdd_history) < 30:
            return "UNKNOWN"

        recent = float(cdd_history.tail(7).mean())
        prior = float(cdd_history.tail(30).head(23).mean())

        if prior <= 0:
            return "UNKNOWN"

        pct_change = (recent - prior) / prior * 100
        if pct_change > 20:
            return "RISING_SHARPLY"
        elif pct_change > 5:
            return "RISING"
        elif pct_change < -20:
            return "FALLING_SHARPLY"
        elif pct_change < -5:
            return "FALLING"
        else:
            return "STABLE"


# ---------------------------------------------------------------------------
# NUPLAnalyzer
# ---------------------------------------------------------------------------

class NUPLAnalyzer:
    """
    Net Unrealized Profit/Loss analysis.

    NUPL = (Market Cap - Realized Cap) / Market Cap
    Measures the aggregate unrealized profit/loss of all BTC holders.
    """

    def compute_nupl(self, market_cap: float, realized_cap: float) -> float:
        """
        NUPL = (Market Cap - Realized Cap) / Market Cap

        Args:
            market_cap:   Current BTC market cap
            realized_cap: Realized Cap (aggregate cost basis)

        Returns:
            NUPL value (-∞ to 1)
        """
        if market_cap <= 0:
            return 0.0
        return (market_cap - realized_cap) / market_cap

    def classify_nupl(self, nupl: float) -> str:
        """
        Classify NUPL into market phase.

        Historical correlation with Bitcoin market cycles:
          < 0.0:       CAPITULATION  — all holders on average underwater
          0.0-0.25:    HOPE/FEAR     — early recovery, still fragile
          0.25-0.50:   OPTIMISM      — growing confidence
          0.50-0.75:   BELIEF        — bull market, euphoria building
          > 0.75:      EUPHORIA_GREED — danger zone, historically near tops

        Args:
            nupl: NUPL value

        Returns:
            Market phase string
        """
        return self.classify_nupl_static(nupl)

    @staticmethod
    def classify_nupl_static(nupl: float) -> str:
        """Static version for use from other classes."""
        if nupl < NUPL_CAPITULATION:
            return "CAPITULATION"
        elif nupl < NUPL_HOPE:
            return "HOPE_FEAR"
        elif nupl < NUPL_OPTIMISM:
            return "OPTIMISM"
        elif nupl < NUPL_BELIEF:
            return "BELIEF"
        else:
            return "EUPHORIA_GREED"

    def compute_nupl_history(
        self, timespan: str = "1years"
    ) -> Tuple["pd.Series", str]:
        """
        Compute NUPL history.

        Returns:
            (nupl_series, current_zone)
        """
        calc = RealizedCapCalculator()
        client = BlockchainInfoClient()

        price_history, volume_history = client.get_price_history(timespan=timespan)
        supply = client.get_total_supply()

        if not HAS_PANDAS or price_history.empty:
            return pd.Series(dtype=float), "UNKNOWN"

        realized_cap_series = calc.compute_realized_cap_proxy(price_history, volume_history)

        # Align
        price_aligned = price_history.reindex(realized_cap_series.index, method="ffill")
        market_cap_series = price_aligned * supply

        nupl_series = (market_cap_series - realized_cap_series) / market_cap_series.replace(0, float("nan"))
        nupl_series.name = "nupl"
        nupl_series = nupl_series.dropna()

        current_nupl = float(nupl_series.iloc[-1]) if not nupl_series.empty else 0.0
        zone = self.classify_nupl(current_nupl)

        return nupl_series, zone


# ---------------------------------------------------------------------------
# ActiveAddressAnalyzer
# ---------------------------------------------------------------------------

class ActiveAddressAnalyzer:
    """
    Bitcoin network activity analysis via active address metrics.
    Growing active addresses = adoption; declining = bear market.
    """

    def __init__(self) -> None:
        self._client = BlockchainInfoClient()

    def get_active_addresses_history(
        self, timespan: str = "1years"
    ) -> "pd.Series":
        """
        Historical unique active address count from Blockchain.info.

        Returns:
            pd.Series indexed by datetime
        """
        return self._client.get_active_addresses_history(timespan=timespan)

    def compute_address_momentum(
        self,
        history: "pd.Series",
        ma_short: int = 7,
        ma_long: int = 90,
    ) -> float:
        """
        Address momentum: short MA / long MA - 1.

        > 0: Short-term address activity above long-term trend (adoption growing)
        < 0: Below trend (declining activity = bear signal)

        Args:
            history:  Historical active address series
            ma_short: Short MA window (default 7 days)
            ma_long:  Long MA window (default 90 days)

        Returns:
            Momentum float (-1 to +1 typical range)
        """
        if not HAS_PANDAS or len(history) < ma_long:
            return 0.0

        short_ma = float(history.tail(ma_short).mean())
        long_ma = float(history.tail(ma_long).mean())

        if long_ma <= 0:
            return 0.0
        return (short_ma - long_ma) / long_ma

    def compute_address_z_score(
        self, current: float, history: "pd.Series"
    ) -> float:
        """
        Z-score: how active is the network vs historical baseline?

        Args:
            current: Current daily active address count
            history: Historical active address series

        Returns:
            Z-score (>2: unusually high activity, <-2: unusually low)
        """
        if not HAS_PANDAS or history.empty:
            return 0.0

        mean = float(history.mean())
        std = float(history.std())
        if std < 1.0:
            return 0.0
        return (current - mean) / std

    def get_address_metrics(
        self, timespan: str = "1years"
    ) -> Dict[str, Any]:
        """
        Full active address analysis.

        Returns:
            dict with current count, momentum, z-score, trend
        """
        history = self.get_active_addresses_history(timespan=timespan)
        if history.empty:
            return {
                "current_addresses": 0,
                "momentum": 0.0,
                "z_score": 0.0,
                "trend": "UNKNOWN",
            }

        current = float(history.iloc[-1])
        momentum = self.compute_address_momentum(history)
        z_score = self.compute_address_z_score(current, history)

        # Classify trend
        if momentum > 0.10:
            trend = "STRONG_GROWTH"
        elif momentum > 0.02:
            trend = "GROWING"
        elif momentum > -0.02:
            trend = "STABLE"
        elif momentum > -0.10:
            trend = "DECLINING"
        else:
            trend = "STRONG_DECLINE"

        return {
            "current_addresses": int(current),
            "7d_avg": int(float(history.tail(7).mean())),
            "90d_avg": int(float(history.tail(90).mean())),
            "momentum": round(momentum, 4),
            "z_score": round(z_score, 3),
            "trend": trend,
            "peak_90d": int(float(history.tail(90).max())),
            "trough_90d": int(float(history.tail(90).min())),
        }


# ---------------------------------------------------------------------------
# OnChainCompositeSignal
# ---------------------------------------------------------------------------

class OnChainCompositeSignal:
    """
    Combine all on-chain signals into one actionable composite score.

    Score 0-100 where:
      > 70: EXTREME_GREED (reduce exposure, historically near tops)
      55-70: GREED
      45-55: NEUTRAL
      30-45: FEAR
      < 30:  EXTREME_FEAR (accumulate, historically near bottoms)

    Weights:
      MVRV Z-Score:     30%
      Puell Multiple:   20%
      SOPR:             20%
      NVT Signal:       15%
      Address Momentum: 15%
    """

    def __init__(self) -> None:
        self._client = BlockchainInfoClient()
        self._mvrv_calc = RealizedCapCalculator()
        self._nvt = NVTAnalyzer()
        self._sopr = SOPRAnalyzer()
        self._puell = PuellMultiple()
        self._addr = ActiveAddressAnalyzer()
        self._nupl = NUPLAnalyzer()

    def _normalize_mvrv_z(self, z_score: float) -> float:
        """
        Normalize MVRV Z-Score to 0-100 scale.
        Z > 7: near top (score 90+), Z < 0: near bottom (score <30)
        """
        # Historical range: approximately -1 to +10
        # Map: -2 → 5, 0 → 30, 3 → 55, 5 → 75, 7+ → 95
        if z_score >= 7.0:
            return 95.0
        elif z_score >= 5.0:
            return 75.0 + (z_score - 5.0) * 10.0
        elif z_score >= 3.0:
            return 55.0 + (z_score - 3.0) * 10.0
        elif z_score >= 0.0:
            return 30.0 + (z_score / 3.0) * 25.0
        else:
            return max(0.0, 30.0 + z_score * 10.0)

    def _normalize_puell(self, puell: float) -> float:
        """
        Normalize Puell Multiple to 0-100.
        > 4 → ~90 (sell zone), < 0.5 → ~10 (buy zone), 1.0 → 50
        """
        if puell >= 4.0:
            return min(95.0, 80.0 + (puell - 4.0) * 5.0)
        elif puell >= 2.0:
            return 65.0 + (puell - 2.0) * 7.5
        elif puell >= 1.0:
            return 50.0 + (puell - 1.0) * 15.0
        elif puell >= 0.5:
            return 20.0 + (puell - 0.5) * 60.0
        else:
            return max(5.0, 20.0 - (0.5 - puell) * 30.0)

    def _normalize_sopr(self, sopr: float) -> float:
        """
        Normalize SOPR to 0-100.
        > 1.05 → greed (>60), = 1.0 → neutral (50), < 0.95 → fear (<35)
        """
        if sopr >= 1.10:
            return min(90.0, 70.0 + (sopr - 1.10) * 200.0)
        elif sopr >= 1.0:
            return 50.0 + (sopr - 1.0) * 200.0
        elif sopr >= 0.95:
            return 30.0 + (sopr - 0.95) * 400.0
        else:
            return max(5.0, 30.0 - (0.95 - sopr) * 400.0)

    def _normalize_nvt(self, nvt: float) -> float:
        """
        Normalize NVT Signal to 0-100.
        High NVT = overvalued = greed. Low NVT = undervalued = fear (buy).
        Note: direction is inverted vs other metrics for consistent scoring.
        """
        if nvt >= NVT_OVERVALUED:
            return min(95.0, 75.0 + (nvt - NVT_OVERVALUED) * 0.1)
        elif nvt >= NVT_UNDERVALUED:
            norm = (nvt - NVT_UNDERVALUED) / (NVT_OVERVALUED - NVT_UNDERVALUED)
            return 25.0 + norm * 50.0
        else:
            return max(5.0, 25.0 - (NVT_UNDERVALUED - nvt) * 0.3)

    def _normalize_address_momentum(self, momentum: float) -> float:
        """
        Normalize address momentum to 0-100.
        Rising addresses = adoption = bullish (higher score).
        """
        # momentum ranges typically -0.3 to +0.3
        normalized = 50.0 + momentum * 200.0
        return max(5.0, min(95.0, normalized))

    def _get_fear_greed(self) -> int:
        """Fetch Fear & Greed Index from Alternative.me."""
        data = _rate_limited_get(FEAR_GREED_URL, params={"limit": "1"})
        if data and isinstance(data, dict):
            fng_data = data.get("data", [{}])
            if fng_data:
                return int(fng_data[0].get("value") or 50)
        return 50

    def compute_composite_bull_bear_score(
        self, timespan: str = "1years"
    ) -> OnChainScore:
        """
        Compute the full composite on-chain bull/bear score.

        Fetches all required data, computes all signals, and combines
        with the specified weights.

        Args:
            timespan: Historical lookback for signal computation

        Returns:
            OnChainScore dataclass with all components
        """
        logger.info("Computing composite on-chain score (timespan=%s)...", timespan)
        now = datetime.now(timezone.utc)

        # 1. MVRV Z-Score (30%)
        logger.info("  Fetching MVRV data...")
        mvrv_data = self._mvrv_calc.compute_full_mvrv_data(timespan=timespan)
        mvrv_z = mvrv_data.mvrv_z_score
        mvrv_component = self._normalize_mvrv_z(mvrv_z)

        # 2. Puell Multiple (20%)
        logger.info("  Computing Puell Multiple...")
        puell_val, puell_zone = self._puell.get_current_puell()
        puell_component = self._normalize_puell(puell_val)

        # 3. SOPR (20%)
        logger.info("  Computing SOPR...")
        price_h, vol_h = self._client.get_price_history(timespan=timespan)
        sopr_series = self._sopr.compute_sopr_proxy(price_h, vol_h)
        current_sopr = float(sopr_series.iloc[-1]) if not sopr_series.empty else 1.0
        sopr_component = self._normalize_sopr(current_sopr)

        # 4. NVT Signal (15%)
        logger.info("  Computing NVT Signal...")
        nvt_series, nvt_signal_series, nvt_regime = self._nvt.compute_nvt_full_history(timespan=timespan)
        current_nvt = float(nvt_signal_series.iloc[-1]) if not nvt_signal_series.empty else 100.0
        nvt_component = self._normalize_nvt(current_nvt)

        # 5. Active Address Momentum (15%)
        logger.info("  Computing address momentum...")
        addr_metrics = self._addr.get_address_metrics(timespan=timespan)
        addr_momentum = float(addr_metrics.get("momentum") or 0.0)
        addr_component = self._normalize_address_momentum(addr_momentum)

        # Fear & Greed
        logger.info("  Fetching Fear & Greed Index...")
        fear_greed = self._get_fear_greed()

        # Weighted composite
        composite = (
            mvrv_component * 0.30
            + puell_component * 0.20
            + sopr_component * 0.20
            + nvt_component * 0.15
            + addr_component * 0.15
        )

        return OnChainScore(
            timestamp=now,
            composite_score=round(composite, 1),
            mvrv_z_score=mvrv_z,
            mvrv_component=round(mvrv_component, 1),
            puell_multiple=puell_val,
            puell_component=round(puell_component, 1),
            sopr_value=round(current_sopr, 4),
            sopr_component=round(sopr_component, 1),
            nvt_signal=round(current_nvt, 1),
            nvt_component=round(nvt_component, 1),
            address_momentum=round(addr_momentum, 4),
            address_component=round(addr_component, 1),
            cycle_position=self._infer_cycle_position(mvrv_z, mvrv_data.nupl, puell_val),
            nupl=mvrv_data.nupl,
            fear_greed_index=fear_greed,
        )

    def _infer_cycle_position(
        self, mvrv_z: float, nupl: float, puell: float
    ) -> str:
        """
        Infer macro market cycle position from multiple signals.

        Returns:
            ACCUMULATION | EARLY_BULL | BULL | LATE_BULL | BEAR
        """
        if mvrv_z < 0 and nupl < 0:
            return "ACCUMULATION"
        elif mvrv_z < 2 and nupl < 0.25:
            return "EARLY_BULL"
        elif mvrv_z < 5 and nupl < 0.6:
            return "BULL"
        elif mvrv_z >= 5 or nupl >= 0.6 or puell >= 3.0:
            return "LATE_BULL"
        else:
            return "BEAR"

    def get_market_cycle_position(self) -> str:
        """
        Convenience method: compute composite score and return cycle position.

        Returns:
            Cycle position string
        """
        score = self.compute_composite_bull_bear_score(timespan="1years")
        return score.cycle_position

    def generate_report(self) -> str:
        """
        Generate a formatted narrative on-chain analytics report.

        Returns:
            Multi-line formatted string
        """
        score = self.compute_composite_bull_bear_score(timespan="1years")
        mvrv_data = self._mvrv_calc.compute_full_mvrv_data(timespan="1years")

        lines = ["=" * 70]
        lines.append("SENTINEL — Bitcoin On-Chain Analytics Report")
        lines.append(f"Generated: {score.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 70)

        # Overall sentiment
        sentiment_emoji = {
            "EXTREME_GREED": "[EXTREME GREED]",
            "GREED": "[GREED]",
            "NEUTRAL": "[NEUTRAL]",
            "FEAR": "[FEAR]",
            "EXTREME_FEAR": "[EXTREME FEAR]",
        }.get(score.sentiment, "[UNKNOWN]")

        lines.append(f"\nComposite Score: {score.composite_score:.1f}/100  {sentiment_emoji}")
        lines.append(f"Cycle Position:  {score.cycle_position}")
        lines.append(f"Fear & Greed:    {score.fear_greed_index}/100")

        # Market overview
        lines.append("\n--- Market Structure ---")
        current_price = self._client.get_current_price()
        if current_price > 0:
            lines.append(f"  BTC Price:          ${current_price:>12,.0f}")
        if mvrv_data.market_cap > 0:
            lines.append(f"  Market Cap:         ${mvrv_data.market_cap/1e9:>10.1f}B")
            lines.append(f"  Realized Cap:       ${mvrv_data.realized_cap/1e9:>10.1f}B")
            lines.append(f"  Realized Price:     ${mvrv_data.realized_price:>12,.0f}")

        # Component breakdown
        lines.append("\n--- On-Chain Signal Components ---")
        components = [
            ("MVRV Z-Score", f"{score.mvrv_z_score:>+7.3f}",
             score.mvrv_component, 30,
             "SELL ZONE" if score.mvrv_z_score > 7 else "NEUTRAL" if score.mvrv_z_score > 0 else "BUY ZONE"),
            ("Puell Multiple", f"{score.puell_multiple:>7.3f}",
             score.puell_component, 20,
             self._puell.classify_puell_zone(score.puell_multiple)),
            ("SOPR", f"{score.sopr_value:>7.4f}",
             score.sopr_component, 20,
             self._sopr.interpret_sopr(score.sopr_value)),
            ("NVT Signal", f"{score.nvt_signal:>7.1f}",
             score.nvt_component, 15,
             self._nvt.classify_nvt_regime(score.nvt_signal)),
            ("Addr Momentum", f"{score.address_momentum:>+7.4f}",
             score.address_component, 15, ""),
        ]

        lines.append(f"  {'Metric':<16} {'Value':>8}  {'Component':>9}  {'Weight':>6}  Zone")
        lines.append(f"  {'-'*16} {'-----':>8}  {'---------':>9}  {'------':>6}  ----")
        for name, val, comp, weight, zone in components:
            lines.append(
                f"  {name:<16} {val:>8}  {comp:>8.1f}   {weight:>5}%  {zone}"
            )

        lines.append(f"\n  {'COMPOSITE':.<16} {'':>8}  {score.composite_score:>8.1f}   {'100':>5}%")

        # NUPL
        lines.append("\n--- NUPL (Unrealized Profit/Loss) ---")
        nupl_zone = NUPLAnalyzer.classify_nupl_static(score.nupl)
        lines.append(f"  NUPL:  {score.nupl:>+.4f}  [{nupl_zone}]")
        nupl_descriptions = {
            "CAPITULATION": "Market in extreme fear; historically excellent long-term entry.",
            "HOPE_FEAR": "Early recovery; fragile confidence, accumulation zone.",
            "OPTIMISM": "Bull market building; healthy risk-on sentiment.",
            "BELIEF": "Strong bull market; approach with position discipline.",
            "EUPHORIA_GREED": "DANGER: Historical top territory. Reduce exposure.",
        }
        lines.append(f"  {nupl_descriptions.get(nupl_zone, '')}")

        # Network stats
        try:
            stats = BlockchainInfoClient().get_network_stats()
            lines.append("\n--- Network Health ---")
            lines.append(f"  Hashrate:           {stats.hashrate_ehash:>8.2f} EH/s")
            lines.append(f"  Mempool TXs:        {stats.mempool_tx_count:>8,}")
            lines.append(f"  Block Interval:     {stats.avg_block_time_minutes:>8.2f} min")
            lines.append(f"  Miners Revenue:     ${stats.miners_revenue_usd:>10,.0f}/day")
        except Exception:
            pass

        # Actionable summary
        lines.append("\n--- Actionable Summary ---")
        if score.composite_score > 70:
            lines.append("  SIGNAL: REDUCE EXPOSURE")
            lines.append("  Rationale: Multiple on-chain metrics in extreme territory.")
            lines.append("  Action: Scale down positions, move profits to stables.")
        elif score.composite_score > 55:
            lines.append("  SIGNAL: HOLD / LIGHT TRIMMING")
            lines.append("  Rationale: Greed signals elevated but not extreme.")
            lines.append("  Action: Maintain core positions, set trailing stops.")
        elif score.composite_score >= 45:
            lines.append("  SIGNAL: HOLD / DCA")
            lines.append("  Rationale: Market near equilibrium.")
            lines.append("  Action: Continue systematic accumulation strategy.")
        elif score.composite_score >= 30:
            lines.append("  SIGNAL: ACCUMULATE")
            lines.append("  Rationale: Fear signals present, long-term opportunity.")
            lines.append("  Action: Increase position size, DCA into weakness.")
        else:
            lines.append("  SIGNAL: STRONG ACCUMULATE")
            lines.append("  Rationale: EXTREME FEAR — historically optimal entry zone.")
            lines.append("  Action: Maximum accumulation within risk parameters.")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OnChainMetricsEngine (Orchestrator)
# ---------------------------------------------------------------------------

class OnChainMetricsEngine:
    """
    Primary orchestrator for all Bitcoin on-chain analytics.
    Aggregates all metrics into dashboards and manages data refresh.
    """

    def __init__(self) -> None:
        self._client = BlockchainInfoClient()
        self._mvrv_calc = RealizedCapCalculator()
        self._nvt = NVTAnalyzer()
        self._sopr = SOPRAnalyzer()
        self._puell = PuellMultiple()
        self._cdd = CoinDaysDestroyed()
        self._nupl = NUPLAnalyzer()
        self._addr = ActiveAddressAnalyzer()
        self._composite = OnChainCompositeSignal()

    def get_full_dashboard(self) -> Dict[str, Any]:
        """
        Compute all current on-chain metrics for a complete dashboard.

        Returns:
            dict with all metric values and interpretations
        """
        logger.info("Building full on-chain dashboard...")

        # Network stats
        stats = self._client.get_network_stats()
        current_price = self._client.get_current_price()

        # MVRV & Realized Cap
        mvrv_data = self._mvrv_calc.compute_full_mvrv_data(timespan="2years")

        # NUPL
        _, nupl_zone = self._nupl.compute_nupl_history(timespan="1years")

        # Puell Multiple
        puell_val, puell_zone = self._puell.get_current_puell()

        # SOPR
        price_h, vol_h = self._client.get_price_history(timespan="1years")
        sopr_series = self._sopr.compute_sopr_proxy(price_h, vol_h)
        current_sopr = float(sopr_series.iloc[-1]) if not sopr_series.empty else 1.0
        sopr_interp = self._sopr.interpret_sopr(current_sopr)

        # NVT
        nvt_s, nvt_sig_s, nvt_regime = self._nvt.compute_nvt_full_history(timespan="1years")
        current_nvt = float(nvt_sig_s.iloc[-1]) if not nvt_sig_s.empty else 100.0

        # Active Addresses
        addr_metrics = self._addr.get_address_metrics(timespan="1years")

        # CDD
        cdd_series = self._cdd.compute_cdd_proxy_history(timespan="1years")
        cdd_trend = self._cdd.compute_cdd_trend(cdd_series)
        bcdd = self._cdd.compute_binary_cdd(cdd_series)
        current_bcdd = float(bcdd.iloc[-1]) if not bcdd.empty else 1.0

        # Composite score
        composite = self._composite.compute_composite_bull_bear_score(timespan="1years")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price_usd": current_price,
            "network": {
                "hashrate_ehash": stats.hashrate_ehash,
                "difficulty": stats.difficulty,
                "mempool_tx_count": stats.mempool_tx_count,
                "avg_block_time_min": stats.avg_block_time_minutes,
                "total_btc": stats.total_btc,
                "miners_revenue_usd_daily": stats.miners_revenue_usd,
            },
            "mvrv": {
                "market_cap": mvrv_data.market_cap,
                "realized_cap": mvrv_data.realized_cap,
                "mvrv_ratio": mvrv_data.mvrv_ratio,
                "mvrv_z_score": mvrv_data.mvrv_z_score,
                "realized_price": mvrv_data.realized_price,
                "signal": (
                    "SELL_ZONE" if mvrv_data.mvrv_ratio > MVRV_TOP_THRESHOLD
                    else "BUY_ZONE" if mvrv_data.mvrv_ratio < MVRV_BOTTOM_THRESHOLD
                    else "NEUTRAL"
                ),
            },
            "nupl": {
                "value": mvrv_data.nupl,
                "zone": nupl_zone,
            },
            "puell_multiple": {
                "value": puell_val,
                "zone": puell_zone,
            },
            "sopr": {
                "value": current_sopr,
                "interpretation": sopr_interp,
            },
            "nvt": {
                "nvt_signal": current_nvt,
                "regime": nvt_regime,
            },
            "active_addresses": addr_metrics,
            "coin_days_destroyed": {
                "trend": cdd_trend,
                "binary_cdd": round(current_bcdd, 3),
                "signal": "BEARISH" if cdd_trend in ("RISING_SHARPLY", "RISING") else "NEUTRAL",
            },
            "composite": {
                "score": composite.composite_score,
                "sentiment": composite.sentiment,
                "cycle_position": composite.cycle_position,
                "fear_greed_index": composite.fear_greed_index,
            },
        }

    def get_historical_signals(
        self, lookback_days: int = 365
    ) -> "pd.DataFrame":
        """
        Build a historical signal DataFrame for all metrics.

        Args:
            lookback_days: Days of history to compute

        Returns:
            DataFrame with one row per day and all signal columns
        """
        if not HAS_PANDAS:
            logger.warning("pandas not available for historical signals")
            return pd.DataFrame() if HAS_PANDAS else []  # type: ignore[return-value]

        timespan_map = {30: "1months", 90: "3months", 180: "6months", 365: "1years", 730: "2years"}
        timespan = "1years"
        for days, ts in sorted(timespan_map.items()):
            if lookback_days <= days:
                timespan = ts
                break

        logger.info("Building historical signals for timespan=%s...", timespan)

        price_h, vol_h = self._client.get_price_history(timespan=timespan)
        if price_h.empty:
            return pd.DataFrame()

        supply = self._client.get_total_supply()

        # Realized Cap proxy
        realized_cap_s = self._mvrv_calc.compute_realized_cap_proxy(price_h, vol_h)

        # Align everything
        price_aligned = price_h.reindex(realized_cap_s.index, method="ffill")
        market_cap_s = price_aligned * supply

        # MVRV history
        mvrv_s = market_cap_s / realized_cap_s.replace(0, float("nan"))
        mvrv_mean = float(mvrv_s.mean())
        mvrv_std = float(mvrv_s.std())
        mvrv_z_s = (mvrv_s - mvrv_mean) / mvrv_std if mvrv_std > 0 else mvrv_s * 0

        # NUPL history
        nupl_s = (market_cap_s - realized_cap_s) / market_cap_s.replace(0, float("nan"))

        # SOPR proxy history
        sopr_s = self._sopr.compute_sopr_proxy(price_h, vol_h)
        sopr_aligned = sopr_s.reindex(realized_cap_s.index, method="ffill")

        # NVT
        tx_vol = self._client.get_transaction_volume_history(timespan=timespan)
        tx_aligned = tx_vol.reindex(realized_cap_s.index, method="ffill")
        nvt_s = market_cap_s / tx_aligned.replace(0, float("nan"))
        nvt_signal_s = self._nvt.compute_nvt_signal(nvt_s)

        # Active addresses
        addr_s = self._addr.get_active_addresses_history(timespan=timespan)
        addr_aligned = addr_s.reindex(realized_cap_s.index, method="ffill")

        # Build DataFrame
        df = pd.DataFrame({
            "price": price_aligned,
            "market_cap": market_cap_s,
            "realized_cap": realized_cap_s,
            "realized_price": realized_cap_s / supply,
            "mvrv_ratio": mvrv_s,
            "mvrv_z_score": mvrv_z_s,
            "nupl": nupl_s,
            "sopr_proxy": sopr_aligned,
            "nvt_ratio": nvt_s,
            "nvt_signal": nvt_signal_s.reindex(realized_cap_s.index, method="ffill"),
            "active_addresses": addr_aligned,
        }).dropna(how="all")

        # Add categorical signals
        df["mvrv_signal"] = df["mvrv_ratio"].apply(
            lambda x: "BUY" if x < 1.0 else ("SELL" if x > 3.0 else "HOLD")
        )
        df["nupl_zone"] = df["nupl"].apply(NUPLAnalyzer.classify_nupl_static)
        df["nvt_regime"] = df["nvt_signal"].apply(
            lambda x: self._nvt.classify_nvt_regime(x) if not math.isnan(x) else "UNKNOWN"
        )

        return df.tail(lookback_days)

    def run_daily_update(self) -> Dict[str, Any]:
        """
        Refresh all on-chain data and return updated dashboard.
        Clears expired cache entries before fetching.

        Returns:
            Updated dashboard dict
        """
        logger.info("Running daily on-chain update...")
        _CACHE.purge_expired()
        dashboard = self.get_full_dashboard()
        logger.info(
            "Daily update complete. BTC: $%s, MVRV: %s, Score: %s",
            f"{dashboard.get('price_usd', 0):,.0f}",
            dashboard.get("mvrv", {}).get("mvrv_ratio", "N/A"),
            dashboard.get("composite", {}).get("score", "N/A"),
        )
        return dashboard

    def export_signals(self, path: str) -> None:
        """
        Export current dashboard signals to JSON file.

        Args:
            path: Output file path
        """
        dashboard = self.get_full_dashboard()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dashboard, f, indent=2, default=str)
        logger.info("On-chain signals exported to %s", path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main() -> None:
    """
    SENTINEL On-Chain Metrics Demo:
      1. Fetch 1-year BTC on-chain data
      2. Compute MVRV Z-Score
      3. Compute NVT Signal
      4. Compute Puell Multiple
      5. Compute SOPR proxy
      6. Compute NUPL and zone
      7. Coin Days Destroyed trend
      8. Active Address momentum
      9. Composite score
      10. Full narrative report
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    _banner("SENTINEL — On-Chain Metrics V3 (dim_108)")
    print("  Fetching 1-year Bitcoin on-chain data from Blockchain.info + CoinGecko...")
    print("  (Free APIs — may take 20-30 seconds)")

    engine = OnChainMetricsEngine()
    client = BlockchainInfoClient()

    # 1. Network stats
    _banner("1. Network Statistics")
    stats = client.get_network_stats()
    current_price = client.get_current_price()
    print(f"  BTC Price:           ${current_price:>12,.0f}")
    print(f"  Hashrate:            {stats.hashrate_ehash:>10.2f} EH/s")
    print(f"  Difficulty:          {stats.difficulty:>10.2e}")
    print(f"  Mempool TXs:         {stats.mempool_tx_count:>10,}")
    print(f"  Block interval:      {stats.avg_block_time_minutes:>10.2f} min")
    print(f"  Total BTC:           {stats.total_btc:>10.4f} M")
    print(f"  Miners Revenue:      ${stats.miners_revenue_usd:>10,.0f}/day")

    # 2. MVRV Z-Score
    _banner("2. MVRV Ratio & Z-Score")
    mvrv_calc = RealizedCapCalculator()
    mvrv_data = mvrv_calc.compute_full_mvrv_data(timespan="2years")
    print(f"  Market Cap:          ${mvrv_data.market_cap/1e9:>10.2f}B")
    print(f"  Realized Cap:        ${mvrv_data.realized_cap/1e9:>10.2f}B")
    print(f"  Realized Price:      ${mvrv_data.realized_price:>12,.0f}")
    print(f"  MVRV Ratio:          {mvrv_data.mvrv_ratio:>10.3f}")
    print(f"  MVRV Z-Score:        {mvrv_data.mvrv_z_score:>10.3f}")
    mvrv_signal = (
        "SELL ZONE (>3.0)" if mvrv_data.mvrv_ratio > 3.0
        else "BUY ZONE (<1.0)" if mvrv_data.mvrv_ratio < 1.0
        else f"NEUTRAL ({MVRV_BOTTOM_THRESHOLD}-{MVRV_TOP_THRESHOLD})"
    )
    z_signal = (
        "EXTREME OVERVALUATION" if mvrv_data.mvrv_z_score > 7
        else "OVERVALUED" if mvrv_data.mvrv_z_score > 3
        else "NEUTRAL" if mvrv_data.mvrv_z_score > 0
        else "UNDERVALUED (ACCUMULATE)"
    )
    print(f"  MVRV Signal:         {mvrv_signal}")
    print(f"  Z-Score Signal:      {z_signal}")

    # 3. NVT Signal
    _banner("3. NVT Ratio & Signal")
    nvt_analyzer = NVTAnalyzer()
    nvt_s, nvt_sig_s, nvt_regime = nvt_analyzer.compute_nvt_full_history(timespan="1years")
    if not nvt_s.empty:
        current_nvt = float(nvt_s.iloc[-1])
        current_nvt_sig = float(nvt_sig_s.iloc[-1]) if not nvt_sig_s.empty else current_nvt
        print(f"  Current NVT:         {current_nvt:>10.1f}")
        print(f"  NVT Signal (90d MA): {current_nvt_sig:>10.1f}")
        print(f"  NVT Regime:          {nvt_regime}")
        print(f"  Threshold:           <{NVT_UNDERVALUED:.0f}=UNDERVALUED, >{NVT_OVERVALUED:.0f}=OVERVALUED")
    else:
        print("  [NVT data unavailable — check network]")

    # 4. Puell Multiple
    _banner("4. Puell Multiple (Miner Revenue)")
    puell = PuellMultiple()
    puell_val, puell_zone = puell.get_current_puell()
    print(f"  Puell Multiple:      {puell_val:>10.3f}")
    print(f"  Zone:                {puell_zone}")
    print(f"  Sell Zone: >{PUELL_SELL_ZONE}  |  Buy Zone: <{PUELL_BUY_ZONE}")

    # 5. SOPR
    _banner("5. SOPR (Spent Output Profit Ratio)")
    sopr_analyzer = SOPRAnalyzer()
    price_h, vol_h = client.get_price_history(timespan="1years")
    sopr_s = sopr_analyzer.compute_sopr_proxy(price_h, vol_h)
    adj_sopr_s = sopr_analyzer.compute_adjusted_sopr(sopr_s)
    if not sopr_s.empty:
        current_sopr = float(sopr_s.iloc[-1])
        current_adj_sopr = float(adj_sopr_s.iloc[-1]) if not adj_sopr_s.empty else current_sopr
        sopr_interp = sopr_analyzer.interpret_sopr(current_sopr)
        print(f"  SOPR Proxy:          {current_sopr:>10.4f}")
        print(f"  Adjusted SOPR:       {current_adj_sopr:>10.4f}")
        print(f"  Interpretation:      {sopr_interp}")
        print(f"  Reference: >1=selling at profit, <1=selling at loss, =1=breakeven")
    else:
        print("  [SOPR data unavailable]")

    # 6. NUPL
    _banner("6. NUPL (Net Unrealized Profit/Loss)")
    nupl_analyzer = NUPLAnalyzer()
    nupl_val = mvrv_data.nupl
    nupl_zone = nupl_analyzer.classify_nupl(nupl_val)
    print(f"  NUPL:                {nupl_val:>+10.4f}")
    print(f"  Zone:                {nupl_zone}")
    print("  Zones: Capitulation<0 | Hope 0-0.25 | Optimism 0.25-0.5 | Belief 0.5-0.75 | Euphoria>0.75")

    # 7. Coin Days Destroyed
    _banner("7. Coin Days Destroyed (CDD)")
    cdd_analyzer = CoinDaysDestroyed()
    cdd_series = cdd_analyzer.compute_cdd_proxy_history(timespan="1years")
    if not cdd_series.empty:
        bcdd_series = cdd_analyzer.compute_binary_cdd(cdd_series)
        cdd_trend = cdd_analyzer.compute_cdd_trend(cdd_series)
        current_bcdd = float(bcdd_series.iloc[-1]) if not bcdd_series.empty else 1.0
        print(f"  CDD Trend:           {cdd_trend}")
        print(f"  Binary CDD (bCDD):   {current_bcdd:>10.3f}  (>1 = above avg old coin activity)")
        print(f"  Signal: Rising CDD = experienced investors active = potentially bearish near tops")
    else:
        print("  [CDD data unavailable]")

    # 8. Active Addresses
    _banner("8. Active Address Momentum")
    addr_analyzer = ActiveAddressAnalyzer()
    addr_metrics = addr_analyzer.get_address_metrics(timespan="1years")
    print(f"  Current Addresses:   {addr_metrics.get('current_addresses', 0):>10,}")
    print(f"  7-day Avg:           {addr_metrics.get('7d_avg', 0):>10,}")
    print(f"  90-day Avg:          {addr_metrics.get('90d_avg', 0):>10,}")
    print(f"  Momentum:            {addr_metrics.get('momentum', 0):>+10.4f}")
    print(f"  Z-Score:             {addr_metrics.get('z_score', 0):>+10.3f}")
    print(f"  Trend:               {addr_metrics.get('trend', 'UNKNOWN')}")

    # 9. Composite Score
    _banner("9. Composite On-Chain Score")
    composite_engine = OnChainCompositeSignal()
    comp_score = composite_engine.compute_composite_bull_bear_score(timespan="1years")
    print(f"  Composite Score:     {comp_score.composite_score:>10.1f}/100")
    print(f"  Sentiment:           {comp_score.sentiment}")
    print(f"  Cycle Position:      {comp_score.cycle_position}")
    print(f"  Fear & Greed:        {comp_score.fear_greed_index:>10}/100")
    print()
    print("  Component Breakdown:")
    print(f"    MVRV Z-Score (30%): {comp_score.mvrv_component:>6.1f}  [z={comp_score.mvrv_z_score:+.3f}]")
    print(f"    Puell Multi  (20%): {comp_score.puell_component:>6.1f}  [{comp_score.puell_multiple:.3f}]")
    print(f"    SOPR         (20%): {comp_score.sopr_component:>6.1f}  [{comp_score.sopr_value:.4f}]")
    print(f"    NVT Signal   (15%): {comp_score.nvt_component:>6.1f}  [{comp_score.nvt_signal:.1f}]")
    print(f"    Addr Momentum(15%): {comp_score.address_component:>6.1f}  [{comp_score.address_momentum:+.4f}]")

    # 10. Full Report
    _banner("10. Full Narrative Report")
    report = composite_engine.generate_report()
    print(report)


if __name__ == "__main__":
    main()
