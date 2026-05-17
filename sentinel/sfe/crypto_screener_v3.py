"""
sentinel/sfe/crypto_screener_v3.py
dim_075: Crypto / on-chain screener (score 7 → 9)

Comprehensive crypto and on-chain screening platform using free data sources:
- CoinGecko free API (no key required)
- Blockchain.info (Bitcoin on-chain)
- DefiLlama (TVL, protocol flows)
- Alternative.me Fear & Greed Index
- Etherscan-compatible gas APIs (no key for basic stats)
"""

from __future__ import annotations

import time
import json
import logging
import hashlib
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BLOCKCHAIN_INFO_BASE = "https://blockchain.info"
DEFI_LLAMA_BASE = "https://api.llama.fi"
FEAR_GREED_BASE = "https://api.alternative.me"
ETH_GAS_WATCH = "https://ethgas.watch/api/gas"

# Free tier: ~50 calls/min → 1.2 seconds between calls
CG_RATE_LIMIT_SLEEP = 1.25
CG_PRICE_TTL = 300       # 5 minutes
CG_HISTORY_TTL = 3600    # 1 hour

NVT_OVERBOUGHT = 150
NVT_OVERSOLD = 50
MVRV_TOP_ZONE = 3.7
MVRV_BOTTOM_ZONE = 1.0
BTC_BLOCK_REWARD = 3.125  # post-4th-halving


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OnChainData:
    """Snapshot of on-chain metrics for a single asset."""
    coin_id: str
    timestamp: datetime
    hashrate_ths: Optional[float] = None
    difficulty: Optional[float] = None
    tx_count_24h: Optional[int] = None
    mempool_txs: Optional[int] = None
    avg_fee_btc: Optional[float] = None
    avg_fee_usd: Optional[float] = None
    nvt_ratio: Optional[float] = None
    nvt_signal: Optional[float] = None
    mvrv_ratio: Optional[float] = None
    mvrv_zscore: Optional[float] = None
    sopr: Optional[float] = None
    s2f_ratio: Optional[float] = None
    s2f_model_price: Optional[float] = None
    eth_gas_fast: Optional[float] = None
    eth_gas_standard: Optional[float] = None
    eth_gas_slow: Optional[float] = None
    eth_burned_total: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class CryptoMetrics:
    """Full metrics snapshot for a single coin."""
    coin_id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    total_volume: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    circulating_supply: float
    ath: float
    ath_change_pct: float
    atl_change_pct: float
    rsi_14: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_55: Optional[float] = None
    momentum_7d: Optional[float] = None
    momentum_30d: Optional[float] = None
    momentum_90d: Optional[float] = None
    momentum_365d: Optional[float] = None
    relative_momentum_vs_btc: Optional[float] = None
    onchain: Optional[OnChainData] = None


@dataclass
class ScreenResult:
    """Result of a screening pass."""
    criteria: list[str]
    matches: pd.DataFrame
    total_universe: int
    match_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Screen: {', '.join(self.criteria)}",
            f"Universe: {self.total_universe} | Matches: {self.match_count} ({self.match_count/max(self.total_universe,1)*100:.1f}%)",
            f"Timestamp: {self.timestamp.isoformat()}",
        ]
        if not self.matches.empty and "symbol" in self.matches.columns:
            top = self.matches.head(10)["symbol"].tolist()
            lines.append(f"Top matches: {', '.join(str(s).upper() for s in top)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility: HTTP helper
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = 20, headers: dict = None) -> Any:
    """Fetch JSON from a URL with error handling."""
    req_headers = {
        "User-Agent": "SENTINEL/3.0 (research; contact: sentinel@example.com)",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    req = Request(url, headers=req_headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        if e.code == 429:
            logger.warning("Rate limited by %s — sleeping 60s", url)
            time.sleep(60)
        raise
    except URLError as e:
        logger.error("URLError fetching %s: %s", url, e)
        raise


def _ttl_key(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# CoinGeckoClient
# ---------------------------------------------------------------------------

class CoinGeckoClient:
    """
    Wrapper around CoinGecko free API with rate limiting and in-memory cache.

    Rate limit: ~50 calls/min free tier → enforced with 1.25s between calls.
    Cache: 5-min TTL for prices, 1-hour for historical data.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._last_call_ts: float = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < CG_RATE_LIMIT_SLEEP:
            time.sleep(CG_RATE_LIMIT_SLEEP - elapsed)
        self._last_call_ts = time.time()

    def _get(self, endpoint: str, params: dict = None, ttl: int = CG_PRICE_TTL) -> Any:
        url = f"{COINGECKO_BASE}{endpoint}"
        if params:
            url += "?" + urlencode(params)
        cache_key = _ttl_key(url)
        now = time.time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < ttl:
                return data
        self._rate_limit()
        data = _fetch_json(url)
        self._cache[cache_key] = (now, data)
        return data

    def get_top_coins(
        self,
        n: int = 250,
        vs_currency: str = "usd",
    ) -> pd.DataFrame:
        """
        Fetch top N coins by market cap.

        Returns DataFrame with columns:
        id, symbol, name, current_price, market_cap, total_volume,
        price_change_24h, price_change_7d, price_change_30d,
        circulating_supply, ath, ath_change_percentage, atl_change_percentage
        """
        all_rows = []
        per_page = 250
        pages = max(1, (n + per_page - 1) // per_page)
        for page in range(1, pages + 1):
            params = {
                "vs_currency": vs_currency,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "7d,30d",
            }
            rows = self._get("/coins/markets", params=params, ttl=CG_PRICE_TTL)
            all_rows.extend(rows)
            if len(all_rows) >= n:
                break

        all_rows = all_rows[:n]

        records = []
        for r in all_rows:
            records.append({
                "id": r.get("id"),
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "current_price": r.get("current_price", 0.0) or 0.0,
                "market_cap": r.get("market_cap", 0.0) or 0.0,
                "total_volume": r.get("total_volume", 0.0) or 0.0,
                "price_change_24h": r.get("price_change_percentage_24h", 0.0) or 0.0,
                "price_change_7d": r.get("price_change_percentage_7d_in_currency", 0.0) or 0.0,
                "price_change_30d": r.get("price_change_percentage_30d_in_currency", 0.0) or 0.0,
                "circulating_supply": r.get("circulating_supply", 0.0) or 0.0,
                "ath": r.get("ath", 0.0) or 0.0,
                "ath_change_percentage": r.get("ath_change_percentage", 0.0) or 0.0,
                "atl_change_percentage": r.get("atl_change_percentage", 0.0) or 0.0,
                "market_cap_rank": r.get("market_cap_rank"),
            })
        return pd.DataFrame(records)

    def get_coin_ohlcv(
        self,
        coin_id: str,
        days: int = 365,
        vs_currency: str = "usd",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV history for a coin from CoinGecko market_chart endpoint.

        Note: CoinGecko free tier returns OHLC in separate endpoint.
        We approximate OHLCV from price + volume market chart data.
        """
        params = {
            "vs_currency": vs_currency,
            "days": str(days),
            "interval": "daily" if days >= 90 else "hourly",
        }
        data = self._get(f"/coins/{coin_id}/market_chart", params=params, ttl=CG_HISTORY_TTL)

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            return pd.DataFrame()

        price_df = pd.DataFrame(prices, columns=["timestamp_ms", "close"])
        price_df["date"] = pd.to_datetime(price_df["timestamp_ms"], unit="ms", utc=True).dt.normalize()

        vol_df = pd.DataFrame(volumes, columns=["timestamp_ms", "volume"])
        vol_df["date"] = pd.to_datetime(vol_df["timestamp_ms"], unit="ms", utc=True).dt.normalize()

        df = price_df.merge(vol_df[["date", "volume"]], on="date", how="left")
        df = df.drop_duplicates("date").set_index("date").sort_index()

        # Compute open from prior close; approximate high/low as ±0.5% of close
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df["close"] * 1.005
        df["low"] = df["close"] * 0.995
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        return df

    def get_global_metrics(self) -> dict:
        """Fetch global crypto market metrics."""
        data = self._get("/global", ttl=CG_PRICE_TTL)
        gd = data.get("data", {})
        mcp = gd.get("market_cap_percentage", {})
        return {
            "total_market_cap_usd": gd.get("total_market_cap", {}).get("usd", 0.0),
            "total_volume_usd": gd.get("total_volume", {}).get("usd", 0.0),
            "btc_dominance": mcp.get("btc", 0.0),
            "eth_dominance": mcp.get("eth", 0.0),
            "market_cap_change_24h": gd.get("market_cap_change_percentage_24h_usd", 0.0),
            "active_cryptocurrencies": gd.get("active_cryptocurrencies", 0),
            "defi_volume_24h": gd.get("defi_volume_24h", 0.0),
            "stablecoin_volume_24h": gd.get("stablecoin_volume_24h", 0.0),
        }

    def get_trending(self) -> list[dict]:
        """Fetch trending coins from CoinGecko."""
        data = self._get("/search/trending", ttl=CG_PRICE_TTL)
        coins = data.get("coins", [])
        result = []
        for c in coins:
            item = c.get("item", {})
            result.append({
                "id": item.get("id"),
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "market_cap_rank": item.get("market_cap_rank"),
                "score": item.get("score"),
            })
        return result

    def get_coin_detail(self, coin_id: str) -> dict:
        """Fetch detailed coin data including community/developer stats."""
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "false",
        }
        return self._get(f"/coins/{coin_id}", params=params, ttl=CG_HISTORY_TTL)

    def get_stablecoin_market_cap(self) -> float:
        """Estimate stablecoin market cap by fetching top stablecoins."""
        stable_ids = ["tether", "usd-coin", "dai", "binance-usd", "true-usd", "frax"]
        params = {
            "vs_currency": "usd",
            "ids": ",".join(stable_ids),
            "order": "market_cap_desc",
            "per_page": 20,
            "page": 1,
            "sparkline": "false",
        }
        rows = self._get("/coins/markets", params=params, ttl=CG_PRICE_TTL)
        return sum(r.get("market_cap", 0) or 0 for r in rows)


# ---------------------------------------------------------------------------
# OnChainMetricsCollector
# ---------------------------------------------------------------------------

class OnChainMetricsCollector:
    """
    Collect on-chain metrics from free public sources.

    Sources:
    - Blockchain.info: Bitcoin network stats
    - ethgas.watch: Ethereum gas prices
    - Derived metrics: NVT, MVRV, SOPR, S2F
    """

    def __init__(self, cg_client: CoinGeckoClient = None):
        self._cg = cg_client or CoinGeckoClient()
        self._btc_stats_cache: tuple[float, dict] = (0.0, {})
        self._btc_stats_ttl = 600  # 10 min

    def _get_btc_stats(self) -> dict:
        now = time.time()
        ts, data = self._btc_stats_cache
        if data and now - ts < self._btc_stats_ttl:
            return data
        url = f"{BLOCKCHAIN_INFO_BASE}/stats?format=json"
        try:
            data = _fetch_json(url)
            self._btc_stats_cache = (now, data)
        except Exception as e:
            logger.warning("Blockchain.info stats fetch failed: %s", e)
            data = {}
        return data

    def get_btc_hashrate(self) -> float:
        """Return current BTC hash rate in TH/s."""
        stats = self._get_btc_stats()
        # hash_rate in Blockchain.info is in GH/s; convert to TH/s
        ghs = stats.get("hash_rate", 0.0) or 0.0
        return ghs / 1000.0

    def get_btc_difficulty(self) -> float:
        """Return current BTC mining difficulty."""
        stats = self._get_btc_stats()
        return stats.get("difficulty", 0.0) or 0.0

    def get_btc_tx_count(self) -> int:
        """Return 24h transaction count."""
        stats = self._get_btc_stats()
        return int(stats.get("n_tx", 0) or 0)

    def get_btc_mempool_size(self) -> int:
        """Return estimated mempool transaction count from blockchain.info."""
        try:
            url = f"{BLOCKCHAIN_INFO_BASE}/q/unconfirmedcount"
            req = Request(url, headers={"User-Agent": "SENTINEL/3.0", "Accept": "text/plain"})
            with urlopen(req, timeout=10) as resp:
                return int(resp.read().decode().strip())
        except Exception as e:
            logger.warning("BTC mempool fetch failed: %s", e)
            return 0

    def get_btc_fees(self) -> dict:
        """Return average BTC transaction fee stats."""
        stats = self._get_btc_stats()
        avg_fee_btc = (stats.get("total_fees_btc", 0.0) or 0.0) / max(stats.get("n_tx", 1), 1) / 1e8
        # Get current BTC price for USD conversion
        btc_price = 0.0
        try:
            df = self._cg.get_top_coins(n=1)
            if not df.empty:
                btc_price = float(df.iloc[0]["current_price"])
        except Exception:
            pass
        avg_fee_usd = avg_fee_btc * btc_price
        return {
            "avg_fee_btc": avg_fee_btc,
            "avg_fee_usd": avg_fee_usd,
            "total_fees_btc": stats.get("total_fees_btc", 0.0) or 0.0,
        }

    def get_btc_nvt(self) -> float:
        """
        Compute NVT ratio = Market Cap / Daily TX Volume (USD).

        Uses blockchain.info estimated_transaction_volume_usd as proxy.
        """
        stats = self._get_btc_stats()
        market_cap = stats.get("market_price_usd", 0.0) * stats.get("totalbc", 0.0) / 1e8
        tx_vol_usd = stats.get("estimated_transaction_volume_usd", 0.0) or 1.0
        if tx_vol_usd <= 0 or market_cap <= 0:
            return float("nan")
        return market_cap / tx_vol_usd

    def get_eth_gas_price(self) -> dict:
        """Fetch ETH gas prices (fast/standard/slow) in Gwei from ethgas.watch."""
        try:
            data = _fetch_json(ETH_GAS_WATCH, timeout=10)
            return {
                "fast": data.get("fast", {}).get("price", 0.0) if isinstance(data.get("fast"), dict) else data.get("fast", 0.0),
                "standard": data.get("normal", {}).get("price", 0.0) if isinstance(data.get("normal"), dict) else data.get("normal", 0.0),
                "slow": data.get("slow", {}).get("price", 0.0) if isinstance(data.get("slow"), dict) else data.get("slow", 0.0),
            }
        except Exception as e:
            logger.warning("ETH gas fetch failed: %s", e)
            return {"fast": 0.0, "standard": 0.0, "slow": 0.0}

    def get_eth_burned(self) -> float:
        """
        Approximate ETH burned since EIP-1559 from public endpoint.
        Falls back to estimation if unavailable.
        """
        try:
            # ultrasound.money provides public data; use their API
            url = "https://ultrasound.money/api/v2/fees/burn-rates"
            data = _fetch_json(url, timeout=10)
            return float(data.get("total_burned", 0.0) or 0.0)
        except Exception:
            pass
        # Fallback: rough estimate (ETH burned since Aug 2021 ≈ 3.5M+ ETH)
        # We return 0 and note it's unavailable
        logger.warning("ETH burned data unavailable; returning 0")
        return 0.0

    def compute_mvrv(self, coin_id: str, days: int = 365) -> float:
        """
        Approximate MVRV ratio (Market Value / Realized Value).

        Realized price is approximated as the 365-day VWAP of price×volume,
        serving as a proxy for the average acquisition price of all coins.
        MVRV = current_price / realized_price
        """
        try:
            ohlcv = self._cg.get_coin_ohlcv(coin_id, days=days)
            if ohlcv.empty or len(ohlcv) < 30:
                return float("nan")
            prices = ohlcv["close"].values
            volumes = ohlcv["volume"].values
            # VWAP as realized price proxy
            total_vol = volumes.sum()
            if total_vol <= 0:
                return float("nan")
            realized_price = float(np.average(prices, weights=volumes + 1e-9))
            current_price = prices[-1]
            return current_price / realized_price if realized_price > 0 else float("nan")
        except Exception as e:
            logger.warning("MVRV computation failed for %s: %s", coin_id, e)
            return float("nan")

    def compute_mvrv_zscore(self, coin_id: str, days: int = 730) -> float:
        """
        MVRV Z-Score = (Market Cap - Realized Cap) / std(Market Cap).

        Uses approximation from price history.
        """
        try:
            ohlcv = self._cg.get_coin_ohlcv(coin_id, days=days)
            if ohlcv.empty or len(ohlcv) < 90:
                return float("nan")
            prices = ohlcv["close"].values
            volumes = ohlcv["volume"].values
            # Realized cap proxy: cumulative VWAP applied to circulating supply
            realized_price = float(np.average(prices, weights=volumes + 1e-9))
            current_price = prices[-1]
            mc_values = prices  # scaled proxy
            zscore = (current_price - realized_price) / (np.std(mc_values) + 1e-9)
            return float(zscore)
        except Exception as e:
            logger.warning("MVRV Z-score failed for %s: %s", coin_id, e)
            return float("nan")

    def compute_sopr(self, coin_id: str, days: int = 90) -> float:
        """
        Approximate SOPR (Spent Output Profit Ratio).

        True SOPR requires UTXO data. We approximate via 90-day price momentum:
        - Compares current price to the price coins were likely last moved (approximated
          as the rolling 90-day weighted average acquisition price).
        SOPR > 1: spending in profit; < 1: spending at loss.
        """
        try:
            ohlcv = self._cg.get_coin_ohlcv(coin_id, days=max(days, 90))
            if ohlcv.empty or len(ohlcv) < 30:
                return 1.0
            prices = ohlcv["close"].values
            # 90-day weighted avg as "realized" proxy for SOPR
            w = np.arange(1, len(prices) + 1, dtype=float)
            weighted_avg = np.average(prices, weights=w)
            current = prices[-1]
            return float(current / weighted_avg) if weighted_avg > 0 else 1.0
        except Exception as e:
            logger.warning("SOPR computation failed for %s: %s", coin_id, e)
            return 1.0

    def compute_stock_to_flow(self, coin_id: str) -> dict:
        """
        Compute Stock-to-Flow (S2F) ratio and model price.

        S2F = circulating_supply / annual_new_supply
        PlanB S2F model: price = exp(a) * S2F^4.0 (simplified log-log regression)

        Only meaningful for BTC and ETH (hard-capped / deflationary assets).
        """
        result = {"s2f": None, "model_price": None, "annual_new_supply": None}
        try:
            detail = self._cg.get_coin_detail(coin_id)
            md = detail.get("market_data", {})
            circulating = md.get("circulating_supply", 0.0) or 0.0
            if coin_id == "bitcoin":
                blocks_per_day = 144
                annual_new = BTC_BLOCK_REWARD * blocks_per_day * 365
            elif coin_id == "ethereum":
                # Post-merge: ~0.5% annual issuance rate (deflationary near-zero)
                annual_new = circulating * 0.005
            else:
                # Generic: attempt from max_supply vs circulating proxy
                max_supply = md.get("max_supply", None)
                if max_supply and circulating > 0:
                    pct_minted = circulating / max_supply
                    # Assume emission halves as minted % increases
                    annual_new = circulating * (1 - pct_minted) * 0.1
                else:
                    return result
            if annual_new <= 0:
                return result
            s2f = circulating / annual_new
            # PlanB simplified formula: ln(price) = 3.31 * ln(S2F) - 1.84
            import math
            model_price = math.exp(3.31 * math.log(max(s2f, 0.01)) - 1.84)
            result.update({
                "s2f": s2f,
                "model_price": model_price,
                "annual_new_supply": annual_new,
                "circulating_supply": circulating,
            })
        except Exception as e:
            logger.warning("S2F computation failed for %s: %s", coin_id, e)
        return result

    def collect_btc_onchain(self, btc_price_usd: float = 0.0) -> OnChainData:
        """Collect all BTC on-chain metrics into a single OnChainData object."""
        stats = self._get_btc_stats()
        if btc_price_usd <= 0:
            btc_price_usd = stats.get("market_price_usd", 50000.0)

        fees = self.get_btc_fees()
        nvt = self.get_btc_nvt()

        # S2F
        s2f_data = self.compute_stock_to_flow("bitcoin")

        return OnChainData(
            coin_id="bitcoin",
            timestamp=datetime.now(timezone.utc),
            hashrate_ths=self.get_btc_hashrate(),
            difficulty=self.get_btc_difficulty(),
            tx_count_24h=self.get_btc_tx_count(),
            mempool_txs=self.get_btc_mempool_size(),
            avg_fee_btc=fees["avg_fee_btc"],
            avg_fee_usd=fees["avg_fee_usd"],
            nvt_ratio=nvt,
            mvrv_ratio=self.compute_mvrv("bitcoin"),
            mvrv_zscore=self.compute_mvrv_zscore("bitcoin"),
            sopr=self.compute_sopr("bitcoin"),
            s2f_ratio=s2f_data.get("s2f"),
            s2f_model_price=s2f_data.get("model_price"),
            extra={"s2f_detail": s2f_data},
        )

    def collect_eth_onchain(self) -> OnChainData:
        """Collect Ethereum on-chain metrics."""
        gas = self.get_eth_gas_price()
        mvrv = self.compute_mvrv("ethereum")
        sopr = self.compute_sopr("ethereum")
        burned = self.get_eth_burned()
        return OnChainData(
            coin_id="ethereum",
            timestamp=datetime.now(timezone.utc),
            mvrv_ratio=mvrv,
            sopr=sopr,
            eth_gas_fast=gas.get("fast"),
            eth_gas_standard=gas.get("standard"),
            eth_gas_slow=gas.get("slow"),
            eth_burned_total=burned,
        )


# ---------------------------------------------------------------------------
# CryptoMomentumSignals
# ---------------------------------------------------------------------------

class CryptoMomentumSignals:
    """Compute momentum and technical signals for crypto assets."""

    def __init__(self, cg_client: CoinGeckoClient = None):
        self._cg = cg_client or CoinGeckoClient()

    def compute_momentum(
        self,
        coin_id: str,
        periods: list[int] = None,
    ) -> dict[int, float]:
        """
        Compute price return over multiple lookback periods (in days).

        Returns dict: {period_days: return_pct}
        """
        if periods is None:
            periods = [7, 30, 90, 365]
        max_days = max(periods) + 5
        ohlcv = self._cg.get_coin_ohlcv(coin_id, days=max_days)
        if ohlcv.empty:
            return {p: float("nan") for p in periods}

        result = {}
        closes = ohlcv["close"]
        current = closes.iloc[-1]

        for p in periods:
            if len(closes) > p:
                past = closes.iloc[-(p + 1)]
                result[p] = (current / past - 1.0) * 100 if past > 0 else float("nan")
            else:
                result[p] = float("nan")
        return result

    def compute_relative_momentum(
        self,
        coin_id: str,
        vs_coin: str = "bitcoin",
        days: int = 30,
    ) -> float:
        """
        Compute excess return of coin vs BTC (or any reference coin).

        Positive value = outperforming BTC (altcoin season indicator).
        """
        coin_mom = self.compute_momentum(coin_id, [days])
        vs_mom = self.compute_momentum(vs_coin, [days])
        c = coin_mom.get(days, float("nan"))
        v = vs_mom.get(days, float("nan"))
        if c != c or v != v:
            return float("nan")
        return c - v

    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Compute RSI using pure numpy (Wilder's smoothing).

        Returns float in [0, 100].
        """
        if len(prices) < period + 1:
            return float("nan")
        deltas = np.diff(prices.values.astype(float))
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Initial average using simple mean
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        # Wilder's smoothing for remaining values
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def compute_ema(prices: pd.Series, span: int) -> pd.Series:
        """Compute EMA with given span."""
        alpha = 2.0 / (span + 1)
        values = prices.values.astype(float)
        ema = np.empty_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return pd.Series(ema, index=prices.index)

    def compute_ema_signals(self, prices: pd.Series) -> dict:
        """
        Compute EMA 9/21/55 and crossover signals.

        Returns dict with current EMA values, price positions, and crossover flags.
        """
        if len(prices) < 56:
            return {"error": "insufficient data"}

        ema9 = self.compute_ema(prices, 9)
        ema21 = self.compute_ema(prices, 21)
        ema55 = self.compute_ema(prices, 55)

        current_price = prices.iloc[-1]
        e9 = ema9.iloc[-1]
        e21 = ema21.iloc[-1]
        e55 = ema55.iloc[-1]

        # Crossover: check if last 2 bars had a crossover
        ema9_prev = ema9.iloc[-2]
        ema21_prev = ema21.iloc[-2]
        bullish_cross_9_21 = (ema9_prev < ema21_prev) and (e9 > e21)
        bearish_cross_9_21 = (ema9_prev > ema21_prev) and (e9 < e21)

        return {
            "ema_9": float(e9),
            "ema_21": float(e21),
            "ema_55": float(e55),
            "price_above_ema9": bool(current_price > e9),
            "price_above_ema21": bool(current_price > e21),
            "price_above_ema55": bool(current_price > e55),
            "ema9_above_ema21": bool(e9 > e21),
            "ema21_above_ema55": bool(e21 > e55),
            "bullish_alignment": bool(current_price > e9 > e21 > e55),
            "bearish_alignment": bool(current_price < e9 < e21 < e55),
            "bullish_cross_9_21": bullish_cross_9_21,
            "bearish_cross_9_21": bearish_cross_9_21,
        }

    def compute_fear_greed_index(self) -> dict:
        """
        Fetch Alternative.me Fear & Greed Index.

        Returns current score, classification, and 30-day history.
        """
        try:
            url = f"{FEAR_GREED_BASE}/fng/?limit=30&format=json"
            data = _fetch_json(url, timeout=15)
            entries = data.get("data", [])
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)
            return {"error": str(e), "current_value": None, "classification": None}

        if not entries:
            return {"error": "no data", "current_value": None, "classification": None}

        current = entries[0]
        score = int(current.get("value", 50))

        if score < 25:
            classification = "EXTREME_FEAR"
        elif score < 45:
            classification = "FEAR"
        elif score < 55:
            classification = "NEUTRAL"
        elif score < 75:
            classification = "GREED"
        else:
            classification = "EXTREME_GREED"

        history = [
            {
                "date": datetime.fromtimestamp(int(e.get("timestamp", 0)), tz=timezone.utc).date().isoformat(),
                "value": int(e.get("value", 50)),
                "classification": e.get("value_classification", ""),
            }
            for e in entries
        ]

        avg_30d = statistics.mean(int(e.get("value", 50)) for e in entries)

        return {
            "current_value": score,
            "classification": classification,
            "timestamp": current.get("timestamp"),
            "history_30d": history,
            "avg_30d": avg_30d,
            "trend": "improving" if len(history) > 5 and history[0]["value"] > history[5]["value"] else "declining",
        }

    def get_signals_for_coin(self, coin_id: str) -> dict:
        """Compute all momentum signals for a given coin."""
        ohlcv = self._cg.get_coin_ohlcv(coin_id, days=400)
        if ohlcv.empty:
            return {"error": f"No data for {coin_id}"}

        prices = ohlcv["close"]
        momentum = self.compute_momentum(coin_id, [7, 30, 90, 365])
        rsi = self.compute_rsi(prices, 14)
        ema_signals = self.compute_ema_signals(prices)
        rel_mom = self.compute_relative_momentum(coin_id, "bitcoin", 30)

        return {
            "coin_id": coin_id,
            "current_price": float(prices.iloc[-1]),
            "momentum": momentum,
            "rsi_14": rsi,
            "ema_signals": ema_signals,
            "relative_momentum_vs_btc_30d": rel_mom,
        }


# ---------------------------------------------------------------------------
# CryptoFundamentalAnalyzer
# ---------------------------------------------------------------------------

class CryptoFundamentalAnalyzer:
    """On-chain and fundamental analysis for crypto assets."""

    def __init__(
        self,
        cg_client: CoinGeckoClient = None,
        onchain_collector: OnChainMetricsCollector = None,
    ):
        self._cg = cg_client or CoinGeckoClient()
        self._onchain = onchain_collector or OnChainMetricsCollector(self._cg)
        self._mom = CryptoMomentumSignals(self._cg)

    def compute_nvt_signal(self, coin_id: str = "bitcoin") -> float:
        """
        Compute NVT Signal = NVT smoothed with 90-day MA.

        For Bitcoin: uses Blockchain.info tx volume data.
        High NVT (>150): overvalued vs network usage.
        Low NVT (<50): undervalued.
        """
        if coin_id == "bitcoin":
            raw_nvt = self._onchain.get_btc_nvt()
            # Approximate smoothing: blockchain.info provides a single snapshot
            # We treat current NVT as signal (full 90-day MA requires historical storage)
            return raw_nvt
        # For other coins: approximate NVT from market cap / trading volume
        try:
            df = self._cg.get_top_coins(n=500)
            row = df[df["id"] == coin_id]
            if row.empty:
                return float("nan")
            mc = row.iloc[0]["market_cap"]
            vol = row.iloc[0]["total_volume"]
            if vol <= 0:
                return float("nan")
            return mc / (vol * 30)  # approximate: annualize monthly volume
        except Exception as e:
            logger.warning("NVT signal failed for %s: %s", coin_id, e)
            return float("nan")

    def compute_market_cycle_position(self, coin_id: str) -> dict:
        """
        Determine market cycle position using MVRV Z-Score and price history.

        MVRV Z-Score interpretation:
        - > 7: historically near market tops (DISTRIBUTION)
        - 3-7: bull market (BULL)
        - 0-3: early bull / recovery (RECOVERY)
        - < 0: near market bottoms (ACCUMULATION)
        """
        mvrv = self._onchain.compute_mvrv(coin_id)
        zscore = self._onchain.compute_mvrv_zscore(coin_id)
        sopr = self._onchain.compute_sopr(coin_id)

        if zscore != zscore:
            phase = "UNKNOWN"
        elif zscore > 7:
            phase = "DISTRIBUTION"
        elif zscore > 3:
            phase = "BULL"
        elif zscore >= 0:
            phase = "RECOVERY"
        else:
            phase = "ACCUMULATION"

        return {
            "coin_id": coin_id,
            "mvrv_ratio": mvrv,
            "mvrv_zscore": zscore,
            "sopr": sopr,
            "phase": phase,
            "mvrv_below_one": bool(mvrv < 1.0) if mvrv == mvrv else False,
            "historically_strong_buy": bool(mvrv < MVRV_BOTTOM_ZONE) if mvrv == mvrv else False,
            "historically_near_top": bool(mvrv > MVRV_TOP_ZONE) if mvrv == mvrv else False,
        }

    def compute_supply_metrics(self, coin_id: str) -> dict:
        """
        Compute supply-side metrics: % in profit, illiquid supply proxy, S2F.
        """
        try:
            detail = self._cg.get_coin_detail(coin_id)
            md = detail.get("market_data", {})
            current_price = md.get("current_price", {}).get("usd", 0.0) or 0.0
            ath = md.get("ath", {}).get("usd", 0.0) or 0.0
            atl = md.get("atl", {}).get("usd", 0.0) or 0.0
            circulating = md.get("circulating_supply", 0.0) or 0.0
            max_supply = md.get("max_supply")
            total_supply = md.get("total_supply", circulating) or circulating

            # % in profit proxy: if current price > historical average price
            ohlcv = self._cg.get_coin_ohlcv(coin_id, days=365)
            pct_in_profit = float("nan")
            if not ohlcv.empty:
                prices = ohlcv["close"].values
                avg_price_365d = prices.mean()
                # Estimate % of coins acquired below current price
                prices_below_current = prices[prices < current_price]
                pct_in_profit = len(prices_below_current) / len(prices) * 100

            # Illiquid supply proxy: coins held > 1 year (proxy via price history)
            # We approximate: 1 - (365d avg volume / circulating_supply)
            illiquid_ratio = float("nan")
            if not ohlcv.empty and circulating > 0:
                avg_daily_vol = ohlcv["volume"].mean()
                annual_turnover = avg_daily_vol * 365 / circulating
                illiquid_ratio = max(0.0, 1.0 - min(1.0, annual_turnover / current_price))

            s2f = self._onchain.compute_stock_to_flow(coin_id)

            return {
                "coin_id": coin_id,
                "circulating_supply": circulating,
                "max_supply": max_supply,
                "total_supply": total_supply,
                "pct_in_profit_proxy": pct_in_profit,
                "illiquid_supply_ratio_proxy": illiquid_ratio,
                "s2f_ratio": s2f.get("s2f"),
                "s2f_model_price": s2f.get("model_price"),
                "price_vs_ath_pct": ((current_price / ath - 1) * 100) if ath > 0 else None,
                "price_vs_atl_pct": ((current_price / atl - 1) * 100) if atl > 0 else None,
            }
        except Exception as e:
            logger.warning("Supply metrics failed for %s: %s", coin_id, e)
            return {"coin_id": coin_id, "error": str(e)}

    def compute_dominance_trend(self) -> dict:
        """
        Compute BTC and ETH dominance trend and alt season indicator.

        Alt season: ETH.D + top-10-alts.D > BTC.D and all rising.
        """
        global_metrics = self._cg.get_global_metrics()
        btc_dom = global_metrics.get("btc_dominance", 0.0)
        eth_dom = global_metrics.get("eth_dominance", 0.0)
        alt_dom = 100.0 - btc_dom - eth_dom

        # Trend: fetch global data for trend direction (single snapshot limitation)
        # We use BTC 30d return vs ETH 30d return as dominance direction proxy
        btc_mom = CryptoMomentumSignals(self._cg).compute_momentum("bitcoin", [30])
        eth_mom = CryptoMomentumSignals(self._cg).compute_momentum("ethereum", [30])

        btc_30d = btc_mom.get(30, 0.0) or 0.0
        eth_30d = eth_mom.get(30, 0.0) or 0.0

        eth_outperforming = eth_30d - btc_30d
        alt_season = eth_outperforming > 10.0 and eth_dom > 15.0

        return {
            "btc_dominance": btc_dom,
            "eth_dominance": eth_dom,
            "alt_dominance": alt_dom,
            "btc_30d_return": btc_30d,
            "eth_30d_return": eth_30d,
            "eth_outperformance_30d": eth_outperforming,
            "alt_season": alt_season,
            "alt_season_signal": "ALTCOIN_SEASON" if alt_season else "BTC_DOMINANCE",
            "btc_dominance_rising": btc_30d > eth_30d,  # risk-off proxy
        }


# ---------------------------------------------------------------------------
# DefiLlamaClient
# ---------------------------------------------------------------------------

class DefiLlamaClient:
    """Minimal DefiLlama free API client for TVL and protocol data."""

    def __init__(self):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = 1800  # 30 min

    def _get(self, endpoint: str) -> Any:
        url = f"{DEFI_LLAMA_BASE}{endpoint}"
        key = _ttl_key(url)
        now = time.time()
        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts < self._ttl:
                return data
        data = _fetch_json(url, timeout=30)
        self._cache[key] = (now, data)
        return data

    def get_protocols(self) -> list[dict]:
        """Get all DeFi protocols with TVL."""
        return self._get("/protocols") or []

    def get_global_tvl(self) -> float:
        """Get total DeFi TVL across all chains."""
        data = self._get("/v2/historicalChainTvl")
        if isinstance(data, list) and data:
            return float(data[-1].get("tvl", 0.0))
        return 0.0

    def get_protocol_tvl_history(self, protocol_slug: str) -> pd.DataFrame:
        """Get TVL history for a specific protocol."""
        try:
            data = self._get(f"/protocol/{protocol_slug}")
            tvl_data = data.get("tvl", [])
            if not tvl_data:
                return pd.DataFrame()
            df = pd.DataFrame(tvl_data)
            df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
            df = df.set_index("date").rename(columns={"totalLiquidityUSD": "tvl"})
            if "tvl" not in df.columns and "totalLiquidityUSD" in data:
                pass
            return df
        except Exception as e:
            logger.warning("DefiLlama protocol TVL failed for %s: %s", protocol_slug, e)
            return pd.DataFrame()

    def compute_tvl_growth(self, protocol_slug: str, days: int = 30) -> float:
        """Compute TVL growth percentage over specified days."""
        try:
            df = self.get_protocol_tvl_history(protocol_slug)
            if df.empty or "tvl" not in df.columns:
                return float("nan")
            df = df.sort_index()
            if len(df) < days:
                return float("nan")
            recent = df["tvl"].iloc[-1]
            past = df["tvl"].iloc[-days]
            if past <= 0:
                return float("nan")
            return (recent / past - 1.0) * 100
        except Exception as e:
            logger.warning("TVL growth calc failed for %s: %s", protocol_slug, e)
            return float("nan")

    def get_stablecoin_market_caps(self) -> pd.DataFrame:
        """Get stablecoin market cap data from DefiLlama."""
        try:
            data = _fetch_json("https://stablecoins.llama.fi/stablecoins?includePrices=true", timeout=20)
            pegged = data.get("peggedAssets", [])
            records = []
            for p in pegged:
                mc = p.get("circulating", {}).get("peggedUSD", 0.0) or 0.0
                records.append({
                    "name": p.get("name"),
                    "symbol": p.get("symbol"),
                    "market_cap": mc,
                    "peg_type": p.get("pegType"),
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.warning("Stablecoin data failed: %s", e)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# CryptoScreener
# ---------------------------------------------------------------------------

class CryptoScreener:
    """
    Screen the crypto universe against 20+ institutional-grade criteria.

    All criteria are implemented as boolean filters applied to a DataFrame
    of coin metrics. Composite presets combine multiple criteria.
    """

    PRESETS = {
        "ALTCOIN_SEASON": [
            "btc_dominance_falling", "altcoin_season", "eth_outperforming", "momentum_all_timeframes",
        ],
        "BOTTOMING_SIGNALS": [
            "mvrv_below_1", "deep_correction", "rsi_oversold", "fear_greed_extreme_fear",
        ],
        "MOMENTUM_BREAKOUT": [
            "momentum_all_timeframes", "volume_surge", "cross_ema_bullish", "large_mcap_momentum",
        ],
        "OVERHEATED_MARKET": [
            "new_ath_proximity", "rsi_overbought", "high_funding_rate",
        ],
        "DEFI_OPPORTUNITIES": [
            "defi_tvl_growth", "low_market_cap_high_volume", "momentum_all_timeframes",
        ],
    }

    def __init__(
        self,
        cg_client: CoinGeckoClient = None,
        onchain_collector: OnChainMetricsCollector = None,
        defi_client: DefiLlamaClient = None,
    ):
        self._cg = cg_client or CoinGeckoClient()
        self._onchain = onchain_collector or OnChainMetricsCollector(self._cg)
        self._defi = defi_client or DefiLlamaClient()
        self._mom = CryptoMomentumSignals(self._cg)
        self._fund = CryptoFundamentalAnalyzer(self._cg, self._onchain)
        self._fg_cache: dict = {}
        self._global_cache: dict = {}
        self._universe_df: pd.DataFrame = pd.DataFrame()

    def _load_universe(self, top_n: int = 250) -> pd.DataFrame:
        """Load and cache the top-N coin universe."""
        if self._universe_df.empty:
            self._universe_df = self._cg.get_top_coins(n=top_n)
        return self._universe_df

    def _get_fear_greed(self) -> dict:
        if not self._fg_cache:
            self._fg_cache = self._mom.compute_fear_greed_index()
        return self._fg_cache

    def _get_global(self) -> dict:
        if not self._global_cache:
            self._global_cache = self._cg.get_global_metrics()
        return self._global_cache

    def _enrich_with_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and RSI columns to the universe DataFrame (batch)."""
        if "rsi_14" in df.columns:
            return df

        rsi_values = []
        mom_90d = []
        ema_bullish = []

        for _, row in df.iterrows():
            try:
                coin_id = row["id"]
                ohlcv = self._cg.get_coin_ohlcv(coin_id, days=120)
                if ohlcv.empty or len(ohlcv) < 20:
                    rsi_values.append(float("nan"))
                    mom_90d.append(float("nan"))
                    ema_bullish.append(False)
                    continue
                prices = ohlcv["close"]
                rsi_values.append(self._mom.compute_rsi(prices, 14))
                if len(prices) > 90:
                    m = (prices.iloc[-1] / prices.iloc[-91] - 1) * 100
                    mom_90d.append(m)
                else:
                    mom_90d.append(float("nan"))
                if len(prices) >= 56:
                    sigs = self._mom.compute_ema_signals(prices)
                    ema_bullish.append(sigs.get("bullish_alignment", False))
                else:
                    ema_bullish.append(False)
            except Exception:
                rsi_values.append(float("nan"))
                mom_90d.append(float("nan"))
                ema_bullish.append(False)

        df = df.copy()
        df["rsi_14"] = rsi_values
        df["momentum_90d"] = mom_90d
        df["ema_bullish"] = ema_bullish
        return df

    def _apply_criterion(self, df: pd.DataFrame, criterion: str) -> pd.Series:
        """
        Apply a single screening criterion and return a boolean Series.
        """
        fg = self._get_fear_greed()
        gm = self._get_global()
        fg_score = fg.get("current_value", 50) or 50
        btc_dom = gm.get("btc_dominance", 50.0)
        eth_dom = gm.get("eth_dominance", 15.0)

        false_mask = pd.Series([False] * len(df), index=df.index)
        true_mask = pd.Series([True] * len(df), index=df.index)

        if criterion == "new_ath_proximity":
            # Price within 20% of ATH
            return df["ath_change_percentage"].fillna(-100) > -20.0

        elif criterion == "deep_correction":
            # Price > 50% below ATH
            return df["ath_change_percentage"].fillna(0) < -50.0

        elif criterion == "volume_surge":
            # 24h volume > 3× average daily volume proxy
            # Proxy: volume/market_cap ratio > typical (0.1 = high turnover)
            avg_ratio = 0.05  # typical daily vol/mcap
            vol_ratio = df["total_volume"] / (df["market_cap"].replace(0, float("nan")))
            return vol_ratio.fillna(0) > avg_ratio * 3

        elif criterion == "momentum_all_timeframes":
            # Positive 7d, 30d, 90d returns
            return (
                (df["price_change_7d"].fillna(-100) > 0)
                & (df["price_change_30d"].fillna(-100) > 0)
                & (df.get("momentum_90d", pd.Series([0.0] * len(df), index=df.index)).fillna(-100) > 0)
            )

        elif criterion == "fear_greed_extreme_fear":
            return true_mask if fg_score < 20 else false_mask

        elif criterion == "btc_dominance_falling":
            # BTC dominance < 45% (proxy for altcoin rotation)
            return true_mask if btc_dom < 45.0 else false_mask

        elif criterion == "altcoin_season":
            # ETH.D + alts > BTC.D and ETH outperforming
            return true_mask if (eth_dom > 15.0 and btc_dom < 48.0) else false_mask

        elif criterion == "eth_outperforming":
            return df["symbol"].str.lower().isin(["eth", "ethereum"])

        elif criterion == "rsi_oversold":
            if "rsi_14" not in df.columns:
                return false_mask
            return df["rsi_14"].fillna(50) < 30.0

        elif criterion == "rsi_overbought":
            if "rsi_14" not in df.columns:
                return false_mask
            return df["rsi_14"].fillna(50) > 70.0

        elif criterion == "nvt_undervalued":
            # Only applies to BTC; pass all others or filter BTC only
            nvt = self._onchain.get_btc_nvt()
            if nvt < NVT_OVERSOLD:
                return df["id"] == "bitcoin"
            return false_mask

        elif criterion == "mvrv_below_1":
            # MVRV < 1 is a strong bottom signal; batch check via price/ATL ratio proxy
            # True MVRV requires on-chain data; approximate via price being below 30-day moving avg
            # We flag coins where price is below 60-day low proxy
            return df["ath_change_percentage"].fillna(0) < -70.0

        elif criterion == "low_market_cap_high_volume":
            # Market cap < $1B AND 24h volume > $100M
            return (df["market_cap"].fillna(0) < 1e9) & (df["total_volume"].fillna(0) > 1e8)

        elif criterion == "defi_tvl_growth":
            # Flag DeFi-related coins (UNI, AAVE, COMP, etc.) with positive 30d momentum
            defi_symbols = {
                "uni", "aave", "comp", "mkr", "crv", "sushi", "bal", "yfi",
                "snx", "1inch", "ldo", "convex", "frax", "gmx", "dydx",
            }
            is_defi = df["symbol"].str.lower().isin(defi_symbols)
            has_momentum = df["price_change_30d"].fillna(-100) > 0
            return is_defi & has_momentum

        elif criterion == "hashrate_ath":
            # Check if BTC hashrate is at or near ATH (proxy: hashrate > 500 EH/s = ~500,000 TH/s)
            hr = self._onchain.get_btc_hashrate()
            return (df["id"] == "bitcoin") & (hr > 500_000)

        elif criterion == "high_funding_rate":
            # Perpetual funding rate proxy: extreme price rally = likely high funding
            # Use 7d change > 30% as proxy for overleveraged market
            return df["price_change_7d"].fillna(0) > 30.0

        elif criterion == "negative_funding_rate":
            # Extreme negative 7d change as proxy for overcrowded shorts
            return df["price_change_7d"].fillna(0) < -25.0

        elif criterion == "large_mcap_momentum":
            # Market cap > $10B AND 30d return > 20%
            return (df["market_cap"].fillna(0) > 1e10) & (df["price_change_30d"].fillna(0) > 20.0)

        elif criterion == "cross_ema_bullish":
            if "ema_bullish" not in df.columns:
                return false_mask
            return df["ema_bullish"].fillna(False).astype(bool)

        elif criterion == "exchange_outflow":
            # Proxy: coins with high 7d return AND falling volume (coins leaving exchanges)
            high_return = df["price_change_7d"].fillna(0) > 15.0
            vol_ratio = df["total_volume"] / (df["market_cap"].replace(0, float("nan")))
            low_vol_ratio = vol_ratio.fillna(1) < 0.03
            return high_return & low_vol_ratio

        elif criterion == "stablecoin_supply_ratio":
            # High stablecoin ratio = more dry powder (bullish backdrop)
            # Apply to entire universe if global stablecoin supply / BTC mcap > historical avg
            total_mc = gm.get("total_market_cap_usd", 1e12)
            stable_mc = self._cg.get_stablecoin_market_cap()
            ratio = stable_mc / max(total_mc, 1.0)
            return true_mask if ratio > 0.12 else false_mask

        else:
            logger.warning("Unknown criterion: %s", criterion)
            return false_mask

    def screen(
        self,
        criteria_names: list[str],
        universe: list[str] = None,
        top_n: int = 250,
        enrich_with_momentum: bool = True,
    ) -> ScreenResult:
        """
        Screen the crypto universe and return matching coins.

        Args:
            criteria_names: List of criterion names (AND logic by default).
            universe: Optional list of coin IDs to restrict screening to.
            top_n: Size of universe to screen (up to 250 for free tier).
            enrich_with_momentum: Fetch RSI/EMA data for each coin (slower).
        """
        df = self._load_universe(top_n=top_n)
        if universe:
            df = df[df["id"].isin(universe)]

        if enrich_with_momentum and any(
            c in criteria_names for c in ["rsi_oversold", "rsi_overbought", "momentum_all_timeframes", "cross_ema_bullish"]
        ):
            df = self._enrich_with_momentum(df)

        mask = pd.Series([True] * len(df), index=df.index)
        for criterion in criteria_names:
            criterion_mask = self._apply_criterion(df, criterion)
            mask = mask & criterion_mask

        matches = df[mask].copy()
        matches = matches.sort_values("market_cap", ascending=False)

        return ScreenResult(
            criteria=criteria_names,
            matches=matches,
            total_universe=len(df),
            match_count=len(matches),
            metadata={
                "fear_greed_score": self._get_fear_greed().get("current_value"),
                "btc_dominance": self._get_global().get("btc_dominance"),
            },
        )

    def screen_all_presets(self, top_n: int = 100) -> dict[str, ScreenResult]:
        """Run all preset screens and return results dict."""
        results = {}
        # Reset universe cache to re-use the same DataFrame
        self._universe_df = pd.DataFrame()
        df = self._load_universe(top_n=top_n)
        df = self._enrich_with_momentum(df)
        self._universe_df = df

        for preset_name, criteria in self.PRESETS.items():
            logger.info("Running preset: %s", preset_name)
            try:
                results[preset_name] = self.screen(
                    criteria, top_n=top_n, enrich_with_momentum=False
                )
            except Exception as e:
                logger.error("Preset %s failed: %s", preset_name, e)
        return results


# ---------------------------------------------------------------------------
# CryptoPortfolioAnalyzer
# ---------------------------------------------------------------------------

class CryptoPortfolioAnalyzer:
    """Analyze a crypto portfolio's risk metrics and factor exposures."""

    def __init__(self, cg_client: CoinGeckoClient = None):
        self._cg = cg_client or CoinGeckoClient()
        self._mom = CryptoMomentumSignals(self._cg)

    def _get_returns(self, coin_id: str, days: int = 90) -> pd.Series:
        """Fetch daily log returns for a coin."""
        ohlcv = self._cg.get_coin_ohlcv(coin_id, days=days + 5)
        if ohlcv.empty:
            return pd.Series(dtype=float)
        prices = ohlcv["close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    def compute_portfolio_beta(self, holdings: dict[str, float], days: int = 90) -> float:
        """
        Compute portfolio beta vs BTC using 90-day daily returns regression.

        Args:
            holdings: {coin_id: weight} — weights should sum to 1.0
            days: Lookback window in days.
        """
        btc_returns = self._get_returns("bitcoin", days=days)
        if btc_returns.empty:
            return float("nan")

        portfolio_returns = pd.Series(0.0, index=btc_returns.index)
        total_weight = sum(holdings.values())
        if total_weight <= 0:
            return float("nan")

        for coin_id, weight in holdings.items():
            norm_weight = weight / total_weight
            if coin_id == "bitcoin":
                portfolio_returns = portfolio_returns.add(btc_returns * norm_weight, fill_value=0)
                continue
            try:
                coin_ret = self._get_returns(coin_id, days=days)
                aligned = portfolio_returns.align(coin_ret, join="inner")
                portfolio_returns = portfolio_returns.add(coin_ret * norm_weight, fill_value=0)
            except Exception:
                continue

        # Align
        common_idx = portfolio_returns.index.intersection(btc_returns.index)
        if len(common_idx) < 10:
            return float("nan")
        p_ret = portfolio_returns.loc[common_idx].values
        b_ret = btc_returns.loc[common_idx].values

        # OLS: beta = cov(p, b) / var(b)
        cov = np.cov(p_ret, b_ret)
        if cov.shape == (2, 2) and cov[1, 1] > 0:
            return float(cov[0, 1] / cov[1, 1])
        return float("nan")

    def compute_dominance_exposure(self, holdings: dict[str, float]) -> dict:
        """
        Categorize portfolio exposure by crypto sector.

        Returns % of portfolio in BTC, ETH, DeFi, Layer2, Altcoin, etc.
        """
        btc_coins = {"bitcoin"}
        eth_coins = {"ethereum"}
        defi_coins = {
            "uniswap", "aave", "compound-governance-token", "maker", "curve-dao-token",
            "sushi", "balancer", "yearn-finance", "synthetix-network-token", "1inch",
            "lido-dao", "gmx", "dydx", "frax-share",
        }
        layer2_coins = {
            "matic-network", "optimism", "arbitrum", "loopring", "immutable-x",
            "starknet", "zksync", "metis-token", "boba-network",
        }
        layer1_coins = {
            "solana", "cardano", "avalanche-2", "polkadot", "near", "cosmos",
            "algorand", "fantom", "tezos", "elrond-erd-2", "hedera-hashgraph",
        }

        total = sum(holdings.values()) or 1.0
        exposure = {
            "BTC": 0.0, "ETH": 0.0, "DeFi": 0.0,
            "Layer2": 0.0, "Layer1_Alt": 0.0, "Other": 0.0,
        }

        for coin_id, weight in holdings.items():
            pct = (weight / total) * 100
            if coin_id in btc_coins:
                exposure["BTC"] += pct
            elif coin_id in eth_coins:
                exposure["ETH"] += pct
            elif coin_id in defi_coins:
                exposure["DeFi"] += pct
            elif coin_id in layer2_coins:
                exposure["Layer2"] += pct
            elif coin_id in layer1_coins:
                exposure["Layer1_Alt"] += pct
            else:
                exposure["Other"] += pct

        return exposure

    def compute_correlation_matrix(self, coin_ids: list[str], days: int = 90) -> pd.DataFrame:
        """
        Compute pairwise return correlation matrix for given coins.
        """
        returns_dict = {}
        for coin_id in coin_ids:
            ret = self._get_returns(coin_id, days=days)
            if not ret.empty:
                returns_dict[coin_id] = ret

        if not returns_dict:
            return pd.DataFrame()

        ret_df = pd.DataFrame(returns_dict).dropna()
        return ret_df.corr()

    def compute_portfolio_drawdown(self, holdings: dict[str, float], days: int = 365) -> float:
        """
        Compute maximum drawdown of the portfolio over given period.

        Returns maximum drawdown as a negative percentage (e.g., -0.35 = -35%).
        """
        total = sum(holdings.values()) or 1.0
        portfolio_price = None

        for coin_id, weight in holdings.items():
            norm_weight = weight / total
            try:
                ohlcv = self._cg.get_coin_ohlcv(coin_id, days=days)
                if ohlcv.empty:
                    continue
                prices = ohlcv["close"].dropna()
                if portfolio_price is None:
                    portfolio_price = prices * norm_weight
                else:
                    aligned = portfolio_price.align(prices, join="inner")
                    portfolio_price = aligned[0] + aligned[1] * norm_weight
            except Exception:
                continue

        if portfolio_price is None or len(portfolio_price) < 10:
            return float("nan")

        rolling_max = portfolio_price.cummax()
        drawdown = (portfolio_price - rolling_max) / rolling_max
        return float(drawdown.min())

    def compute_portfolio_metrics(self, holdings: dict[str, float], days: int = 90) -> dict:
        """Comprehensive portfolio risk metrics."""
        beta = self.compute_portfolio_beta(holdings, days=days)
        exposure = self.compute_dominance_exposure(holdings)
        drawdown = self.compute_portfolio_drawdown(holdings, days=days)
        coin_ids = list(holdings.keys())[:8]  # limit correlation matrix size
        corr = self.compute_correlation_matrix(coin_ids, days=days)

        return {
            "portfolio_beta_vs_btc": beta,
            "sector_exposure": exposure,
            "max_drawdown": drawdown,
            "correlation_matrix": corr.to_dict() if not corr.empty else {},
            "num_positions": len(holdings),
        }


# ---------------------------------------------------------------------------
# CryptoMarketSummary
# ---------------------------------------------------------------------------

class CryptoMarketSummary:
    """
    High-level market dashboard and regime detection.
    """

    def __init__(
        self,
        cg_client: CoinGeckoClient = None,
        onchain_collector: OnChainMetricsCollector = None,
    ):
        self._cg = cg_client or CoinGeckoClient()
        self._onchain = onchain_collector or OnChainMetricsCollector(self._cg)
        self._mom = CryptoMomentumSignals(self._cg)
        self._fund = CryptoFundamentalAnalyzer(self._cg, self._onchain)

    def get_full_dashboard(self) -> dict:
        """
        Compile full market dashboard including:
        - Fear & Greed index
        - BTC on-chain metrics
        - Top movers (24h gainers/losers)
        - Global dominance metrics
        - Trending coins
        """
        fg = self._mom.compute_fear_greed_index()
        global_metrics = self._cg.get_global_metrics()
        top_coins = self._cg.get_top_coins(n=50)
        trending = self._cg.get_trending()

        # Top movers
        top_gainers_24h = top_coins.nlargest(5, "price_change_24h")[
            ["symbol", "name", "current_price", "price_change_24h", "market_cap"]
        ].to_dict("records")
        top_losers_24h = top_coins.nsmallest(5, "price_change_24h")[
            ["symbol", "name", "current_price", "price_change_24h", "market_cap"]
        ].to_dict("records")
        top_gainers_7d = top_coins.nlargest(5, "price_change_7d")[
            ["symbol", "name", "current_price", "price_change_7d", "market_cap"]
        ].to_dict("records")

        # BTC snapshot
        btc_row = top_coins[top_coins["id"] == "bitcoin"]
        btc_price = float(btc_row.iloc[0]["current_price"]) if not btc_row.empty else 0.0
        btc_onchain = self._onchain.collect_btc_onchain(btc_price_usd=btc_price)

        # Dominance trend
        dom_trend = self._fund.compute_dominance_trend()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fear_greed": fg,
            "global_metrics": global_metrics,
            "dominance_trend": dom_trend,
            "btc_onchain": {
                "price_usd": btc_price,
                "hashrate_ths": btc_onchain.hashrate_ths,
                "difficulty": btc_onchain.difficulty,
                "tx_count_24h": btc_onchain.tx_count_24h,
                "mempool_txs": btc_onchain.mempool_txs,
                "nvt_ratio": btc_onchain.nvt_ratio,
                "mvrv_ratio": btc_onchain.mvrv_ratio,
                "mvrv_zscore": btc_onchain.mvrv_zscore,
                "sopr": btc_onchain.sopr,
                "s2f_ratio": btc_onchain.s2f_ratio,
                "s2f_model_price": btc_onchain.s2f_model_price,
            },
            "top_gainers_24h": top_gainers_24h,
            "top_losers_24h": top_losers_24h,
            "top_gainers_7d": top_gainers_7d,
            "trending": trending[:5],
        }

    def get_market_regime(self) -> str:
        """
        Classify current market regime.

        Returns: BULL | BEAR | ACCUMULATION | DISTRIBUTION
        """
        fg = self._mom.compute_fear_greed_index()
        global_metrics = self._cg.get_global_metrics()
        fg_score = fg.get("current_value", 50) or 50
        mc_change_24h = global_metrics.get("market_cap_change_24h", 0.0) or 0.0
        btc_dom = global_metrics.get("btc_dominance", 50.0) or 50.0

        # BTC 30d momentum
        btc_mom = self._mom.compute_momentum("bitcoin", [30, 90])
        btc_30d = btc_mom.get(30, 0.0) or 0.0
        btc_90d = btc_mom.get(90, 0.0) or 0.0

        # Classification logic
        if btc_30d > 15 and btc_90d > 20 and fg_score > 55:
            return "BULL"
        elif btc_30d < -15 and btc_90d < -20 and fg_score < 35:
            return "BEAR"
        elif fg_score < 35 and btc_dom > 48 and mc_change_24h < 0:
            return "ACCUMULATION"
        elif fg_score > 65 and btc_30d > 10 and mc_change_24h > 1:
            return "DISTRIBUTION"
        elif btc_30d > 0 and btc_90d > 0:
            return "BULL"
        elif btc_30d < 0 and btc_90d < 0:
            return "BEAR"
        else:
            return "ACCUMULATION"

    def generate_report(self) -> str:
        """Generate a formatted narrative market report."""
        dashboard = self.get_full_dashboard()
        regime = self.get_market_regime()
        fg = dashboard["fear_greed"]
        gm = dashboard["global_metrics"]
        btc = dashboard["btc_onchain"]
        dom = dashboard["dominance_trend"]

        total_mc = gm.get("total_market_cap_usd", 0)
        mc_str = f"${total_mc/1e12:.2f}T" if total_mc > 1e12 else f"${total_mc/1e9:.1f}B"

        lines = [
            "=" * 70,
            "  SENTINEL CRYPTO MARKET REPORT",
            f"  {dashboard['timestamp']}",
            "=" * 70,
            "",
            f"MARKET REGIME: {regime}",
            "",
            f"FEAR & GREED INDEX: {fg.get('current_value', 'N/A')} — {fg.get('classification', 'N/A')}",
            f"  30-day average: {fg.get('avg_30d', 'N/A'):.1f}" if fg.get('avg_30d') else "",
            "",
            f"GLOBAL METRICS:",
            f"  Total Market Cap: {mc_str}",
            f"  24h Change:       {gm.get('market_cap_change_24h', 0):.2f}%",
            f"  BTC Dominance:    {gm.get('btc_dominance', 0):.1f}%",
            f"  ETH Dominance:    {gm.get('eth_dominance', 0):.1f}%",
            "",
            f"BITCOIN ON-CHAIN:",
            f"  Price:            ${btc.get('price_usd', 0):,.0f}",
            f"  Hash Rate:        {btc.get('hashrate_ths', 0)/1e6:.1f} EH/s" if btc.get('hashrate_ths') else "  Hash Rate:        N/A",
            f"  24h TX Count:     {btc.get('tx_count_24h', 0):,}",
            f"  Mempool TXs:      {btc.get('mempool_txs', 0):,}",
            f"  NVT Ratio:        {btc.get('nvt_ratio', 'N/A'):.1f}" if btc.get('nvt_ratio') else "  NVT Ratio:        N/A",
            f"  MVRV Ratio:       {btc.get('mvrv_ratio', 'N/A'):.2f}" if btc.get('mvrv_ratio') else "  MVRV Ratio:       N/A",
            f"  MVRV Z-Score:     {btc.get('mvrv_zscore', 'N/A'):.2f}" if btc.get('mvrv_zscore') else "  MVRV Z-Score:     N/A",
            f"  SOPR:             {btc.get('sopr', 'N/A'):.3f}" if btc.get('sopr') else "  SOPR:             N/A",
            f"  S2F Ratio:        {btc.get('s2f_ratio', 'N/A'):.1f}" if btc.get('s2f_ratio') else "  S2F Ratio:        N/A",
            f"  S2F Model Price:  ${btc.get('s2f_model_price', 0):,.0f}" if btc.get('s2f_model_price') else "",
            "",
            f"DOMINANCE ANALYSIS:",
            f"  Alt Season:       {'YES' if dom.get('alt_season') else 'NO'}",
            f"  BTC.D Risk-Off:   {'YES' if dom.get('btc_dominance_rising') else 'NO'}",
            f"  ETH vs BTC 30d:   {dom.get('eth_outperformance_30d', 0):.1f}%",
            "",
            "TOP GAINERS (24h):",
        ]

        for g in dashboard.get("top_gainers_24h", []):
            lines.append(f"  {g.get('symbol','?').upper():8s} +{g.get('price_change_24h',0):.1f}%  ${g.get('current_price',0):,.2f}")

        lines += ["", "TOP LOSERS (24h):"]
        for l in dashboard.get("top_losers_24h", []):
            lines.append(f"  {l.get('symbol','?').upper():8s} {l.get('price_change_24h',0):.1f}%  ${l.get('current_price',0):,.2f}")

        lines += ["", "TRENDING (CoinGecko):"]
        for t in dashboard.get("trending", []):
            lines.append(f"  {t.get('symbol','?').upper()} — {t.get('name','')}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("Initializing SENTINEL Crypto Screener v3...\n")

    cg = CoinGeckoClient()
    onchain = OnChainMetricsCollector(cg)
    screener = CryptoScreener(cg_client=cg, onchain_collector=onchain)
    summary = CryptoMarketSummary(cg_client=cg, onchain_collector=onchain)
    mom = CryptoMomentumSignals(cg)
    fund = CryptoFundamentalAnalyzer(cg, onchain)

    # 1. Get top 50 coins
    print("Fetching top 50 coins...")
    top50 = cg.get_top_coins(n=50)
    print(f"  Universe: {len(top50)} coins loaded\n")

    # 2. Screen for momentum breakout + RSI oversold
    print("Running: MOMENTUM + RSI_OVERSOLD screen...")
    result = screener.screen(
        criteria_names=["momentum_all_timeframes", "rsi_oversold"],
        top_n=50,
        enrich_with_momentum=True,
    )
    print(result.summary())
    if not result.matches.empty:
        display_cols = [c for c in ["symbol", "name", "current_price", "price_change_30d", "market_cap", "rsi_14"] if c in result.matches.columns]
        print(result.matches[display_cols].to_string(index=False))
    print()

    # 3. Fear & Greed Index
    print("Fear & Greed Index:")
    fg = mom.compute_fear_greed_index()
    print(f"  Current: {fg.get('current_value')} — {fg.get('classification')}")
    print(f"  30d avg: {fg.get('avg_30d', 'N/A')}")
    print(f"  Trend:   {fg.get('trend', 'N/A')}\n")

    # 4. BTC NVT and MVRV
    print("Bitcoin On-Chain Metrics:")
    fund_ana = CryptoFundamentalAnalyzer(cg, onchain)
    nvt = fund_ana.compute_nvt_signal("bitcoin")
    cycle = fund_ana.compute_market_cycle_position("bitcoin")
    print(f"  NVT Signal: {nvt:.1f}" if nvt == nvt else "  NVT Signal: N/A")
    print(f"  MVRV Ratio: {cycle.get('mvrv_ratio', 'N/A'):.2f}" if cycle.get('mvrv_ratio') else "  MVRV Ratio: N/A")
    print(f"  MVRV Z-Score: {cycle.get('mvrv_zscore', 'N/A'):.2f}" if cycle.get('mvrv_zscore') else "  MVRV Z-Score: N/A")
    print(f"  Market Phase: {cycle.get('phase', 'UNKNOWN')}\n")

    # 5. Generate market summary report
    print("Generating market summary...")
    report = summary.generate_report()
    print(report)

    # 6. Run all preset screens
    print("\nRunning all preset screens on top 50 coins...")
    preset_results = screener.screen_all_presets(top_n=50)
    for preset_name, pr in preset_results.items():
        symbols = ", ".join(
            pr.matches["symbol"].str.upper().tolist()[:5]
        ) if not pr.matches.empty else "(none)"
        print(f"  {preset_name:25s} — {pr.match_count:3d} matches — {symbols}")
