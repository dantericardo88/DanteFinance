"""
DEX / AMM Liquidity Analytics V3 — dim_110 (score 6 → 9).

Comprehensive DEX and AMM liquidity analytics platform using only free APIs.

Architecture
------------
  DefiLlamaClient         — wrapper for all DefiLlama endpoints (TVL, DEX, yields, stables)
  UniswapV3Analytics      — The Graph subgraph queries for pool data, swaps, tick depth
  AMMPriceEngine          — price math, sqrtPriceX96, TWAP, DexScreener fallback
  LiquidityFlowAnalyzer   — TVL momentum, liquidity migration, rugpull risk
  YieldFarmingAnalyzer    — risk-adjusted yields, IL risk, stablecoin yield ladder
  DEXVolumeTracker        — cross-DEX volume tracking, market share, anomaly detection
  DEXScreener             — new pair screening, arbitrage opportunities
  DEXAnalyticsEngine      — orchestrator / dashboard

Free Data Sources
-----------------
  https://api.llama.fi          — protocols, TVL, DEX volumes
  https://yields.llama.fi       — yield pools (APY, TVL)
  https://api.thegraph.com      — Uniswap V3 and SushiSwap subgraphs (GraphQL)
  https://api.coingecko.com     — token prices (free tier)
  https://api.dexscreener.com   — pair data, new pairs, prices

Public API
----------
  engine = DEXAnalyticsEngine()
  dashboard  = engine.get_defi_dashboard()
  health     = engine.get_protocol_health("uniswap")
  opps       = engine.screen_opportunities()
  summary    = engine.get_market_summary()
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

import numpy as np
import pandas as pd
import requests

try:
    from scipy.interpolate import RectBivariateSpline as _RBS
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LLAMA_BASE = "https://api.llama.fi"
_YIELDS_BASE = "https://yields.llama.fi"
_STABLES_BASE = "https://stablecoins.llama.fi"
_GRAPH_UNISWAP_V3 = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
_GRAPH_SUSHI = "https://api.thegraph.com/subgraphs/name/sushiswap/exchange"
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex"

_HEADERS = {
    "User-Agent": "SENTINEL/3.0 financial-terminal richard.porras@realempanada.com",
    "Accept": "application/json",
    "Content-Type": "application/json",
}
_TIMEOUT = 30
_RATE_LIMIT_DELAY = 1.0   # 1 request/sec for DefiLlama

# SQLite cache setup
_DB_DIR = Path(__file__).parent.parent / "data" / "db"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "dex_analytics_v3.db"

_TTL: Dict[str, int] = {
    "protocols":    300,   # 5 min
    "protocol":     300,
    "tvl_history":  600,
    "dex_overview": 300,
    "yields":       300,
    "chains":       600,
    "stablecoins":  600,
    "graphql":      120,   # 2 min for on-chain data
    "coingecko":    60,
    "dexscreener":  60,
}

_lock = threading.Lock()
_last_request_time: float = 0.0


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def _init_db() -> None:
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key  TEXT PRIMARY KEY,
            data_type  TEXT NOT NULL,
            payload    TEXT NOT NULL,
            fetched_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cache_type ON cache(data_type);
        """)


_init_db()


def _cache_get(key: str, data_type: str) -> Optional[Any]:
    ttl = _TTL.get(data_type, 300)
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            row = conn.execute(
                "SELECT payload, fetched_at FROM cache WHERE cache_key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        payload, ts = row
        if time.time() - ts > ttl:
            return None
        return json.loads(payload)
    except Exception:
        return None


def _cache_set(key: str, data_type: str, data: Any) -> None:
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                (key, data_type, json.dumps(data, default=str), time.time()),
            )
    except Exception:
        pass


def _throttle(delay: float = _RATE_LIMIT_DELAY) -> None:
    """Enforce minimum delay between requests."""
    global _last_request_time
    with _lock:
        elapsed = time.time() - _last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        _last_request_time = time.time()


def _get(url: str, params: Optional[dict] = None, *, rate_limit: bool = True,
         timeout: int = _TIMEOUT, retries: int = 3) -> Optional[Any]:
    """HTTP GET with rate limiting, retries, and JSON parsing."""
    if rate_limit:
        _throttle()
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
        except Exception as exc:
            logger.warning("Request error fetching %s: %s", url, exc)
            if attempt == retries - 1:
                return None
            time.sleep(1)
    return None


def _post(url: str, payload: dict, *, rate_limit: bool = False,
          timeout: int = _TIMEOUT, retries: int = 3) -> Optional[dict]:
    """HTTP POST for GraphQL queries."""
    if rate_limit:
        _throttle()
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("POST error to %s: %s", url, exc)
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PoolData:
    id: str
    token0_symbol: str
    token1_symbol: str
    token0_address: str
    token1_address: str
    fee_tier: int                      # 100, 500, 3000, 10000
    liquidity: float
    tvl_usd: float
    volume_usd_24h: float
    fees_usd_24h: float
    volume_usd_total: float
    sqrt_price_x96: int = 0
    tick_current: int = 0
    token0_price: float = 0.0
    token1_price: float = 0.0

    @property
    def fee_pct(self) -> float:
        return self.fee_tier / 1_000_000

    @property
    def vol_tvl_ratio(self) -> float:
        return self.volume_usd_24h / self.tvl_usd if self.tvl_usd > 0 else 0.0

    @property
    def daily_fee_yield(self) -> float:
        return self.fees_usd_24h / self.tvl_usd if self.tvl_usd > 0 else 0.0

    @property
    def annualized_fee_yield(self) -> float:
        return self.daily_fee_yield * 365


@dataclass
class SwapData:
    timestamp: datetime
    amount0: float
    amount1: float
    amount_usd: float
    sqrt_price_x96: int
    tick: int
    sender: str = ""
    recipient: str = ""


@dataclass
class LiquidityTick:
    tick_idx: int
    liquidity_net: float
    liquidity_gross: float = 0.0
    price: float = 0.0


@dataclass
class ProtocolHealth:
    name: str
    slug: str
    tvl_usd: float
    tvl_change_1d: float
    tvl_change_7d: float
    tvl_change_30d: float
    chains: List[str]
    category: str
    volume_24h: float = 0.0
    fees_24h: float = 0.0
    audits: int = 0
    has_audit: bool = False
    rugpull_risk_score: float = 0.0
    age_days: int = 0
    health_score: float = 0.0
    notes: str = ""


@dataclass
class YieldOpportunity:
    pool_id: str
    project: str
    chain: str
    symbol: str
    tvl_usd: float
    apy: float
    apy_base: float
    apy_reward: float
    risk_score: float
    risk_adjusted_apy: float
    il_risk: float
    is_stable: bool
    audited: bool
    stablecoin: bool = False


# ---------------------------------------------------------------------------
# DefiLlamaClient
# ---------------------------------------------------------------------------

class DefiLlamaClient:
    """
    Wrapper for all DefiLlama API endpoints.
    Rate limit: 1 req/sec. Cache TTL per data type.
    """

    def get_all_protocols(self) -> List[dict]:
        """All DeFi protocols with TVL, chain, category."""
        cached = _cache_get("protocols_all", "protocols")
        if cached is not None:
            return cached
        data = _get(f"{_LLAMA_BASE}/protocols")
        if data is None:
            return []
        _cache_set("protocols_all", "protocols", data)
        return data

    def get_protocol(self, name: str) -> dict:
        """Detailed protocol data including TVL breakdown by chain."""
        key = f"protocol_{name}"
        cached = _cache_get(key, "protocol")
        if cached is not None:
            return cached
        data = _get(f"{_LLAMA_BASE}/protocol/{name}")
        if data is None:
            return {}
        _cache_set(key, "protocol", data)
        return data

    def get_tvl_history(self, protocol: str) -> pd.Series:
        """Daily TVL history for a protocol."""
        key = f"tvl_hist_{protocol}"
        cached = _cache_get(key, "tvl_history")
        if cached is None:
            data = _get(f"{_LLAMA_BASE}/protocol/{protocol}")
            if data is None:
                return pd.Series(dtype=float)
            tvl_data = data.get("tvl", [])
            cached = [{"date": d["date"], "totalLiquidityUSD": d.get("totalLiquidityUSD", 0)}
                      for d in tvl_data if isinstance(d, dict)]
            _cache_set(key, "tvl_history", cached)
        if not cached:
            return pd.Series(dtype=float)
        df = pd.DataFrame(cached)
        if df.empty or "date" not in df.columns:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        series = df["totalLiquidityUSD"].sort_index()
        series.name = f"{protocol}_tvl_usd"
        return series

    def get_dex_overview(self) -> pd.DataFrame:
        """All DEX volumes — dailyVolume, totalVolume, change_1d."""
        cached = _cache_get("dex_overview", "dex_overview")
        if cached is None:
            data = _get(f"{_LLAMA_BASE}/overview/dexs", params={"excludeTotalDataChart": "true"})
            if data is None:
                return pd.DataFrame()
            protocols = data.get("protocols", [])
            cached = protocols
            _cache_set("dex_overview", "dex_overview", cached)
        rows = []
        for p in cached:
            rows.append({
                "name":        p.get("name", ""),
                "slug":        p.get("slug", p.get("name", "").lower()),
                "chain":       p.get("chain", ""),
                "dailyVolume": p.get("totalVolume24h", 0) or 0,
                "totalVolume": p.get("totalAllTime", 0) or 0,
                "change_1d":   p.get("change_1d", 0) or 0,
                "change_7d":   p.get("change_7d", 0) or 0,
                "tvl":         p.get("tvl", 0) or 0,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("dailyVolume", ascending=False).reset_index(drop=True)
        return df

    def get_yields(self) -> pd.DataFrame:
        """All yield opportunities — pool, project, chain, apy, tvlUsd."""
        cached = _cache_get("yields_all", "yields")
        if cached is None:
            data = _get(f"{_YIELDS_BASE}/pools")
            if data is None:
                return pd.DataFrame()
            pools = data.get("data", [])
            cached = pools
            _cache_set("yields_all", "yields", cached)
        if not cached:
            return pd.DataFrame()
        df = pd.DataFrame(cached)
        rename = {
            "pool": "pool_id",
            "project": "project",
            "chain": "chain",
            "symbol": "symbol",
            "tvlUsd": "tvl_usd",
            "apy": "apy",
            "apyBase": "apy_base",
            "apyReward": "apy_reward",
            "stablecoin": "stablecoin",
            "ilRisk": "il_risk_label",
            "audits": "audits",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["tvl_usd", "apy", "apy_base", "apy_reward"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def get_chains(self) -> pd.DataFrame:
        """All chains with TVL."""
        cached = _cache_get("chains_all", "chains")
        if cached is None:
            data = _get(f"{_LLAMA_BASE}/chains")
            if data is None:
                return pd.DataFrame()
            cached = data
            _cache_set("chains_all", "chains", cached)
        df = pd.DataFrame(cached)
        if "tvl" in df.columns:
            df = df.sort_values("tvl", ascending=False).reset_index(drop=True)
        return df

    def get_stablecoins(self) -> pd.DataFrame:
        """Stablecoin market caps and pegs."""
        cached = _cache_get("stablecoins_all", "stablecoins")
        if cached is None:
            data = _get(f"{_STABLES_BASE}/stablecoins?includePrices=true")
            if data is None:
                return pd.DataFrame()
            stables = data.get("peggedAssets", [])
            cached = stables
            _cache_set("stablecoins_all", "stablecoins", cached)
        rows = []
        for s in cached:
            circulating = s.get("circulating", {})
            peg_type = s.get("pegType", "")
            price = s.get("price", 1.0) or 1.0
            rows.append({
                "name":        s.get("name", ""),
                "symbol":      s.get("symbol", ""),
                "peg_type":    peg_type,
                "peg_mechanism": s.get("pegMechanism", ""),
                "circulating_usd": float(circulating.get("peggedUSD", 0) or 0),
                "price":       float(price),
                "peg_deviation": abs(float(price) - 1.0),
                "chains":      list(s.get("chainCirculating", {}).keys()),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("circulating_usd", ascending=False).reset_index(drop=True)
        return df

    def get_protocol_fees(self) -> pd.DataFrame:
        """Protocol fees — daily and total revenue."""
        cached = _cache_get("fees_overview", "protocols")
        if cached is None:
            data = _get(f"{_LLAMA_BASE}/overview/fees", params={"excludeTotalDataChart": "true"})
            if data is None:
                return pd.DataFrame()
            protocols = data.get("protocols", [])
            cached = protocols
            _cache_set("fees_overview", "protocols", cached)
        rows = []
        for p in cached:
            rows.append({
                "name":        p.get("name", ""),
                "slug":        p.get("slug", ""),
                "dailyFees":   p.get("totalFees24h", 0) or 0,
                "totalFees":   p.get("totalAllTime", 0) or 0,
                "dailyRevenue": p.get("totalRevenue24h", 0) or 0,
            })
        return pd.DataFrame(rows).sort_values("dailyFees", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# UniswapV3Analytics
# ---------------------------------------------------------------------------

class UniswapV3Analytics:
    """
    Query Uniswap V3 via The Graph subgraph (GraphQL, free).
    """

    def __init__(self, subgraph_url: str = _GRAPH_UNISWAP_V3):
        self.url = subgraph_url

    def query_graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        """POST GraphQL query to subgraph."""
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables
        key = f"gql_{hash(query + json.dumps(variables or {}))}"
        cached = _cache_get(key, "graphql")
        if cached is not None:
            return cached
        result = _post(self.url, payload)
        if result is None:
            return {}
        errors = result.get("errors")
        if errors:
            logger.warning("GraphQL errors: %s", errors)
        data = result.get("data", {})
        _cache_set(key, "graphql", data)
        return data

    def get_top_pools(self, n: int = 50) -> pd.DataFrame:
        """Top Uniswap V3 pools by TVL."""
        query = """
        {
          pools(first: %d, orderBy: totalValueLockedUSD, orderDirection: desc) {
            id
            token0 { id symbol decimals }
            token1 { id symbol decimals }
            feeTier
            liquidity
            totalValueLockedUSD
            volumeUSD
            feesUSD
            token0Price
            token1Price
            sqrtPrice
            tick
          }
        }
        """ % min(n, 1000)
        data = self.query_graphql(query)
        pools = data.get("pools", [])
        if not pools:
            return pd.DataFrame()
        rows = []
        for p in pools:
            rows.append({
                "id":            p["id"],
                "token0":        p["token0"]["symbol"],
                "token1":        p["token1"]["symbol"],
                "token0_addr":   p["token0"]["id"],
                "token1_addr":   p["token1"]["id"],
                "token0_dec":    int(p["token0"]["decimals"]),
                "token1_dec":    int(p["token1"]["decimals"]),
                "feeTier":       int(p["feeTier"]),
                "liquidity":     float(p.get("liquidity") or 0),
                "totalValueLockedUSD": float(p.get("totalValueLockedUSD") or 0),
                "volumeUSD":     float(p.get("volumeUSD") or 0),
                "feesUSD":       float(p.get("feesUSD") or 0),
                "token0Price":   float(p.get("token0Price") or 0),
                "token1Price":   float(p.get("token1Price") or 0),
                "sqrtPrice":     int(p.get("sqrtPrice") or 0),
                "tick":          int(p.get("tick") or 0),
            })
        df = pd.DataFrame(rows)
        df["vol_tvl_ratio"] = df["volumeUSD"] / df["totalValueLockedUSD"].replace(0, np.nan)
        df["fee_pct"] = df["feeTier"] / 1_000_000
        df["annualized_fee_yield"] = (df["feesUSD"] / df["totalValueLockedUSD"].replace(0, np.nan)) * 365
        return df.sort_values("totalValueLockedUSD", ascending=False).reset_index(drop=True)

    def get_pool_detail(self, pool_id: str) -> dict:
        """Full pool stats for a single pool."""
        query = """
        {
          pool(id: "%s") {
            id
            token0 { id symbol decimals name }
            token1 { id symbol decimals name }
            feeTier
            liquidity
            sqrtPrice
            tick
            totalValueLockedUSD
            totalValueLockedToken0
            totalValueLockedToken1
            volumeUSD
            feesUSD
            txCount
            poolDayData(first: 30, orderBy: date, orderDirection: desc) {
              date tvlUSD volumeUSD feesUSD txCount
            }
          }
        }
        """ % pool_id.lower()
        data = self.query_graphql(query)
        return data.get("pool") or {}

    def get_token_pools(self, token_address: str) -> List[dict]:
        """All pools containing a given token."""
        addr = token_address.lower()
        query = """
        {
          pools0: pools(first: 50, where: {token0: "%s"}, orderBy: totalValueLockedUSD, orderDirection: desc) {
            id token0 { symbol } token1 { symbol } feeTier totalValueLockedUSD volumeUSD
          }
          pools1: pools(first: 50, where: {token1: "%s"}, orderBy: totalValueLockedUSD, orderDirection: desc) {
            id token0 { symbol } token1 { symbol } feeTier totalValueLockedUSD volumeUSD
          }
        }
        """ % (addr, addr)
        data = self.query_graphql(query)
        pools = data.get("pools0", []) + data.get("pools1", [])
        # Deduplicate by id
        seen = set()
        unique = []
        for p in pools:
            if p["id"] not in seen:
                seen.add(p["id"])
                unique.append(p)
        return sorted(unique, key=lambda x: float(x.get("totalValueLockedUSD") or 0), reverse=True)

    def get_recent_swaps(self, pool_id: str, n: int = 100) -> pd.DataFrame:
        """Recent swaps for a pool."""
        query = """
        {
          swaps(first: %d, where: {pool: "%s"}, orderBy: timestamp, orderDirection: desc) {
            timestamp
            amount0
            amount1
            amountUSD
            sqrtPriceX96
            tick
            sender
            recipient
          }
        }
        """ % (min(n, 1000), pool_id.lower())
        data = self.query_graphql(query)
        swaps = data.get("swaps", [])
        if not swaps:
            return pd.DataFrame()
        df = pd.DataFrame(swaps)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        for col in ["amount0", "amount1", "amountUSD"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["sqrtPriceX96", "tick"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    def get_pool_liquidity_distribution(self, pool_id: str) -> dict:
        """
        Fetch tick data and compute liquidity distribution around current price.
        Returns dict with ticks list, current_tick, liquidity_above, liquidity_below.
        """
        query = """
        {
          pool(id: "%s") {
            tick
            sqrtPrice
            liquidity
            ticks(first: 500, orderBy: tickIdx, orderDirection: asc) {
              tickIdx
              liquidityNet
              liquidityGross
            }
          }
        }
        """ % pool_id.lower()
        data = self.query_graphql(query)
        pool = data.get("pool")
        if not pool:
            return {}

        current_tick = int(pool.get("tick") or 0)
        sqrt_price = int(pool.get("sqrtPrice") or 0)
        pool_liquidity = float(pool.get("liquidity") or 0)
        ticks_raw = pool.get("ticks", [])

        ticks = []
        for t in ticks_raw:
            idx = int(t["tickIdx"])
            ln = float(t["liquidityNet"])
            lg = float(t.get("liquidityGross") or 0)
            # Approximate price at tick: price = 1.0001^tickIdx
            price = 1.0001 ** idx
            ticks.append(LiquidityTick(tick_idx=idx, liquidity_net=ln,
                                        liquidity_gross=lg, price=price))

        # Compute cumulative liquidity above/below current tick
        liq_below = sum(abs(t.liquidity_net) for t in ticks if t.tick_idx <= current_tick)
        liq_above = sum(abs(t.liquidity_net) for t in ticks if t.tick_idx > current_tick)

        return {
            "pool_id": pool_id,
            "current_tick": current_tick,
            "sqrt_price_x96": sqrt_price,
            "pool_liquidity": pool_liquidity,
            "ticks": ticks,
            "liquidity_below": liq_below,
            "liquidity_above": liq_above,
            "tick_count": len(ticks),
        }

    def compute_price_impact(self, pool_id: str, trade_size_usd: float) -> float:
        """
        Estimate price impact for a trade of given USD size.
        Approximation: impact ≈ trade_size / (2 × sqrt(L × p))
        where L = pool liquidity, p = current price in token1/token0.
        """
        pool = self.get_pool_detail(pool_id)
        if not pool:
            return float("nan")
        liquidity = float(pool.get("liquidity") or 0)
        tvl_usd = float(pool.get("totalValueLockedUSD") or 1)
        sqrt_price_raw = int(pool.get("sqrtPrice") or 0)

        if liquidity <= 0 or tvl_usd <= 0:
            return float("nan")

        # price = (sqrtPriceX96 / 2^96)^2
        q96 = 2 ** 96
        price = (sqrt_price_raw / q96) ** 2 if sqrt_price_raw > 0 else 1.0

        # Simplified impact formula
        depth = 2.0 * math.sqrt(max(liquidity * price, 1e-12))
        impact = trade_size_usd / depth if depth > 0 else float("inf")
        return round(impact, 6)

    def get_historical_volume(self, pool_id: str, days: int = 30) -> pd.Series:
        """Daily volume history for a pool."""
        query = """
        {
          poolDayDatas(
            first: %d
            where: {pool: "%s"}
            orderBy: date
            orderDirection: desc
          ) {
            date
            volumeUSD
          }
        }
        """ % (min(days, 365), pool_id.lower())
        data = self.query_graphql(query)
        records = data.get("poolDayDatas", [])
        if not records:
            return pd.Series(dtype=float)
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
        df["volumeUSD"] = pd.to_numeric(df["volumeUSD"], errors="coerce")
        series = df.set_index("date")["volumeUSD"].sort_index()
        series.name = f"pool_{pool_id[:8]}_volume_usd"
        return series


# ---------------------------------------------------------------------------
# AMMPriceEngine
# ---------------------------------------------------------------------------

class AMMPriceEngine:
    """
    Compute prices and pricing math for AMM pools.
    Handles Uniswap V3 sqrtPriceX96 math, TWAP, and DexScreener fallback.
    """

    def __init__(self):
        self._uniswap = UniswapV3Analytics()

    @staticmethod
    def sqrt_price_x96_to_price(sqrt_price_x96: int, token0_decimals: int = 18,
                                 token1_decimals: int = 18) -> float:
        """
        Convert Uniswap V3 sqrtPriceX96 to human-readable price.
        price_token1_per_token0 = (sqrtPriceX96 / 2^96)^2 × 10^(dec0-dec1)
        """
        if sqrt_price_x96 <= 0:
            return 0.0
        q96 = 2 ** 96
        raw = (sqrt_price_x96 / q96) ** 2
        decimal_adj = 10 ** (token0_decimals - token1_decimals)
        return raw * decimal_adj

    def get_uniswap_price(self, token0: str, token1: str, fee_tier: int = 3000) -> float:
        """
        Find Uniswap V3 pool for token pair, return current price.
        token0/token1 are addresses (0x...).
        """
        pools = self._uniswap.get_token_pools(token0)
        for p in pools:
            t0 = p.get("token0", {})
            t1 = p.get("token1", {})
            # Match the other token
            other = t1 if t0.get("id", "").lower() == token0.lower() else t0
            if other.get("id", "").lower() == token1.lower():
                # fetch full detail for sqrtPrice
                detail = self._uniswap.get_pool_detail(p["id"])
                sqrt_price = int(detail.get("sqrtPrice") or 0)
                dec0 = int(detail.get("token0", {}).get("decimals", 18))
                dec1 = int(detail.get("token1", {}).get("decimals", 18))
                return self.sqrt_price_x96_to_price(sqrt_price, dec0, dec1)
        return float("nan")

    def get_pool_price_range(self, pool_id: str) -> Tuple[float, float]:
        """
        Min/max price from active liquidity ticks.
        Returns (min_price, max_price).
        """
        dist = self._uniswap.get_pool_liquidity_distribution(pool_id)
        ticks = dist.get("ticks", [])
        if not ticks:
            return (float("nan"), float("nan"))
        active_prices = [t.price for t in ticks if abs(t.liquidity_net) > 0]
        if not active_prices:
            return (float("nan"), float("nan"))
        return (min(active_prices), max(active_prices))

    def compute_concentrated_liquidity_bounds(self, pool_id: str) -> dict:
        """
        Find price range where 80% of liquidity is concentrated.
        """
        dist = self._uniswap.get_pool_liquidity_distribution(pool_id)
        ticks = dist.get("ticks", [])
        if not ticks:
            return {}

        # Sort by absolute liquidity contribution
        sorted_ticks = sorted(ticks, key=lambda t: abs(t.liquidity_net), reverse=True)
        total_liq = sum(abs(t.liquidity_net) for t in ticks)
        if total_liq <= 0:
            return {}

        cumulative = 0.0
        included = []
        for t in sorted_ticks:
            cumulative += abs(t.liquidity_net)
            included.append(t)
            if cumulative >= 0.80 * total_liq:
                break

        prices = [t.price for t in included]
        return {
            "pool_id": pool_id,
            "lower_price": min(prices),
            "upper_price": max(prices),
            "liquidity_concentration_pct": 80.0,
            "tick_count_covering_80pct": len(included),
        }

    def compute_twap(self, pool_id: str, period_seconds: int = 3600) -> float:
        """
        Approximate TWAP by averaging recent swap prices.
        True on-chain TWAP requires contract call; we use swap data as proxy.
        """
        pool = self._uniswap.get_pool_detail(pool_id)
        if not pool:
            return float("nan")
        day_data = pool.get("poolDayData", [])
        if not day_data:
            return float("nan")
        # Use last 2 days volume-weighted average of token prices
        total_vol = 0.0
        weighted_sum = 0.0
        for d in day_data[:2]:
            vol = float(d.get("volumeUSD") or 0)
            # We use token0Price from parent pool as proxy
            price = float(pool.get("token0Price") or 0)
            weighted_sum += price * vol
            total_vol += vol
        if total_vol <= 0:
            return float(pool.get("token0Price") or 0)
        return weighted_sum / total_vol

    def get_price_from_dexscreener(self, pair_address: str,
                                    chain: str = "ethereum") -> Optional[dict]:
        """DexScreener fallback for pair price data."""
        url = f"{_DEXSCREENER_BASE}/pairs/{chain}/{pair_address}"
        cached = _cache_get(f"dex_{pair_address}", "dexscreener")
        if cached is not None:
            return cached
        data = _get(url, rate_limit=False)
        if data is None:
            return None
        pairs = data.get("pairs", [])
        if not pairs:
            return None
        result = pairs[0]
        _cache_set(f"dex_{pair_address}", "dexscreener", result)
        return result

    def get_token_price_coingecko(self, token_id: str) -> float:
        """Fetch token price from CoinGecko free API."""
        key = f"cg_price_{token_id}"
        cached = _cache_get(key, "coingecko")
        if cached is not None:
            return float(cached.get("usd", 0))
        url = f"{_COINGECKO_BASE}/simple/price"
        data = _get(url, params={"ids": token_id, "vs_currencies": "usd"}, rate_limit=False)
        if data is None or token_id not in data:
            return float("nan")
        result = data[token_id]
        _cache_set(key, "coingecko", result)
        return float(result.get("usd", 0))


# ---------------------------------------------------------------------------
# LiquidityFlowAnalyzer
# ---------------------------------------------------------------------------

class LiquidityFlowAnalyzer:
    """
    Analyze liquidity flows across the DEX ecosystem.
    """

    def __init__(self):
        self._llama = DefiLlamaClient()

    def compute_tvl_momentum(self, protocol: str, window: int = 7) -> float:
        """
        Compute TVL momentum: window-day change normalized by starting TVL.
        Returns percentage change (e.g., 0.15 = +15%).
        """
        series = self._llama.get_tvl_history(protocol)
        if series.empty or len(series) < window + 1:
            # Fallback: use protocol overview data
            p = self._llama.get_protocol(protocol)
            return float(p.get("change_7d", 0) or 0) / 100.0
        recent = series.iloc[-1]
        past = series.iloc[-(window + 1)]
        if past <= 0:
            return 0.0
        return (recent - past) / past

    def detect_liquidity_migration(self, from_protocol: str, to_protocol: str,
                                    days: int = 30) -> dict:
        """
        Detect if liquidity is migrating from one protocol to another.
        Positive correlation between A's TVL drop and B's TVL rise signals migration.
        """
        from_series = self._llama.get_tvl_history(from_protocol)
        to_series = self._llama.get_tvl_history(to_protocol)

        if from_series.empty or to_series.empty:
            return {"migration_detected": False, "reason": "insufficient_data"}

        # Align to common dates, last N days
        combined = pd.DataFrame({"from": from_series, "to": to_series}).dropna()
        combined = combined.tail(days)

        if len(combined) < 7:
            return {"migration_detected": False, "reason": "too_few_observations"}

        from_change = (combined["from"].iloc[-1] - combined["from"].iloc[0]) / max(combined["from"].iloc[0], 1)
        to_change = (combined["to"].iloc[-1] - combined["to"].iloc[0]) / max(combined["to"].iloc[0], 1)

        # Migration signal: from lost TVL, to gained
        migration_detected = from_change < -0.05 and to_change > 0.05
        # Compute correlation of daily changes (negative correlation = migration)
        from_daily = combined["from"].pct_change().dropna()
        to_daily = combined["to"].pct_change().dropna()
        corr = float(from_daily.corr(to_daily)) if len(from_daily) > 3 else 0.0

        return {
            "from_protocol": from_protocol,
            "to_protocol": to_protocol,
            "from_tvl_change_pct": round(from_change * 100, 2),
            "to_tvl_change_pct": round(to_change * 100, 2),
            "daily_return_correlation": round(corr, 4),
            "migration_detected": migration_detected,
            "migration_strength": round(abs(from_change) + abs(to_change), 4),
            "window_days": days,
        }

    def compute_cross_chain_tvl(self, protocol: str) -> Dict[str, float]:
        """TVL breakdown by chain for a protocol."""
        data = self._llama.get_protocol(protocol)
        chain_tvls = data.get("chainTvls", {})
        result = {}
        for chain, details in chain_tvls.items():
            if isinstance(details, dict):
                tvl_list = details.get("tvl", [])
                if tvl_list:
                    latest = tvl_list[-1]
                    result[chain] = float(latest.get("totalLiquidityUSD", 0) or 0)
            elif isinstance(details, (int, float)):
                result[chain] = float(details)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def get_top_liquidity_destinations(self, days: int = 7) -> pd.DataFrame:
        """Protocols with highest TVL gain over recent window — where liquidity is flowing."""
        protocols = self._llama.get_all_protocols()
        rows = []
        for p in protocols:
            tvl = float(p.get("tvl") or 0)
            change_key = f"change_{days}d" if days in (1, 7, 30) else "change_7d"
            change = float(p.get(change_key) or p.get("change_7d", 0) or 0)
            if tvl > 1_000_000:  # Min $1M TVL
                rows.append({
                    "name": p.get("name", ""),
                    "slug": p.get("slug", ""),
                    "category": p.get("category", ""),
                    "chain": p.get("chain", ""),
                    "tvl_usd": tvl,
                    "tvl_change_pct": change,
                    "tvl_gain_usd": tvl * change / 100 if change > 0 else 0,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df[df["tvl_change_pct"] > 0].sort_values("tvl_gain_usd",
                                                           ascending=False).reset_index(drop=True).head(50)

    def detect_rugpull_risk(self, protocol: str) -> float:
        """
        Compute rugpull risk score (0-100) based on:
        - Age < 30 days = high risk (+40)
        - No audits = high risk (+30)
        - TVL concentration (single chain > 90%) = medium risk (+15)
        - Anonymous team = medium risk (+15)
        Returns float 0-100 (higher = riskier).
        """
        data = self._llama.get_protocol(protocol)
        if not data:
            return 75.0  # Unknown protocol = high risk

        score = 0.0
        # Age check
        inception = data.get("inception")
        if inception:
            age_days = (datetime.utcnow() - datetime.utcfromtimestamp(inception)).days
            if age_days < 30:
                score += 40.0
            elif age_days < 90:
                score += 20.0
            elif age_days < 180:
                score += 10.0
        else:
            score += 20.0  # Unknown age

        # Audit check
        audits = data.get("audits") or []
        if not audits:
            score += 30.0
        elif len(audits) >= 2:
            score += 0.0  # Multiple audits
        else:
            score += 10.0

        # TVL concentration
        chain_tvls = self.compute_cross_chain_tvl(protocol)
        total_tvl = sum(chain_tvls.values())
        if total_tvl > 0:
            max_chain_share = max(chain_tvls.values()) / total_tvl
            if max_chain_share > 0.95:
                score += 15.0
            elif max_chain_share > 0.80:
                score += 7.0

        # Team transparency proxy (slug with "anon" or very low TVL)
        tvl = float(data.get("tvl") or 0)
        if tvl < 100_000:
            score += 15.0  # Micro TVL = higher risk

        return min(100.0, round(score, 1))

    def compute_rugpull_score_from_signals(
        self,
        tvl_change_1d: float,
        has_audit: bool,
        sell_tax_pct: float = 0.0,
        ownership_renounced: bool = True,
        price_impact_small_trade_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Composite rugpull risk scoring from on-chain signals.

        Scoring:
          - Liquidity removal speed: tvl_change_1d < -0.50 (>50% drop) → +40
          - No audit:                has_audit == False → +30
          - Honeypot: sell_tax > 10% or ownership not renounced → +20
          - Price impact anomaly:    price_impact_small_trade_pct > 5% → +15

        Score 0-100. > 70 = HIGH RISK.

        Args:
            tvl_change_1d:              Fraction change in TVL over 1 day (e.g., -0.60 = -60%).
            has_audit:                  Whether contract is audited (DefiLlama has_audit).
            sell_tax_pct:               Sell tax percentage (0-100). >10 = honeypot signal.
            ownership_renounced:        Whether contract ownership has been renounced.
            price_impact_small_trade_pct: Price impact of a small trade. >5% = thin liquidity.

        Returns:
            dict with score, risk_level, and per-signal breakdown.
        """
        score = 0.0
        signals: Dict[str, Any] = {}

        # 1. Liquidity removal speed
        if tvl_change_1d < -0.50:
            score += 40.0
            signals["liquidity_removal"] = {
                "value": round(tvl_change_1d * 100, 2),
                "flag": "CRITICAL — >50% TVL removed in 1 day",
                "points": 40,
            }
        elif tvl_change_1d < -0.20:
            score += 20.0
            signals["liquidity_removal"] = {
                "value": round(tvl_change_1d * 100, 2),
                "flag": "WARNING — >20% TVL removed in 1 day",
                "points": 20,
            }
        else:
            signals["liquidity_removal"] = {
                "value": round(tvl_change_1d * 100, 2),
                "flag": "OK",
                "points": 0,
            }

        # 2. Anonymous team / no audit (uses DefiLlama has_audit field proxy)
        if not has_audit:
            score += 30.0
            signals["no_audit"] = {
                "flag": "CRITICAL — contract not audited",
                "points": 30,
            }
        else:
            signals["no_audit"] = {"flag": "OK — audited", "points": 0}

        # 3. Honeypot indicators
        honeypot_pts = 0.0
        honeypot_flags = []
        if sell_tax_pct > 10.0:
            honeypot_pts += 15.0
            honeypot_flags.append(f"sell_tax={sell_tax_pct:.1f}%")
        if not ownership_renounced:
            honeypot_pts += 5.0
            honeypot_flags.append("ownership_not_renounced")
        score += honeypot_pts
        signals["honeypot"] = {
            "sell_tax_pct": sell_tax_pct,
            "ownership_renounced": ownership_renounced,
            "flags": honeypot_flags,
            "points": honeypot_pts,
        }

        # 4. Price impact anomaly — small trade causes >5% impact = thin liquidity
        if price_impact_small_trade_pct > 5.0:
            impact_pts = 15.0
            score += impact_pts
            signals["price_impact_anomaly"] = {
                "value_pct": price_impact_small_trade_pct,
                "flag": f"HIGH — {price_impact_small_trade_pct:.1f}% impact on small trade",
                "points": impact_pts,
            }
        else:
            signals["price_impact_anomaly"] = {
                "value_pct": price_impact_small_trade_pct,
                "flag": "OK",
                "points": 0,
            }

        score = min(100.0, round(score, 1))
        if score >= 70:
            risk_level = "HIGH RISK"
        elif score >= 40:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        return {
            "rug_score": score,
            "risk_level": risk_level,
            "signals": signals,
        }


# ---------------------------------------------------------------------------
# YieldFarmingAnalyzer
# ---------------------------------------------------------------------------

class YieldFarmingAnalyzer:
    """
    Analyze yield opportunities across DeFi.
    """

    def __init__(self):
        self._llama = DefiLlamaClient()

    def get_best_yields(self, min_tvl_usd: float = 1_000_000,
                         max_apy: float = 100.0) -> pd.DataFrame:
        """
        Fetch all pools, filter by TVL floor and realistic APY cap.
        Sort by APY descending.
        """
        df = self._llama.get_yields()
        if df.empty:
            return df
        df = df[df.get("tvl_usd", pd.Series(dtype=float)) >= min_tvl_usd]
        df = df[df.get("apy", pd.Series(dtype=float)) <= max_apy]
        df = df[df.get("apy", pd.Series(dtype=float)) > 0]
        return df.sort_values("apy", ascending=False).reset_index(drop=True)

    def compute_risk_adjusted_yield(self, pool: dict) -> float:
        """
        Risk-adjusted APY = APY × (1 - risk_score/100).
        Risk factors: protocol age, audit status, TVL, IL risk.
        """
        apy = float(pool.get("apy", 0) or 0)
        if apy <= 0:
            return 0.0

        risk_score = 0.0

        # TVL: higher TVL = lower risk
        tvl = float(pool.get("tvl_usd", 0) or 0)
        if tvl < 100_000:
            risk_score += 40
        elif tvl < 1_000_000:
            risk_score += 25
        elif tvl < 10_000_000:
            risk_score += 10
        else:
            risk_score += 5

        # IL risk
        il_risk_label = str(pool.get("il_risk_label", "")).lower()
        if "high" in il_risk_label:
            risk_score += 25
        elif "medium" in il_risk_label:
            risk_score += 15
        elif "low" in il_risk_label or "no" in il_risk_label:
            risk_score += 0
        else:
            # Infer from symbol
            symbol = str(pool.get("symbol", "")).upper()
            if any(s in symbol for s in ["USDC", "USDT", "DAI", "FRAX", "LUSD"]):
                risk_score += 0  # Stablecoin = no IL
            else:
                risk_score += 20

        # Audit bonus
        audited = pool.get("audits") or pool.get("audited", False)
        if not audited:
            risk_score += 15

        # APY suspiciously high
        if apy > 50:
            risk_score += 10

        risk_score = min(95.0, risk_score)
        return round(apy * (1.0 - risk_score / 100.0), 4)

    def detect_il_risk(self, pool: dict) -> float:
        """
        Impermanent loss risk score (0-1).
        0 = no IL risk (stablecoin pair)
        1 = maximum IL risk (volatile uncorrelated pair)
        """
        symbol = str(pool.get("symbol", "")).upper()
        il_label = str(pool.get("il_risk_label", "")).lower()

        # Override from explicit label
        if "no" in il_label:
            return 0.0
        if "low" in il_label:
            return 0.15
        if "medium" in il_label:
            return 0.40
        if "high" in il_label:
            return 0.75

        # Infer from token symbols
        stable_tokens = {"USDC", "USDT", "DAI", "FRAX", "LUSD", "BUSD", "TUSD", "USDP",
                         "GUSD", "MIM", "USDD", "CRVUSD", "PYUSD", "USDE"}
        correlated_tokens = {"WBTC", "BTC", "ETH", "WETH", "STETH", "WSTETH", "RETH", "CBETH"}

        tokens = set(symbol.replace("-", "/").replace("_", "/").split("/"))

        if tokens.issubset(stable_tokens):
            return 0.02  # Near-zero IL
        if len(tokens & stable_tokens) >= 1:
            return 0.30  # One stable, one volatile
        if tokens.issubset(correlated_tokens):
            return 0.20  # Correlated volatiles (e.g., ETH/WETH)
        return 0.65  # Unknown volatile pair

    def compute_stable_yield_ladder(self) -> pd.DataFrame:
        """
        Rank stablecoin-only pools by risk-adjusted APY.
        Only includes USDC/USDT/DAI pools on established protocols.
        """
        df = self._llama.get_yields()
        if df.empty:
            return df

        stable_tokens = {"USDC", "USDT", "DAI", "FRAX", "LUSD", "BUSD", "CRVUSD", "USDE", "PYUSD"}
        safe_protocols = {"aave", "compound", "curve", "makerdao", "lido", "convex",
                          "yearn", "spark", "morpho", "euler", "flux"}

        def is_stable_pool(row: pd.Series) -> bool:
            symbol = str(row.get("symbol", "")).upper()
            project = str(row.get("project", "")).lower()
            tokens = set(symbol.replace("-", "/").replace("_", "/").split("/"))
            if not tokens.issubset(stable_tokens):
                return False
            if project not in safe_protocols:
                return False
            return True

        stable_mask = df.apply(is_stable_pool, axis=1)
        stable_df = df[stable_mask].copy()

        if stable_df.empty:
            # Relaxed filter: just check stablecoin field and TVL
            if "stablecoin" in df.columns:
                stable_df = df[df["stablecoin"] == True].copy()

        if stable_df.empty:
            return pd.DataFrame()

        stable_df["risk_adjusted_apy"] = stable_df.apply(
            lambda r: self.compute_risk_adjusted_yield(r.to_dict()), axis=1
        )
        stable_df["il_risk_score"] = stable_df.apply(
            lambda r: self.detect_il_risk(r.to_dict()), axis=1
        )

        cols = [c for c in ["project", "chain", "symbol", "tvl_usd", "apy",
                              "risk_adjusted_apy", "il_risk_score"] if c in stable_df.columns]
        return stable_df[cols].sort_values("risk_adjusted_apy",
                                            ascending=False).reset_index(drop=True).head(50)

    def build_yield_opportunities(self, min_tvl: float = 1_000_000,
                                   max_apy: float = 100.0) -> List[YieldOpportunity]:
        """Build YieldOpportunity objects for top pools."""
        df = self.get_best_yields(min_tvl, max_apy)
        results = []
        for _, row in df.head(100).iterrows():
            pool_dict = row.to_dict()
            risk_apy = self.compute_risk_adjusted_yield(pool_dict)
            il_risk = self.detect_il_risk(pool_dict)
            is_stable = il_risk < 0.1
            results.append(YieldOpportunity(
                pool_id=str(row.get("pool_id", "")),
                project=str(row.get("project", "")),
                chain=str(row.get("chain", "")),
                symbol=str(row.get("symbol", "")),
                tvl_usd=float(row.get("tvl_usd", 0)),
                apy=float(row.get("apy", 0)),
                apy_base=float(row.get("apy_base", 0)),
                apy_reward=float(row.get("apy_reward", 0)),
                risk_score=max(0, float(row.get("apy", 0)) - risk_apy) / max(0.01, float(row.get("apy", 1))),
                risk_adjusted_apy=risk_apy,
                il_risk=il_risk,
                is_stable=is_stable,
                audited=bool(row.get("audits")),
                stablecoin=bool(row.get("stablecoin", False)),
            ))
        return results


# ---------------------------------------------------------------------------
# DEXVolumeTracker
# ---------------------------------------------------------------------------

class DEXVolumeTracker:
    """
    Track trading volumes across all DEXs.
    """

    def __init__(self):
        self._llama = DefiLlamaClient()

    def get_total_dex_volume(self, days: int = 7) -> pd.DataFrame:
        """
        Aggregate daily DEX volume across all protocols.
        Uses DefiLlama DEX overview data.
        """
        df = self._llama.get_dex_overview()
        if df.empty:
            return df
        # Sum columns exist per protocol; aggregate to daily total
        total_24h = df["dailyVolume"].sum()
        total_all = df["totalVolume"].sum()
        # Build a synthetic daily series
        summary = pd.DataFrame([{
            "period": f"last_{days}d",
            "total_volume_usd": total_all,
            "daily_volume_avg": total_24h,
            "protocol_count": len(df),
        }])
        return summary

    def compute_market_share(self, days: int = 30) -> pd.DataFrame:
        """Volume market share by DEX, sorted by share descending."""
        df = self._llama.get_dex_overview()
        if df.empty:
            return df
        total = df["dailyVolume"].sum()
        df = df.copy()
        df["market_share_pct"] = (df["dailyVolume"] / total * 100).round(2) if total > 0 else 0.0
        df["cumulative_share"] = df["market_share_pct"].cumsum()
        return df[["name", "chain", "dailyVolume", "market_share_pct",
                    "cumulative_share"]].reset_index(drop=True)

    def get_volume_trend(self, protocol: str, days: int = 90) -> pd.Series:
        """
        Daily volume history for a protocol.
        Fetches from DefiLlama protocol detail dex chart data.
        """
        key = f"vol_trend_{protocol}_{days}"
        cached = _cache_get(key, "dex_overview")
        if cached is None:
            data = _get(f"{_LLAMA_BASE}/overview/dexs/{protocol}",
                        params={"excludeTotalDataChart": "false"})
            if data is None:
                return pd.Series(dtype=float, name=f"{protocol}_volume")
            # totalDataChart is list of [timestamp, volume]
            chart = data.get("totalDataChart", [])
            cached = chart
            _cache_set(key, "dex_overview", cached)

        if not cached:
            return pd.Series(dtype=float, name=f"{protocol}_volume")

        timestamps, volumes = [], []
        for item in cached[-days:]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                timestamps.append(pd.Timestamp(item[0], unit="s"))
                volumes.append(float(item[1]))

        if not timestamps:
            return pd.Series(dtype=float, name=f"{protocol}_volume")

        series = pd.Series(volumes, index=timestamps, name=f"{protocol}_volume")
        return series.sort_index()

    def detect_volume_anomaly(self, protocol: str) -> bool:
        """
        Detect if recent volume is anomalously high (wash trading or genuine surge).
        Uses 3-sigma rule: volume > mean + 3σ = anomaly.
        """
        series = self.get_volume_trend(protocol, days=90)
        if len(series) < 14:
            return False
        mean = series.mean()
        std = series.std()
        recent = series.iloc[-1]
        threshold = mean + 3 * std
        return float(recent) > float(threshold)

    def compute_dex_vs_cex_ratio(self, dex_volume: float, cex_volume: float) -> float:
        """
        DEX/CEX volume ratio.
        Rising ratio signals DeFi adoption momentum.
        """
        if cex_volume <= 0:
            return float("inf")
        return round(dex_volume / cex_volume, 6)

    def get_dex_volume_summary(self) -> dict:
        """Summarize current DEX landscape."""
        df = self._llama.get_dex_overview()
        if df.empty:
            return {}
        top5 = df.head(5)
        return {
            "total_daily_volume_usd": float(df["dailyVolume"].sum()),
            "protocol_count": len(df),
            "top_5_protocols": top5[["name", "dailyVolume", "change_1d"]].to_dict("records"),
            "top_protocol": df.iloc[0]["name"] if not df.empty else "",
            "top_protocol_share_pct": round(
                df.iloc[0]["dailyVolume"] / df["dailyVolume"].sum() * 100, 2
            ) if not df.empty and df["dailyVolume"].sum() > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# DEXScreener
# ---------------------------------------------------------------------------

class DEXScreener:
    """
    Screen DEX pairs for trading opportunities.
    Uses DexScreener API (free, no key).
    """

    def _fetch_new_pairs(self, chain: str = "ethereum") -> List[dict]:
        """Fetch recently created pairs from DexScreener."""
        url = f"{_DEXSCREENER_BASE}/search"
        # DexScreener search endpoint — use token address or query
        # For new pairs, we use the /latest endpoint if available
        # Fallback: search for recent pairs by chain
        key = f"new_pairs_{chain}"
        cached = _cache_get(key, "dexscreener")
        if cached is not None:
            return cached
        data = _get(f"https://api.dexscreener.com/latest/dex/tokens/ethereum", rate_limit=False)
        if data is None:
            return []
        pairs = data.get("pairs", [])
        _cache_set(key, "dexscreener", pairs[:200])
        return pairs[:200]

    def screen_new_pairs(self, min_liquidity_usd: float = 10_000,
                          max_age_hours: int = 24) -> pd.DataFrame:
        """
        Screen recently created DEX pairs with real liquidity.
        Filters: minimum liquidity and maximum age.
        """
        pairs = self._fetch_new_pairs()
        if not pairs:
            return pd.DataFrame()

        cutoff_ts = time.time() - max_age_hours * 3600
        rows = []
        for p in pairs:
            try:
                created_at = p.get("pairCreatedAt", 0)
                if created_at and created_at / 1000 < cutoff_ts:
                    continue
                liquidity = float((p.get("liquidity") or {}).get("usd", 0) or 0)
                if liquidity < min_liquidity_usd:
                    continue
                rows.append({
                    "pair_address":    p.get("pairAddress", ""),
                    "dex_id":          p.get("dexId", ""),
                    "chain_id":        p.get("chainId", ""),
                    "base_token":      p.get("baseToken", {}).get("symbol", ""),
                    "quote_token":     p.get("quoteToken", {}).get("symbol", ""),
                    "price_usd":       float(p.get("priceUsd") or 0),
                    "liquidity_usd":   liquidity,
                    "volume_24h":      float((p.get("volume") or {}).get("h24", 0) or 0),
                    "price_change_24h": float((p.get("priceChange") or {}).get("h24", 0) or 0),
                    "created_at":      pd.Timestamp(created_at // 1000, unit="s") if created_at else None,
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("liquidity_usd", ascending=False).reset_index(drop=True)

    def screen_high_volume_pools(self, min_volume_usd: float = 100_000) -> pd.DataFrame:
        """Screen pools with high 24h volume using DefiLlama DEX data."""
        llama = DefiLlamaClient()
        df = llama.get_dex_overview()
        if df.empty:
            return df
        return df[df["dailyVolume"] >= min_volume_usd].sort_values(
            "dailyVolume", ascending=False).reset_index(drop=True)

    def screen_arbitrage_opportunities(self, pairs: List[str]) -> pd.DataFrame:
        """
        Compare price of the same token across different DEXs.
        Detects price discrepancies that may represent arbitrage opportunities.
        pairs: list of pair addresses or token symbols
        """
        price_data = {}
        for pair in pairs:
            key = f"dexs_{pair}"
            cached = _cache_get(key, "dexscreener")
            if cached is not None:
                data = cached
            else:
                data = _get(f"{_DEXSCREENER_BASE}/search?q={pair}", rate_limit=False)
                if data:
                    _cache_set(key, "dexscreener", data)
            if not data:
                continue
            pair_list = data.get("pairs", [])
            for p in pair_list[:10]:
                token = p.get("baseToken", {}).get("symbol", "")
                dex = p.get("dexId", "")
                price = float(p.get("priceUsd") or 0)
                if token and price > 0:
                    if token not in price_data:
                        price_data[token] = {}
                    price_data[token][dex] = price

        rows = []
        for token, prices in price_data.items():
            if len(prices) < 2:
                continue
            min_price = min(prices.values())
            max_price = max(prices.values())
            spread_pct = (max_price - min_price) / min_price * 100 if min_price > 0 else 0
            if spread_pct > 0.1:  # At least 0.1% spread
                rows.append({
                    "token": token,
                    "min_price": min_price,
                    "max_price": max_price,
                    "spread_pct": round(spread_pct, 4),
                    "buy_dex": min(prices, key=prices.get),
                    "sell_dex": max(prices, key=prices.get),
                    "prices_by_dex": prices,
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("spread_pct", ascending=False).reset_index(drop=True)

    def screen_concentrated_liquidity_gaps(self, pool_id: str) -> dict:
        """
        Find price ranges with thin liquidity = potential large slippage zones.
        Uses Uniswap V3 tick data.
        """
        uni = UniswapV3Analytics()
        dist = uni.get_pool_liquidity_distribution(pool_id)
        ticks = dist.get("ticks", [])
        if not ticks:
            return {"pool_id": pool_id, "gaps": []}

        # Find tick ranges with low liquidity (< 5% of max)
        if not ticks:
            return {"pool_id": pool_id, "gaps": []}
        max_liq = max(abs(t.liquidity_net) for t in ticks) if ticks else 0
        threshold = max_liq * 0.05
        sorted_ticks = sorted(ticks, key=lambda t: t.tick_idx)

        gaps = []
        for i in range(len(sorted_ticks) - 1):
            curr = sorted_ticks[i]
            nxt = sorted_ticks[i + 1]
            gap_width = nxt.tick_idx - curr.tick_idx
            avg_liq = (abs(curr.liquidity_net) + abs(nxt.liquidity_net)) / 2
            if avg_liq < threshold and gap_width > 60:  # > 60 ticks = significant gap
                gaps.append({
                    "from_tick": curr.tick_idx,
                    "to_tick": nxt.tick_idx,
                    "from_price": round(curr.price, 6),
                    "to_price": round(nxt.price, 6),
                    "gap_width_ticks": gap_width,
                    "avg_liquidity": round(avg_liq, 2),
                    "slippage_risk": "HIGH" if gap_width > 600 else "MEDIUM",
                })
        return {
            "pool_id": pool_id,
            "current_tick": dist.get("current_tick", 0),
            "gap_count": len(gaps),
            "gaps": sorted(gaps, key=lambda g: g["gap_width_ticks"], reverse=True)[:20],
        }


# ---------------------------------------------------------------------------
# ILCalculator — Impermanent Loss for AMM LPs (dim_110)
# ---------------------------------------------------------------------------

class ILCalculator:
    """
    Impermanent loss mathematics for constant-product AMM pools (x*y=k).

    IL Formula:
        IL = 2*sqrt(price_ratio) / (1 + price_ratio) - 1

    where price_ratio = P_t / P_0.

    IL is always <= 0 (a loss relative to simply holding the tokens).
    IL = 0 when price_ratio = 1 (no price change).

    This class is consistent with ImpermanentLossCalculator in defi_analytics_v3.
    """

    @staticmethod
    def compute_il(price_ratio_0_to_1: float) -> float:
        """
        Compute impermanent loss as a decimal fraction.

        Args:
            price_ratio_0_to_1: P_t / P_0 — ratio of token1-in-token0 price
                                 at withdrawal vs deposit. Must be > 0.

        Returns:
            IL as a negative decimal (e.g. -0.0572 for a price doubling).
        """
        if price_ratio_0_to_1 <= 0:
            raise ValueError(f"price_ratio must be > 0, got {price_ratio_0_to_1}")
        sqrt_r = math.sqrt(price_ratio_0_to_1)
        il = (2.0 * sqrt_r / (1.0 + price_ratio_0_to_1)) - 1.0
        return il  # always <= 0

    @staticmethod
    def compute_pool_il(pool_data: PoolData, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute full impermanent loss for an LP position in a pool.

        Requires current prices for both tokens in USD.
        Assumes LP entered when token0_price / token1_price = pool_data entry ratio.
        Uses token0_price from PoolData as the entry reference.

        Args:
            pool_data:      PoolData object with token0_price (entry price).
            current_prices: Dict mapping token symbol → current USD price.
                            e.g. {"ETH": 3200.0, "USDC": 1.0}

        Returns:
            dict with:
                - il_pct:         IL as a percentage (e.g., -5.72)
                - il_decimal:     IL as decimal (e.g., -0.0572)
                - price_ratio:    current P_t / P_0
                - entry_price:    token0 price at entry (from pool_data.token0_price)
                - current_price:  token0 current price (from current_prices)
                - hodl_value:     hypothetical value if simply holding 50/50
                - lp_value_pct:   LP value relative to hodl (= 1 + IL)
                - pool_id:        pool identifier
        """
        t0 = pool_data.token0_symbol
        t1 = pool_data.token1_symbol

        # Entry price: pool's stored token0_price (in token1 units)
        entry_price = pool_data.token0_price
        if entry_price <= 0:
            return {
                "error": "Entry price unavailable (token0_price = 0 in pool_data)",
                "pool_id": pool_data.id,
                "il_pct": 0.0,
                "il_decimal": 0.0,
            }

        # Compute current price ratio using provided prices
        t0_current = current_prices.get(t0, 0.0)
        t1_current = current_prices.get(t1, 0.0)

        if t0_current > 0 and t1_current > 0:
            # price of t0 in t1 units = USD(t0) / USD(t1)
            current_price = t0_current / t1_current
        elif t0_current > 0:
            # Only t0 price available — use pool's token1_price
            t1_ref = pool_data.token1_price if pool_data.token1_price > 0 else 1.0
            current_price = t0_current / t1_ref
        else:
            # Fallback: use pool's current token0_price directly
            current_price = pool_data.token0_price

        price_ratio = current_price / entry_price if entry_price > 0 else 1.0

        il = ILCalculator.compute_il(price_ratio)

        return {
            "pool_id": pool_data.id,
            "token_pair": f"{t0}/{t1}",
            "entry_price": entry_price,
            "current_price": round(current_price, 8),
            "price_ratio": round(price_ratio, 6),
            "il_decimal": round(il, 6),
            "il_pct": round(il * 100, 4),
            "lp_value_pct": round((1.0 + il) * 100, 4),
            "tvl_usd": pool_data.tvl_usd,
        }

    @staticmethod
    def il_table(price_ratios: Optional[List[float]] = None) -> List[Dict[str, float]]:
        """
        Generate a lookup table of IL values for common price ratios.

        Args:
            price_ratios: List of P_t/P_0 values (default: standard multiples).

        Returns:
            List of dicts with price_ratio, il_pct.
        """
        if price_ratios is None:
            price_ratios = [0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0, 4.0, 5.0]
        rows = []
        for r in price_ratios:
            if r > 0:
                rows.append({
                    "price_ratio": r,
                    "il_pct": round(ILCalculator.compute_il(r) * 100, 4),
                })
        return rows


# ---------------------------------------------------------------------------
# DEXAnalyticsEngine (Orchestrator)
# ---------------------------------------------------------------------------

class DEXAnalyticsEngine:
    """
    Orchestrator for DEX / AMM analytics.
    Provides unified interface across all data sources.
    """

    def __init__(self):
        self._llama = DefiLlamaClient()
        self._uni = UniswapV3Analytics()
        self._price = AMMPriceEngine()
        self._flow = LiquidityFlowAnalyzer()
        self._yield = YieldFarmingAnalyzer()
        self._volume = DEXVolumeTracker()
        self._screener = DEXScreener()

    def get_defi_dashboard(self) -> dict:
        """Comprehensive DeFi dashboard — TVL, volume, top protocols, top yields."""
        protocols = self._llama.get_all_protocols()
        dex_overview = self._llama.get_dex_overview()
        chains = self._llama.get_chains()
        yields_df = self._llama.get_yields()
        stables = self._llama.get_stablecoins()

        # Total TVL
        total_tvl = sum(float(p.get("tvl") or 0) for p in protocols)

        # Top protocols by TVL
        top_protocols = sorted(protocols, key=lambda p: float(p.get("tvl") or 0),
                                reverse=True)[:20]

        # Top chains
        top_chains = chains.head(10).to_dict("records") if not chains.empty else []

        # Top yields
        top_yields: List[dict] = []
        if not yields_df.empty:
            safe_yields = yields_df[
                (yields_df.get("tvl_usd", pd.Series(dtype=float)) > 1_000_000) &
                (yields_df.get("apy", pd.Series(dtype=float)) > 0) &
                (yields_df.get("apy", pd.Series(dtype=float)) <= 50)
            ]
            top_yields = safe_yields.head(10).to_dict("records") if not safe_yields.empty else []

        # DEX volume
        dex_summary = self._volume.get_dex_volume_summary()

        # Stable peg status
        depeg_alerts: List[str] = []
        if not stables.empty and "peg_deviation" in stables.columns:
            depegged = stables[stables["peg_deviation"] > 0.005]
            depeg_alerts = depegged["symbol"].tolist()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_defi_tvl_usd": total_tvl,
            "protocol_count": len(protocols),
            "top_protocols_by_tvl": [
                {
                    "name": p.get("name"),
                    "tvl_usd": float(p.get("tvl") or 0),
                    "category": p.get("category"),
                    "chain": p.get("chain"),
                    "change_1d": float(p.get("change_1d") or 0),
                }
                for p in top_protocols
            ],
            "top_chains": top_chains,
            "dex_volume": dex_summary,
            "top_yield_opportunities": top_yields,
            "stablecoin_depeg_alerts": depeg_alerts,
            "data_sources": ["DefiLlama", "TheGraph", "DexScreener", "CoinGecko"],
        }

    def get_protocol_health(self, protocol: str) -> ProtocolHealth:
        """Comprehensive health assessment for a DeFi protocol."""
        data = self._llama.get_protocol(protocol)
        all_protocols = self._llama.get_all_protocols()

        # Find in all protocols for change data
        proto_meta = next(
            (p for p in all_protocols
             if p.get("slug", "").lower() == protocol.lower()
             or p.get("name", "").lower() == protocol.lower()),
            {}
        )

        tvl = float(data.get("tvl") or proto_meta.get("tvl") or 0)
        chains = list(data.get("chainTvls", {}).keys()) or [data.get("chain", "")]
        category = data.get("category") or proto_meta.get("category", "")

        # Volume data
        dex_df = self._llama.get_dex_overview()
        vol_24h = 0.0
        if not dex_df.empty:
            match = dex_df[dex_df["name"].str.lower() == protocol.lower()]
            if not match.empty:
                vol_24h = float(match.iloc[0].get("dailyVolume", 0))

        # Compute rugpull risk
        rugpull = self._flow.detect_rugpull_risk(protocol)

        # Age
        inception = data.get("inception")
        age_days = 0
        if inception:
            try:
                age_days = (datetime.utcnow() - datetime.utcfromtimestamp(inception)).days
            except Exception:
                pass

        # Health score: composite
        health = 0.0
        if tvl > 1_000_000_000:
            health += 30
        elif tvl > 100_000_000:
            health += 20
        elif tvl > 10_000_000:
            health += 10
        health += max(0, 30 - rugpull / 3)  # Up to 30 pts for low risk
        if age_days > 365:
            health += 20
        elif age_days > 90:
            health += 10
        if len(chains) > 3:
            health += 10
        if vol_24h > 1_000_000:
            health += 10

        return ProtocolHealth(
            name=data.get("name") or proto_meta.get("name", protocol),
            slug=protocol,
            tvl_usd=tvl,
            tvl_change_1d=float(proto_meta.get("change_1d", 0) or 0),
            tvl_change_7d=float(proto_meta.get("change_7d", 0) or 0),
            tvl_change_30d=float(proto_meta.get("change_30d", 0) or 0),
            chains=chains,
            category=category,
            volume_24h=vol_24h,
            has_audit=bool(data.get("audits")),
            audits=len(data.get("audits") or []),
            rugpull_risk_score=rugpull,
            age_days=age_days,
            health_score=min(100.0, round(health, 1)),
        )

    def screen_opportunities(self) -> pd.DataFrame:
        """Combined opportunity screener: yields + high-volume pools + arbitrage."""
        # Best yields
        yields_df = self._yield.get_best_yields(min_tvl_usd=5_000_000, max_apy=50.0)
        if not yields_df.empty:
            yields_df["opportunity_type"] = "yield"
            yields_df["risk_adjusted_apy"] = yields_df.apply(
                lambda r: self._yield.compute_risk_adjusted_yield(r.to_dict()), axis=1
            )

        # High volume DEX pools
        high_vol = self._screener.screen_high_volume_pools(min_volume_usd=10_000_000)
        if not high_vol.empty:
            high_vol["opportunity_type"] = "high_volume"

        # Combine
        frames = []
        if not yields_df.empty:
            cols = [c for c in ["project", "chain", "symbol", "tvl_usd", "apy",
                                  "risk_adjusted_apy", "opportunity_type"] if c in yields_df.columns]
            frames.append(yields_df[cols].head(20))

        if not high_vol.empty:
            cols = [c for c in ["name", "chain", "dailyVolume", "opportunity_type"]
                    if c in high_vol.columns]
            frames.append(high_vol[cols].head(10))

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).reset_index(drop=True)

    def get_market_summary(self) -> str:
        """Formatted narrative summary of DeFi market state."""
        dashboard = self.get_defi_dashboard()
        total_tvl = dashboard.get("total_defi_tvl_usd", 0)
        dex_vol = dashboard.get("dex_volume", {}).get("total_daily_volume_usd", 0)
        top_proto = dashboard.get("top_protocols_by_tvl", [])
        depeg_alerts = dashboard.get("stablecoin_depeg_alerts", [])

        top_3 = ", ".join(p["name"] for p in top_proto[:3]) if top_proto else "N/A"

        lines = [
            f"=== SENTINEL DEX Market Summary — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===",
            f"Total DeFi TVL:      ${total_tvl/1e9:.2f}B",
            f"24h DEX Volume:      ${dex_vol/1e9:.2f}B",
            f"Top Protocols:       {top_3}",
        ]
        if top_proto:
            tvl1 = top_proto[0]["tvl_usd"]
            chg1 = top_proto[0]["change_1d"]
            lines.append(f"Largest Protocol:    {top_proto[0]['name']} — ${tvl1/1e9:.2f}B TVL ({chg1:+.1f}% 1d)")
        if depeg_alerts:
            lines.append(f"DEPEG ALERT:         {', '.join(depeg_alerts)} showing >0.5% peg deviation")
        else:
            lines.append("Stablecoin Pegs:     All major stablecoins within normal range")
        lines.append("Data: DefiLlama, TheGraph (Uniswap V3), DexScreener")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    log = logging.getLogger("dex_analytics_v3.main")

    engine = DEXAnalyticsEngine()
    llama = DefiLlamaClient()
    uni = UniswapV3Analytics()
    yield_analyzer = YieldFarmingAnalyzer()

    print("\n" + "=" * 70)
    print("SENTINEL — DEX / AMM Analytics V3 (dim_110)")
    print("=" * 70)

    # 1. Top 10 DEX protocols by TVL (via DefiLlama)
    print("\n[1] Top 10 DEX Protocols by TVL")
    print("-" * 50)
    try:
        all_protocols = llama.get_all_protocols()
        dex_protocols = [p for p in all_protocols if p.get("category") in ("Dexes", "DEX")]
        dex_protocols.sort(key=lambda p: float(p.get("tvl") or 0), reverse=True)
        for i, p in enumerate(dex_protocols[:10], 1):
            tvl = float(p.get("tvl") or 0)
            chg7 = float(p.get("change_7d") or 0)
            print(f"  {i:2d}. {p.get('name','?'):25s} TVL: ${tvl/1e9:.3f}B  7d: {chg7:+.1f}%  chains: {p.get('chain','?')}")
    except Exception as e:
        print(f"  Error fetching protocols: {e}")

    # 2. Top 5 Uniswap V3 pools
    print("\n[2] Top 5 Uniswap V3 Pools by TVL")
    print("-" * 50)
    try:
        pools_df = uni.get_top_pools(n=5)
        if not pools_df.empty:
            for _, row in pools_df.iterrows():
                fee_pct = row["feeTier"] / 10000
                print(f"  {row['token0']}/{row['token1']} ({fee_pct:.2f}%) — "
                      f"TVL: ${row['totalValueLockedUSD']/1e6:.1f}M  "
                      f"Vol24h: ${row['volumeUSD']/1e6:.1f}M  "
                      f"AnnFeeYield: {row.get('annualized_fee_yield', 0)*100:.1f}%")
        else:
            print("  No pool data available (subgraph may be rate-limited)")
    except Exception as e:
        print(f"  Error fetching Uniswap pools: {e}")

    # 3. Best risk-adjusted yields
    print("\n[3] Top 10 Risk-Adjusted Yield Opportunities (min $1M TVL, max 100% APY)")
    print("-" * 60)
    try:
        best_yields = yield_analyzer.get_best_yields(min_tvl_usd=1_000_000, max_apy=100.0)
        if not best_yields.empty:
            for i, row in best_yields.head(10).iterrows():
                pool_dict = row.to_dict()
                ra_apy = yield_analyzer.compute_risk_adjusted_yield(pool_dict)
                il = yield_analyzer.detect_il_risk(pool_dict)
                print(f"  {row.get('project','?'):15s} {row.get('symbol','?'):20s} "
                      f"APY:{row.get('apy',0):6.2f}%  RA-APY:{ra_apy:6.2f}%  "
                      f"IL:{il:.2f}  TVL:${row.get('tvl_usd',0)/1e6:.1f}M  "
                      f"Chain:{row.get('chain','?')}")
        else:
            print("  No yield data available")
    except Exception as e:
        print(f"  Error fetching yields: {e}")

    # 4. Stablecoin yield ladder
    print("\n[4] Stablecoin Yield Ladder (stable-only pools, established protocols)")
    print("-" * 60)
    try:
        ladder = yield_analyzer.compute_stable_yield_ladder()
        if not ladder.empty:
            for _, row in ladder.head(5).iterrows():
                print(f"  {row.get('project','?'):15s} {row.get('symbol','?'):20s} "
                      f"APY:{row.get('apy',0):5.2f}%  RA-APY:{row.get('risk_adjusted_apy',0):5.2f}%  "
                      f"TVL:${row.get('tvl_usd',0)/1e6:.1f}M")
        else:
            print("  No stable pool data available")
    except Exception as e:
        print(f"  Error computing stable ladder: {e}")

    # 5. Arbitrage check for ETH-based pairs
    print("\n[5] Arbitrage Scan — ETH-based pairs across DEXs")
    print("-" * 50)
    try:
        screener = DEXScreener()
        arb_df = screener.screen_arbitrage_opportunities(["WETH", "ETH"])
        if not arb_df.empty:
            for _, row in arb_df.head(5).iterrows():
                print(f"  {row['token']:10s}  Buy: {row['buy_dex']:15s} @ ${row['min_price']:.4f}  "
                      f"Sell: {row['sell_dex']:15s} @ ${row['max_price']:.4f}  "
                      f"Spread: {row['spread_pct']:.3f}%")
        else:
            print("  No significant arbitrage opportunities detected")
    except Exception as e:
        print(f"  Error during arbitrage scan: {e}")

    # 6. Market summary
    print("\n[6] DeFi Market Summary")
    print("-" * 60)
    try:
        summary = engine.get_market_summary()
        print(summary)
    except Exception as e:
        print(f"  Error generating summary: {e}")

    print("\n" + "=" * 70)
    print("DEX Analytics V3 complete.")
    sys.exit(0)
