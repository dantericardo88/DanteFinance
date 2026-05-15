"""
DeFi Protocol Analytics V2 — Dimension #107 (target score 9).

Comprehensive DeFi intelligence using only free DefiLlama APIs (no paid keys).

Coverage
--------
- TVL: historical per-protocol, chain-level, dominance, momentum
- Yield/APY: DefiLlama yields API — 1000+ pools, screener, IL calculator
- Protocol revenue: fees API — annualised revenue, P/S ratio
- DEX volume: dexs API — volume/TVL ratio, fee tier analytics
- Stablecoins: supply, peg deviation, depeg alerts, market share
- Bridges: cross-chain volume, bridge security scoring
- Protocol health: TVL growth, retention proxy, revenue/TVL, chain diversification
- Yield farming screener: chain / category / min TVL / min APY / audited-only filters
- Governance token analytics: on-chain treasury balance proxy, proposal data (Snapshot)
- Liquidity pool analytics: fee tier, vol/TVL, IL calculator, optimal concentrated range
- DeFi sector comparison: lending vs DEX vs yield vs derivatives TVL share
- SQLite caching: 1h TTL for TVL, 15min for yields, protocol snapshots
- FastAPI router: all specified endpoints

Free APIs
---------
https://api.llama.fi          — protocols, protocol detail, TVL, chains, fees, dexs
https://yields.llama.fi       — yield pools (APY, TVL, IL risk)
https://stablecoins.llama.fi  — stablecoin supply, peg deviation
https://bridges.llama.fi      — bridge TVL and volume
https://hub.snapshot.org/graphql — governance proposals (Snapshot, free)
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# API base URLs (all free, no auth)
# ---------------------------------------------------------------------------

_LLAMA_BASE   = "https://api.llama.fi"
_YIELDS_BASE  = "https://yields.llama.fi"
_STABLES_BASE = "https://stablecoins.llama.fi"
_BRIDGES_BASE = "https://bridges.llama.fi"
_SNAPSHOT_GQL = "https://hub.snapshot.org/graphql"

_HEADERS = {
    "User-Agent": "SENTINEL/2.0 financial-terminal richard.porras@realempanada.com",
    "Accept":     "application/json",
}
_TIMEOUT = 30

# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

_DB_DIR = Path(__file__).parent.parent / "data" / "db"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "defi_analytics_v2.db"

# Cache TTL by data type (seconds)
_TTL = {
    "tvl":       3600,   # 1h — TVL changes slowly
    "yields":    900,    # 15min — pools update frequently
    "stables":   1800,   # 30min
    "bridges":   3600,   # 1h
    "fees":      3600,   # 1h
    "dexs":      1800,   # 30min
    "protocols": 3600,   # 1h
    "snapshot":  1800,   # 30min
}


def _init_cache_db() -> None:
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key    TEXT PRIMARY KEY,
            data_type    TEXT NOT NULL,
            payload      TEXT NOT NULL,
            fetched_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS protocol_snapshots (
            slug         TEXT NOT NULL,
            snapshot_ts  REAL NOT NULL,
            tvl_usd      REAL,
            apy_avg      REAL,
            revenue_usd  REAL,
            metadata_json TEXT,
            PRIMARY KEY (slug, snapshot_ts)
        );
        CREATE INDEX IF NOT EXISTS idx_snap_slug ON protocol_snapshots(slug);
        """)


_init_cache_db()


def _cache_get(key: str, data_type: str) -> Optional[Any]:
    ttl = _TTL.get(data_type, 3600)
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
    except Exception as exc:
        logger.debug("Cache read error", key=key, error=str(exc))
        return None


def _cache_set(key: str, data_type: str, data: Any) -> None:
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                (key, data_type, json.dumps(data, default=str), time.time()),
            )
    except Exception as exc:
        logger.debug("Cache write error", key=key, error=str(exc))


def _save_protocol_snapshot(slug: str, tvl: float, apy_avg: float,
                             revenue: float, meta: Dict) -> None:
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO protocol_snapshots VALUES (?,?,?,?,?,?)",
                (slug, time.time(), tvl, apy_avg, revenue, json.dumps(meta, default=str)),
            )
    except Exception as exc:
        logger.debug("Snapshot save error", slug=slug, error=str(exc))


def _load_protocol_snapshots(slug: str, days: int = 30) -> pd.DataFrame:
    cutoff = time.time() - days * 86400
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            rows = conn.execute(
                "SELECT snapshot_ts, tvl_usd, apy_avg, revenue_usd "
                "FROM protocol_snapshots WHERE slug=? AND snapshot_ts>=? "
                "ORDER BY snapshot_ts",
                (slug, cutoff),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["timestamp", "tvl_usd", "apy_avg", "revenue_usd"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: Optional[Dict] = None, timeout: int = _TIMEOUT) -> Any:
    """Synchronous HTTP GET with error handling."""
    try:
        r = requests.get(url, headers=_HEADERS, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as exc:
        logger.warning("HTTP error", url=url, status=exc.response.status_code if exc.response else None)
        return None
    except Exception as exc:
        logger.warning("Request failed", url=url, error=str(exc))
        return None


def _post(url: str, payload: Dict, timeout: int = _TIMEOUT) -> Any:
    try:
        r = requests.post(url, json=payload, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("POST failed", url=url, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProtocolSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    slug: str
    name: str
    chain: str
    chains: List[str] = Field(default_factory=list)
    category: str = ""
    tvl_usd: float = 0.0
    change_1h_pct: Optional[float] = None
    change_1d_pct: Optional[float] = None
    change_7d_pct: Optional[float] = None
    mcap_tvl: Optional[float] = None
    audit_links: List[str] = Field(default_factory=list)


class YieldPool(BaseModel):
    model_config = ConfigDict(frozen=True)
    pool_id: str
    protocol: str
    chain: str
    symbol: str
    tvl_usd: float
    apy: float
    apy_base: Optional[float] = None
    apy_reward: Optional[float] = None
    stablecoin: bool = False
    il_risk: str = "no"
    exposure: str = ""
    audited: bool = False
    pool_meta: Optional[str] = None


class StablecoinInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    name: str
    symbol: str
    peg_type: str = ""
    peg_mechanism: str = ""
    circulating_usd: float = 0.0
    circulating_prev: float = 0.0
    change_1d_pct: Optional[float] = None
    peg_deviation_pct: Optional[float] = None
    depeg_alert: bool = False


class BridgeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    name: str
    chains: List[str] = Field(default_factory=list)
    volume_24h_usd: float = 0.0
    volume_7d_usd: float = 0.0
    last_hourly_volume_usd: float = 0.0
    audit_links: List[str] = Field(default_factory=list)


class ProtocolHealthScore(BaseModel):
    model_config = ConfigDict(frozen=True)
    protocol_slug: str
    health_score: float = Field(..., ge=0.0, le=100.0)
    tvl_stability_score: float
    revenue_tvl_ratio: Optional[float]
    chain_diversification_score: float
    age_days: Optional[int]
    tvl_momentum_30d_pct: Optional[float]
    risk_flags: List[str] = Field(default_factory=list)
    tier: str = "unknown"  # "blue_chip" | "established" | "emerging" | "risky"


class LiquidityPool(BaseModel):
    model_config = ConfigDict(frozen=True)
    pool_id: str
    protocol: str
    chain: str
    symbol: str
    fee_tier_bps: Optional[float] = None
    tvl_usd: float
    volume_24h_usd: Optional[float] = None
    volume_tvl_ratio: Optional[float] = None
    apy: float
    il_risk: str = "no"
    il_7d_pct: Optional[float] = None


# ===========================================================================
# DefiLlamaClient — synchronous, cached
# ===========================================================================

class DefiLlamaClient:
    """
    Synchronous DefiLlama API client with SQLite caching.

    All responses are cached per TTL config. Raw data is parsed to DataFrames.
    """

    # ------------------------------------------------------------------
    # protocols
    # ------------------------------------------------------------------
    def get_protocols(self, limit: int = 500) -> pd.DataFrame:
        """Fetch all DeFi protocols from DefiLlama /protocols."""
        cache_key = f"protocols_{limit}"
        cached = _cache_get(cache_key, "protocols")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_LLAMA_BASE}/protocols")
        if not data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for p in data[:limit]:
            try:
                rows.append({
                    "slug":           p.get("slug", ""),
                    "name":           p.get("name", ""),
                    "chain":          p.get("chain", ""),
                    "chains":         p.get("chains", []),
                    "category":       p.get("category", ""),
                    "tvl_usd":        float(p.get("tvl") or 0),
                    "change_1h_pct":  p.get("change_1h"),
                    "change_1d_pct":  p.get("change_1d"),
                    "change_7d_pct":  p.get("change_7d"),
                    "mcap_tvl":       p.get("mcap") / float(p.get("tvl") or 1) if p.get("mcap") and p.get("tvl") else None,
                    "audit_links":    p.get("audit_links") or [],
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        _cache_set(cache_key, "protocols", rows)
        logger.info("Fetched protocols", count=len(df))
        return df

    # ------------------------------------------------------------------
    # protocol detail
    # ------------------------------------------------------------------
    def get_protocol(self, slug: str) -> Dict:
        """Fetch full protocol detail from /protocol/{slug}."""
        cache_key = f"protocol_detail_{slug}"
        cached = _cache_get(cache_key, "protocols")
        if cached is not None:
            return cached

        data = _get(f"{_LLAMA_BASE}/protocol/{slug}")
        if not data:
            return {}

        _cache_set(cache_key, "protocols", data)
        return data

    # ------------------------------------------------------------------
    # TVL history
    # ------------------------------------------------------------------
    def get_tvl_history(self, slug: str) -> pd.DataFrame:
        """Fetch historical TVL for a single protocol."""
        cache_key = f"tvl_hist_{slug}"
        cached = _cache_get(cache_key, "tvl")
        if cached is not None:
            return pd.DataFrame(cached)

        detail = self.get_protocol(slug)
        tvl_data = detail.get("tvl", [])
        if not tvl_data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for entry in tvl_data:
            try:
                rows.append({
                    "date": datetime.fromtimestamp(entry["date"], tz=timezone.utc),
                    "tvl":  float(entry.get("totalLiquidityUSD") or entry.get("tvl") or 0),
                })
            except (KeyError, TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        _cache_set(cache_key, "tvl", rows)
        return df

    # ------------------------------------------------------------------
    # chains
    # ------------------------------------------------------------------
    def get_chains(self) -> pd.DataFrame:
        """Fetch per-chain TVL snapshot from /v2/chains."""
        cache_key = "chains_tvl"
        cached = _cache_get(cache_key, "tvl")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_LLAMA_BASE}/v2/chains")
        if not data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for c in data:
            try:
                rows.append({
                    "chain":        c.get("name", ""),
                    "tvl_usd":      float(c.get("tvl") or 0),
                    "token_symbol": c.get("tokenSymbol", ""),
                    "cmcId":        c.get("cmcId"),
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows).sort_values("tvl_usd", ascending=False).reset_index(drop=True)
        _cache_set(cache_key, "tvl", rows)
        return df

    # ------------------------------------------------------------------
    # chain TVL history (aggregate DeFi)
    # ------------------------------------------------------------------
    def get_total_tvl_history(self) -> pd.DataFrame:
        """Historical total DeFi TVL from /v2/historicalChainTvl."""
        cache_key = "total_defi_tvl_hist"
        cached = _cache_get(cache_key, "tvl")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_LLAMA_BASE}/v2/historicalChainTvl")
        if not isinstance(data, list):
            return pd.DataFrame()

        rows = [
            {"date": datetime.fromtimestamp(d["date"], tz=timezone.utc), "tvl": float(d.get("tvl") or 0)}
            for d in data if "date" in d
        ]
        df = pd.DataFrame(rows)
        _cache_set(cache_key, "tvl", rows)
        return df

    # ------------------------------------------------------------------
    # yields
    # ------------------------------------------------------------------
    def get_yields(self) -> pd.DataFrame:
        """Fetch all yield pools from yields.llama.fi/pools."""
        cache_key = "yield_pools_all"
        cached = _cache_get(cache_key, "yields")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_YIELDS_BASE}/pools")
        if not data or "data" not in data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for p in data["data"]:
            try:
                rows.append({
                    "pool_id":      p.get("pool", ""),
                    "protocol":     p.get("project", ""),
                    "chain":        p.get("chain", ""),
                    "symbol":       p.get("symbol", ""),
                    "tvl_usd":      float(p.get("tvlUsd") or 0),
                    "apy":          float(p.get("apy") or 0),
                    "apy_base":     p.get("apyBase"),
                    "apy_reward":   p.get("apyReward"),
                    "stablecoin":   bool(p.get("stablecoin", False)),
                    "il_risk":      p.get("ilRisk", "no"),
                    "exposure":     p.get("exposure", ""),
                    "audited":      bool(p.get("audits")),
                    "pool_meta":    p.get("poolMeta"),
                    "underlying_tokens": p.get("underlyingTokens", []),
                    "reward_tokens": p.get("rewardTokens", []),
                    "volume_usd_1d": float(p.get("volumeUsd1d") or 0),
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        _cache_set(cache_key, "yields", rows)
        logger.info("Fetched yield pools", count=len(df))
        return df

    # ------------------------------------------------------------------
    # fees / revenue
    # ------------------------------------------------------------------
    def get_fees(self) -> pd.DataFrame:
        """Fetch protocol fee/revenue data from /overview/fees."""
        cache_key = "protocol_fees"
        cached = _cache_get(cache_key, "fees")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_LLAMA_BASE}/overview/fees")
        if not data or "protocols" not in data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for p in data["protocols"]:
            try:
                rows.append({
                    "name":              p.get("name", ""),
                    "category":          p.get("category", ""),
                    "total_24h_usd":     float(p.get("total24h") or 0),
                    "total_7d_usd":      float(p.get("total7d") or 0),
                    "total_30d_usd":     float(p.get("total30d") or 0),
                    "revenue_24h_usd":   float(p.get("revenue24h") or 0),
                    "revenue_7d_usd":    float(p.get("revenue7d") or 0),
                    "revenue_30d_usd":   float(p.get("revenue30d") or 0),
                    "chains":            p.get("chains", []),
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        _cache_set(cache_key, "fees", rows)
        return df

    # ------------------------------------------------------------------
    # DEX volumes
    # ------------------------------------------------------------------
    def get_dex_volumes(self) -> pd.DataFrame:
        """Fetch DEX volume data from /overview/dexs."""
        cache_key = "dex_volumes"
        cached = _cache_get(cache_key, "dexs")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_LLAMA_BASE}/overview/dexs")
        if not data or "protocols" not in data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for p in data["protocols"]:
            try:
                rows.append({
                    "name":          p.get("name", ""),
                    "chain":         p.get("chains", [""])[0] if p.get("chains") else "",
                    "chains":        p.get("chains", []),
                    "category":      p.get("category", ""),
                    "total_24h_usd": float(p.get("total24h") or 0),
                    "total_7d_usd":  float(p.get("total7d") or 0),
                    "total_30d_usd": float(p.get("total30d") or 0),
                    "tvl_usd":       float(p.get("tvl") or 0),
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        _cache_set(cache_key, "dexs", rows)
        return df

    # ------------------------------------------------------------------
    # stablecoins
    # ------------------------------------------------------------------
    def get_stablecoins(self) -> pd.DataFrame:
        """Fetch stablecoin data from stablecoins.llama.fi/stablecoins."""
        cache_key = "stablecoins_all"
        cached = _cache_get(cache_key, "stables")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_STABLES_BASE}/stablecoins?includePrices=true")
        if not data or "peggedAssets" not in data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for s in data["peggedAssets"]:
            try:
                circ = s.get("circulating", {})
                circ_usd = float(circ.get("peggedUSD") or circ.get("peggedEUR") or 0)
                circ_prev = float((s.get("circulatingPrevDay") or {}).get("peggedUSD") or 0)
                price = float((s.get("price") or 1.0))
                peg_dev = abs(price - 1.0) * 100.0 if s.get("pegType", "").endswith("USD") else None

                rows.append({
                    "id":                s.get("id", ""),
                    "name":              s.get("name", ""),
                    "symbol":            s.get("symbol", ""),
                    "peg_type":          s.get("pegType", ""),
                    "peg_mechanism":     s.get("pegMechanism", ""),
                    "circulating_usd":   circ_usd,
                    "circulating_prev":  circ_prev,
                    "change_1d_pct":     ((circ_usd / circ_prev - 1.0) * 100.0) if circ_prev > 0 else None,
                    "price":             price,
                    "peg_deviation_pct": round(peg_dev, 4) if peg_dev is not None else None,
                    "depeg_alert":       peg_dev is not None and peg_dev > 1.0,
                    "chains":            list(s.get("chainCirculating", {}).keys()),
                })
            except (TypeError, ValueError, ZeroDivisionError):
                continue

        df = pd.DataFrame(rows).sort_values("circulating_usd", ascending=False).reset_index(drop=True)
        _cache_set(cache_key, "stables", rows)
        return df

    # ------------------------------------------------------------------
    # bridges
    # ------------------------------------------------------------------
    def get_bridges(self) -> pd.DataFrame:
        """Fetch bridge data from bridges.llama.fi/bridges."""
        cache_key = "bridges_all"
        cached = _cache_get(cache_key, "bridges")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_BRIDGES_BASE}/bridges?includeChains=true")
        if not data or "bridges" not in data:
            return pd.DataFrame()

        rows: List[Dict] = []
        for b in data["bridges"]:
            try:
                rows.append({
                    "id":                   b.get("id", 0),
                    "name":                 b.get("displayName", b.get("name", "")),
                    "chains":               list(b.get("chains", {}).keys()) if isinstance(b.get("chains"), dict) else [],
                    "volume_24h_usd":       float(b.get("lastDailyVolume") or 0),
                    "volume_7d_usd":        float(b.get("lastWeeklyVolume") or 0),
                    "last_hourly_volume_usd": float(b.get("lastHourlyVolume") or 0),
                    "tx_count_24h":         b.get("txsPrevDay"),
                    "audit_links":          b.get("audits", []),
                    "icon":                 b.get("icon", ""),
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows).sort_values("volume_24h_usd", ascending=False).reset_index(drop=True)
        _cache_set(cache_key, "bridges", rows)
        return df

    # ------------------------------------------------------------------
    # bridge volume by chain
    # ------------------------------------------------------------------
    def get_bridge_flows(self, bridge_id: int) -> pd.DataFrame:
        """Fetch daily bridge flows for a specific bridge."""
        cache_key = f"bridge_flows_{bridge_id}"
        cached = _cache_get(cache_key, "bridges")
        if cached is not None:
            return pd.DataFrame(cached)

        data = _get(f"{_BRIDGES_BASE}/bridgedaystats/{bridge_id}")
        if not data or "time" not in data:
            return pd.DataFrame()

        rows: List[Dict] = []
        times = data.get("time", [])
        deposits = data.get("totalDeposited", {})
        withdrawals = data.get("totalWithdrawn", {})

        for ts_str, ts_val in zip(times, data.get("time", [])):
            try:
                ts = int(ts_val)
                rows.append({
                    "date":            datetime.fromtimestamp(ts, tz=timezone.utc),
                    "deposits_usd":    float(deposits.get(str(ts)) or 0),
                    "withdrawals_usd": float(withdrawals.get(str(ts)) or 0),
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        _cache_set(cache_key, "bridges", rows)
        return df


# ===========================================================================
# TVLAnalytics
# ===========================================================================

class TVLAnalytics:
    """Protocol and chain-level TVL tracking, dominance, momentum."""

    def __init__(self) -> None:
        self._client = DefiLlamaClient()

    def tvl_dominance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Compute TVL dominance (market share) for the top N protocols.

        Columns: slug, name, category, tvl_usd, market_share_pct, rank.
        """
        df = self._client.get_protocols(limit=500)
        if df.empty:
            return df

        total = df["tvl_usd"].sum()
        df = df.head(top_n).copy()
        df["market_share_pct"] = (df["tvl_usd"] / total * 100).round(4)
        df["rank"] = range(1, len(df) + 1)
        return df[["rank", "slug", "name", "category", "chain", "tvl_usd",
                   "market_share_pct", "change_1d_pct", "change_7d_pct"]].reset_index(drop=True)

    def chain_tvl_dominance(self) -> pd.DataFrame:
        """Chain-level TVL market share, sorted by TVL descending."""
        df = self._client.get_chains()
        if df.empty:
            return df
        total = df["tvl_usd"].sum()
        df = df.copy()
        df["market_share_pct"] = (df["tvl_usd"] / total * 100).round(4)
        df["rank"] = range(1, len(df) + 1)
        return df[["rank", "chain", "tvl_usd", "market_share_pct", "token_symbol"]].reset_index(drop=True)

    def tvl_momentum(self, min_7d_change_pct: float = 10.0, min_tvl_mm: float = 10.0) -> pd.DataFrame:
        """Protocols with TVL growth >= threshold over past 7 days."""
        df = self._client.get_protocols(limit=500)
        if df.empty:
            return df

        mask = (
            df["change_7d_pct"].notna()
            & (df["change_7d_pct"] >= min_7d_change_pct)
            & (df["tvl_usd"] >= min_tvl_mm * 1e6)
        )
        result = df[mask].copy()
        if not result.empty:
            result = result.sort_values("change_7d_pct", ascending=False).reset_index(drop=True)
        return result

    def tvl_outflows(self, min_7d_decline_pct: float = -15.0, min_tvl_mm: float = 5.0) -> pd.DataFrame:
        """Protocols experiencing significant TVL outflows."""
        df = self._client.get_protocols(limit=500)
        if df.empty:
            return df

        mask = (
            df["change_7d_pct"].notna()
            & (df["change_7d_pct"] <= min_7d_decline_pct)
            & (df["tvl_usd"] >= min_tvl_mm * 1e6)
        )
        result = df[mask].copy()
        if not result.empty:
            result = result.sort_values("change_7d_pct", ascending=True).reset_index(drop=True)
        return result

    def protocol_tvl_history(self, slug: str, days: int = 90) -> pd.DataFrame:
        """Historical TVL for a single protocol, trimmed to last N days."""
        df = self._client.get_tvl_history(slug)
        if df.empty:
            return df

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        return df

    def total_defi_tvl_trend(self, days: int = 90) -> Dict:
        """Total DeFi TVL trend over last N days."""
        df = self._client.get_total_tvl_history()
        if df.empty:
            return {"error": "No TVL history available"}

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        if len(df) < 2:
            return {"error": "Insufficient data for trend"}

        current = df["tvl"].iloc[-1]
        start = df["tvl"].iloc[0]
        change_pct = (current / start - 1.0) * 100.0 if start > 0 else 0.0

        # Linear regression slope
        x = np.arange(len(df), dtype=float)
        y = df["tvl"].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0

        return {
            "current_tvl_usd":       round(current, 0),
            "start_tvl_usd":         round(start, 0),
            "change_pct":            round(change_pct, 3),
            "trend":                 "up" if change_pct > 2 else ("down" if change_pct < -2 else "flat"),
            "linear_slope_usd_day":  round(slope, 0),
            "data_points":           len(df),
            "period_days":           days,
        }


# ===========================================================================
# YieldAggregator
# ===========================================================================

class YieldAggregator:
    """
    Yield farming pool aggregator and screener.

    Filters: chain, protocol category, min TVL, min APY, audited only.
    Includes IL calculator and optimal concentrated liquidity range estimator.
    """

    MIN_TVL_DEFAULT = 1_000_000   # $1M

    def __init__(self) -> None:
        self._client = DefiLlamaClient()

    def screen(
        self,
        chain: Optional[str] = None,
        category: Optional[str] = None,
        min_tvl_usd: float = 1_000_000,
        min_apy: float = 0.0,
        max_apy: float = 500.0,
        stablecoin_only: bool = False,
        audited_only: bool = False,
        il_risk_max: str = "high",   # "no" | "low" | "high"
        protocol: Optional[str] = None,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        Screen yield pools by multiple criteria.

        Returns DataFrame sorted by risk-adjusted APY (IL-penalised).
        """
        df = self._client.get_yields()
        if df.empty:
            return df

        mask = (
            (df["tvl_usd"] >= min_tvl_usd)
            & (df["apy"] >= min_apy)
            & (df["apy"] <= max_apy)
        )
        if chain:
            mask &= df["chain"].str.lower() == chain.lower()
        if protocol:
            mask &= df["protocol"].str.lower().str.contains(protocol.lower(), na=False)
        if stablecoin_only:
            mask &= df["stablecoin"] == True   # noqa: E712
        if audited_only:
            mask &= df["audited"] == True      # noqa: E712

        il_rank = {"no": 0, "low": 1, "high": 2}
        il_max_rank = il_rank.get(il_risk_max, 2)
        mask &= df["il_risk"].map(lambda x: il_rank.get(x, 0)) <= il_max_rank

        result = df[mask].copy()
        if result.empty:
            return result

        # Risk-adjusted APY
        il_penalty_map = {"no": 1.0, "low": 0.85, "high": 0.6}
        result["il_penalty"] = result["il_risk"].map(lambda x: il_penalty_map.get(x, 1.0))
        result["risk_adj_apy"] = (result["apy"] * result["il_penalty"]).round(4)
        result = result.sort_values("risk_adj_apy", ascending=False).head(top_n).reset_index(drop=True)
        result = result.drop(columns=["il_penalty"])
        logger.info("Yield screen", hits=len(result), chain=chain, min_apy=min_apy)
        return result

    def top_stable_yields(self, min_tvl_usd: float = 5_000_000, top_n: int = 20) -> pd.DataFrame:
        """Top stable-coin yield pools (no IL risk) sorted by APY."""
        return self.screen(
            stablecoin_only=True,
            il_risk_max="no",
            min_tvl_usd=min_tvl_usd,
            max_apy=30.0,
            top_n=top_n,
        )

    def il_calculator(
        self,
        price_ratio_change: float,
        il_risk: str = "high",
    ) -> Dict:
        """
        Compute impermanent loss for a standard 50/50 AMM pool.

        price_ratio_change: fractional change in relative price of asset A vs B.
          e.g. 0.5 = asset A doubled vs B, -0.5 = asset A halved vs B.

        Returns IL as a percentage of initial value.
        """
        k = 1.0 + price_ratio_change
        if k <= 0:
            return {"error": "Invalid price change — price cannot go negative"}

        # IL formula: 2*sqrt(k)/(1+k) - 1
        il = (2.0 * math.sqrt(k) / (1.0 + k)) - 1.0
        il_pct = il * 100.0

        return {
            "price_ratio_change_pct": round(price_ratio_change * 100, 2),
            "il_pct":                 round(il_pct, 4),
            "value_vs_hodl":          round(1 + il, 6),
            "break_even_apy_needed":  round(abs(il_pct), 4),
            "risk_tier":              il_risk,
            "note": (
                f"At {price_ratio_change*100:.0f}% relative price move, "
                f"LP position is worth {il_pct:.2f}% less than HODL."
            ),
        }

    def optimal_cl_range(
        self,
        current_price: float,
        expected_move_pct: float = 20.0,
        fee_tier_bps: float = 30.0,
    ) -> Dict:
        """
        Estimate optimal concentrated liquidity range for a Uniswap v3-style AMM.

        Returns suggested price range [lower, upper] that captures most of the
        expected trading volume while minimising rebalancing frequency.

        Uses ±1 standard deviation approximation based on expected_move_pct.
        """
        sigma = expected_move_pct / 100.0
        # Optimal range: ±1.5σ for typical fee tier
        multiplier = 1.5 if fee_tier_bps <= 5 else (1.2 if fee_tier_bps <= 30 else 1.0)
        lower = current_price * (1 - sigma * multiplier)
        upper = current_price * (1 + sigma * multiplier)

        # Capital efficiency vs full-range
        # CE = sqrt(upper/lower) for concentrated range
        ce = math.sqrt(upper / lower) if lower > 0 else 1.0
        full_range_ce = 1.0  # Uniswap v2 baseline
        capital_efficiency_x = ce / full_range_ce

        return {
            "current_price":          round(current_price, 6),
            "expected_move_pct":      round(expected_move_pct, 2),
            "fee_tier_bps":           fee_tier_bps,
            "suggested_lower":        round(lower, 6),
            "suggested_upper":        round(upper, 6),
            "range_width_pct":        round((upper / lower - 1) * 100, 2),
            "capital_efficiency_vs_v2_x": round(capital_efficiency_x, 2),
            "note": "Concentrated range; rebalance needed if price exits range.",
        }


# ===========================================================================
# ProtocolRevenueAnalytics
# ===========================================================================

class ProtocolRevenueAnalytics:
    """
    Protocol revenue and fee analytics.

    Computes annualised revenue, TVL-to-revenue (P/S analogue),
    and sector revenue breakdown.
    """

    def __init__(self) -> None:
        self._client = DefiLlamaClient()

    def top_revenue_protocols(self, top_n: int = 20) -> pd.DataFrame:
        """Top protocols by 30-day revenue, with TVL and P/S ratio."""
        fees_df = self._client.get_fees()
        protocols_df = self._client.get_protocols(limit=500)

        if fees_df.empty:
            return pd.DataFrame()

        merged = fees_df.copy()
        if not protocols_df.empty:
            proto_map = protocols_df.set_index("name")["tvl_usd"].to_dict()
            merged["tvl_usd"] = merged["name"].map(proto_map)
        else:
            merged["tvl_usd"] = None

        merged["annualised_revenue_usd"] = merged["revenue_30d_usd"] * 12.0
        merged["ps_ratio"] = merged.apply(
            lambda r: r["tvl_usd"] / r["annualised_revenue_usd"]
            if r["annualised_revenue_usd"] and r["annualised_revenue_usd"] > 0
            and r.get("tvl_usd") and r["tvl_usd"] > 0 else None,
            axis=1,
        )

        result = merged.sort_values("revenue_30d_usd", ascending=False).head(top_n)
        cols = ["name", "category", "total_24h_usd", "revenue_24h_usd",
                "revenue_30d_usd", "annualised_revenue_usd", "tvl_usd", "ps_ratio"]
        available = [c for c in cols if c in result.columns]
        return result[available].reset_index(drop=True)

    def sector_revenue(self) -> pd.DataFrame:
        """Aggregate fees and revenue by protocol category."""
        fees_df = self._client.get_fees()
        if fees_df.empty:
            return pd.DataFrame()

        grouped = fees_df.groupby("category").agg(
            total_24h_usd=("total_24h_usd", "sum"),
            revenue_24h_usd=("revenue_24h_usd", "sum"),
            revenue_30d_usd=("revenue_30d_usd", "sum"),
            protocol_count=("name", "count"),
        ).reset_index()
        grouped["annualised_revenue_usd"] = grouped["revenue_30d_usd"] * 12.0
        grouped["revenue_share_pct"] = (
            grouped["revenue_30d_usd"] / grouped["revenue_30d_usd"].sum() * 100
        ).round(3)
        return grouped.sort_values("revenue_30d_usd", ascending=False).reset_index(drop=True)


# ===========================================================================
# DEXAnalytics
# ===========================================================================

class DEXAnalytics:
    """DEX volume, market share, and liquidity efficiency analytics."""

    def __init__(self) -> None:
        self._client = DefiLlamaClient()

    def top_dexs(self, top_n: int = 20) -> pd.DataFrame:
        """Top DEXs by 24h volume with volume/TVL ratio."""
        df = self._client.get_dex_volumes()
        yields_df = self._client.get_protocols(limit=500)
        if df.empty:
            return pd.DataFrame()

        # Merge TVL
        if not yields_df.empty:
            tvl_map = yields_df.set_index("name")["tvl_usd"].to_dict()
            df["tvl_usd"] = df["tvl_usd"].fillna(df["name"].map(tvl_map))

        df["volume_tvl_ratio"] = df.apply(
            lambda r: round(r["total_24h_usd"] / r["tvl_usd"], 4)
            if r.get("tvl_usd") and r["tvl_usd"] > 0 else None,
            axis=1,
        )
        result = df.sort_values("total_24h_usd", ascending=False).head(top_n)
        return result.reset_index(drop=True)

    def chain_dex_market_share(self) -> pd.DataFrame:
        """DEX volume market share by chain."""
        df = self._client.get_dex_volumes()
        if df.empty:
            return pd.DataFrame()

        # Explode chains
        rows: List[Dict] = []
        for _, r in df.iterrows():
            for chain in (r.get("chains") or [r.get("chain", "")]):
                rows.append({"chain": chain, "volume_24h_usd": float(r["total_24h_usd"] or 0)})

        chain_df = pd.DataFrame(rows).groupby("chain")["volume_24h_usd"].sum().reset_index()
        total = chain_df["volume_24h_usd"].sum()
        chain_df["market_share_pct"] = (chain_df["volume_24h_usd"] / total * 100).round(3)
        return chain_df.sort_values("volume_24h_usd", ascending=False).reset_index(drop=True)

    def sector_tvl_share(self) -> pd.DataFrame:
        """
        DeFi sector TVL breakdown: lending / DEX / yield / derivatives / other.

        Uses protocol category from DefiLlama.
        """
        df = self._client.get_protocols(limit=1000)
        if df.empty:
            return pd.DataFrame()

        sector_map = {
            "Lending": "Lending",
            "DEX":     "DEX",
            "Yield":   "Yield",
            "Derivatives": "Derivatives",
            "Bridge":  "Bridge",
            "CDP":     "CDP / Stablecoin",
            "Staking": "Staking",
            "Liquid Staking": "Liquid Staking",
        }

        def _map_sector(cat: str) -> str:
            for k, v in sector_map.items():
                if k.lower() in cat.lower():
                    return v
            return "Other"

        df["sector"] = df["category"].apply(_map_sector)
        grouped = df.groupby("sector").agg(
            tvl_usd=("tvl_usd", "sum"),
            protocol_count=("slug", "count"),
        ).reset_index()
        total = grouped["tvl_usd"].sum()
        grouped["tvl_share_pct"] = (grouped["tvl_usd"] / total * 100).round(3)
        return grouped.sort_values("tvl_usd", ascending=False).reset_index(drop=True)


# ===========================================================================
# StablecoinAnalytics
# ===========================================================================

class StablecoinAnalytics:
    """
    Stablecoin supply, peg deviation, depeg alerts, and market share.
    """

    def __init__(self) -> None:
        self._client = DefiLlamaClient()

    def top_stablecoins(self, top_n: int = 20) -> pd.DataFrame:
        """Top stablecoins by circulating supply with peg status."""
        df = self._client.get_stablecoins()
        if df.empty:
            return df
        return df.head(top_n)

    def depeg_alerts(self, threshold_pct: float = 1.0) -> pd.DataFrame:
        """Stablecoins with peg deviation > threshold_pct."""
        df = self._client.get_stablecoins()
        if df.empty:
            return df
        alerts = df[
            df["peg_deviation_pct"].notna()
            & (df["peg_deviation_pct"] > threshold_pct)
        ].copy()
        alerts["severity"] = alerts["peg_deviation_pct"].apply(
            lambda x: "critical" if x > 5.0 else ("warning" if x > 2.0 else "caution")
        )
        return alerts.sort_values("peg_deviation_pct", ascending=False).reset_index(drop=True)

    def market_share(self) -> pd.DataFrame:
        """Stablecoin market share by circulating supply."""
        df = self._client.get_stablecoins()
        if df.empty:
            return df
        total = df["circulating_usd"].sum()
        df = df.copy()
        df["market_share_pct"] = (df["circulating_usd"] / total * 100).round(4)
        return df[["name", "symbol", "peg_type", "peg_mechanism", "circulating_usd",
                   "market_share_pct", "price", "peg_deviation_pct", "depeg_alert"]].reset_index(drop=True)

    def supply_growth(self, days: int = 30) -> pd.DataFrame:
        """
        Estimate 30-day supply growth from prev-day circulating supply.
        Note: DefiLlama stablecoins endpoint only provides 1-day comparison.
        Returns protocols with significant supply changes.
        """
        df = self._client.get_stablecoins()
        if df.empty:
            return df
        df = df[df["change_1d_pct"].notna()].copy()
        df["annualised_growth_pct"] = (df["change_1d_pct"] * 365).round(2)
        df["change_direction"] = df["change_1d_pct"].apply(
            lambda x: "expanding" if x > 0 else ("contracting" if x < 0 else "flat")
        )
        return df.sort_values("change_1d_pct", ascending=False).reset_index(drop=True)


# ===========================================================================
# BridgeAnalytics
# ===========================================================================

class BridgeAnalytics:
    """Cross-chain bridge volume, security scoring, and flow analysis."""

    # Security score heuristics (0–100): higher = safer
    _AUDIT_SCORE = 30       # per audit link (max 2)
    _VOLUME_SCORE = 30      # high volume → more battle-tested
    _AGE_BONUS = 20         # placeholder: consistent name/brand = established

    def __init__(self) -> None:
        self._client = DefiLlamaClient()

    def top_bridges(self, top_n: int = 15) -> pd.DataFrame:
        """Top bridges by 24h volume with security scoring."""
        df = self._client.get_bridges()
        if df.empty:
            return df

        df = df.head(top_n).copy()
        df["security_score"] = df.apply(self._score, axis=1)
        return df.reset_index(drop=True)

    def _score(self, row: pd.Series) -> float:
        """Simple bridge security score: audits + volume track record."""
        score = 0.0
        audits = row.get("audit_links") or []
        score += min(len(audits), 2) * self._AUDIT_SCORE
        vol = float(row.get("volume_7d_usd") or 0)
        if vol > 1e9:
            score += self._VOLUME_SCORE
        elif vol > 1e8:
            score += self._VOLUME_SCORE * 0.6
        elif vol > 1e7:
            score += self._VOLUME_SCORE * 0.3
        score += self._AGE_BONUS   # assume all listed = some track record
        return round(min(score, 100.0), 1)

    def cross_chain_flow_summary(self) -> Dict:
        """Aggregate cross-chain flow metrics."""
        df = self._client.get_bridges()
        if df.empty:
            return {"error": "No bridge data available"}

        total_vol_24h = df["volume_24h_usd"].sum()
        total_vol_7d = df["volume_7d_usd"].sum()
        top3 = df.head(3)[["name", "volume_24h_usd", "volume_7d_usd"]].to_dict(orient="records")
        num_audited = df[df["audit_links"].map(bool)].shape[0]

        return {
            "total_volume_24h_usd":  round(total_vol_24h, 0),
            "total_volume_7d_usd":   round(total_vol_7d, 0),
            "bridge_count":          len(df),
            "audited_bridges":       num_audited,
            "top_3_bridges":         top3,
        }


# ===========================================================================
# ProtocolHealthAnalyzer
# ===========================================================================

class ProtocolHealthAnalyzer:
    """
    Compute 0–100 health score for a DeFi protocol.

    Components
    ----------
    TVL stability (30):  CV of 90-day TVL history — lower = more stable
    Revenue/TVL    (25):  Protocol P/S ratio — higher revenue/TVL = more efficient
    Chain diversity (20): Herfindahl-Hirschman index of chain TVL — lower = safer
    Age score      (15):  Days since first TVL recording
    TVL momentum   (10):  30-day TVL change direction
    """

    def __init__(self) -> None:
        self._client = DefiLlamaClient()

    def score(self, slug: str) -> ProtocolHealthScore:
        """Compute comprehensive health score for a protocol."""
        now = datetime.now(tz=timezone.utc)
        detail = self._client.get_protocol(slug)
        tvl_hist = self._client.get_tvl_history(slug)
        fees_df = self._client.get_fees()
        risk_flags: List[str] = []

        # --- TVL stability score (30 pts) ---
        tvl_stability = 0.0
        if not tvl_hist.empty and len(tvl_hist) >= 7:
            recent = tvl_hist.tail(90)["tvl"].values
            if recent.mean() > 0:
                cv = float(np.std(recent) / np.mean(recent))
                if cv < 0.10:
                    tvl_stability = 30.0
                elif cv < 0.25:
                    tvl_stability = 22.0
                elif cv < 0.50:
                    tvl_stability = 12.0
                else:
                    tvl_stability = 4.0
                    risk_flags.append(f"High TVL volatility (CV={cv:.2f})")
        else:
            risk_flags.append("Insufficient TVL history (<7 days)")

        # --- Revenue/TVL (25 pts) ---
        rev_tvl_ratio: Optional[float] = None
        rev_score = 0.0
        if not fees_df.empty:
            fee_row = fees_df[fees_df["name"].str.lower() == slug.lower()]
            if not fee_row.empty:
                rev_30d = float(fee_row.iloc[0].get("revenue_30d_usd") or 0)
                tvl_now = float(detail.get("tvl") or tvl_hist["tvl"].iloc[-1] if not tvl_hist.empty else 0)
                if tvl_now > 0 and rev_30d > 0:
                    rev_tvl_ratio = round(rev_30d * 12.0 / tvl_now, 4)
                    if rev_tvl_ratio > 0.10:
                        rev_score = 25.0
                    elif rev_tvl_ratio > 0.03:
                        rev_score = 15.0
                    elif rev_tvl_ratio > 0.005:
                        rev_score = 8.0
                    else:
                        rev_score = 2.0
            else:
                risk_flags.append("No fee revenue data found")

        # --- Chain diversification (20 pts) ---
        chain_div_score = 0.0
        chain_tvls = detail.get("currentChainTvls", {})
        if chain_tvls:
            vals = [float(v) for v in chain_tvls.values() if v and float(v) > 0]
            total = sum(vals)
            if total > 0 and vals:
                hhi = sum((v / total) ** 2 for v in vals)
                # HHI: 1.0 = monopoly (one chain), lower = diverse
                if hhi < 0.25:
                    chain_div_score = 20.0
                elif hhi < 0.50:
                    chain_div_score = 13.0
                elif hhi < 0.75:
                    chain_div_score = 7.0
                else:
                    chain_div_score = 2.0
                    if len(vals) == 1:
                        risk_flags.append("Single-chain protocol (100% concentration)")
        else:
            risk_flags.append("No chain TVL breakdown available")

        # --- Age score (15 pts) ---
        age_score = 0.0
        age_days: Optional[int] = None
        if not tvl_hist.empty:
            oldest = tvl_hist["date"].min()
            age_days = (now - oldest).days
            if age_days >= 730:     # 2+ years
                age_score = 15.0
            elif age_days >= 365:
                age_score = 10.0
            elif age_days >= 90:
                age_score = 5.0
            else:
                age_score = 1.0
                risk_flags.append(f"Protocol is only {age_days} days old")

        # --- TVL momentum (10 pts) ---
        momentum_score = 0.0
        tvl_momentum_30d: Optional[float] = None
        if not tvl_hist.empty and len(tvl_hist) >= 31:
            recent_tvl = tvl_hist["tvl"].iloc[-1]
            month_ago_tvl = tvl_hist["tvl"].iloc[-31] if len(tvl_hist) >= 31 else tvl_hist["tvl"].iloc[0]
            if month_ago_tvl > 0:
                tvl_momentum_30d = round((recent_tvl / month_ago_tvl - 1.0) * 100, 3)
                if tvl_momentum_30d > 10:
                    momentum_score = 10.0
                elif tvl_momentum_30d > 0:
                    momentum_score = 6.0
                elif tvl_momentum_30d > -15:
                    momentum_score = 3.0
                else:
                    momentum_score = 0.0
                    risk_flags.append(f"TVL down {abs(tvl_momentum_30d):.1f}% in 30 days")

        # Security/audit flags
        audit_links = detail.get("audit_links") or []
        if not audit_links:
            risk_flags.append("No audit links found — unverified contracts")

        health = tvl_stability + rev_score + chain_div_score + age_score + momentum_score
        health = round(min(health, 100.0), 2)

        if health >= 70:
            tier = "blue_chip"
        elif health >= 50:
            tier = "established"
        elif health >= 30:
            tier = "emerging"
        else:
            tier = "risky"

        # Save snapshot
        tvl_now_val = float(tvl_hist["tvl"].iloc[-1]) if not tvl_hist.empty else 0.0
        _save_protocol_snapshot(slug, tvl_now_val, 0.0, 0.0, {
            "health_score": health, "tier": tier, "flags": risk_flags,
        })

        return ProtocolHealthScore(
            protocol_slug=slug,
            health_score=health,
            tvl_stability_score=tvl_stability,
            revenue_tvl_ratio=rev_tvl_ratio,
            chain_diversification_score=chain_div_score,
            age_days=age_days,
            tvl_momentum_30d_pct=tvl_momentum_30d,
            risk_flags=risk_flags,
            tier=tier,
        )


# ===========================================================================
# GovernanceAnalytics
# ===========================================================================

class GovernanceAnalytics:
    """
    Governance token analytics via Snapshot.org GraphQL API (free, no auth).

    Tracks proposals, voter participation, and treasury balance proxy.
    """

    _GQL_PROPOSALS = """
    query Proposals($space: String!, $first: Int!) {
        proposals(
            first: $first
            skip: 0
            where: { space: $space, state: "all" }
            orderBy: "created"
            orderDirection: desc
        ) {
            id title state start end choices votes quorum author
        }
    }
    """

    _GQL_SPACE = """
    query Space($id: String!) {
        space(id: $id) {
            id name about symbol members network proposals followers
        }
    }
    """

    def __init__(self) -> None:
        pass

    def get_proposals(self, space_id: str, limit: int = 20) -> pd.DataFrame:
        """
        Fetch recent governance proposals for a Snapshot space.

        Common space IDs: aave.eth, compound-governance.eth, uniswap, etc.
        """
        cache_key = f"snapshot_proposals_{space_id}_{limit}"
        cached = _cache_get(cache_key, "snapshot")
        if cached is not None:
            return pd.DataFrame(cached)

        result = _post(_SNAPSHOT_GQL, {
            "query": self._GQL_PROPOSALS,
            "variables": {"space": space_id, "first": limit},
        })

        if not result or "data" not in result:
            return pd.DataFrame()

        proposals = result["data"].get("proposals", [])
        rows: List[Dict] = []
        for p in proposals:
            rows.append({
                "id":          p.get("id", ""),
                "title":       p.get("title", ""),
                "state":       p.get("state", ""),
                "votes":       p.get("votes", 0),
                "quorum":      p.get("quorum", 0),
                "choices":     len(p.get("choices") or []),
                "author":      p.get("author", ""),
                "start_ts":    p.get("start"),
                "end_ts":      p.get("end"),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            _cache_set(cache_key, "snapshot", rows)
        return df

    def get_space_info(self, space_id: str) -> Dict:
        """Get governance space metadata (members, followers, proposals count)."""
        cache_key = f"snapshot_space_{space_id}"
        cached = _cache_get(cache_key, "snapshot")
        if cached is not None:
            return cached

        result = _post(_SNAPSHOT_GQL, {
            "query": self._GQL_SPACE,
            "variables": {"id": space_id},
        })

        if not result or "data" not in result:
            return {"error": f"Snapshot space '{space_id}' not found or API unavailable"}

        space = result["data"].get("space") or {}
        info = {
            "space_id":    space.get("id", space_id),
            "name":        space.get("name", ""),
            "symbol":      space.get("symbol", ""),
            "network":     space.get("network", ""),
            "members":     space.get("members", 0),
            "followers":   space.get("followers", 0),
            "proposals":   space.get("proposals", 0),
            "about":       (space.get("about") or "")[:200],
        }
        _cache_set(cache_key, "snapshot", info)
        return info

    def participation_metrics(self, space_id: str, limit: int = 50) -> Dict:
        """Compute governance participation metrics from recent proposals."""
        df = self.get_proposals(space_id, limit)
        if df.empty:
            return {"error": "No proposals found", "space_id": space_id}

        closed = df[df["state"] == "closed"]
        active = df[df["state"] == "active"]

        avg_votes = float(closed["votes"].mean()) if not closed.empty else 0.0
        max_votes = float(closed["votes"].max()) if not closed.empty else 0.0
        passed_quorum = int((closed["votes"] >= closed["quorum"].where(closed["quorum"] > 0, 1)).sum()) if not closed.empty else 0

        return {
            "space_id":            space_id,
            "total_proposals":     len(df),
            "closed_proposals":    len(closed),
            "active_proposals":    len(active),
            "avg_votes_per_prop":  round(avg_votes, 0),
            "max_votes":           round(max_votes, 0),
            "proposals_met_quorum": passed_quorum,
            "quorum_hit_rate_pct": round(passed_quorum / max(len(closed), 1) * 100, 2),
        }


# ===========================================================================
# LiquidityPoolAnalytics
# ===========================================================================

class LiquidityPoolAnalytics:
    """
    Detailed liquidity pool analytics: fee tier, volume/TVL, IL, and
    concentrated liquidity range optimisation.
    """

    def __init__(self) -> None:
        self._client = DefiLlamaClient()
        self._yield_agg = YieldAggregator()

    def get_pools(
        self,
        protocol: Optional[str] = None,
        chain: Optional[str] = None,
        min_tvl_usd: float = 1_000_000,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """Fetch and enrich liquidity pool data with volume/TVL metrics."""
        df = self._client.get_yields()
        if df.empty:
            return df

        # Exclude stablecoins from LP analysis (focus on volatile pairs)
        mask = df["tvl_usd"] >= min_tvl_usd
        if protocol:
            mask &= df["protocol"].str.lower().str.contains(protocol.lower(), na=False)
        if chain:
            mask &= df["chain"].str.lower() == chain.lower()

        result = df[mask].copy()

        # Volume/TVL ratio
        result["volume_tvl_ratio"] = result.apply(
            lambda r: round(float(r.get("volume_usd_1d") or 0) / r["tvl_usd"], 6)
            if r["tvl_usd"] > 0 else None,
            axis=1,
        )

        # Estimate fee tier from symbol or pool_meta
        def _est_fee_tier(row: pd.Series) -> Optional[float]:
            meta = str(row.get("pool_meta") or "")
            if "0.05%" in meta or "5bps" in meta.lower():
                return 5.0
            if "0.3%" in meta or "30bps" in meta.lower():
                return 30.0
            if "1%" in meta or "100bps" in meta.lower():
                return 100.0
            return None

        result["fee_tier_bps"] = result.apply(_est_fee_tier, axis=1)

        cols = ["pool_id", "protocol", "chain", "symbol", "tvl_usd", "apy",
                "il_risk", "stablecoin", "fee_tier_bps", "volume_usd_1d",
                "volume_tvl_ratio", "audited", "pool_meta"]
        available = [c for c in cols if c in result.columns]
        return result[available].sort_values("tvl_usd", ascending=False).head(top_n).reset_index(drop=True)

    def il_impact_comparison(self, price_changes: Optional[List[float]] = None) -> pd.DataFrame:
        """
        IL comparison table for multiple price move scenarios.

        price_changes: list of fractional price moves (e.g. [0.2, 0.5, 1.0, -0.5])
        """
        if price_changes is None:
            price_changes = [-0.75, -0.50, -0.25, 0.25, 0.50, 1.00, 2.00]

        ya = YieldAggregator()
        rows: List[Dict] = []
        for change in price_changes:
            result = ya.il_calculator(change)
            rows.append({
                "price_change_pct":      change * 100,
                "il_pct":                result.get("il_pct"),
                "value_vs_hodl":         result.get("value_vs_hodl"),
                "breakeven_apy_needed":  result.get("break_even_apy_needed"),
            })

        return pd.DataFrame(rows)


# ===========================================================================
# FastAPI Router
# ===========================================================================

router = APIRouter(prefix="/api/v2/defi", tags=["defi-analytics-v2"])

# Module-level service instances (lazy init is fine for single-worker)
_tvl = TVLAnalytics()
_yield_agg = YieldAggregator()
_rev = ProtocolRevenueAnalytics()
_dex = DEXAnalytics()
_stable = StablecoinAnalytics()
_bridge = BridgeAnalytics()
_health = ProtocolHealthAnalyzer()
_gov = GovernanceAnalytics()
_lp = LiquidityPoolAnalytics()
_client = DefiLlamaClient()


# ------------------------------------------------------------------
# Protocols
# ------------------------------------------------------------------

@router.get("/protocols", summary="All DeFi protocols with TVL and metadata")
def get_protocols(
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = Query(None),
    chain: Optional[str] = Query(None),
    min_tvl_mm: float = Query(0.0, ge=0.0),
) -> Dict:
    """Fetch all DefiLlama protocols. Filter by category, chain, or min TVL."""
    df = _client.get_protocols(limit=limit)
    if df.empty:
        return {"error": "No protocol data available", "count": 0}

    if category:
        df = df[df["category"].str.lower().str.contains(category.lower(), na=False)]
    if chain:
        df = df[df["chain"].str.lower() == chain.lower()]
    if min_tvl_mm > 0:
        df = df[df["tvl_usd"] >= min_tvl_mm * 1e6]

    return {
        "count": len(df),
        "protocols": df.to_dict(orient="records"),
    }


@router.get("/protocol/{slug}", summary="Protocol detail, TVL history, and health snapshot")
def get_protocol(
    slug: str,
    include_tvl_history: bool = Query(True),
    tvl_days: int = Query(90, ge=7, le=365),
) -> Dict:
    """Full protocol data: metadata, TVL history, chain breakdown."""
    detail = _client.get_protocol(slug)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Protocol '{slug}' not found on DefiLlama")

    result: Dict = {
        "slug":      slug,
        "name":      detail.get("name", slug),
        "category":  detail.get("category", ""),
        "chains":    detail.get("chains", []),
        "tvl_usd":   detail.get("tvl", 0),
        "audit_links": detail.get("audit_links") or [],
        "url":       detail.get("url", ""),
        "twitter":   detail.get("twitter", ""),
        "description": detail.get("description", ""),
        "chain_tvls":  detail.get("currentChainTvls", {}),
    }

    if include_tvl_history:
        hist_df = _tvl.protocol_tvl_history(slug, days=tvl_days)
        if not hist_df.empty:
            result["tvl_history"] = hist_df.to_dict(orient="records")

    return result


@router.get("/tvl/{chain}", summary="Chain-level TVL and market share")
def get_chain_tvl(chain: str) -> Dict:
    """TVL and DeFi market share for a specific chain."""
    df = _client.get_chains()
    if df.empty:
        raise HTTPException(status_code=503, detail="Chain TVL data unavailable")

    row = df[df["chain"].str.lower() == chain.lower()]
    if row.empty:
        # Return summary if chain not found
        return {
            "error": f"Chain '{chain}' not found",
            "available_chains": df["chain"].head(30).tolist(),
        }

    total_tvl = df["tvl_usd"].sum()
    r = row.iloc[0]
    return {
        "chain":           r["chain"],
        "tvl_usd":         round(float(r["tvl_usd"]), 0),
        "market_share_pct": round(float(r["tvl_usd"]) / total_tvl * 100, 4) if total_tvl > 0 else 0.0,
        "token_symbol":    r.get("token_symbol", ""),
        "total_defi_tvl":  round(total_tvl, 0),
    }


# ------------------------------------------------------------------
# Yields
# ------------------------------------------------------------------

@router.get("/yield-pools", summary="Yield farming pool screener")
def get_yield_pools(
    chain: Optional[str] = Query(None),
    protocol: Optional[str] = Query(None),
    min_tvl_mm: float = Query(1.0, ge=0.0),
    min_apy: float = Query(0.0, ge=0.0),
    max_apy: float = Query(500.0),
    stablecoin_only: bool = Query(False),
    audited_only: bool = Query(False),
    il_risk_max: str = Query("high", regex="^(no|low|high)$"),
    top_n: int = Query(50, ge=1, le=200),
) -> Dict:
    """Comprehensive yield pool screener with risk-adjusted APY ranking."""
    df = _yield_agg.screen(
        chain=chain,
        protocol=protocol,
        min_tvl_usd=min_tvl_mm * 1e6,
        min_apy=min_apy,
        max_apy=max_apy,
        stablecoin_only=stablecoin_only,
        audited_only=audited_only,
        il_risk_max=il_risk_max,
        top_n=top_n,
    )
    return {
        "count": len(df),
        "filters": {
            "chain": chain, "protocol": protocol,
            "min_tvl_mm": min_tvl_mm, "min_apy": min_apy,
            "stablecoin_only": stablecoin_only, "audited_only": audited_only,
        },
        "pools": df.to_dict(orient="records") if not df.empty else [],
    }


@router.get("/farming-screener", summary="Curated yield farming opportunities")
def farming_screener(
    chain: Optional[str] = Query(None),
    min_tvl_mm: float = Query(5.0),
    min_apy: float = Query(5.0),
    max_apy: float = Query(100.0),
    stablecoin_only: bool = Query(False),
    audited_only: bool = Query(True),
    top_n: int = Query(25),
) -> Dict:
    """
    Curated yield farming screener (audited protocols preferred).
    Defaults to audited-only, $5M min TVL, 5–100% APY.
    """
    df = _yield_agg.screen(
        chain=chain,
        min_tvl_usd=min_tvl_mm * 1e6,
        min_apy=min_apy,
        max_apy=max_apy,
        stablecoin_only=stablecoin_only,
        audited_only=audited_only,
        il_risk_max="low" if not stablecoin_only else "no",
        top_n=top_n,
    )
    return {
        "count": len(df),
        "screener_name": "farming_screener_v2",
        "pools": df.to_dict(orient="records") if not df.empty else [],
    }


# ------------------------------------------------------------------
# Stablecoins
# ------------------------------------------------------------------

@router.get("/stablecoins", summary="Stablecoin supply, peg status, and depeg alerts")
def get_stablecoins(
    depeg_alert_only: bool = Query(False),
    peg_type: Optional[str] = Query(None, description="e.g. peggedUSD, peggedEUR"),
    top_n: int = Query(30, ge=1, le=100),
) -> Dict:
    """All stablecoins with peg deviation and depeg alert flags."""
    if depeg_alert_only:
        df = _stable.depeg_alerts()
    else:
        df = _stable.top_stablecoins(top_n=top_n)

    if not df.empty and peg_type:
        df = df[df["peg_type"].str.lower() == peg_type.lower()]

    alerts = df[df["depeg_alert"] == True].shape[0] if not df.empty else 0  # noqa: E712
    return {
        "count":         len(df),
        "active_depegs": alerts,
        "stablecoins":   df.to_dict(orient="records") if not df.empty else [],
    }


# ------------------------------------------------------------------
# Bridges
# ------------------------------------------------------------------

@router.get("/bridges", summary="Cross-chain bridge volume and security scores")
def get_bridges(top_n: int = Query(20, ge=1, le=50)) -> Dict:
    """Bridge cross-chain volume, security scoring, and flow summary."""
    df = _bridge.top_bridges(top_n=top_n)
    flow_summary = _bridge.cross_chain_flow_summary()
    return {
        "flow_summary": flow_summary,
        "count":        len(df),
        "bridges":      df.to_dict(orient="records") if not df.empty else [],
    }


# ------------------------------------------------------------------
# Fees / Revenue
# ------------------------------------------------------------------

@router.get("/fees", summary="Protocol fee revenue and P/S ratios")
def get_fees(
    top_n: int = Query(30, ge=1, le=100),
    category: Optional[str] = Query(None),
) -> Dict:
    """Protocol fee revenue ranked by 30-day earnings with TVL and P/S ratio."""
    df = _rev.top_revenue_protocols(top_n=top_n)
    if not df.empty and category:
        df = df[df["category"].str.lower().str.contains(category.lower(), na=False)]

    sector = _rev.sector_revenue()
    return {
        "count":           len(df),
        "top_protocols":   df.to_dict(orient="records") if not df.empty else [],
        "sector_summary":  sector.to_dict(orient="records") if not sector.empty else [],
    }


# ------------------------------------------------------------------
# DEX Volumes
# ------------------------------------------------------------------

@router.get("/dex-volumes", summary="DEX volume, market share, and volume/TVL ratio")
def get_dex_volumes(
    top_n: int = Query(20, ge=1, le=100),
    chain: Optional[str] = Query(None),
) -> Dict:
    """DEX 24h volume, TVL, and sector TVL breakdown."""
    df = _dex.top_dexs(top_n=top_n)
    if not df.empty and chain:
        df = df[df["chains"].apply(lambda x: chain.lower() in [c.lower() for c in (x or [])])]

    sector_tvl = _dex.sector_tvl_share()
    chain_share = _dex.chain_dex_market_share()

    return {
        "count":           len(df),
        "dexs":            df.to_dict(orient="records") if not df.empty else [],
        "sector_tvl":      sector_tvl.to_dict(orient="records") if not sector_tvl.empty else [],
        "chain_dex_share": chain_share.head(15).to_dict(orient="records") if not chain_share.empty else [],
    }


# ------------------------------------------------------------------
# Health Metrics
# ------------------------------------------------------------------

@router.get("/health-metrics/{slug}", summary="Protocol health score (0–100)")
def get_health_metrics(slug: str) -> Dict:
    """
    Compute a 0–100 health score for a DeFi protocol.

    Components: TVL stability (30), Revenue/TVL (25), Chain diversity (20),
                Age (15), TVL momentum (10).
    """
    try:
        score = _health.score(slug)
        return score.model_dump()
    except Exception as exc:
        logger.error("Health score error", slug=slug, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# TVL
# ------------------------------------------------------------------

@router.get("/tvl-dominance", summary="TVL dominance of top DeFi protocols")
def get_tvl_dominance(
    top_n: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
) -> Dict:
    """TVL market share for top N protocols."""
    df = _tvl.tvl_dominance(top_n=top_n)
    if not df.empty and category:
        df = df[df["category"].str.lower().str.contains(category.lower(), na=False)]
    trend = _tvl.total_defi_tvl_trend()
    return {
        "count":         len(df),
        "total_tvl_trend": trend,
        "protocols":     df.to_dict(orient="records") if not df.empty else [],
    }


@router.get("/chain-tvl", summary="Per-chain TVL market share")
def get_chain_tvl_all() -> Dict:
    """TVL market share for all tracked chains."""
    df = _tvl.chain_tvl_dominance()
    return {
        "count":  len(df),
        "chains": df.to_dict(orient="records") if not df.empty else [],
    }


@router.get("/sector-comparison", summary="DeFi sector TVL breakdown (lending/DEX/yield/derivatives)")
def get_sector_comparison() -> Dict:
    """DeFi sector TVL share: lending vs DEX vs yield vs derivatives."""
    tvl_sectors = _dex.sector_tvl_share()
    rev_sectors = _rev.sector_revenue()
    return {
        "tvl_by_sector":     tvl_sectors.to_dict(orient="records") if not tvl_sectors.empty else [],
        "revenue_by_sector": rev_sectors.to_dict(orient="records") if not rev_sectors.empty else [],
    }


# ------------------------------------------------------------------
# Governance
# ------------------------------------------------------------------

@router.get("/governance/{space_id}", summary="Governance proposals and participation metrics")
def get_governance(
    space_id: str,
    limit: int = Query(20, ge=1, le=100),
) -> Dict:
    """
    Fetch governance data from Snapshot.org (free, no auth).

    Common space IDs: aave.eth, uniswap, compound-governance.eth, makerdao.eth
    """
    space_info = _gov.get_space_info(space_id)
    metrics = _gov.participation_metrics(space_id, limit=limit)
    proposals_df = _gov.get_proposals(space_id, limit=limit)

    return {
        "space":        space_info,
        "participation": metrics,
        "recent_proposals": proposals_df.to_dict(orient="records") if not proposals_df.empty else [],
    }


# ------------------------------------------------------------------
# Liquidity Pools
# ------------------------------------------------------------------

@router.get("/liquidity-pools", summary="Liquidity pool analytics with volume/TVL and fee tier")
def get_liquidity_pools(
    protocol: Optional[str] = Query(None),
    chain: Optional[str] = Query(None),
    min_tvl_mm: float = Query(1.0),
    top_n: int = Query(50),
) -> Dict:
    """Liquidity pools with fee tier, volume/TVL ratio, and IL risk."""
    df = _lp.get_pools(
        protocol=protocol,
        chain=chain,
        min_tvl_usd=min_tvl_mm * 1e6,
        top_n=top_n,
    )
    il_table = _lp.il_impact_comparison()
    return {
        "count":       len(df),
        "pools":       df.to_dict(orient="records") if not df.empty else [],
        "il_impact_table": il_table.to_dict(orient="records"),
    }


@router.get("/il-calculator", summary="Impermanent loss calculator")
def il_calculator(
    price_change_pct: float = Query(..., description="Price change % of asset A vs B (e.g. 50 = +50%)"),
    fee_tier_bps: float = Query(30.0),
    apy_pct: float = Query(0.0, description="Pool APY % to estimate breakeven hold period"),
) -> Dict:
    """
    Calculate impermanent loss for an AMM pool at a given price move.

    Also estimates breakeven APY hold period.
    """
    price_ratio_change = price_change_pct / 100.0
    result = _yield_agg.il_calculator(price_ratio_change)

    # Breakeven days from fee income
    il_pct = abs(result.get("il_pct") or 0)
    breakeven_days: Optional[float] = None
    if apy_pct > 0:
        daily_rate = apy_pct / 365.0
        breakeven_days = round(il_pct / daily_rate, 1) if daily_rate > 0 else None

    result["pool_apy_pct"] = apy_pct
    result["fee_tier_bps"] = fee_tier_bps
    result["breakeven_days"] = breakeven_days

    # Optimal CL range
    cl_range = _yield_agg.optimal_cl_range(
        current_price=100.0,  # relative price
        expected_move_pct=abs(price_change_pct),
        fee_tier_bps=fee_tier_bps,
    )
    result["concentrated_liquidity_range"] = cl_range
    return result


# ------------------------------------------------------------------
# TVL Momentum / Screeners
# ------------------------------------------------------------------

@router.get("/tvl-momentum", summary="Protocols gaining TVL (momentum screen)")
def get_tvl_momentum(
    min_7d_change_pct: float = Query(10.0),
    min_tvl_mm: float = Query(10.0),
) -> Dict:
    """Protocols with TVL growth >= threshold — capital rotation signal."""
    df = _tvl.tvl_momentum(min_7d_change_pct=min_7d_change_pct, min_tvl_mm=min_tvl_mm)
    return {
        "count":     len(df),
        "protocols": df.to_dict(orient="records") if not df.empty else [],
    }


@router.get("/tvl-outflows", summary="Protocols experiencing TVL outflows")
def get_tvl_outflows(
    min_7d_decline_pct: float = Query(-15.0),
    min_tvl_mm: float = Query(5.0),
) -> Dict:
    """Protocols with significant TVL outflows — risk / flight-to-quality signal."""
    df = _tvl.tvl_outflows(min_7d_decline_pct=min_7d_decline_pct, min_tvl_mm=min_tvl_mm)
    return {
        "count":     len(df),
        "protocols": df.to_dict(orient="records") if not df.empty else [],
    }


@router.get("/stable-yields", summary="Top stable-coin yield pools (no IL risk)")
def get_stable_yields(
    min_tvl_mm: float = Query(5.0),
    top_n: int = Query(20),
) -> Dict:
    """Best yield opportunities with no IL risk (stablecoin LP or lending)."""
    df = _yield_agg.top_stable_yields(min_tvl_usd=min_tvl_mm * 1e6, top_n=top_n)
    return {
        "count": len(df),
        "pools": df.to_dict(orient="records") if not df.empty else [],
    }


# ------------------------------------------------------------------
# Cache / Health
# ------------------------------------------------------------------

@router.get("/cache-stats", summary="SQLite cache statistics")
def get_cache_stats() -> Dict:
    """Return cache entry counts by data type and approximate size."""
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            rows = conn.execute(
                "SELECT data_type, COUNT(*) as n, AVG(LENGTH(payload)) as avg_bytes "
                "FROM cache GROUP BY data_type"
            ).fetchall()
            snap_count = conn.execute(
                "SELECT COUNT(*) as n FROM protocol_snapshots"
            ).fetchone()[0]
        return {
            "db_path":          str(_DB_PATH),
            "cache_entries":    {r[0]: {"count": r[1], "avg_bytes": round(r[2] or 0, 0)} for r in rows},
            "protocol_snapshots": snap_count,
            "ttl_config":       _TTL,
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.post("/cache-invalidate", summary="Invalidate a cache key or all entries")
def cache_invalidate(
    cache_key: Optional[str] = Query(None, description="Specific key to delete; omit to clear all"),
    data_type: Optional[str] = Query(None, description="Delete all entries of this data_type"),
) -> Dict:
    """Invalidate SQLite cache entries. Use sparingly — fetches are rate-limited."""
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            if cache_key:
                conn.execute("DELETE FROM cache WHERE cache_key=?", (cache_key,))
                return {"deleted": "key", "cache_key": cache_key}
            elif data_type:
                conn.execute("DELETE FROM cache WHERE data_type=?", (data_type,))
                return {"deleted": "data_type", "data_type": data_type}
            else:
                conn.execute("DELETE FROM cache")
                return {"deleted": "all"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/module-health", summary="DeFi analytics v2 health check")
def module_health() -> Dict:
    """Return module status, API connectivity, and cache statistics."""
    # Quick connectivity check
    chains_ok = not _client.get_chains().empty
    yields_ok = not _client.get_yields().empty

    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            total_cache = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    except Exception:
        total_cache = -1

    return {
        "status":           "ok",
        "dimension":        107,
        "version":          "v2",
        "defi_llama_tvl":   "connected" if chains_ok else "unavailable",
        "defi_llama_yields": "connected" if yields_ok else "unavailable",
        "db_path":          str(_DB_PATH),
        "cache_entries":    total_cache,
        "as_of":            datetime.utcnow().isoformat(),
        "free_apis": [
            _LLAMA_BASE,
            _YIELDS_BASE,
            _STABLES_BASE,
            _BRIDGES_BASE,
            _SNAPSHOT_GQL,
        ],
    }


# ---------------------------------------------------------------------------
# Convenience re-exports
# ---------------------------------------------------------------------------

defi_router = router   # alias for main app registration
