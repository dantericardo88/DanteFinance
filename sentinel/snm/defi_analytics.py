"""
DeFi protocol analytics via DefiLlama free API — Dimensions #107 and #110.

Aggregates TVL, chain activity, stablecoin supply, yield opportunities, and
DEX volume data from the DefiLlama family of APIs. No API key required.

Base URLs:
  - https://api.llama.fi        (protocols, chains, DEX volume)
  - https://yields.llama.fi     (yield pools)
  - https://stablecoins.llama.fi (stablecoin supply)

Score target: SENTINEL 9, Bloomberg 2 (Bloomberg has no DeFi TVL dashboard).
Dim 110 (DEX analytics): SENTINEL sovereign — real-time DEX volume by protocol
and chain, not available in Bloomberg at any tier.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIMEOUT = 30.0          # seconds
_CACHE_TTL = 300         # 5-minute in-memory cache (seconds)
_MAX_RETRIES = 3
_RETRY_SLEEP = 2.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProtocolData(BaseModel):
    name: str
    slug: str
    chain: Optional[str] = None
    chains: list[str] = []
    tvl: Optional[float] = None
    tvl_change_1d: Optional[float] = None
    tvl_change_7d: Optional[float] = None
    mcap: Optional[float] = None
    mcap_tvl_ratio: Optional[float] = None
    category: Optional[str] = None
    token: Optional[str] = None


class ChainData(BaseModel):
    name: str
    tvl: float
    tvl_change_1d: Optional[float] = None
    tvl_change_7d: Optional[float] = None
    protocols_count: Optional[int] = None


class StablecoinData(BaseModel):
    name: str
    symbol: str
    peg_type: str        # "fiat" | "crypto" | "algorithmic"
    circulating: float   # total supply in USD
    circulating_change_7d: Optional[float] = None
    peg_deviation: Optional[float] = None   # % deviation from $1.00


class YieldOpportunity(BaseModel):
    protocol: str
    chain: str
    pool: str
    apy: float
    tvl_usd: Optional[float] = None
    il_risk: str         # "none" | "low" | "medium" | "high"
    reward_tokens: list[str] = []


class DeFiDashboard(BaseModel):
    total_defi_tvl: float
    top_protocols: list[ProtocolData]
    top_chains: list[ChainData]
    stablecoins: list[StablecoinData]
    top_yields: list[YieldOpportunity]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Dim 110 — DEX Pydantic models
# ---------------------------------------------------------------------------

class DEXProtocol(BaseModel):
    name: str
    chains: list[str] = []
    volume_24h: Optional[float] = None
    volume_7d: Optional[float] = None
    volume_all_time: Optional[float] = None
    change_1d_pct: Optional[float] = None


class DEXDashboard(BaseModel):
    total_dex_volume_24h: float
    top_protocols: list[DEXProtocol]
    volume_by_chain: dict[str, float]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    age = datetime.now(timezone.utc).timestamp() - ts
    if age > _CACHE_TTL:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value: object) -> None:
    _cache[key] = (datetime.now(timezone.utc).timestamp(), value)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _fetch_json(client: httpx.AsyncClient, url: str) -> Optional[object]:
    """GET url, return parsed JSON or None on error. Retries up to _MAX_RETRIES."""
    cached = _cache_get(url)
    if cached is not None:
        logger.debug("defi_analytics cache hit: %s", url)
        return cached

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(url, timeout=_TIMEOUT)
            if resp.status_code == 429:
                wait = _RETRY_SLEEP * attempt
                logger.warning("defi_analytics 429 on %s, sleeping %.1fs", url, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            _cache_set(url, data)
            return data
        except httpx.TimeoutException:
            logger.warning("defi_analytics timeout on %s (attempt %d)", url, attempt)
        except httpx.HTTPStatusError as exc:
            logger.error("defi_analytics HTTP %d on %s: %s", exc.response.status_code, url, exc)
            return None
        except Exception as exc:
            logger.error("defi_analytics error on %s: %s", url, exc)
            return None
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_SLEEP)

    return None


# ---------------------------------------------------------------------------
# IL-risk heuristic
# ---------------------------------------------------------------------------

def _il_risk(pool: str, symbol: str) -> str:
    """Estimate impermanent-loss risk from pool name / symbol string."""
    stable_tokens = {"usdc", "usdt", "dai", "frax", "lusd", "susd", "busd", "eurs", "crvusd"}
    tokens = {t.lower() for t in symbol.replace("-", "/").split("/")}
    if tokens.issubset(stable_tokens):
        return "none"
    if tokens & stable_tokens:
        return "low"
    pool_lower = pool.lower()
    if any(t in pool_lower for t in ["eth/btc", "btc/eth"]):
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# DefiLlamaClient
# ---------------------------------------------------------------------------

class DefiLlamaClient:
    BASE = "https://api.llama.fi"
    YIELDS_BASE = "https://yields.llama.fi"
    STABLECOINS_BASE = "https://stablecoins.llama.fi"

    # ------------------------------------------------------------------
    async def get_all_protocols(self) -> list[ProtocolData]:
        """GET /protocols — all DeFi protocols with TVL."""
        async with httpx.AsyncClient() as client:
            data = await _fetch_json(client, f"{self.BASE}/protocols")

        if not isinstance(data, list):
            logger.error("get_all_protocols: unexpected response type")
            return []

        result: list[ProtocolData] = []
        for item in data:
            try:
                tvl = item.get("tvl")
                mcap = item.get("mcap")
                mcap_tvl = None
                if mcap and tvl and tvl > 0:
                    mcap_tvl = round(mcap / tvl, 4)

                result.append(ProtocolData(
                    name=item.get("name", ""),
                    slug=item.get("slug", item.get("name", "").lower().replace(" ", "-")),
                    chain=item.get("chain"),
                    chains=item.get("chains", []) or [],
                    tvl=float(tvl) if tvl is not None else None,
                    tvl_change_1d=item.get("change_1d"),
                    tvl_change_7d=item.get("change_7d"),
                    mcap=float(mcap) if mcap is not None else None,
                    mcap_tvl_ratio=mcap_tvl,
                    category=item.get("category"),
                    token=item.get("symbol"),
                ))
            except Exception as exc:
                logger.debug("defi_analytics protocol parse error: %s", exc)

        return result

    # ------------------------------------------------------------------
    async def get_protocol_tvl_history(self, protocol_slug: str) -> list[dict]:
        """GET /protocol/{protocol} — historical TVL for one protocol."""
        url = f"{self.BASE}/protocol/{protocol_slug}"
        async with httpx.AsyncClient() as client:
            data = await _fetch_json(client, url)

        if not isinstance(data, dict):
            return []

        # DefiLlama returns tvl as a list of {date, totalLiquidityUSD}
        tvl_list = data.get("tvl", [])
        if not isinstance(tvl_list, list):
            return []

        history: list[dict] = []
        for entry in tvl_list:
            try:
                history.append({
                    "date": datetime.utcfromtimestamp(entry["date"]).date().isoformat(),
                    "tvl_usd": float(entry.get("totalLiquidityUSD", 0)),
                })
            except Exception:
                pass
        return history

    # ------------------------------------------------------------------
    async def get_chain_tvl(self) -> list[ChainData]:
        """GET /v2/chains — TVL by chain."""
        url = f"{self.BASE}/v2/chains"
        async with httpx.AsyncClient() as client:
            data = await _fetch_json(client, url)

        if not isinstance(data, list):
            return []

        result: list[ChainData] = []
        for item in data:
            try:
                result.append(ChainData(
                    name=item.get("name", item.get("gecko_id", "")),
                    tvl=float(item.get("tvl", 0)),
                    tvl_change_1d=item.get("change_1d"),
                    tvl_change_7d=item.get("change_7d"),
                    protocols_count=item.get("protocols"),
                ))
            except Exception as exc:
                logger.debug("chain parse error: %s", exc)

        # Sort descending by TVL
        result.sort(key=lambda c: c.tvl or 0.0, reverse=True)
        return result

    # ------------------------------------------------------------------
    async def get_stablecoins(self) -> list[StablecoinData]:
        """GET stablecoins.llama.fi/stablecoins — stablecoin supply."""
        url = f"{self.STABLECOINS_BASE}/stablecoins?includePrices=true"
        async with httpx.AsyncClient() as client:
            data = await _fetch_json(client, url)

        if not isinstance(data, dict):
            return []

        peggedAssets = data.get("peggedAssets", [])
        if not isinstance(peggedAssets, list):
            return []

        result: list[StablecoinData] = []
        for item in peggedAssets:
            try:
                # circulating supply nested under peggedUSD key
                circ_obj = item.get("circulating", {})
                circulating = float(circ_obj.get("peggedUSD", 0) or 0)

                circ_7d_ago_obj = item.get("circulatingPrevWeek", {})
                circ_7d_ago = float(circ_7d_ago_obj.get("peggedUSD", 0) or 0) if circ_7d_ago_obj else None
                change_7d = None
                if circ_7d_ago and circ_7d_ago > 0:
                    change_7d = round((circulating - circ_7d_ago) / circ_7d_ago * 100, 2)

                # Peg deviation from price (if price is provided)
                price = item.get("price")
                peg_deviation = None
                if price is not None:
                    peg_deviation = round((float(price) - 1.0) * 100, 4)

                # Map peg mechanism
                peg_mechanism = item.get("pegMechanism", "").lower()
                if "fiat" in peg_mechanism or "backed" in peg_mechanism:
                    peg_type = "fiat"
                elif "algo" in peg_mechanism:
                    peg_type = "algorithmic"
                else:
                    peg_type = "crypto"

                result.append(StablecoinData(
                    name=item.get("name", ""),
                    symbol=item.get("symbol", ""),
                    peg_type=peg_type,
                    circulating=circulating,
                    circulating_change_7d=change_7d,
                    peg_deviation=peg_deviation,
                ))
            except Exception as exc:
                logger.debug("stablecoin parse error: %s", exc)

        result.sort(key=lambda s: s.circulating, reverse=True)
        return result

    # ------------------------------------------------------------------
    async def get_yields(
        self,
        min_tvl: float = 1_000_000,
        min_apy: float = 3.0,
    ) -> list[YieldOpportunity]:
        """GET yields.llama.fi/pools — yield farming opportunities."""
        url = f"{self.YIELDS_BASE}/pools"
        async with httpx.AsyncClient() as client:
            data = await _fetch_json(client, url)

        if not isinstance(data, dict):
            return []

        pools = data.get("data", [])
        if not isinstance(pools, list):
            return []

        result: list[YieldOpportunity] = []
        for pool in pools:
            try:
                tvl = pool.get("tvlUsd") or 0.0
                apy = pool.get("apy") or 0.0
                if tvl < min_tvl or apy < min_apy:
                    continue

                symbol = pool.get("symbol", "")
                pool_id = pool.get("pool", "")
                reward_tokens = pool.get("rewardTokens") or []
                if not isinstance(reward_tokens, list):
                    reward_tokens = []

                result.append(YieldOpportunity(
                    protocol=pool.get("project", ""),
                    chain=pool.get("chain", ""),
                    pool=symbol or pool_id,
                    apy=round(float(apy), 4),
                    tvl_usd=float(tvl),
                    il_risk=_il_risk(pool_id, symbol),
                    reward_tokens=[str(t) for t in reward_tokens],
                ))
            except Exception as exc:
                logger.debug("yield pool parse error: %s", exc)

        # Sort by APY descending
        result.sort(key=lambda y: y.apy, reverse=True)
        return result

    # ------------------------------------------------------------------
    # Dim 110 — DEX analytics
    # ------------------------------------------------------------------

    async def get_dex_overview(self, top_n: int = 15) -> list[dict]:
        """Top DEX protocols by 24h volume.

        Fetches from https://api.llama.fi/overview/dexs with chart data
        excluded for a lean response. Returns a list of raw protocol dicts
        sorted descending by total24h volume.
        """
        url = (
            f"{self.BASE}/overview/dexs"
            "?excludeTotalDataChart=true"
            "&excludeTotalDataChartBreakdown=true"
        )
        async with httpx.AsyncClient() as client:
            data = await _fetch_json(client, url)

        if not isinstance(data, dict):
            logger.error("get_dex_overview: unexpected response type")
            return []

        protocols: list[dict] = data.get("protocols", [])
        if not isinstance(protocols, list):
            return []

        # Sort by 24h volume descending (None treated as 0)
        protocols.sort(key=lambda p: float(p.get("total24h") or 0), reverse=True)
        return protocols[:top_n]

    # ------------------------------------------------------------------
    async def get_chain_dex_volume(self) -> dict[str, float]:
        """DEX volume aggregated by chain over the last 24 hours.

        Fetches from https://api.llama.fi/overview/dexs (chart breakdown
        included so we can read the per-chain breakdown field), then sums
        each protocol's chain-level 24h volumes into a single dict.

        Returns:
            Mapping of chain_name → total_24h_volume_usd, sorted descending.
        """
        url = (
            f"{self.BASE}/overview/dexs"
            "?excludeTotalDataChart=true"
        )
        async with httpx.AsyncClient() as client:
            data = await _fetch_json(client, url)

        if not isinstance(data, dict):
            logger.error("get_chain_dex_volume: unexpected response type")
            return {}

        protocols: list[dict] = data.get("protocols", [])
        if not isinstance(protocols, list):
            return {}

        chain_volumes: dict[str, float] = {}
        for protocol in protocols:
            # breakdown24h is a dict of {chain: volume}
            breakdown = protocol.get("breakdown24h") or {}
            if not isinstance(breakdown, dict):
                continue
            for chain, vol in breakdown.items():
                try:
                    chain_volumes[chain] = chain_volumes.get(chain, 0.0) + float(vol or 0)
                except (TypeError, ValueError):
                    pass

        # If breakdown is absent, fall back to per-protocol chains list with
        # total24h evenly distributed (best-effort approximation).
        if not chain_volumes:
            for protocol in protocols:
                total_24h = float(protocol.get("total24h") or 0)
                chains: list[str] = protocol.get("chains") or []
                if chains and total_24h > 0:
                    share = total_24h / len(chains)
                    for chain in chains:
                        chain_volumes[chain] = chain_volumes.get(chain, 0.0) + share

        # Sort descending by volume
        return dict(
            sorted(chain_volumes.items(), key=lambda kv: kv[1], reverse=True)
        )

    # ------------------------------------------------------------------
    async def get_dashboard(self, top_n: int = 20) -> DeFiDashboard:
        """Full dashboard: protocols + chains + stables + yields in parallel."""
        protocols_task = asyncio.create_task(self.get_all_protocols())
        chains_task = asyncio.create_task(self.get_chain_tvl())
        stables_task = asyncio.create_task(self.get_stablecoins())
        yields_task = asyncio.create_task(self.get_yields())

        protocols, chains, stables, yields_ = await asyncio.gather(
            protocols_task, chains_task, stables_task, yields_task,
            return_exceptions=False,
        )

        # Sort protocols by TVL and take top_n
        protocols_sorted = sorted(
            [p for p in protocols if p.tvl is not None],
            key=lambda p: p.tvl or 0.0,
            reverse=True,
        )
        total_tvl = sum(p.tvl or 0.0 for p in protocols_sorted)

        return DeFiDashboard(
            total_defi_tvl=round(total_tvl, 2),
            top_protocols=protocols_sorted[:top_n],
            top_chains=chains[:top_n],
            stablecoins=stables[:top_n],
            top_yields=yields_[:top_n],
            generated_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Dim 110 — standalone DEX dashboard function
# ---------------------------------------------------------------------------

async def get_dex_dashboard(top_n: int = 15) -> DEXDashboard:
    """
    Build a DEX volume dashboard using DefiLlamaClient.

    Fetches the top DEX protocols by 24h volume and per-chain volume breakdown
    in parallel, then assembles a DEXDashboard.

    Args:
        top_n: Number of top protocols to include.

    Returns:
        DEXDashboard with total_dex_volume_24h, top_protocols, and
        volume_by_chain populated.
    """
    client = DefiLlamaClient()

    overview_task = asyncio.create_task(client.get_dex_overview(top_n=top_n))
    chain_vol_task = asyncio.create_task(client.get_chain_dex_volume())

    raw_protocols, volume_by_chain = await asyncio.gather(
        overview_task, chain_vol_task, return_exceptions=False
    )

    top_protocols: list[DEXProtocol] = []
    for p in raw_protocols:
        try:
            top_protocols.append(DEXProtocol(
                name=p.get("name", ""),
                chains=p.get("chains") or [],
                volume_24h=float(p["total24h"]) if p.get("total24h") is not None else None,
                volume_7d=float(p["total7d"]) if p.get("total7d") is not None else None,
                volume_all_time=float(p["totalAllTime"]) if p.get("totalAllTime") is not None else None,
                change_1d_pct=float(p["change_1d"]) if p.get("change_1d") is not None else None,
            ))
        except Exception as exc:
            logger.debug("get_dex_dashboard: protocol parse error: %s", exc)

    total_dex_volume_24h = sum(
        p.volume_24h for p in top_protocols if p.volume_24h is not None
    )

    logger.info(
        "get_dex_dashboard: top_n=%d total_24h=%.0f chains=%d",
        len(top_protocols), total_dex_volume_24h, len(volume_by_chain),
    )

    return DEXDashboard(
        total_dex_volume_24h=round(total_dex_volume_24h, 2),
        top_protocols=top_protocols,
        volume_by_chain=volume_by_chain,
        generated_at=datetime.now(timezone.utc),
    )
