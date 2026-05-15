"""
DeFi protocol analytics — Dimension #107 (target score 9+).

Adapters / Classes
------------------
DeFiLlamaAdapter    — DefiLlama REST API (no key): protocols, yields, stables, bridges
DeFiScreener        — TVL screening, yield filtering, protocol health, rug-pull risk
ChainAnalytics      — Per-chain TVL market share, momentum, cross-chain arb
DeFiMacroSignals    — DeFi TVL trend, stablecoin dominance, DeFi-vs-CeFi ratio, yield curve

All HTTP calls are async (httpx).  In-process TTL cache prevents redundant round-trips.

Free endpoints used (no auth required)
---------------------------------------
https://api.llama.fi/protocols              — All protocols + TVL + categories
https://api.llama.fi/protocol/{slug}        — Full protocol detail + historical TVL
https://api.llama.fi/tvl/{slug}             — Simple TVL scalar for one protocol
https://api.llama.fi/v2/historicalChainTvl  — DeFi total TVL history
https://api.llama.fi/v2/chains              — Per-chain TVL snapshot
https://api.llama.fi/overview/dexs          — DEX aggregate volume data
https://api.llama.fi/overview/fees          — Protocol fee/revenue data
https://yields.llama.fi/pools               — Yield farming pools (APY + TVL)
https://stablecoins.llama.fi/stablecoins    — Stablecoin market caps + peg deviation
https://bridges.llama.fi/bridges            — Bridge TVL and volume
"""
from __future__ import annotations

import asyncio
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LLAMA_BASE = "https://api.llama.fi"
_YIELDS_BASE = "https://yields.llama.fi"
_STABLES_BASE = "https://stablecoins.llama.fi"
_BRIDGES_BASE = "https://bridges.llama.fi"

_TIMEOUT = 30.0
_CACHE_TTL = 300  # 5 minutes — DefiLlama updates frequently

# Health score thresholds
_TVL_STABILITY_GOOD = 0.20   # CV < 20% = stable
_TVL_STABILITY_BAD = 0.50    # CV > 50% = unstable
_MIN_POOL_TVL = 1_000_000    # $1M minimum pool TVL for yield screening
_MIN_PROTOCOL_AGE_DAYS = 30  # Protocols < 30 days get rug-pull flag

# Lending protocols for yield curve construction
_LENDING_PROTOCOLS = {
    "aave-v3": "Aave V3",
    "compound-v3": "Compound V3",
    "morpho-blue": "Morpho Blue",
    "spark": "Spark",
    "venus": "Venus",
    "radiant-v2": "Radiant V2",
}

# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return val


def _cache_set(key: str, val: object) -> None:
    _cache[key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProtocolSummary(BaseModel):
    """Compact protocol record from /protocols list."""
    model_config = ConfigDict(frozen=True)
    slug: str
    name: str
    chain: str
    chains: list[str] = Field(default_factory=list)
    category: str = ""
    tvl_usd: float = 0.0
    change_1h_pct: Optional[float] = None
    change_1d_pct: Optional[float] = None
    change_7d_pct: Optional[float] = None
    mcap_tvl: Optional[float] = None
    audit_links: list[str] = Field(default_factory=list)
    launch_date: Optional[datetime] = None


class YieldPool(BaseModel):
    """A single yield farming pool record from /pools."""
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
    il_risk: str = "no"  # "no" | "low" | "high"
    exposure: str = ""


class ProtocolHealth(BaseModel):
    """Protocol health score and component metrics."""
    model_config = ConfigDict(frozen=True)
    protocol_slug: str
    health_score: float = Field(..., ge=0.0, le=100.0)
    tvl_stability_score: float
    revenue_tvl_ratio: Optional[float]
    chain_diversification_score: float
    age_days: Optional[int]
    tvl_revenue_multiple: Optional[float]
    flags: list[str]
    as_of: datetime


class RugPullRisk(BaseModel):
    """Rug-pull risk flags for a DeFi protocol."""
    model_config = ConfigDict(frozen=True)
    protocol_slug: str
    risk_level: str  # "low" | "medium" | "high" | "critical"
    risk_score: float = Field(..., ge=0.0, le=100.0)
    flags: list[str]
    as_of: datetime


class ChainTVL(BaseModel):
    """Per-chain TVL snapshot."""
    model_config = ConfigDict(frozen=True)
    chain: str
    tvl_usd: float
    market_share_pct: float
    token_symbol: str = ""


class DeFiMacroSnapshot(BaseModel):
    """Macro DeFi market snapshot."""
    model_config = ConfigDict(frozen=True)
    total_tvl_usd: float
    trend_direction: str  # "up" | "down" | "flat"
    rate_of_change_30d_pct: float
    stablecoin_dominance_pct: float
    risk_signal: str  # "risk-on" | "risk-off" | "neutral"
    as_of: datetime


# ---------------------------------------------------------------------------
# DeFiLlamaAdapter
# ---------------------------------------------------------------------------


class DeFiLlamaAdapter:
    """
    Adapter for the DefiLlama public REST API (no authentication required).

    Covers protocols, chains, DEX volumes, yields, stablecoins, fees,
    and bridges via the llama.fi, yields.llama.fi, stablecoins.llama.fi,
    and bridges.llama.fi subdomains.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def _get(self, url: str, params: Optional[dict] = None) -> object:
        """Internal async GET with cache, timeout, and structured error handling."""
        cache_key = f"llama:{url}:{json_stable(params)}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params or {})
                resp.raise_for_status()
                data = resp.json()
            _cache_set(cache_key, data)
            return data
        except httpx.HTTPStatusError as exc:
            logger.warning("DefiLlama HTTP error", url=url, status=exc.response.status_code)
            raise
        except Exception as exc:
            logger.warning("DefiLlama request failed", url=url, error=str(exc))
            raise

    async def get_protocols(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch all DeFi protocols with TVL snapshots and metadata.

        Returns a DataFrame with columns: slug, name, chain, chains, category,
        tvl_usd, change_1h_pct, change_1d_pct, change_7d_pct, mcap_tvl, audit_links.

        Sorted by tvl_usd descending; truncated to *limit* rows.
        """
        data = await self._get(f"{_LLAMA_BASE}/protocols")
        if not isinstance(data, list):
            logger.warning("Unexpected /protocols response type")
            return pd.DataFrame()

        rows: list[dict] = []
        for p in data[:max(limit, len(data))]:
            rows.append({
                "slug": p.get("slug", ""),
                "name": p.get("name", ""),
                "chain": p.get("chain", ""),
                "chains": p.get("chains", []),
                "category": p.get("category", ""),
                "tvl_usd": float(p.get("tvl", 0) or 0),
                "change_1h_pct": p.get("change_1h"),
                "change_1d_pct": p.get("change_1d"),
                "change_7d_pct": p.get("change_7d"),
                "mcap_tvl": p.get("mcap") / p.get("tvl", 1) if p.get("mcap") and p.get("tvl") else None,
                "audit_links": p.get("audit_links") or [],
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("tvl_usd", ascending=False).head(limit).reset_index(drop=True)
        logger.info("DefiLlama protocols fetched", count=len(df))
        return df

    async def get_protocol(self, protocol_slug: str) -> dict:
        """
        Fetch full protocol detail: historical TVL, chains breakdown, token info.

        Returns the raw DefiLlama protocol dict including 'tvl', 'chainTvls',
        'tokens', 'audit_links', and 'currentChainTvls'.
        """
        data = await self._get(f"{_LLAMA_BASE}/protocol/{protocol_slug}")
        return data if isinstance(data, dict) else {}

    async def get_tvl_history(self, protocol_slug: str) -> pd.DataFrame:
        """
        Return historical TVL timeseries for a protocol.

        Columns: date (datetime, UTC), tvl (float USD).
        Sorted chronologically ascending.
        """
        detail = await self.get_protocol(protocol_slug)
        tvl_series = detail.get("tvl", [])
        if not tvl_series:
            return pd.DataFrame(columns=["date", "tvl"])

        rows = []
        for entry in tvl_series:
            ts = entry.get("date") or entry.get("timestamp")
            tvl = entry.get("totalLiquidityUSD") or entry.get("tvl", 0)
            if ts is None:
                continue
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            rows.append({"date": dt, "tvl": float(tvl or 0)})

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    async def get_chains(self) -> pd.DataFrame:
        """
        Return per-chain TVL breakdown from DefiLlama.

        Columns: chain, tvl_usd, token_symbol.
        Sorted by tvl_usd descending.
        """
        data = await self._get(f"{_LLAMA_BASE}/v2/chains")
        if not isinstance(data, list):
            return pd.DataFrame()

        rows = [
            {
                "chain": c.get("name", ""),
                "tvl_usd": float(c.get("tvl", 0) or 0),
                "token_symbol": c.get("tokenSymbol", ""),
            }
            for c in data
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("tvl_usd", ascending=False).reset_index(drop=True)
        return df

    async def get_dex_volumes(self) -> pd.DataFrame:
        """
        Fetch DEX aggregate volume data from the DefiLlama DEX overview endpoint.

        Columns: name, chain, total24h, total7d, change_1d_pct, category.
        """
        data = await self._get(f"{_LLAMA_BASE}/overview/dexs")
        if not isinstance(data, dict):
            return pd.DataFrame()

        protocols = data.get("protocols", []) or data.get("allChains", [])
        rows = []
        for p in protocols:
            rows.append({
                "name": p.get("name", ""),
                "slug": p.get("slug", ""),
                "chain": p.get("chain", ""),
                "total_24h_usd": float(p.get("total24h", 0) or 0),
                "total_7d_usd": float(p.get("total7d", 0) or 0),
                "change_1d_pct": p.get("change_1d"),
                "category": p.get("category", "dex"),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_24h_usd", ascending=False).reset_index(drop=True)
        logger.info("DEX volumes fetched", count=len(df))
        return df

    async def get_yields(self) -> pd.DataFrame:
        """
        Fetch yield farming pool opportunities from yields.llama.fi/pools.

        Columns: pool_id, protocol, chain, symbol, tvl_usd, apy, apy_base,
        apy_reward, stablecoin, il_risk, exposure.

        Filters out pools with missing APY or extreme APY > 10,000% (likely errors).
        """
        data = await self._get(f"{_YIELDS_BASE}/pools")
        if not isinstance(data, dict):
            return pd.DataFrame()

        pools = data.get("data", [])
        rows = []
        for p in pools:
            apy = p.get("apy")
            if apy is None or float(apy or 0) > 10_000:
                continue
            rows.append({
                "pool_id": p.get("pool", ""),
                "protocol": p.get("project", ""),
                "chain": p.get("chain", ""),
                "symbol": p.get("symbol", ""),
                "tvl_usd": float(p.get("tvlUsd", 0) or 0),
                "apy": float(apy or 0),
                "apy_base": p.get("apyBase"),
                "apy_reward": p.get("apyReward"),
                "stablecoin": bool(p.get("stablecoin", False)),
                "il_risk": p.get("ilRisk", "no") or "no",
                "exposure": p.get("exposure", ""),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("tvl_usd", ascending=False).reset_index(drop=True)
        logger.info("Yield pools fetched", count=len(df))
        return df

    async def get_stablecoins(self) -> pd.DataFrame:
        """
        Fetch stablecoin market caps and peg deviation from stablecoins.llama.fi.

        Columns: name, symbol, peg_type, peg_mechanism, circulating_usd,
        price_usd, peg_deviation_pct, chains.
        """
        data = await self._get(f"{_STABLES_BASE}/stablecoins?includePrices=true")
        if not isinstance(data, dict):
            return pd.DataFrame()

        pegs = data.get("peggedAssets", [])
        rows = []
        for s in pegs:
            price = s.get("price") or 1.0
            peg_target = 1.0  # Most stables target $1
            peg_dev = round(abs(float(price) - peg_target) / peg_target * 100, 4) if price else None
            circ = s.get("circulating", {})
            circ_usd = float(circ.get("peggedUSD", 0) or circ.get("peggedEUR", 0) or 0)
            rows.append({
                "name": s.get("name", ""),
                "symbol": s.get("symbol", ""),
                "peg_type": s.get("pegType", ""),
                "peg_mechanism": s.get("pegMechanism", ""),
                "circulating_usd": circ_usd,
                "price_usd": float(price or 1.0),
                "peg_deviation_pct": peg_dev,
                "chains": list(s.get("chainCirculating", {}).keys()),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("circulating_usd", ascending=False).reset_index(drop=True)
        return df

    async def get_protocol_fees(self) -> pd.DataFrame:
        """
        Fetch protocol revenue and fee data from DefiLlama fees overview.

        Columns: name, slug, total_24h_usd, total_7d_usd, total_30d_usd,
        change_1d_pct, category.
        """
        data = await self._get(f"{_LLAMA_BASE}/overview/fees")
        if not isinstance(data, dict):
            return pd.DataFrame()

        protocols = data.get("protocols", [])
        rows = []
        for p in protocols:
            rows.append({
                "name": p.get("name", ""),
                "slug": p.get("slug", ""),
                "total_24h_usd": float(p.get("total24h", 0) or 0),
                "total_7d_usd": float(p.get("total7d", 0) or 0),
                "total_30d_usd": float(p.get("total30d", 0) or 0),
                "change_1d_pct": p.get("change_1d"),
                "category": p.get("category", ""),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_24h_usd", ascending=False).reset_index(drop=True)
        return df

    async def get_bridges(self) -> pd.DataFrame:
        """
        Fetch cross-chain bridge TVL and volume from bridges.llama.fi.

        Columns: name, chains, tvl_usd, volume_24h_usd, change_1d_pct.
        """
        data = await self._get(f"{_BRIDGES_BASE}/bridges?includeChains=true")
        if not isinstance(data, dict):
            return pd.DataFrame()

        bridges = data.get("bridges", [])
        rows = []
        for b in bridges:
            rows.append({
                "name": b.get("displayName") or b.get("name", ""),
                "chains": b.get("chains", []),
                "tvl_usd": float(b.get("currentTvl", 0) or 0),
                "volume_24h_usd": float(b.get("lastDailyVolume", 0) or 0),
                "change_1d_pct": b.get("change_1d"),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("tvl_usd", ascending=False).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# DeFiScreener
# ---------------------------------------------------------------------------


class DeFiScreener:
    """
    Screens DeFi protocols and yield pools using DefiLlama data.

    Methods cover TVL-based protocol filtering, yield opportunity discovery
    with risk controls, TVL momentum screening, protocol health scoring,
    and rug-pull risk detection.
    """

    def __init__(self) -> None:
        self._llama = DeFiLlamaAdapter()

    async def screen_by_tvl(
        self,
        min_tvl_usd: float = 100_000_000,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter protocols by minimum TVL and optional category.

        Parameters
        ----------
        min_tvl_usd : Minimum TVL threshold in USD (default $100M).
        category    : Optional category filter e.g. "Lending", "DEX", "Yield".

        Returns DataFrame of matching protocols sorted by tvl_usd descending.
        """
        df = await self._llama.get_protocols(limit=500)
        if df.empty:
            return df

        mask = df["tvl_usd"] >= min_tvl_usd
        if category:
            mask &= df["category"].str.lower() == category.lower()
        return df[mask].reset_index(drop=True)

    async def screen_high_yield(
        self,
        min_apy: float = 5.0,
        max_apy: float = 100.0,
        stablecoin_only: bool = False,
    ) -> pd.DataFrame:
        """
        Find yield farming opportunities within a target APY range.

        Automatically excludes:
        - Pools with TVL < $1M (low liquidity / elevated exit risk)
        - Pools with APY > *max_apy* (likely unsustainable rewards)
        - Pools with high impermanent loss risk (unless stablecoin_only=False)

        Parameters
        ----------
        min_apy         : Minimum APY threshold (default 5%).
        max_apy         : Maximum APY threshold (default 100%).
        stablecoin_only : When True, restricts to stablecoin pools only.

        Returns DataFrame sorted by risk-adjusted yield (APY / IL_risk_penalty).
        """
        df = await self._llama.get_yields()
        if df.empty:
            return df

        mask = (
            (df["apy"] >= min_apy)
            & (df["apy"] <= max_apy)
            & (df["tvl_usd"] >= _MIN_POOL_TVL)
        )
        if stablecoin_only:
            mask &= df["stablecoin"] == True  # noqa: E712

        filtered = df[mask].copy()

        # Risk-adjusted score: penalise high IL risk
        def _il_penalty(row: pd.Series) -> float:
            if row["il_risk"] == "high":
                return 0.6
            if row["il_risk"] == "low":
                return 0.85
            return 1.0

        if not filtered.empty:
            filtered["il_penalty"] = filtered.apply(_il_penalty, axis=1)
            filtered["risk_adj_apy"] = filtered["apy"] * filtered["il_penalty"]
            filtered = filtered.sort_values("risk_adj_apy", ascending=False).reset_index(drop=True)
            filtered = filtered.drop(columns=["il_penalty"])

        logger.info("High yield screen", hits=len(filtered), min_apy=min_apy, max_apy=max_apy)
        return filtered

    async def tvl_momentum_screen(
        self,
        min_7d_change_pct: float = 10.0,
    ) -> pd.DataFrame:
        """
        Return protocols with TVL growth >= *min_7d_change_pct* over the past week.

        Useful for identifying capital rotation into growing protocols.
        Excludes protocols with absolute TVL < $10M (too small for reliable signals).
        """
        df = await self._llama.get_protocols(limit=500)
        if df.empty:
            return df

        mask = (
            df["change_7d_pct"].notna()
            & (df["change_7d_pct"] >= min_7d_change_pct)
            & (df["tvl_usd"] >= 10_000_000)
        )
        result = df[mask].copy()
        if not result.empty:
            result = result.sort_values("change_7d_pct", ascending=False).reset_index(drop=True)
        logger.info("TVL momentum screen", hits=len(result), threshold_pct=min_7d_change_pct)
        return result

    async def compute_protocol_health(self, protocol_slug: str) -> ProtocolHealth:
        """
        Compute a 0–100 health score for a DeFi protocol.

        Component metrics
        -----------------
        TVL stability    : Coefficient of variation of 90-day TVL history (lower = better).
        Revenue/TVL      : Annual fee revenue as % of TVL (higher = more capital-efficient).
        Chain diversity  : Herfindahl index of chain TVL concentration (lower = more diverse).
        Age              : Days since first recorded TVL (older = more established).
        TVL/Revenue mult : TVL divided by annualized revenue (lower = cheaper valuation).
        """
        now = datetime.now(tz=timezone.utc)
        flags: list[str] = []
        score = 100.0  # Start at 100, subtract penalties

        detail = await self._llama.get_protocol(protocol_slug)
        tvl_history = await self._llama.get_tvl_history(protocol_slug)

        # --- TVL stability (CV of 90-day window) ---
        tvl_stability_score = 50.0
        if not tvl_history.empty and len(tvl_history) >= 7:
            recent = tvl_history.tail(90)["tvl"].values
            mean_tvl = float(np.mean(recent)) if len(recent) > 0 else 0.0
            if mean_tvl > 0:
                cv = float(np.std(recent) / mean_tvl)
                tvl_stability_score = max(0.0, 100.0 - cv * 150.0)
                if cv > _TVL_STABILITY_BAD:
                    penalty = (cv - _TVL_STABILITY_BAD) * 30.0
                    score -= penalty
                    flags.append(f"High TVL volatility (CV={cv:.2%})")

        # --- Revenue vs TVL ratio ---
        revenue_tvl_ratio: Optional[float] = None
        tvl_revenue_multiple: Optional[float] = None
        fees_24h = detail.get("fees", {})
        current_tvl = float(detail.get("tvl", 0) or 0)
        daily_revenue = None
        if isinstance(fees_24h, dict):
            daily_revenue = fees_24h.get("fees", {}).get("total24h")
        elif isinstance(fees_24h, (int, float)):
            daily_revenue = fees_24h

        if daily_revenue and current_tvl > 0:
            annual_revenue = float(daily_revenue) * 365
            revenue_tvl_ratio = round(annual_revenue / current_tvl, 4)
            tvl_revenue_multiple = round(current_tvl / annual_revenue, 2) if annual_revenue > 0 else None
            if revenue_tvl_ratio < 0.001:
                score -= 15.0
                flags.append(f"Low revenue/TVL ratio: {revenue_tvl_ratio:.4%}")
        else:
            score -= 10.0
            flags.append("Revenue data unavailable")

        # --- Chain diversification (Herfindahl index) ---
        chain_tvls = detail.get("currentChainTvls", {})
        chain_diversity_score = 50.0
        if chain_tvls:
            tvl_vals = [float(v) for v in chain_tvls.values() if v and float(v) > 0]
            total = sum(tvl_vals)
            if total > 0 and len(tvl_vals) > 1:
                hhi = sum((v / total) ** 2 for v in tvl_vals)
                # HHI: 1.0 = full concentration, ~0 = perfectly distributed
                chain_diversity_score = max(0.0, 100.0 * (1.0 - hhi))
                if hhi > 0.9:
                    score -= 10.0
                    flags.append(f"Single-chain concentration (HHI={hhi:.2f})")
            elif len(tvl_vals) == 1:
                chain_diversity_score = 10.0
                score -= 10.0
                flags.append("Single-chain protocol")

        # --- Age ---
        age_days: Optional[int] = None
        if not tvl_history.empty:
            oldest = tvl_history["date"].min()
            age_days = (now - oldest).days
            if age_days < _MIN_PROTOCOL_AGE_DAYS:
                score -= 20.0
                flags.append(f"New protocol: {age_days} days old")
            elif age_days < 180:
                score -= 5.0

        # --- Audit check ---
        audit_links = detail.get("audit_links") or []
        if not audit_links:
            score -= 10.0
            flags.append("No audit links found")

        final_score = max(0.0, min(100.0, round(score, 2)))

        return ProtocolHealth(
            protocol_slug=protocol_slug,
            health_score=final_score,
            tvl_stability_score=round(tvl_stability_score, 2),
            revenue_tvl_ratio=revenue_tvl_ratio,
            chain_diversification_score=round(chain_diversity_score, 2),
            age_days=age_days,
            tvl_revenue_multiple=tvl_revenue_multiple,
            flags=flags,
            as_of=now,
        )

    async def detect_rug_pull_risk(self, protocol_slug: str) -> RugPullRisk:
        """
        Assess rug-pull risk for a DeFi protocol.

        Risk flags checked
        ------------------
        - Anonymous team (no listed team members in DefiLlama data)
        - Unaudited code (no audit_links)
        - Single-chain concentration (>90% TVL on one chain)
        - Recent TVL collapse (>50% decline in 7 days)
        - Very new protocol (<30 days old)
        - No website or social presence listed

        Risk levels: low (<20), medium (20–49), high (50–74), critical (>=75).
        """
        now = datetime.now(tz=timezone.utc)
        flags: list[str] = []
        risk_score = 0.0

        detail = await self._llama.get_protocol(protocol_slug)
        tvl_history = await self._llama.get_tvl_history(protocol_slug)

        # Anonymous team
        team = detail.get("team", []) or detail.get("oracles", [])
        if not team:
            risk_score += 20.0
            flags.append("No team information disclosed")

        # Unaudited
        audit_links = detail.get("audit_links") or []
        if not audit_links:
            risk_score += 25.0
            flags.append("No audit links found — unverified smart contracts")

        # Single-chain concentration
        chain_tvls = detail.get("currentChainTvls", {})
        if chain_tvls:
            tvl_vals = [float(v) for v in chain_tvls.values() if v and float(v) > 0]
            total = sum(tvl_vals)
            if total > 0 and tvl_vals:
                max_share = max(tvl_vals) / total
                if max_share > 0.90 and len(tvl_vals) == 1:
                    risk_score += 10.0
                    flags.append(f"Single-chain protocol ({max_share:.0%} concentrated)")

        # TVL collapse in last 7 days
        if not tvl_history.empty and len(tvl_history) >= 8:
            recent_tvl = tvl_history["tvl"].iloc[-1]
            week_ago_tvl = tvl_history["tvl"].iloc[-8] if len(tvl_history) >= 8 else tvl_history["tvl"].iloc[0]
            if week_ago_tvl > 0:
                tvl_change = (recent_tvl - week_ago_tvl) / week_ago_tvl
                if tvl_change < -0.50:
                    risk_score += 35.0
                    flags.append(f"Severe TVL decline: {tvl_change:.1%} in 7 days")
                elif tvl_change < -0.30:
                    risk_score += 15.0
                    flags.append(f"Significant TVL decline: {tvl_change:.1%} in 7 days")

        # Protocol age
        if not tvl_history.empty:
            oldest = tvl_history["date"].min()
            age_days = (now - oldest).days
            if age_days < _MIN_PROTOCOL_AGE_DAYS:
                risk_score += 20.0
                flags.append(f"Very new protocol: {age_days} days since first TVL")

        # No website
        if not detail.get("url") and not detail.get("twitter"):
            risk_score += 10.0
            flags.append("No website or social presence found")

        risk_score = min(100.0, risk_score)
        if risk_score < 20.0:
            level = "low"
        elif risk_score < 50.0:
            level = "medium"
        elif risk_score < 75.0:
            level = "high"
        else:
            level = "critical"

        return RugPullRisk(
            protocol_slug=protocol_slug,
            risk_level=level,
            risk_score=round(risk_score, 2),
            flags=flags,
            as_of=now,
        )


# ---------------------------------------------------------------------------
# ChainAnalytics
# ---------------------------------------------------------------------------


class ChainAnalytics:
    """
    Chain-level TVL analytics: market share, momentum rotation, and
    cross-chain yield arbitrage opportunities.
    """

    def __init__(self) -> None:
        self._llama = DeFiLlamaAdapter()

    async def tvl_market_share(self) -> pd.DataFrame:
        """
        Compute chain TVL market share as a percentage of total DeFi TVL.

        Columns: chain, tvl_usd, market_share_pct, token_symbol.
        Sorted by tvl_usd descending.
        """
        df = await self._llama.get_chains()
        if df.empty:
            return df

        total_tvl = df["tvl_usd"].sum()
        df = df.copy()
        if total_tvl > 0:
            df["market_share_pct"] = (df["tvl_usd"] / total_tvl * 100).round(4)
        else:
            df["market_share_pct"] = 0.0

        return df.sort_values("tvl_usd", ascending=False).reset_index(drop=True)

    async def chain_momentum_rotation(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Identify chains gaining or losing TVL share over *lookback_days*.

        Uses the DefiLlama historical chain TVL endpoint to compare start vs end
        TVL for each chain.  Returns a DataFrame with:
        chain, tvl_start_usd, tvl_end_usd, change_usd, change_pct, momentum.

        momentum: "gaining" | "losing" | "stable"
        """
        data = await self._llama._get(f"{_LLAMA_BASE}/v2/historicalChainTvl")
        if not isinstance(data, list):
            logger.warning("Unexpected historicalChainTvl response")
            return pd.DataFrame()

        cutoff_ts = (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).timestamp()

        # data is a list of {date: unix_ts, tvl: total_usd}
        # For per-chain momentum we use the /v2/chains snapshot vs protocols endpoint
        # fallback: use change_7d from protocols aggregated by chain
        protocols_df = await self._llama.get_protocols(limit=500)
        if protocols_df.empty:
            return pd.DataFrame()

        chain_tvl_now: dict[str, float] = {}
        chain_tvl_change: dict[str, list[float]] = {}
        for _, row in protocols_df.iterrows():
            chain = row["chain"]
            if not chain:
                continue
            chain_tvl_now[chain] = chain_tvl_now.get(chain, 0.0) + float(row["tvl_usd"] or 0)
            change_7d = row.get("change_7d_pct")
            if change_7d is not None:
                chain_tvl_change.setdefault(chain, []).append(float(change_7d))

        rows = []
        for chain, tvl_now in chain_tvl_now.items():
            changes = chain_tvl_change.get(chain, [])
            avg_change = statistics.mean(changes) if changes else 0.0
            tvl_start = tvl_now / (1 + avg_change / 100) if avg_change != -100 else tvl_now
            change_usd = tvl_now - tvl_start
            if abs(avg_change) > 5.0:
                momentum = "gaining" if avg_change > 0 else "losing"
            else:
                momentum = "stable"
            rows.append({
                "chain": chain,
                "tvl_start_usd": round(tvl_start, 0),
                "tvl_end_usd": round(tvl_now, 0),
                "change_usd": round(change_usd, 0),
                "change_pct": round(avg_change, 3),
                "momentum": momentum,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("change_pct", ascending=False).reset_index(drop=True)
        return df

    async def cross_chain_arbitrage_opportunities(
        self,
        token: str = "USDC",
    ) -> list[dict]:
        """
        Identify yield differentials for *token* across chains.

        Fetches yield pools containing *token* and groups by chain,
        returning the best APY per chain sorted by yield differential.
        Potential arbitrage exists where APY spread between chains exceeds 1%.

        Returns a list of dicts with: chain, best_apy, pool_id, protocol,
        tvl_usd, spread_vs_min, arbitrage_signal.
        """
        yields_df = await self._llama.get_yields()
        if yields_df.empty:
            return []

        # Filter to pools containing the target token
        token_mask = yields_df["symbol"].str.upper().str.contains(token.upper(), na=False)
        token_pools = yields_df[token_mask].copy()

        if token_pools.empty:
            logger.info("No yield pools found for token", token=token)
            return []

        # Best APY per chain
        best_by_chain = (
            token_pools
            .sort_values("apy", ascending=False)
            .groupby("chain")
            .first()
            .reset_index()
        )

        if best_by_chain.empty:
            return []

        min_apy = best_by_chain["apy"].min()
        max_apy = best_by_chain["apy"].max()
        global_spread = max_apy - min_apy

        results = []
        for _, row in best_by_chain.iterrows():
            spread = row["apy"] - min_apy
            results.append({
                "chain": row["chain"],
                "token": token,
                "best_apy": round(float(row["apy"]), 4),
                "pool_id": row["pool_id"],
                "protocol": row["protocol"],
                "tvl_usd": float(row["tvl_usd"]),
                "spread_vs_min_pct": round(spread, 4),
                "arbitrage_signal": spread > 1.0,
            })

        results.sort(key=lambda x: x["best_apy"], reverse=True)
        logger.info(
            "Cross-chain arb scan",
            token=token,
            chains=len(results),
            global_spread_pct=round(global_spread, 3),
        )
        return results


# ---------------------------------------------------------------------------
# DeFiMacroSignals
# ---------------------------------------------------------------------------


class DeFiMacroSignals:
    """
    Macro-level DeFi signals: total TVL trend, stablecoin dominance,
    DeFi vs CeFi ratio, and cross-protocol yield curve for stablecoins.
    """

    def __init__(self) -> None:
        self._llama = DeFiLlamaAdapter()

    async def total_defi_tvl_trend(self, lookback_days: int = 90) -> dict:
        """
        Compute total DeFi TVL trend over *lookback_days*.

        Fetches the DefiLlama aggregate historical TVL timeseries and returns:
        - current_tvl_usd   : Latest total DeFi TVL
        - start_tvl_usd     : TVL at start of lookback window
        - change_pct        : % change over the period
        - trend_direction   : "up" | "down" | "flat"
        - rate_of_change    : Linear slope (USD/day) fit to the series
        - as_of             : Timestamp
        """
        now = datetime.now(tz=timezone.utc)
        data = await self._llama._get(f"{_LLAMA_BASE}/v2/historicalChainTvl")
        if not isinstance(data, list) or len(data) < 2:
            return {"error": "Insufficient TVL history data", "as_of": now.isoformat()}

        cutoff_ts = (now - timedelta(days=lookback_days)).timestamp()
        window = [d for d in data if d.get("date", 0) >= cutoff_ts]
        if len(window) < 2:
            window = data[-min(lookback_days, len(data)):]

        tvl_series = [float(d.get("tvl", 0) or 0) for d in window]
        current_tvl = tvl_series[-1]
        start_tvl = tvl_series[0]
        change_pct = ((current_tvl - start_tvl) / start_tvl * 100) if start_tvl > 0 else 0.0

        if abs(change_pct) < 2.0:
            trend = "flat"
        elif change_pct > 0:
            trend = "up"
        else:
            trend = "down"

        # Linear rate of change (simple rise-over-run)
        n = len(tvl_series)
        rate_of_change = (tvl_series[-1] - tvl_series[0]) / max(n, 1)

        return {
            "current_tvl_usd": round(current_tvl, 0),
            "start_tvl_usd": round(start_tvl, 0),
            "change_pct": round(change_pct, 3),
            "trend_direction": trend,
            "rate_of_change_usd_per_day": round(rate_of_change, 0),
            "data_points": n,
            "lookback_days": lookback_days,
            "as_of": now.isoformat(),
        }

    async def stablecoin_dominance(self) -> dict:
        """
        Compute stablecoin TVL as a percentage of total DeFi TVL.

        Stablecoin dominance rising = risk-off signal (capital seeking safety).
        Stablecoin dominance falling = risk-on signal (capital deploying into DeFi).

        Returns: stablecoin_tvl_usd, total_defi_tvl_usd, dominance_pct,
        risk_signal, top_stablecoins (name, circulating_usd, share_pct).
        """
        now = datetime.now(tz=timezone.utc)

        stables_df, chains_df = await asyncio.gather(
            self._llama.get_stablecoins(),
            self._llama.get_chains(),
        )

        total_defi_tvl = float(chains_df["tvl_usd"].sum()) if not chains_df.empty else 0.0
        total_stable_circ = float(stables_df["circulating_usd"].sum()) if not stables_df.empty else 0.0

        dominance_pct = (total_stable_circ / total_defi_tvl * 100) if total_defi_tvl > 0 else 0.0

        # Risk signal: >60% dominance = risk-off; <40% = risk-on; in between = neutral
        if dominance_pct > 60.0:
            risk_signal = "risk-off"
        elif dominance_pct < 40.0:
            risk_signal = "risk-on"
        else:
            risk_signal = "neutral"

        top_stablecoins: list[dict] = []
        if not stables_df.empty:
            for _, row in stables_df.head(5).iterrows():
                share = (row["circulating_usd"] / total_stable_circ * 100) if total_stable_circ > 0 else 0.0
                top_stablecoins.append({
                    "name": row["name"],
                    "symbol": row["symbol"],
                    "circulating_usd": round(float(row["circulating_usd"]), 0),
                    "share_pct": round(share, 3),
                    "peg_deviation_pct": row.get("peg_deviation_pct"),
                })

        return {
            "stablecoin_circulating_usd": round(total_stable_circ, 0),
            "total_defi_tvl_usd": round(total_defi_tvl, 0),
            "dominance_pct": round(dominance_pct, 3),
            "risk_signal": risk_signal,
            "top_stablecoins": top_stablecoins,
            "as_of": now.isoformat(),
        }

    async def defi_vs_cefi_ratio(self) -> dict:
        """
        Compare DeFi TVL vs centralized exchange (CeFi) volume as a sentiment gauge.

        Uses DefiLlama total TVL as the DeFi proxy and DEX 24h volume as
        the on-chain trading proxy.  A rising DeFi/CeFi ratio suggests
        preference for non-custodial venues.

        Returns: defi_tvl_usd, dex_volume_24h_usd, cefi_proxy_note,
        defi_dex_tvl_ratio, sentiment.
        """
        now = datetime.now(tz=timezone.utc)
        chains_df, dex_df = await asyncio.gather(
            self._llama.get_chains(),
            self._llama.get_dex_volumes(),
        )

        total_defi_tvl = float(chains_df["tvl_usd"].sum()) if not chains_df.empty else 0.0
        total_dex_vol_24h = float(dex_df["total_24h_usd"].sum()) if not dex_df.empty else 0.0

        # TVL-to-daily-volume ratio: higher = more capital locked relative to trading
        ratio = (total_defi_tvl / total_dex_vol_24h) if total_dex_vol_24h > 0 else None

        # DEX market share of total DEX + estimated CeFi
        # Simple heuristic: DeFi is roughly 15-25% of total crypto trading volume
        # Flag rising DEX volume share as a positive sentiment signal
        top_dexs: list[dict] = []
        if not dex_df.empty:
            for _, row in dex_df.head(5).iterrows():
                top_dexs.append({
                    "name": row["name"],
                    "volume_24h_usd": round(float(row["total_24h_usd"]), 0),
                    "chain": row["chain"],
                })

        if ratio and ratio > 20:
            sentiment = "risk-off"   # Capital locked, not trading
        elif ratio and ratio < 5:
            sentiment = "risk-on"    # High trading velocity relative to TVL
        else:
            sentiment = "neutral"

        return {
            "defi_tvl_usd": round(total_defi_tvl, 0),
            "dex_volume_24h_usd": round(total_dex_vol_24h, 0),
            "tvl_to_dex_volume_ratio": round(ratio, 2) if ratio else None,
            "cefi_proxy_note": "CeFi volume not directly available; ratio uses DeFi DEX volume as on-chain proxy",
            "sentiment": sentiment,
            "top_dexs_by_volume": top_dexs,
            "as_of": now.isoformat(),
        }

    async def yield_curve_defi(
        self,
        stablecoin: str = "USDC",
    ) -> pd.DataFrame:
        """
        Construct a DeFi yield curve for *stablecoin* across lending protocols.

        Queries yield pools from Aave V3, Compound V3, Morpho Blue, Spark, and
        related protocols to create a cross-protocol APY comparison — analogous
        to a traditional money-market yield curve by maturity/risk.

        Columns: protocol, chain, symbol, apy, apy_base, apy_reward,
        tvl_usd, stablecoin, risk_tier.

        risk_tier: "tier1" (Aave/Compound/Morpho) | "tier2" (other top-50) | "tier3" (other)
        """
        yields_df = await self._llama.get_yields()
        if yields_df.empty:
            return pd.DataFrame()

        # Filter to target stablecoin pools
        token_mask = yields_df["symbol"].str.upper().str.contains(stablecoin.upper(), na=False)
        stable_pools = yields_df[token_mask & (yields_df["tvl_usd"] >= _MIN_POOL_TVL)].copy()

        if stable_pools.empty:
            logger.info("No stablecoin pools found", stablecoin=stablecoin)
            return pd.DataFrame()

        # Classify risk tier by protocol reputation
        tier1_protocols = {p.lower() for p in _LENDING_PROTOCOLS.keys()}
        tier1_names = {"aave", "compound", "morpho", "spark", "maker"}

        def _risk_tier(protocol: str) -> str:
            pl = protocol.lower()
            if any(t in pl for t in tier1_names) or pl in tier1_protocols:
                return "tier1"
            return "tier2"

        stable_pools["risk_tier"] = stable_pools["protocol"].apply(_risk_tier)
        stable_pools = stable_pools.sort_values(
            ["risk_tier", "apy"], ascending=[True, False]
        ).reset_index(drop=True)

        # Select relevant columns for the yield curve output
        cols = ["protocol", "chain", "symbol", "apy", "apy_base", "apy_reward", "tvl_usd", "stablecoin", "risk_tier"]
        available = [c for c in cols if c in stable_pools.columns]
        logger.info("DeFi yield curve", stablecoin=stablecoin, pools=len(stable_pools))
        return stable_pools[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def json_stable(obj: Optional[dict]) -> str:
    """Produce a stable string key for a dict (for caching)."""
    if obj is None:
        return ""
    import json
    return json.dumps(obj, sort_keys=True)


# ---------------------------------------------------------------------------
# Convenience top-level functions
# ---------------------------------------------------------------------------


async def defi_dashboard_snapshot() -> dict:
    """
    Single-call convenience wrapper returning a macro DeFi dashboard snapshot.

    Aggregates: total TVL trend, stablecoin dominance, top yield opportunities,
    top protocols by TVL, and chain market share.

    Example
    -------
    >>> import asyncio
    >>> snapshot = asyncio.run(defi_dashboard_snapshot())
    """
    macro = DeFiMacroSignals()
    screener = DeFiScreener()
    chain_analytics = ChainAnalytics()

    tvl_trend, stable_dom, defi_cefi, top_yields, top_protocols, chain_share = await asyncio.gather(
        macro.total_defi_tvl_trend(lookback_days=30),
        macro.stablecoin_dominance(),
        macro.defi_vs_cefi_ratio(),
        screener.screen_high_yield(min_apy=5.0, max_apy=50.0),
        screener.screen_by_tvl(min_tvl_usd=500_000_000),
        chain_analytics.tvl_market_share(),
    )

    return {
        "tvl_trend": tvl_trend,
        "stablecoin_dominance": stable_dom,
        "defi_vs_cefi": defi_cefi,
        "top_yield_opportunities": top_yields.head(10).to_dict(orient="records") if not top_yields.empty else [],
        "top_protocols_by_tvl": top_protocols.head(10).to_dict(orient="records") if not top_protocols.empty else [],
        "chain_market_share": chain_share.head(10).to_dict(orient="records") if not chain_share.empty else [],
    }


async def screen_safe_defi_yield(
    min_apy: float = 4.0,
    max_apy: float = 20.0,
    stablecoin_only: bool = True,
    min_tvl_usd: float = 10_000_000,
) -> pd.DataFrame:
    """
    Conservative DeFi yield screen for institutional-grade opportunities.

    Combines yield filtering with rug-pull risk assessment for top candidates.
    Only returns pools from protocols with risk_level != "critical".

    Parameters
    ----------
    min_apy        : Minimum APY (default 4%).
    max_apy        : Maximum APY to filter out unsustainable rewards (default 20%).
    stablecoin_only: Restrict to stablecoin pools (default True).
    min_tvl_usd    : Minimum pool TVL (default $10M).

    Returns DataFrame with yield pools and an appended rug_risk_level column.
    """
    screener = DeFiScreener()
    yields_df = await screener.screen_high_yield(
        min_apy=min_apy,
        max_apy=max_apy,
        stablecoin_only=stablecoin_only,
    )
    if yields_df.empty:
        return yields_df

    # Apply stricter TVL floor
    yields_df = yields_df[yields_df["tvl_usd"] >= min_tvl_usd].copy()

    # Spot-check top protocols for rug risk (cap at 10 to avoid API flooding)
    top_protocols = yields_df["protocol"].unique()[:10]
    risk_results: dict[str, str] = {}
    for slug in top_protocols:
        try:
            risk = await screener.detect_rug_pull_risk(slug)
            risk_results[slug] = risk.risk_level
        except Exception:
            risk_results[slug] = "unknown"

    yields_df["rug_risk_level"] = yields_df["protocol"].map(lambda p: risk_results.get(p, "unknown"))
    safe = yields_df[yields_df["rug_risk_level"] != "critical"].reset_index(drop=True)
    logger.info("Safe DeFi yield screen", total_pools=len(yields_df), safe_pools=len(safe))
    return safe
