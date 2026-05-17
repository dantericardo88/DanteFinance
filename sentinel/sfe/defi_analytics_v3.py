"""
sentinel/sfe/defi_analytics_v3.py
dim_107: DeFi Protocol Analytics — Deep Intelligence (score 8 → 9)

Comprehensive DeFi analytics platform covering:
  - Protocol quality scoring (revenue, TVL quality, security, governance)
  - Lending market risk (utilization, liquidation cascades)
  - Yield optimization with risk adjustment
  - Bridge flow analysis and cross-chain capital rotation
  - Governance token valuation
  - DeFi health index

Free Data Sources (no API keys required)
-----------------------------------------
  https://api.llama.fi          — protocols, TVL, fees, revenue
  https://bridges.llama.fi      — bridge TVL and volumes
  https://stablecoins.llama.fi  — stablecoin market data
  https://yields.llama.fi       — yield pools
  https://api.thegraph.com      — Aave V3, Compound V3 subgraphs
  https://api.coingecko.com     — governance token prices (free tier)

Public API
----------
  engine = DeFiMarketMonitor()
  dashboard = engine.get_defi_market_dashboard()
  report    = engine.generate_weekly_report()

  scorer = ProtocolQualityScorer()
  score  = scorer.compute_composite_score("aave")

  lender = LendingProtocolAnalyzer()
  risk   = lender.detect_high_utilization_risk(markets_df)
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

DEFILLAMA_BASE = "https://api.llama.fi"
DEFILLAMA_YIELDS = "https://yields.llama.fi"
BRIDGES_BASE = "https://bridges.llama.fi"
STABLECOINS_BASE = "https://stablecoins.llama.fi"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
THEGRAPH_BASE = "https://api.thegraph.com/subgraphs/name"

AAVE_V3_SUBGRAPH = f"{THEGRAPH_BASE}/aave/protocol-v3"
COMPOUND_V3_SUBGRAPH = f"{THEGRAPH_BASE}/compound-finance/compound-v3"

RATE_LIMIT_DELAY = 1.1  # seconds between requests
CACHE_TTL_SECONDS = 600  # 10 minutes

# Known major hacks (protocol -> USD amount stolen)
KNOWN_HACKS: Dict[str, float] = {
    "ronin": 625_000_000,
    "poly-network": 611_000_000,
    "bnb-bridge": 586_000_000,
    "wormhole": 320_000_000,
    "nomad": 190_000_000,
    "beanstalk": 182_000_000,
    "euler": 197_000_000,
    "curve": 70_000_000,
    "harvest": 34_000_000,
    "badgerdao": 120_000_000,
    "vulcan-forged": 140_000_000,
    "cream": 130_000_000,
}

# Known protocol audits (protocol slug -> audit count)
KNOWN_AUDITS: Dict[str, int] = {
    "aave": 12,
    "uniswap": 8,
    "compound": 10,
    "makerdao": 15,
    "curve": 7,
    "convex-finance": 5,
    "lido": 9,
    "balancer": 6,
    "yearn-finance": 8,
    "synthetix": 7,
    "sushi": 4,
    "pancakeswap": 3,
    "gmx": 4,
    "dydx": 5,
    "frax": 5,
}

# Governance token CoinGecko IDs
PROTOCOL_COINGECKO_IDS: Dict[str, str] = {
    "aave": "aave",
    "uniswap": "uniswap",
    "compound": "compound-governance-token",
    "makerdao": "maker",
    "curve": "curve-dao-token",
    "convex-finance": "convex-finance",
    "lido": "lido-dao",
    "balancer": "balancer",
    "yearn-finance": "yearn-finance",
    "synthetix": "havven",
    "sushi": "sushi",
    "gmx": "gmx",
    "dydx": "dydx",
    "frax": "frax-share",
    "pancakeswap": "pancakeswap-token",
}

# Token emission schedules (protocol -> annual_inflation_pct approximation)
TOKEN_EMISSION_SCHEDULES: Dict[str, float] = {
    "aave": 2.0,
    "uniswap": 4.0,
    "compound": 5.0,
    "makerdao": 0.5,
    "curve": 15.0,
    "convex-finance": 20.0,
    "lido": 1.5,
    "balancer": 8.0,
    "yearn-finance": 1.0,
    "sushi": 10.0,
    "pancakeswap": 25.0,
    "gmx": 3.0,
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProtocolScore:
    """Composite quality score for a DeFi protocol."""
    protocol: str
    tvl_quality: float = 0.0
    revenue_quality: float = 0.0
    security_score: float = 0.0
    governance_score: float = 0.0
    composite: float = 0.0
    tier: str = "D"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.composite = (
            self.tvl_quality * 0.25
            + self.revenue_quality * 0.35
            + self.security_score * 0.25
            + self.governance_score * 0.15
        )
        if self.composite >= 85:
            self.tier = "S"
        elif self.composite >= 70:
            self.tier = "A"
        elif self.composite >= 55:
            self.tier = "B"
        elif self.composite >= 40:
            self.tier = "C"
        else:
            self.tier = "D"


@dataclass
class YieldAllocation:
    """Optimal yield allocation for a capital amount."""
    protocol: str
    pool: str
    chain: str
    apy: float
    risk_adjusted_apy: float
    allocation_pct: float
    allocation_usd: float
    protocol_tier: str
    il_risk: float = 0.0
    notes: str = ""


@dataclass
class BridgeFlow:
    """Bridge TVL and volume data."""
    name: str
    tvl: float
    volume_24h: float
    chains: List[str]
    change_7d_pct: float = 0.0
    stress_signal: bool = False


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _SimpleCache:
    """Thread-safe in-memory TTL cache."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS) -> None:
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, ts = entry
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (value, time.time())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_CACHE = _SimpleCache()
_LAST_REQUEST: float = 0.0
_REQUEST_LOCK = threading.Lock()


def _rate_limited_get(url: str, params: Optional[Dict] = None, timeout: int = 20) -> Optional[Dict]:
    """Rate-limited GET with caching and retries."""
    global _LAST_REQUEST
    cache_key = url + str(sorted((params or {}).items()))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _REQUEST_LOCK:
        wait = RATE_LIMIT_DELAY - (time.time() - _LAST_REQUEST)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST = time.time()

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": "SENTINEL/3.0 DeFiAnalytics"})
            resp.raise_for_status()
            data = resp.json()
            _CACHE.set(cache_key, data)
            return data
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                logger.warning("HTTP error fetching %s: %s", url, e)
                return None
        except Exception as e:
            logger.warning("Error fetching %s (attempt %d): %s", url, attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


def _graphql_query(url: str, query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
    """Execute a GraphQL query."""
    cache_key = url + query + str(variables)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    global _LAST_REQUEST
    with _REQUEST_LOCK:
        wait = RATE_LIMIT_DELAY - (time.time() - _LAST_REQUEST)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST = time.time()

    payload: Dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=20,
                headers={"Content-Type": "application/json", "User-Agent": "SENTINEL/3.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            _CACHE.set(cache_key, data)
            return data
        except Exception as e:
            logger.warning("GraphQL error at %s (attempt %d): %s", url, attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# DefiLlamaAdvancedClient
# ---------------------------------------------------------------------------

class DefiLlamaAdvancedClient:
    """
    Extended DefiLlama API wrapper with fees, revenue, bridges,
    stablecoins, chain TVL history, and yield pools.
    Rate-limited to 1 req/sec with 10-min cache.
    """

    # ------------------------------------------------------------------
    # Protocol fees & revenue
    # ------------------------------------------------------------------

    def get_protocol_fees(self, protocol: str, period: str = "daily") -> "pd.DataFrame":
        """
        Fetch fee timeseries for a specific protocol.

        Args:
            protocol: DefiLlama slug (e.g. 'aave', 'uniswap')
            period:   'daily' | 'weekly' | 'monthly'

        Returns:
            DataFrame with columns [date, fees, revenue]
        """
        url = f"{DEFILLAMA_BASE}/summary/fees/{protocol}"
        params = {"dataType": "dailyFees"}
        data = _rate_limited_get(url, params)
        if not data:
            logger.warning("No fee data for %s", protocol)
            if HAS_PANDAS:
                return pd.DataFrame(columns=["date", "fees", "revenue"])
            return []  # type: ignore[return-value]

        # Parse totalDataChart: list of [timestamp, value]
        chart = data.get("totalDataChart", []) or data.get("totalDataChartBreakdown", [])

        rows = []
        for entry in chart:
            if isinstance(entry, list) and len(entry) >= 2:
                ts, val = entry[0], entry[1]
                dt = datetime.fromtimestamp(ts, tz=timezone.utc) if isinstance(ts, (int, float)) else ts
                rows.append({"date": dt, "fees": float(val)})

        if not HAS_PANDAS:
            return rows  # type: ignore[return-value]

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values("date").reset_index(drop=True)

        # Resample if weekly/monthly
        if period == "weekly":
            df = df.set_index("date").resample("W").sum().reset_index()
        elif period == "monthly":
            df = df.set_index("date").resample("ME").sum().reset_index()

        # Revenue timeseries (protocol's share)
        rev_data = _rate_limited_get(url, {"dataType": "dailyRevenue"})
        rev_rows = {}
        if rev_data:
            for entry in (rev_data.get("totalDataChart") or []):
                if isinstance(entry, list) and len(entry) >= 2:
                    ts, val = entry[0], entry[1]
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    rev_rows[dt] = float(val)

        df["revenue"] = df["date"].map(rev_rows).fillna(0.0)
        return df

    def get_all_protocol_fees(self) -> "pd.DataFrame":
        """
        Fetch fee summary for ALL DeFi protocols.

        Returns:
            DataFrame with columns:
            [name, category, chains, totalFees24h, totalRevenue24h, totalAllTime]
        """
        url = f"{DEFILLAMA_BASE}/overview/fees"
        data = _rate_limited_get(url)
        if not data:
            if HAS_PANDAS:
                return pd.DataFrame()
            return []  # type: ignore[return-value]

        protocols = data.get("protocols", [])
        rows = []
        for p in protocols:
            rows.append({
                "name": p.get("name", ""),
                "slug": p.get("module", p.get("name", "")).lower().replace(" ", "-"),
                "category": p.get("category", ""),
                "chains": p.get("chains", []),
                "totalFees24h": float(p.get("total24h") or 0),
                "totalRevenue24h": float(p.get("revenue24h") or p.get("totalRevenue24h") or 0),
                "totalAllTime": float(p.get("totalAllTime") or 0),
                "mcap": float(p.get("mcap") or 0),
            })

        if not HAS_PANDAS:
            return rows  # type: ignore[return-value]

        df = pd.DataFrame(rows)
        df = df.sort_values("totalFees24h", ascending=False).reset_index(drop=True)
        return df

    def get_protocol_revenue(self) -> "pd.DataFrame":
        """
        Revenue = fees kept by protocol (not distributed to LPs).
        Adds Price-to-Fees ratio column.

        Returns:
            DataFrame sorted by Revenue/TVL efficiency.
        """
        url = f"{DEFILLAMA_BASE}/overview/fees"
        params = {"dataType": "dailyRevenue"}
        data = _rate_limited_get(url, params)

        fee_df = self.get_all_protocol_fees()

        if not data or not HAS_PANDAS:
            return fee_df

        rev_protocols = {p.get("name", ""): p for p in data.get("protocols", [])}

        rows = []
        for _, row in fee_df.iterrows():
            name = row["name"]
            rev_p = rev_protocols.get(name, {})
            daily_rev = float(rev_p.get("total24h") or row.get("totalRevenue24h", 0))
            annual_rev = daily_rev * 365
            annual_fees = row["totalFees24h"] * 365
            mcap = row["mcap"]

            pf_ratio = mcap / annual_fees if annual_fees > 0 else float("inf")
            rev_share = daily_rev / row["totalFees24h"] if row["totalFees24h"] > 0 else 0

            rows.append({
                "name": name,
                "slug": row["slug"],
                "category": row["category"],
                "annualFees": annual_fees,
                "annualRevenue": annual_rev,
                "mcap": mcap,
                "PF_ratio": pf_ratio,
                "revenueFeeShare": rev_share,
            })

        df = pd.DataFrame(rows)
        df = df[df["annualFees"] > 0].sort_values("PF_ratio").reset_index(drop=True)
        return df

    def get_bridge_flows(self) -> "pd.DataFrame":
        """
        Fetch bridge TVL and volume data.

        Returns:
            DataFrame with columns [name, tvl, volume24h, chains, change7d]
        """
        url = f"{BRIDGES_BASE}/bridges"
        data = _rate_limited_get(url, {"includeChains": "true"})
        if not data:
            if HAS_PANDAS:
                return pd.DataFrame()
            return []  # type: ignore[return-value]

        bridges = data if isinstance(data, list) else data.get("bridges", [])
        rows = []
        for b in bridges:
            rows.append({
                "name": b.get("displayName", b.get("name", "")),
                "slug": b.get("name", ""),
                "tvl": float(b.get("currentTvl") or b.get("tvl") or 0),
                "volume24h": float(b.get("lastDailyVolume") or 0),
                "chains": b.get("chains", []),
                "change7d": float(b.get("change_7d") or 0),
            })

        if not HAS_PANDAS:
            return rows  # type: ignore[return-value]

        df = pd.DataFrame(rows)
        df = df.sort_values("tvl", ascending=False).reset_index(drop=True)
        return df

    def get_chain_tvl_history(self, chain: str) -> "pd.Series":
        """
        Historical TVL for a specific chain.

        Args:
            chain: Chain name (e.g. 'Ethereum', 'Arbitrum', 'Base')

        Returns:
            pd.Series indexed by datetime
        """
        url = f"{DEFILLAMA_BASE}/v2/historicalChainTvl/{chain}"
        data = _rate_limited_get(url)
        if not data or not HAS_PANDAS:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        entries = data if isinstance(data, list) else []
        dates, values = [], []
        for entry in entries:
            if isinstance(entry, dict):
                ts = entry.get("date")
                val = entry.get("tvl")
                if ts and val is not None:
                    dates.append(datetime.fromtimestamp(float(ts), tz=timezone.utc))
                    values.append(float(val))

        return pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()

    def get_stablecoin_breakdown(self) -> "pd.DataFrame":
        """
        Stablecoin market data: market cap, peg stability, chain distribution.

        Returns:
            DataFrame sorted by circulating supply
        """
        url = f"{STABLECOINS_BASE}/stablecoins"
        params = {"includePrices": "true"}
        data = _rate_limited_get(url, params)
        if not data or not HAS_PANDAS:
            return pd.DataFrame() if HAS_PANDAS else []  # type: ignore[return-value]

        pegged = data.get("peggedAssets", [])
        rows = []
        for s in pegged:
            circulating = float(s.get("circulating", {}).get("peggedUSD", 0) or 0)
            price = float((s.get("price") or 1.0))
            peg_dev = abs(price - 1.0) if price > 0 else 1.0
            chains = list(s.get("chainCirculating", {}).keys())

            rows.append({
                "name": s.get("name", ""),
                "symbol": s.get("symbol", ""),
                "pegType": s.get("pegType", ""),
                "circulating_usd": circulating,
                "price": price,
                "peg_deviation": peg_dev,
                "chains": chains,
                "chain_count": len(chains),
                "is_stable": peg_dev < 0.01,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("circulating_usd", ascending=False).reset_index(drop=True)
        return df

    def get_yield_pools(
        self,
        min_tvl: float = 1_000_000,
        max_apy: float = 100.0,
        chains: Optional[List[str]] = None,
    ) -> "pd.DataFrame":
        """
        Fetch yield farming opportunities from DefiLlama.

        Returns:
            DataFrame sorted by TVL-adjusted APY
        """
        url = f"{DEFILLAMA_YIELDS}/pools"
        data = _rate_limited_get(url)
        if not data or not HAS_PANDAS:
            return pd.DataFrame() if HAS_PANDAS else []  # type: ignore[return-value]

        pools = data.get("data", [])
        rows = []
        for p in pools:
            tvl = float(p.get("tvlUsd") or 0)
            apy = float(p.get("apy") or 0)
            if tvl < min_tvl or apy > max_apy or apy <= 0:
                continue
            chain = p.get("chain", "")
            if chains and chain not in chains:
                continue

            il_risk = p.get("ilRisk", "no")
            il_score = 0.3 if il_risk == "yes" else 0.0

            rows.append({
                "pool": p.get("pool", ""),
                "project": p.get("project", ""),
                "symbol": p.get("symbol", ""),
                "chain": chain,
                "tvl": tvl,
                "apy": apy,
                "apy_base": float(p.get("apyBase") or 0),
                "apy_reward": float(p.get("apyReward") or 0),
                "il_risk": il_risk,
                "il_score": il_score,
                "stablecoin": bool(p.get("stablecoin", False)),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("apy", ascending=False).reset_index(drop=True)
        return df

    def get_protocols(self) -> "pd.DataFrame":
        """
        Fetch all DeFi protocols with TVL data.

        Returns:
            DataFrame with protocol metadata
        """
        url = f"{DEFILLAMA_BASE}/protocols"
        data = _rate_limited_get(url)
        if not data or not HAS_PANDAS:
            return pd.DataFrame() if HAS_PANDAS else []  # type: ignore[return-value]

        rows = []
        for p in (data if isinstance(data, list) else []):
            rows.append({
                "name": p.get("name", ""),
                "slug": p.get("slug", ""),
                "category": p.get("category", ""),
                "chains": p.get("chains", []),
                "tvl": float(p.get("tvl") or 0),
                "change_1d": float(p.get("change_1d") or 0),
                "change_7d": float(p.get("change_7d") or 0),
                "mcap": float(p.get("mcap") or 0),
                "fdv": float(p.get("fdv") or 0),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("tvl", ascending=False).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# ProtocolQualityScorer
# ---------------------------------------------------------------------------

class ProtocolQualityScorer:
    """
    Multi-dimensional DeFi protocol quality scoring.
    Produces a 0-100 composite score across four dimensions:
      TVL quality (25%), Revenue quality (35%), Security (25%), Governance (15%)
    """

    def __init__(self) -> None:
        self._llama = DefiLlamaAdvancedClient()
        self._protocols_cache: Optional["pd.DataFrame"] = None
        self._fees_cache: Optional["pd.DataFrame"] = None

    def _get_protocols(self) -> "pd.DataFrame":
        if self._protocols_cache is None:
            self._protocols_cache = self._llama.get_protocols()
        return self._protocols_cache

    def _get_fees(self) -> "pd.DataFrame":
        if self._fees_cache is None:
            self._fees_cache = self._llama.get_all_protocol_fees()
        return self._fees_cache

    def _find_protocol(self, protocol: str) -> Dict[str, Any]:
        """Find protocol row by slug or name (fuzzy)."""
        df = self._get_protocols()
        if df.empty:
            return {}
        slug_lower = protocol.lower()
        mask = (
            df["slug"].str.lower().str.contains(slug_lower, na=False)
            | df["name"].str.lower().str.contains(slug_lower, na=False)
        )
        matches = df[mask]
        if matches.empty:
            return {}
        return matches.iloc[0].to_dict()

    def _find_fees(self, protocol: str) -> Dict[str, Any]:
        """Find fee row by name."""
        df = self._get_fees()
        if df.empty:
            return {}
        slug_lower = protocol.lower()
        mask = (
            df["name"].str.lower().str.contains(slug_lower, na=False)
            | df.get("slug", pd.Series(dtype=str)).str.lower().str.contains(slug_lower, na=False)
        )
        matches = df[mask] if "slug" in df.columns else df[df["name"].str.lower().str.contains(slug_lower, na=False)]
        if matches.empty:
            return {}
        return matches.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Revenue quality
    # ------------------------------------------------------------------

    def compute_revenue_quality(self, protocol: str) -> Dict[str, Any]:
        """
        Revenue quality dimensions:
          - P/F ratio: market_cap / annual_fees (lower = more undervalued)
          - Revenue/TVL: annual_revenue / TVL (higher = more capital efficient)
          - Revenue share: protocol_revenue / total_fees
        Returns score 0-100.
        """
        p_data = self._find_protocol(protocol)
        f_data = self._find_fees(protocol)

        tvl = float(p_data.get("tvl") or 0)
        mcap = float(p_data.get("mcap") or f_data.get("mcap") or 0)
        fees_24h = float(f_data.get("totalFees24h") or 0)
        rev_24h = float(f_data.get("totalRevenue24h") or 0)

        annual_fees = fees_24h * 365
        annual_rev = rev_24h * 365

        # P/F ratio — lower is better (like low P/E)
        pf_ratio = mcap / annual_fees if annual_fees > 0 and mcap > 0 else None
        # Score: <20 = 90pts, 20-50 = 70pts, 50-100 = 50pts, >100 = 20pts
        if pf_ratio is None:
            pf_score = 30.0
        elif pf_ratio < 10:
            pf_score = 95.0
        elif pf_ratio < 20:
            pf_score = 80.0
        elif pf_ratio < 50:
            pf_score = 65.0
        elif pf_ratio < 100:
            pf_score = 45.0
        else:
            pf_score = 20.0

        # Revenue/TVL ratio — higher = more capital efficient
        rev_tvl = annual_rev / tvl if tvl > 0 else 0
        # Score: >0.1 (10%) = excellent, >0.05 = good, >0.01 = fair
        if rev_tvl > 0.15:
            rtv_score = 95.0
        elif rev_tvl > 0.05:
            rtv_score = 75.0
        elif rev_tvl > 0.01:
            rtv_score = 50.0
        elif rev_tvl > 0:
            rtv_score = 30.0
        else:
            rtv_score = 10.0

        # Protocol revenue share
        rev_share = rev_24h / fees_24h if fees_24h > 0 else 0
        rs_score = min(100.0, rev_share * 100)  # Higher share = more goes to treasury

        score = pf_score * 0.4 + rtv_score * 0.4 + rs_score * 0.2

        return {
            "score": round(score, 1),
            "pf_ratio": pf_ratio,
            "pf_score": pf_score,
            "revenue_tvl_ratio": rev_tvl,
            "rtv_score": rtv_score,
            "revenue_fee_share": rev_share,
            "rs_score": rs_score,
            "annual_fees": annual_fees,
            "annual_revenue": annual_rev,
            "tvl": tvl,
            "mcap": mcap,
        }

    # ------------------------------------------------------------------
    # TVL quality
    # ------------------------------------------------------------------

    def compute_tvl_quality(self, protocol: str) -> Dict[str, Any]:
        """
        TVL quality dimensions:
          - TVL size (absolute credibility)
          - TVL stability (7d change)
          - Chain diversification
          - Revenue-to-TVL (organic vs mercenary capital)
        Returns score 0-100.
        """
        p_data = self._find_protocol(protocol)
        tvl = float(p_data.get("tvl") or 0)
        change_1d = float(p_data.get("change_1d") or 0)
        change_7d = float(p_data.get("change_7d") or 0)
        chains = p_data.get("chains", []) or []

        # TVL size score
        if tvl > 5_000_000_000:
            size_score = 95.0
        elif tvl > 1_000_000_000:
            size_score = 85.0
        elif tvl > 500_000_000:
            size_score = 70.0
        elif tvl > 100_000_000:
            size_score = 55.0
        elif tvl > 10_000_000:
            size_score = 35.0
        else:
            size_score = 10.0

        # TVL stability (penalize large swings → mercenary capital)
        abs_change_7d = abs(change_7d)
        if abs_change_7d < 5:
            stab_score = 90.0
        elif abs_change_7d < 15:
            stab_score = 70.0
        elif abs_change_7d < 30:
            stab_score = 45.0
        else:
            stab_score = 15.0

        # Bonus: positive trend
        trend_bonus = 10.0 if change_7d > 5 else 0.0

        # Chain diversification
        chain_count = len(chains) if chains else 1
        div_score = min(90.0, 40.0 + chain_count * 8.0)

        score = (
            size_score * 0.45
            + stab_score * 0.35
            + div_score * 0.20
            + trend_bonus * 0.0
        )
        score = min(100.0, score + trend_bonus * 0.15)

        return {
            "score": round(score, 1),
            "tvl": tvl,
            "change_1d_pct": change_1d,
            "change_7d_pct": change_7d,
            "chain_count": chain_count,
            "size_score": size_score,
            "stability_score": stab_score,
            "diversification_score": div_score,
        }

    # ------------------------------------------------------------------
    # Security score
    # ------------------------------------------------------------------

    def compute_security_score(self, protocol: str) -> float:
        """
        Security score based on:
          - Audit count (KNOWN_AUDITS)
          - Known hacks (KNOWN_HACKS)
          - Protocol age (inferred)
          - Bug bounty (hardcoded for major protocols)
        Returns 0-100.
        """
        slug = protocol.lower().replace(" ", "-")

        # Hack penalty
        hack_amount = 0.0
        for hacked_protocol, amount in KNOWN_HACKS.items():
            if hacked_protocol in slug or slug in hacked_protocol:
                hack_amount = max(hack_amount, amount)

        if hack_amount > 500_000_000:
            hack_penalty = 70.0
        elif hack_amount > 100_000_000:
            hack_penalty = 50.0
        elif hack_amount > 10_000_000:
            hack_penalty = 25.0
        else:
            hack_penalty = 0.0

        # Audit score
        audit_count = 0
        for known_slug, count in KNOWN_AUDITS.items():
            if known_slug in slug or slug in known_slug:
                audit_count = count
                break

        if audit_count >= 8:
            audit_score = 85.0
        elif audit_count >= 4:
            audit_score = 65.0
        elif audit_count >= 2:
            audit_score = 45.0
        elif audit_count >= 1:
            audit_score = 25.0
        else:
            audit_score = 10.0

        # Well-known protocols: assume bug bounty + timelock
        well_known = set(KNOWN_AUDITS.keys())
        bounty_bonus = 10.0 if any(k in slug or slug in k for k in well_known) else 0.0

        score = max(0.0, audit_score + bounty_bonus - hack_penalty)
        return round(min(100.0, score), 1)

    # ------------------------------------------------------------------
    # Governance score
    # ------------------------------------------------------------------

    def compute_governance_score(self, protocol: str) -> float:
        """
        Governance quality (partially hardcoded for known protocols):
          - Decentralized voting (DAO)
          - Multisig + timelock
          - Active governance participation
          - Treasury size vs FDV
        Returns 0-100.
        """
        slug = protocol.lower().replace(" ", "-")

        # Well-established DAOs with strong governance
        top_tier_dao = {"makerdao", "aave", "uniswap", "compound", "lido", "frax"}
        mid_tier_dao = {"curve", "balancer", "yearn-finance", "synthetix", "gmx", "dydx"}
        low_tier = {"pancakeswap", "sushi"}

        base_score = 40.0
        for dao in top_tier_dao:
            if dao in slug or slug in dao:
                base_score = 85.0
                break
        else:
            for dao in mid_tier_dao:
                if dao in slug or slug in dao:
                    base_score = 65.0
                    break
            else:
                for dao in low_tier:
                    if dao in slug or slug in dao:
                        base_score = 35.0
                        break

        # Timelock bonus for major protocols
        timelock_protocols = {"aave", "compound", "uniswap", "makerdao", "lido", "curve"}
        timelock_bonus = 5.0 if any(p in slug for p in timelock_protocols) else 0.0

        return round(min(100.0, base_score + timelock_bonus), 1)

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def compute_composite_score(self, protocol: str) -> ProtocolScore:
        """
        Compute weighted composite quality score for a DeFi protocol.

        Weights: TVL (25%) + Revenue (35%) + Security (25%) + Governance (15%)
        Tier: S (85+), A (70-85), B (55-70), C (40-55), D (<40)
        """
        tvl_q = self.compute_tvl_quality(protocol)
        rev_q = self.compute_revenue_quality(protocol)
        sec = self.compute_security_score(protocol)
        gov = self.compute_governance_score(protocol)

        score = ProtocolScore(
            protocol=protocol,
            tvl_quality=tvl_q["score"],
            revenue_quality=rev_q["score"],
            security_score=sec,
            governance_score=gov,
            details={
                "tvl_detail": tvl_q,
                "revenue_detail": rev_q,
                "security_detail": {"score": sec},
                "governance_detail": {"score": gov},
            },
        )
        return score

    def score_top_protocols(self, top_n: int = 20) -> List[ProtocolScore]:
        """Score the top N protocols by TVL."""
        df = self._get_protocols()
        if df.empty:
            return []
        slugs = df.head(top_n)["slug"].tolist()
        scores = []
        for slug in slugs:
            try:
                s = self.compute_composite_score(slug)
                scores.append(s)
            except Exception as e:
                logger.warning("Failed to score %s: %s", slug, e)
        scores.sort(key=lambda x: x.composite, reverse=True)
        return scores


# ---------------------------------------------------------------------------
# LendingProtocolAnalyzer
# ---------------------------------------------------------------------------

class LendingProtocolAnalyzer:
    """
    Deep analytics for lending protocols: Aave, Compound, MakerDAO.
    Uses The Graph subgraphs for on-chain market data.
    """

    AAVE_QUERY = """
    {
      markets(first: 50, orderBy: totalDepositBalanceUSD, orderDirection: desc) {
        id
        name
        inputToken {
          symbol
          decimals
        }
        totalDepositBalanceUSD
        totalBorrowBalanceUSD
        rates(where: {side: BORROWER}) {
          rate
          type
        }
        liquidationThreshold
        maximumLTV
        isActive
      }
    }
    """

    AAVE_SUPPLY_QUERY = """
    {
      markets(first: 50) {
        id
        inputToken { symbol }
        totalDepositBalanceUSD
        totalBorrowBalanceUSD
        rates(where: {side: LENDER}) {
          rate
          type
        }
      }
    }
    """

    def __init__(self) -> None:
        self._llama = DefiLlamaAdvancedClient()

    def query_aave_markets(self) -> "pd.DataFrame":
        """
        Query Aave V3 markets via The Graph.

        Returns:
            DataFrame: [asset, totalLiquidity, totalBorrows, utilizationRate,
                       liquidationThreshold, maxLTV, borrowRate, supplyRate]
        """
        result = _graphql_query(AAVE_V3_SUBGRAPH, self.AAVE_QUERY)
        if not result or not HAS_PANDAS:
            return pd.DataFrame() if HAS_PANDAS else []  # type: ignore[return-value]

        markets = (result.get("data") or {}).get("markets", [])
        rows = []
        for m in markets:
            deposit = float(m.get("totalDepositBalanceUSD") or 0)
            borrow = float(m.get("totalBorrowBalanceUSD") or 0)
            util = borrow / deposit if deposit > 0 else 0.0

            rates = m.get("rates", [])
            borrow_rate = 0.0
            for r in rates:
                if r.get("type") == "VARIABLE":
                    borrow_rate = float(r.get("rate") or 0) / 100
                    break

            rows.append({
                "asset": m.get("inputToken", {}).get("symbol", "?"),
                "market_id": m.get("id", ""),
                "totalLiquidity": deposit,
                "totalBorrows": borrow,
                "utilizationRate": round(util, 4),
                "liquidationThreshold": float(m.get("liquidationThreshold") or 0) / 10000,
                "maxLTV": float(m.get("maximumLTV") or 0) / 10000,
                "borrowRate": borrow_rate,
                "isActive": bool(m.get("isActive", True)),
            })

        df = pd.DataFrame(rows)
        df = df[df["totalLiquidity"] > 0].sort_values("totalLiquidity", ascending=False)
        return df.reset_index(drop=True)

    def compute_utilization_rate(self, total_borrows: float, total_supply: float) -> float:
        """Utilization = total_borrows / total_supply. Returns 0-1 float."""
        if total_supply <= 0:
            return 0.0
        return min(1.0, total_borrows / total_supply)

    def detect_high_utilization_risk(
        self, markets: "pd.DataFrame", threshold: float = 0.85
    ) -> List[str]:
        """
        Identify assets with utilization > threshold.
        High utilization means depositors can't withdraw — liquidity crunch risk.

        Returns:
            List of asset names above threshold
        """
        if markets.empty or "utilizationRate" not in markets.columns:
            return []
        high_util = markets[markets["utilizationRate"] > threshold]
        return high_util["asset"].tolist()

    def compute_liquidation_risk(self, health_factors: List[float]) -> Dict[str, Any]:
        """
        Aggregate liquidation risk from a distribution of position health factors.

        Health factor < 1.0 → position is liquidatable.
        Health factor 1.0-1.2 → at-risk with moderate price move.

        Returns:
            dict with liquidatable_pct, at_risk_pct, cascade_risk_level
        """
        if not health_factors:
            return {"liquidatable_pct": 0, "at_risk_pct": 0, "cascade_risk_level": "UNKNOWN"}

        n = len(health_factors)
        liquidatable = sum(1 for h in health_factors if h < 1.0)
        at_risk = sum(1 for h in health_factors if 1.0 <= h < 1.2)

        liq_pct = liquidatable / n
        risk_pct = at_risk / n

        if liq_pct > 0.1 or risk_pct > 0.3:
            cascade = "CRITICAL"
        elif liq_pct > 0.05 or risk_pct > 0.15:
            cascade = "HIGH"
        elif liq_pct > 0.01 or risk_pct > 0.08:
            cascade = "MODERATE"
        else:
            cascade = "LOW"

        return {
            "total_positions": n,
            "liquidatable_count": liquidatable,
            "at_risk_count": at_risk,
            "liquidatable_pct": round(liq_pct * 100, 2),
            "at_risk_pct": round(risk_pct * 100, 2),
            "avg_health_factor": sum(health_factors) / n,
            "cascade_risk_level": cascade,
        }

    def estimate_cascade_liquidation(
        self, protocol: str, price_drop_pct: float
    ) -> float:
        """
        Estimate USD value of collateral liquidated if ETH drops by price_drop_pct%.

        Model:
          - ETH collateral C with liquidation threshold LT
          - Position healthy when: C × price_new ≥ debt / LT
          - Cascade: C × (1 - drop) < debt / LT → liquidation triggered

        Args:
            protocol:       Protocol slug (used to get TVL)
            price_drop_pct: e.g. 20.0 for -20%

        Returns:
            Estimated USD liquidated (rough approximation)
        """
        llama = DefiLlamaAdvancedClient()
        protocols_df = llama.get_protocols()

        if protocols_df.empty:
            return 0.0

        mask = protocols_df["slug"].str.lower().str.contains(protocol.lower(), na=False)
        prot_rows = protocols_df[mask]
        if prot_rows.empty:
            return 0.0

        tvl = float(prot_rows.iloc[0]["tvl"])

        # Assume 60% of TVL is ETH collateral (rough Aave average)
        eth_collateral = tvl * 0.60

        # Average LTV ratio in lending protocols: ~70%
        avg_ltv = 0.70
        # Liquidation threshold slightly above LTV: ~80%
        liq_threshold = 0.80

        # Positions with LTV between 70-80% are at risk on a price drop
        # These are the "close to liquidation" positions
        at_risk_fraction = (price_drop_pct / 100) * (1 / (liq_threshold - avg_ltv))
        at_risk_fraction = max(0.0, min(1.0, at_risk_fraction))

        estimated_liquidated = eth_collateral * at_risk_fraction

        logger.info(
            "Cascade estimate for %s at -%g%% ETH drop: ~$%.0f liquidated",
            protocol, price_drop_pct, estimated_liquidated
        )
        return estimated_liquidated

    def get_lending_market_summary(self) -> Dict[str, Any]:
        """
        Summarize current state of major lending protocols.
        Falls back to DefiLlama TVL data if Graph unavailable.
        """
        markets = self.query_aave_markets()

        if not markets.empty:
            high_util = self.detect_high_utilization_risk(markets)
            total_liq = float(markets["totalLiquidity"].sum())
            total_borrow = float(markets["totalBorrows"].sum())
            avg_util = self.compute_utilization_rate(total_borrow, total_liq)

            return {
                "source": "aave_v3_subgraph",
                "total_liquidity_usd": total_liq,
                "total_borrows_usd": total_borrow,
                "avg_utilization": round(avg_util, 4),
                "high_utilization_assets": high_util,
                "top_markets": markets.head(5).to_dict(orient="records"),
            }

        # Fallback: DefiLlama
        llama = DefiLlamaAdvancedClient()
        df = llama.get_protocols()
        lending = df[df["category"].str.lower().str.contains("lending|cdp", na=False)]
        return {
            "source": "defillama_fallback",
            "top_lending_protocols": lending.head(10)[["name", "tvl", "change_7d"]].to_dict(orient="records"),
        }


# ---------------------------------------------------------------------------
# YieldOptimizerEngine
# ---------------------------------------------------------------------------

class YieldOptimizerEngine:
    """
    Find and optimize DeFi yield strategies with risk-adjustment,
    gas cost analysis, and portfolio allocation.
    """

    def __init__(self) -> None:
        self._llama = DefiLlamaAdvancedClient()
        self._scorer = ProtocolQualityScorer()

    def get_risk_adjusted_yields(
        self,
        min_tvl: float = 10_000_000,
        max_apy: float = 50.0,
        chains: Optional[List[str]] = None,
    ) -> "pd.DataFrame":
        """
        Fetch and risk-adjust yield opportunities.

        Risk adjustment: raw_apy × protocol_quality × (1 - il_penalty)
        Protocol quality: from ProtocolQualityScorer (0-1 normalized)
        IL penalty: 0.3 for volatile pairs, 0 for stables

        Returns:
            DataFrame sorted by risk_adjusted_apy (desc)
        """
        pools = self._llama.get_yield_pools(min_tvl=min_tvl, max_apy=max_apy, chains=chains)
        if pools.empty:
            return pd.DataFrame() if HAS_PANDAS else []  # type: ignore[return-value]

        # Get protocol scores for risk adjustment
        protocol_scores: Dict[str, float] = {}

        def _get_score(proj: str) -> float:
            if proj not in protocol_scores:
                try:
                    s = self._scorer.compute_composite_score(proj)
                    protocol_scores[proj] = s.composite / 100.0
                except Exception:
                    protocol_scores[proj] = 0.4  # unknown = conservative default
            return protocol_scores[proj]

        rows = []
        for _, pool in pools.iterrows():
            proj = pool["project"]
            qual = _get_score(proj)
            il_penalty = float(pool.get("il_score", 0.0))
            raw_apy = float(pool["apy"])
            risk_adj = raw_apy * qual * (1.0 - il_penalty)

            rows.append({
                "pool": pool["pool"],
                "project": proj,
                "symbol": pool["symbol"],
                "chain": pool["chain"],
                "tvl": pool["tvl"],
                "apy": raw_apy,
                "apy_base": pool.get("apy_base", 0),
                "apy_reward": pool.get("apy_reward", 0),
                "il_risk": pool.get("il_risk", "no"),
                "protocol_quality": round(qual, 3),
                "risk_adjusted_apy": round(risk_adj, 2),
                "stablecoin": pool.get("stablecoin", False),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("risk_adjusted_apy", ascending=False).reset_index(drop=True)
        return df

    def compute_optimal_allocation(
        self, capital: float, risk_tolerance: str = "medium"
    ) -> Dict[str, Any]:
        """
        Compute optimal capital allocation across DeFi yield opportunities.

        Constraints:
          - Max 30% in single protocol
          - Max 50% in single chain
          - Protocol tier: conservative ≥ C, aggressive ≥ D
          - Conservative: stable-only pools
          - Aggressive: any pool

        Args:
            capital:         Total capital in USD
            risk_tolerance:  'conservative' | 'medium' | 'aggressive'

        Returns:
            dict with allocations list and portfolio metrics
        """
        max_apy = {"conservative": 15.0, "medium": 30.0, "aggressive": 50.0}[risk_tolerance]
        stables_only = risk_tolerance == "conservative"
        min_protocol_quality = {"conservative": 0.55, "medium": 0.40, "aggressive": 0.25}[risk_tolerance]

        pools = self.get_risk_adjusted_yields(max_apy=max_apy)
        if pools.empty:
            return {"allocations": [], "error": "No yield data available"}

        if stables_only:
            pools = pools[pools["stablecoin"] == True]  # noqa: E712

        pools = pools[pools["protocol_quality"] >= min_protocol_quality]

        if pools.empty:
            return {"allocations": [], "error": "No pools meet quality criteria"}

        # Greedy allocation: fill from highest risk-adjusted yield
        # Subject to 30% single protocol, 50% single chain constraints
        allocations: List[YieldAllocation] = []
        protocol_allocated: Dict[str, float] = {}
        chain_allocated: Dict[str, float] = {}
        remaining = capital

        for _, pool in pools.iterrows():
            if remaining <= 0:
                break

            proj = str(pool["project"])
            chain = str(pool["chain"])

            max_proto = capital * 0.30
            max_chain = capital * 0.50

            proto_used = protocol_allocated.get(proj, 0.0)
            chain_used = chain_allocated.get(chain, 0.0)

            available_proto = max_proto - proto_used
            available_chain = max_chain - chain_used
            available = min(remaining, available_proto, available_chain)

            if available < 1000:  # min $1k allocation
                continue

            alloc_usd = min(available, capital * 0.25)  # max 25% in single pool
            alloc_pct = alloc_usd / capital * 100

            # Get tier
            qual = float(pool["protocol_quality"])
            if qual >= 0.85:
                tier = "S"
            elif qual >= 0.70:
                tier = "A"
            elif qual >= 0.55:
                tier = "B"
            elif qual >= 0.40:
                tier = "C"
            else:
                tier = "D"

            allocations.append(YieldAllocation(
                protocol=proj,
                pool=str(pool["symbol"]),
                chain=chain,
                apy=float(pool["apy"]),
                risk_adjusted_apy=float(pool["risk_adjusted_apy"]),
                allocation_pct=round(alloc_pct, 1),
                allocation_usd=round(alloc_usd, 0),
                protocol_tier=tier,
                il_risk=float(pool.get("il_risk", "no") == "yes"),
            ))

            protocol_allocated[proj] = proto_used + alloc_usd
            chain_allocated[chain] = chain_used + alloc_usd
            remaining -= alloc_usd

        if not allocations:
            return {"allocations": [], "capital": capital, "risk_tolerance": risk_tolerance}

        total_alloc = sum(a.allocation_usd for a in allocations)
        wavg_apy = sum(a.apy * a.allocation_usd for a in allocations) / total_alloc if total_alloc > 0 else 0
        wavg_risk_adj = sum(a.risk_adjusted_apy * a.allocation_usd for a in allocations) / total_alloc if total_alloc > 0 else 0

        return {
            "capital": capital,
            "risk_tolerance": risk_tolerance,
            "deployed_usd": round(total_alloc, 0),
            "cash_remaining": round(remaining, 0),
            "weighted_avg_apy": round(wavg_apy, 2),
            "weighted_risk_adj_apy": round(wavg_risk_adj, 2),
            "annual_yield_usd": round(total_alloc * wavg_apy / 100, 0),
            "allocations": [
                {
                    "protocol": a.protocol,
                    "pool": a.pool,
                    "chain": a.chain,
                    "apy": a.apy,
                    "risk_adjusted_apy": a.risk_adjusted_apy,
                    "allocation_pct": a.allocation_pct,
                    "allocation_usd": a.allocation_usd,
                    "tier": a.protocol_tier,
                }
                for a in allocations
            ],
        }

    def detect_yield_arb(
        self, protocol1: str, protocol2: str, asset: str
    ) -> Dict[str, Any]:
        """
        Detect yield arbitrage: borrow on protocol1, lend on protocol2.

        Args:
            protocol1: Protocol to borrow from (lower rate)
            protocol2: Protocol to lend to (higher rate)
            asset:     Asset symbol (e.g. 'USDC')

        Returns:
            dict with spread, net_apy, and feasibility
        """
        pools = self.get_risk_adjusted_yields()
        if pools.empty:
            return {"feasible": False, "reason": "No pool data"}

        p1_pools = pools[
            pools["project"].str.lower().str.contains(protocol1.lower(), na=False)
            & pools["symbol"].str.upper().str.contains(asset.upper(), na=False)
        ]
        p2_pools = pools[
            pools["project"].str.lower().str.contains(protocol2.lower(), na=False)
            & pools["symbol"].str.upper().str.contains(asset.upper(), na=False)
        ]

        if p1_pools.empty or p2_pools.empty:
            return {
                "feasible": False,
                "reason": f"No {asset} pools found for one or both protocols",
                "protocol1_pools": len(p1_pools),
                "protocol2_pools": len(p2_pools),
            }

        lend_rate = float(p2_pools.iloc[0]["apy"])
        # Borrow rate is typically 1.5-3x the supply rate in lending protocols
        borrow_rate = float(p1_pools.iloc[0]["apy"]) * 2.0  # approximation

        spread = lend_rate - borrow_rate
        gas_cost_annual_pct = 0.5  # ~$200 in gas / $40k capital = 0.5%

        feasible = spread > 1.0  # Need >1% spread to be worth it

        return {
            "protocol1": protocol1,
            "protocol2": protocol2,
            "asset": asset,
            "estimated_borrow_rate": round(borrow_rate, 2),
            "lend_rate": round(lend_rate, 2),
            "gross_spread_pct": round(spread, 2),
            "gas_cost_annual_pct": gas_cost_annual_pct,
            "net_apy_pct": round(spread - gas_cost_annual_pct, 2),
            "feasible": feasible,
            "min_capital_usd": 10_000 if feasible else None,
            "notes": "Borrow rate approximated as 2x supply rate; verify on-chain before executing",
        }

    def estimate_gas_cost_impact(
        self, apy: float, capital: float, chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Estimate impact of gas costs on yield.

        Assumptions:
          - Ethereum: ~100k gas total (deposit + withdraw), ~$20-50 at normal gas
          - L2s (Arbitrum, Optimism, Base): ~$0.10-1.00 per tx
          - Break-even: gas < 10% of annual yield

        Args:
            apy:      Annual percentage yield (e.g. 5.0 for 5%)
            capital:  Capital amount in USD
            chain:    Chain name

        Returns:
            dict with gas costs, break-even capital, net yield
        """
        # Gas cost estimates per round trip (deposit + 1 year rebalance + withdraw)
        gas_costs: Dict[str, float] = {
            "ethereum": 150.0,   # $150 total round trip
            "arbitrum": 3.0,
            "optimism": 3.0,
            "base": 1.5,
            "polygon": 0.5,
            "bsc": 1.0,
            "avalanche": 2.0,
        }

        chain_lower = chain.lower()
        gas_usd = gas_costs.get(chain_lower, 50.0)

        annual_yield = capital * apy / 100
        gas_impact_pct = gas_usd / annual_yield * 100 if annual_yield > 0 else float("inf")
        net_yield = annual_yield - gas_usd
        net_apy = (net_yield / capital * 100) if capital > 0 else 0

        # Break-even: where gas = 10% of annual yield → capital = gas / (apy × 0.10)
        break_even_capital = gas_usd / (apy / 100 * 0.10) if apy > 0 else float("inf")

        return {
            "chain": chain,
            "apy": apy,
            "capital": capital,
            "annual_yield_gross": round(annual_yield, 2),
            "gas_cost_usd": gas_usd,
            "gas_impact_pct": round(gas_impact_pct, 1),
            "annual_yield_net": round(net_yield, 2),
            "net_apy": round(net_apy, 2),
            "break_even_capital_usd": round(break_even_capital, 0),
            "viable": capital >= break_even_capital,
            "recommendation": (
                "VIABLE" if capital >= break_even_capital
                else f"Increase capital to ${break_even_capital:,.0f} for gas efficiency"
            ),
        }


# ---------------------------------------------------------------------------
# BridgeFlowAnalyzer
# ---------------------------------------------------------------------------

class BridgeFlowAnalyzer:
    """
    Analyze cross-chain capital flows via bridge TVL and volume data.
    Detect capital rotation, chain momentum, and bridge stress signals.
    """

    def __init__(self) -> None:
        self._llama = DefiLlamaAdvancedClient()

    def get_bridge_rankings(self) -> "pd.DataFrame":
        """
        Rank bridges by TVL.

        Returns:
            DataFrame: [name, tvl, volume24h, chains, change7d, stress_signal]
        """
        df = self._llama.get_bridge_flows()
        if df.empty:
            return df

        # Add stress signal
        df["stress_signal"] = (
            (df["change7d"] < -20) & (df["volume24h"] > df["volume24h"].quantile(0.7))
        )
        return df

    def detect_capital_rotation(self, days: int = 7) -> Dict[str, Any]:
        """
        Detect which chains are gaining or losing TVL (capital rotation signals).

        Methodology:
          - Compare current TVL vs historical for major chains
          - Net gainer: TVL up >5% in period
          - Net loser: TVL down >5%
          - Migration signal: systematic Ethereum→L2 movement

        Args:
            days: Lookback period (used for context)

        Returns:
            dict with gainers, losers, dominant_flow
        """
        chains = ["Ethereum", "Arbitrum", "Base", "Optimism", "Polygon", "BNB", "Avalanche", "Solana"]
        chain_changes: Dict[str, float] = {}

        for chain in chains:
            try:
                history = self._llama.get_chain_tvl_history(chain)
                if history.empty or len(history) < 2:
                    continue
                latest = float(history.iloc[-1])
                prior = float(history.iloc[-min(days, len(history) - 1)])
                if prior > 0:
                    change_pct = (latest - prior) / prior * 100
                    chain_changes[chain] = round(change_pct, 2)
            except Exception as e:
                logger.debug("Chain TVL fetch failed for %s: %s", chain, e)

        if not chain_changes:
            return {"error": "Insufficient chain TVL data", "period_days": days}

        gainers = {k: v for k, v in chain_changes.items() if v > 2}
        losers = {k: v for k, v in chain_changes.items() if v < -2}

        # Detect L1 → L2 migration
        eth_loss = chain_changes.get("Ethereum", 0)
        l2_gains = [chain_changes.get(c, 0) for c in ["Arbitrum", "Base", "Optimism"]]
        l2_migration = eth_loss < -3 and any(g > 3 for g in l2_gains)

        dominant_flow = "STABLE"
        if l2_migration:
            dominant_flow = "ETH_TO_L2_MIGRATION"
        elif gainers and not losers:
            dominant_flow = "BROAD_INFLOW"
        elif losers and not gainers:
            dominant_flow = "BROAD_OUTFLOW"
        elif gainers:
            top_gainer = max(gainers.items(), key=lambda x: x[1])
            dominant_flow = f"ROTATION_TO_{top_gainer[0].upper()}"

        return {
            "period_days": days,
            "chain_changes": chain_changes,
            "gainers": gainers,
            "losers": losers,
            "dominant_flow": dominant_flow,
            "l2_migration_signal": l2_migration,
        }

    def compute_chain_net_flows(self, chain: str, days: int = 30) -> "pd.Series":
        """
        Compute net capital inflow to a chain over time.

        Returns:
            pd.Series of daily TVL changes (proxy for net flows)
        """
        history = self._llama.get_chain_tvl_history(chain)
        if history.empty or not HAS_PANDAS:
            return pd.Series(dtype=float) if HAS_PANDAS else {}  # type: ignore[return-value]

        history = history.tail(days)
        net_flows = history.diff().dropna()
        net_flows.name = f"{chain}_net_flow"
        return net_flows

    def detect_bridge_stress(self, bridge: str) -> bool:
        """
        Detect potential bridge stress: rapid TVL decline + high withdrawal volume.

        Signals:
          - TVL 7d change < -20%
          - High withdrawal relative to TVL

        Returns:
            True if stress signals detected
        """
        df = self.get_bridge_rankings()
        if df.empty:
            return False

        mask = df["name"].str.lower().str.contains(bridge.lower(), na=False)
        bridge_rows = df[mask]
        if bridge_rows.empty:
            return False

        row = bridge_rows.iloc[0]
        change_7d = float(row.get("change7d", 0))
        tvl = float(row.get("tvl", 0))
        vol_24h = float(row.get("volume24h", 0))

        # Stress: TVL declining fast AND withdrawal volume is high (>5% of TVL/day)
        tvl_decline = change_7d < -20
        high_outflow = tvl > 0 and vol_24h / tvl > 0.05

        is_stressed = bool(tvl_decline or high_outflow)

        if is_stressed:
            logger.warning(
                "Bridge stress detected for %s: 7d_change=%.1f%%, vol/TVL=%.2f%%",
                bridge, change_7d, vol_24h / tvl * 100 if tvl > 0 else 0
            )

        return is_stressed

    def get_cross_chain_summary(self) -> Dict[str, Any]:
        """Comprehensive cross-chain flow summary."""
        rankings = self.get_bridge_rankings()
        rotation = self.detect_capital_rotation(days=7)

        top_bridges = []
        if not rankings.empty:
            for _, row in rankings.head(10).iterrows():
                top_bridges.append({
                    "name": row["name"],
                    "tvl": row["tvl"],
                    "volume_24h": row["volume24h"],
                    "change_7d": row["change7d"],
                    "stress": bool(row.get("stress_signal", False)),
                })

        return {
            "top_bridges": top_bridges,
            "capital_rotation": rotation,
            "total_bridge_tvl": float(rankings["tvl"].sum()) if not rankings.empty else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# GovernanceTokenAnalyzer
# ---------------------------------------------------------------------------

class GovernanceTokenAnalyzer:
    """
    Analyze DeFi governance token valuation and emission pressure.
    Uses CoinGecko free tier for price/market data.
    """

    def __init__(self) -> None:
        self._llama = DefiLlamaAdvancedClient()

    def get_governance_token_metrics(self, protocol: str) -> Dict[str, Any]:
        """
        Fetch governance token price and valuation metrics.

        Metrics:
          - Price, market cap, FDV, 24h volume (CoinGecko)
          - FDV/TVL: if >> 1, token may be overvalued vs protocol utility
          - P/F ratio: FDV / annualized fees

        Args:
            protocol: Protocol slug (e.g. 'aave', 'uniswap')

        Returns:
            dict with price, mcap, fdv, fdv_tvl_ratio, pf_ratio
        """
        slug = protocol.lower().replace(" ", "-")
        cg_id = PROTOCOL_COINGECKO_IDS.get(slug)
        if not cg_id:
            # Try to find it
            for k, v in PROTOCOL_COINGECKO_IDS.items():
                if k in slug or slug in k:
                    cg_id = v
                    break

        if not cg_id:
            return {"error": f"No CoinGecko ID found for {protocol}", "protocol": protocol}

        url = f"{COINGECKO_BASE}/coins/{cg_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
        }
        data = _rate_limited_get(url, params)
        if not data:
            return {"error": "CoinGecko unavailable", "protocol": protocol}

        md = data.get("market_data", {})
        price = float((md.get("current_price") or {}).get("usd") or 0)
        mcap = float((md.get("market_cap") or {}).get("usd") or 0)
        fdv = float((md.get("fully_diluted_valuation") or {}).get("usd") or 0)
        vol_24h = float((md.get("total_volume") or {}).get("usd") or 0)

        # Get TVL from DefiLlama
        protocols_df = self._llama.get_protocols()
        tvl = 0.0
        if not protocols_df.empty:
            mask = protocols_df["slug"].str.lower().str.contains(slug, na=False)
            rows = protocols_df[mask]
            if not rows.empty:
                tvl = float(rows.iloc[0]["tvl"])

        # Get fees from DefiLlama
        fees_df = self._llama.get_all_protocol_fees()
        annual_fees = 0.0
        if not fees_df.empty:
            mask = fees_df["name"].str.lower().str.contains(slug.replace("-", " "), na=False)
            rows = fees_df[mask]
            if not rows.empty:
                annual_fees = float(rows.iloc[0]["totalFees24h"]) * 365

        fdv_tvl = fdv / tvl if tvl > 0 else float("inf")
        pf_ratio = fdv / annual_fees if annual_fees > 0 else float("inf")

        # FDV/TVL interpretation
        if fdv_tvl < 0.5:
            fdv_tvl_signal = "UNDERVALUED_vs_PROTOCOL"
        elif fdv_tvl < 2.0:
            fdv_tvl_signal = "FAIR_VALUE"
        elif fdv_tvl < 5.0:
            fdv_tvl_signal = "PREMIUM"
        else:
            fdv_tvl_signal = "HIGHLY_OVERVALUED_vs_TVL"

        return {
            "protocol": protocol,
            "coingecko_id": cg_id,
            "price_usd": price,
            "market_cap": mcap,
            "fdv": fdv,
            "volume_24h": vol_24h,
            "tvl": tvl,
            "fdv_tvl_ratio": round(fdv_tvl, 2),
            "fdv_tvl_signal": fdv_tvl_signal,
            "pf_ratio": round(pf_ratio, 1),
            "annual_fees_usd": annual_fees,
        }

    def detect_emission_pressure(self, protocol: str) -> Dict[str, Any]:
        """
        Assess token emission pressure and death spiral risk.

        Factors:
          - Annual emission rate (from TOKEN_EMISSION_SCHEDULES)
          - Token inflation vs TVL growth
          - High emission + declining TVL = death spiral risk

        Returns:
            dict with emission_rate, pressure_level, death_spiral_risk
        """
        slug = protocol.lower().replace(" ", "-")
        annual_inflation = 0.0
        for k, v in TOKEN_EMISSION_SCHEDULES.items():
            if k in slug or slug in k:
                annual_inflation = v
                break

        # Get TVL change
        protocols_df = self._llama.get_protocols()
        tvl_change_7d = 0.0
        if not protocols_df.empty:
            mask = protocols_df["slug"].str.lower().str.contains(slug, na=False)
            rows = protocols_df[mask]
            if not rows.empty:
                tvl_change_7d = float(rows.iloc[0]["change_7d"])

        # Pressure assessment
        if annual_inflation > 20:
            pressure = "CRITICAL"
        elif annual_inflation > 10:
            pressure = "HIGH"
        elif annual_inflation > 5:
            pressure = "MODERATE"
        else:
            pressure = "LOW"

        # Death spiral: high inflation AND TVL declining
        death_spiral = annual_inflation > 15 and tvl_change_7d < -10
        if annual_inflation > 25 and tvl_change_7d < -5:
            death_spiral = True

        return {
            "protocol": protocol,
            "annual_emission_rate_pct": annual_inflation,
            "emission_pressure": pressure,
            "tvl_change_7d_pct": tvl_change_7d,
            "death_spiral_risk": death_spiral,
            "death_spiral_signal": "ACTIVE" if death_spiral else "NONE",
            "notes": (
                "High token inflation dilutes holders. "
                "Combined with TVL decline creates unsustainable tokenomics."
                if death_spiral else
                "Token emission within acceptable range."
            ),
        }


# ---------------------------------------------------------------------------
# DeFiMarketMonitor
# ---------------------------------------------------------------------------

class DeFiMarketMonitor:
    """
    Top-level DeFi market monitoring and reporting.
    Orchestrates all sub-analyzers into unified dashboards and reports.
    """

    def __init__(self) -> None:
        self._llama = DefiLlamaAdvancedClient()
        self._scorer = ProtocolQualityScorer()
        self._lender = LendingProtocolAnalyzer()
        self._yield_engine = YieldOptimizerEngine()
        self._bridge = BridgeFlowAnalyzer()
        self._gov = GovernanceTokenAnalyzer()

    def get_defi_market_dashboard(self) -> Dict[str, Any]:
        """
        Full DeFi market dashboard.

        Returns:
            dict with TVL, fees, top protocols, yield opportunities, bridge flows
        """
        logger.info("Building DeFi market dashboard...")

        # Top protocols by TVL
        protocols = self._llama.get_protocols()
        top_protocols = []
        if not protocols.empty:
            for _, row in protocols.head(10).iterrows():
                top_protocols.append({
                    "name": row["name"],
                    "tvl": row["tvl"],
                    "change_7d_pct": row["change_7d"],
                    "chains": row["chains"],
                    "category": row["category"],
                })

        # Fee leaders
        fees_df = self._llama.get_all_protocol_fees()
        fee_leaders = []
        if not fees_df.empty:
            for _, row in fees_df.head(5).iterrows():
                fee_leaders.append({
                    "name": row["name"],
                    "fees_24h": row["totalFees24h"],
                    "revenue_24h": row["totalRevenue24h"],
                })

        # Top yields
        yields = self._yield_engine.get_risk_adjusted_yields(min_tvl=50_000_000, max_apy=25.0)
        top_yields = []
        if not yields.empty:
            for _, row in yields.head(5).iterrows():
                top_yields.append({
                    "project": row["project"],
                    "symbol": row["symbol"],
                    "chain": row["chain"],
                    "apy": row["apy"],
                    "risk_adj_apy": row["risk_adjusted_apy"],
                })

        # Bridge flows
        bridge_summary = self._bridge.get_cross_chain_summary()

        # DeFi health
        health = self.compute_defi_health_index()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "defi_health_index": health,
            "top_protocols_by_tvl": top_protocols,
            "fee_leaders": fee_leaders,
            "top_risk_adjusted_yields": top_yields,
            "bridge_summary": bridge_summary,
        }

    def get_sector_breakdown(self) -> "pd.DataFrame":
        """
        TVL breakdown by DeFi sector: DEX, Lending, CDP, Derivatives, Bridge.

        Returns:
            DataFrame with sector TVL aggregates
        """
        protocols = self._llama.get_protocols()
        if protocols.empty:
            return pd.DataFrame() if HAS_PANDAS else []  # type: ignore[return-value]

        sector_map = {
            "dexes": "DEX",
            "dex": "DEX",
            "lending": "Lending",
            "cdp": "CDP",
            "derivatives": "Derivatives",
            "bridge": "Bridge",
            "liquid staking": "Liquid_Staking",
            "yield": "Yield",
            "yield aggregator": "Yield",
            "options": "Derivatives",
        }

        protocols["sector"] = protocols["category"].str.lower().map(
            lambda c: next((v for k, v in sector_map.items() if k in str(c).lower()), "Other")
        )

        sector_tvl = (
            protocols.groupby("sector")["tvl"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        total_tvl = sector_tvl["tvl"].sum()
        sector_tvl["tvl_share_pct"] = (sector_tvl["tvl"] / total_tvl * 100).round(1)
        return sector_tvl

    def compute_defi_health_index(self) -> float:
        """
        Composite DeFi health metric (0-100):
          - TVL momentum (40%): 7d TVL change across top protocols
          - Fee sustainability (35%): revenue/TVL ratio for top protocols
          - Bridge health (25%): no stressed bridges

        Returns:
            Float 0-100 (higher = healthier ecosystem)
        """
        score_parts = []
        weights = []

        # 1. TVL momentum
        try:
            protocols = self._llama.get_protocols()
            if not protocols.empty:
                top20 = protocols.head(20)
                avg_change_7d = float(top20["change_7d"].mean())
                # Map: +10% → 90, 0% → 60, -10% → 30
                tvl_score = max(0.0, min(100.0, 60.0 + avg_change_7d * 3.0))
                score_parts.append(tvl_score)
                weights.append(0.40)
        except Exception:
            pass

        # 2. Fee sustainability
        try:
            fees_df = self._llama.get_protocol_revenue()
            if not fees_df.empty:
                top_fees = fees_df[fees_df["annualFees"] > 0].head(20)
                if not top_fees.empty:
                    avg_rev_tvl = float(top_fees["revenueFeeShare"].mean())
                    # >40% rev share = sustainable
                    fee_score = min(100.0, avg_rev_tvl * 200)
                    score_parts.append(fee_score)
                    weights.append(0.35)
        except Exception:
            pass

        # 3. Bridge health
        try:
            bridges = self._llama.get_bridge_flows()
            if not bridges.empty:
                stressed_pct = float((bridges["change7d"] < -20).mean())
                bridge_score = max(0.0, 100.0 - stressed_pct * 300)
                score_parts.append(bridge_score)
                weights.append(0.25)
        except Exception:
            pass

        if not score_parts:
            return 50.0  # Default neutral

        # Weighted average
        total_weight = sum(weights)
        health = sum(s * w for s, w in zip(score_parts, weights)) / total_weight
        return round(health, 1)

    def generate_weekly_report(self) -> str:
        """
        Generate a formatted narrative DeFi weekly report.

        Returns:
            Formatted string report
        """
        lines = ["=" * 70]
        lines.append("SENTINEL — DeFi Market Weekly Report")
        lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 70)

        # Health index
        health = self.compute_defi_health_index()
        health_label = (
            "HEALTHY" if health >= 70 else
            "NEUTRAL" if health >= 40 else
            "STRESSED"
        )
        lines.append(f"\nDeFi Health Index: {health:.1f}/100 [{health_label}]")

        # Sector breakdown
        try:
            sectors = self.get_sector_breakdown()
            lines.append("\n--- TVL by Sector ---")
            if HAS_PANDAS and not sectors.empty:
                for _, row in sectors.iterrows():
                    tvl_b = row["tvl"] / 1e9
                    lines.append(f"  {row['sector']:<20} ${tvl_b:>8.2f}B  ({row['tvl_share_pct']:.1f}%)")
        except Exception as e:
            lines.append(f"  [Sector data unavailable: {e}]")

        # Fee leaders
        try:
            fees_df = self._llama.get_all_protocol_fees()
            lines.append("\n--- Top Fee Generators (24h) ---")
            if not fees_df.empty:
                for _, row in fees_df.head(8).iterrows():
                    lines.append(
                        f"  {row['name']:<22} Fees: ${row['totalFees24h']:>12,.0f} "
                        f"Revenue: ${row['totalRevenue24h']:>10,.0f}"
                    )
        except Exception as e:
            lines.append(f"  [Fee data unavailable: {e}]")

        # Capital rotation
        try:
            rotation = self._bridge.detect_capital_rotation(days=7)
            lines.append(f"\n--- Capital Rotation ---")
            lines.append(f"  Dominant Flow: {rotation.get('dominant_flow', 'N/A')}")
            lines.append(f"  L2 Migration: {'YES' if rotation.get('l2_migration_signal') else 'NO'}")
            gainers = rotation.get("gainers", {})
            losers = rotation.get("losers", {})
            if gainers:
                lines.append(f"  Gaining Chains: {', '.join(f'{k}({v:+.1f}%)' for k, v in list(gainers.items())[:4])}")
            if losers:
                lines.append(f"  Losing Chains:  {', '.join(f'{k}({v:+.1f}%)' for k, v in list(losers.items())[:4])}")
        except Exception as e:
            lines.append(f"  [Rotation data unavailable: {e}]")

        # Risk alerts
        lines.append("\n--- Risk Alerts ---")
        try:
            bridges = self._llama.get_bridge_flows()
            if not bridges.empty:
                stressed = bridges[bridges["change7d"] < -20]
                if not stressed.empty:
                    for _, row in stressed.head(3).iterrows():
                        lines.append(
                            f"  [BRIDGE STRESS] {row['name']}: TVL 7d change {row['change7d']:+.1f}%"
                        )
                else:
                    lines.append("  No major bridge stress signals detected.")
        except Exception:
            lines.append("  Bridge stress check unavailable.")

        try:
            # Check for high-emission protocols
            for slug in list(TOKEN_EMISSION_SCHEDULES.keys())[:5]:
                emission = self._gov.detect_emission_pressure(slug)
                if emission.get("death_spiral_risk"):
                    lines.append(
                        f"  [EMISSION RISK] {slug}: {emission['annual_emission_rate_pct']:.0f}% annual inflation, "
                        f"TVL {emission['tvl_change_7d_pct']:+.1f}%"
                    )
        except Exception:
            pass

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main() -> None:
    """
    SENTINEL DeFi Analytics Demo:
      1. Score top 20 DeFi protocols
      2. Risk-adjusted yield ladder
      3. Aave cascade liquidation at -20% ETH
      4. Bridge flow analysis
      5. DeFi health report
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")

    _banner("SENTINEL — DeFi Analytics V3 (dim_107)")

    # 1. Protocol scoring
    _banner("1. Protocol Quality Scores — Top 20 by TVL")
    scorer = ProtocolQualityScorer()
    scores = scorer.score_top_protocols(top_n=20)
    if scores:
        print(f"  {'Protocol':<25} {'Score':>6}  {'Tier'}  {'TVL':>12}  {'Rev':>10}")
        print(f"  {'-'*25} {'-----':>6}  {'----'}  {'-----':>12}  {'---':>10}")
        for s in scores[:15]:
            det = s.details.get("tvl_detail", {})
            rev_det = s.details.get("revenue_detail", {})
            tvl = det.get("tvl", 0)
            rev = rev_det.get("annual_revenue", 0)
            print(
                f"  {s.protocol:<25} {s.composite:>6.1f}  [{s.tier}]  "
                f"${tvl/1e6:>9.0f}M  ${rev/1e6:>7.0f}M"
            )
    else:
        print("  [No protocol data available — check network]")

    # 2. Risk-adjusted yield ladder
    _banner("2. Risk-Adjusted Yield Ladder (min TVL $50M)")
    engine = YieldOptimizerEngine()
    yields = engine.get_risk_adjusted_yields(min_tvl=50_000_000, max_apy=30.0)
    if HAS_PANDAS and not yields.empty:
        top_yields = yields.head(15)
        print(f"  {'Protocol':<18} {'Pool':<20} {'Chain':<12} {'APY':>7}  {'Risk-Adj':>9}  {'IL'}")
        print(f"  {'-'*18} {'-'*20} {'-'*12} {'---':>7}  {'--------':>9}  {'--'}")
        for _, row in top_yields.iterrows():
            il = "YES" if row["il_risk"] == "yes" else " no"
            print(
                f"  {str(row['project']):<18} {str(row['symbol'])[:20]:<20} "
                f"{str(row['chain'])[:12]:<12} {row['apy']:>6.1f}%  "
                f"{row['risk_adjusted_apy']:>8.1f}%  {il}"
            )
    else:
        print("  [No yield data — check network]")

    # 3. Aave cascade liquidation at -20% ETH
    _banner("3. Aave Cascade Liquidation Risk — ETH -20%")
    lender = LendingProtocolAnalyzer()
    markets = lender.query_aave_markets()
    if HAS_PANDAS and not markets.empty:
        print(f"  Aave V3 markets loaded: {len(markets)} assets")
        high_util = lender.detect_high_utilization_risk(markets, threshold=0.80)
        print(f"  High utilization assets (>80%): {high_util or 'None'}")
        for _, row in markets.head(8).iterrows():
            util_pct = row["utilizationRate"] * 100
            flag = " [HIGH]" if util_pct > 80 else ""
            print(
                f"    {str(row['asset']):<10} Liq=${row['totalLiquidity']/1e6:>8.1f}M  "
                f"Borrows=${row['totalBorrows']/1e6:>7.1f}M  Util={util_pct:>5.1f}%{flag}"
            )
    else:
        print("  [Aave subgraph unavailable — using TVL approximation]")

    estimated_liq = lender.estimate_cascade_liquidation("aave", price_drop_pct=20.0)
    print(f"\n  ETH -20% cascade estimate: ~${estimated_liq/1e9:.2f}B liquidated from Aave TVL")

    # 4. Bridge flow analysis
    _banner("4. Bridge Flow Analysis")
    bridge_analyzer = BridgeFlowAnalyzer()
    rankings = bridge_analyzer.get_bridge_rankings()
    if HAS_PANDAS and not rankings.empty:
        print(f"  {'Bridge':<22} {'TVL':>12}  {'Vol 24h':>12}  {'7d%':>7}  {'Stress'}")
        print(f"  {'-'*22} {'---':>12}  {'-------':>12}  {'---':>7}  {'------'}")
        for _, row in rankings.head(10).iterrows():
            stress = "[!]" if row.get("stress_signal") else "   "
            print(
                f"  {str(row['name'])[:22]:<22} ${row['tvl']/1e6:>9.0f}M  "
                f"${row['volume24h']/1e6:>9.0f}M  {row['change7d']:>+6.1f}%  {stress}"
            )

    rotation = bridge_analyzer.detect_capital_rotation(days=7)
    print(f"\n  Capital flow: {rotation.get('dominant_flow', 'N/A')}")
    print(f"  L2 migration signal: {'YES' if rotation.get('l2_migration_signal') else 'NO'}")

    # 5. Weekly report
    _banner("5. DeFi Market Report")
    monitor = DeFiMarketMonitor()
    report = monitor.generate_weekly_report()
    print(report)


if __name__ == "__main__":
    main()
