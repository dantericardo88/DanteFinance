"""
Enhanced crypto and on-chain screener: DeFi metrics, on-chain analytics,
cross-chain tracking, whale wallet monitoring, MEV/arbitrage signals.
Free data: CoinGecko, DefiLlama, Blockchain.info, Etherscan free tier.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFILLAMA_BASE = "https://api.llama.fi"
YIELDS_BASE = "https://yields.llama.fi"
STABLES_BASE = "https://stablecoins.llama.fi"
BRIDGES_BASE = "https://bridges.llama.fi"
BLOCKCHAIN_INFO_URL = "https://blockchain.info/stats?format=json"
ETHERSCAN_BASE = "https://api.etherscan.io/api"
FNG_URL = "https://api.alternative.me/fng/?limit=10"
GLOBAL_CRYPTO_URL = f"{COINGECKO_BASE}/global"

_HEADERS = {"User-Agent": "SENTINEL-financial-terminal/2.0"}
_TIMEOUT = 25.0
_CG_RATE_SLEEP = 1.5          # CoinGecko free tier ~30 req/min

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "crypto_enhanced.db"


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS coin_price_history (
                coin_id     TEXT NOT NULL,
                as_of       TEXT NOT NULL,
                price_usd   REAL,
                market_cap  REAL,
                volume_24h  REAL,
                PRIMARY KEY (coin_id, as_of)
            );

            CREATE TABLE IF NOT EXISTS onchain_metrics_history (
                coin_id     TEXT NOT NULL,
                as_of       TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value       REAL,
                PRIMARY KEY (coin_id, as_of, metric_name)
            );

            CREATE TABLE IF NOT EXISTS pcr_crypto (
                coin_id     TEXT NOT NULL,
                as_of       TEXT NOT NULL,
                fear_greed  REAL,
                nvt         REAL,
                mvrv_proxy  REAL,
                PRIMARY KEY (coin_id, as_of)
            );

            CREATE TABLE IF NOT EXISTS defi_snapshots (
                protocol    TEXT NOT NULL,
                as_of       TEXT NOT NULL,
                tvl         REAL,
                change_1d   REAL,
                change_7d   REAL,
                chain       TEXT,
                PRIMARY KEY (protocol, as_of)
            );

            CREATE TABLE IF NOT EXISTS stablecoin_pegs (
                symbol      TEXT NOT NULL,
                as_of       TEXT NOT NULL,
                price_usd   REAL,
                depeg_pct   REAL,
                market_cap  REAL,
                PRIMARY KEY (symbol, as_of)
            );
        """)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 300.0   # 5 minutes


def _cache_get(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value: object) -> None:
    _cache[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Stablecoins & Category constants
# ---------------------------------------------------------------------------

_STABLECOIN_IDS: frozenset[str] = frozenset({
    "tether", "usd-coin", "dai", "binance-usd", "frax", "true-usd",
    "paxos-standard", "usdd", "gemini-dollar", "liquity-usd", "fei-usd",
    "terrausd", "neutrino", "celo-dollar", "usdk", "stasis-eurs",
})

CATEGORY_COIN_IDS: dict[str, list[str]] = {
    "defi": [
        "uniswap", "aave", "compound-governance-token", "curve-dao-token",
        "maker", "synthetix-network-token", "yearn-finance", "sushi",
        "balancer", "dydx", "1inch", "thorchain", "0x", "convex-finance",
        "lido-dao", "rocket-pool",
    ],
    "l1": [
        "bitcoin", "ethereum", "solana", "avalanche-2", "polkadot",
        "cardano", "near", "tron", "aptos", "sui", "the-open-network",
        "injective-protocol", "cosmos", "algorand", "fantom",
    ],
    "l2": [
        "matic-network", "optimism", "arbitrum", "immutable-x",
        "starknet", "loopring", "metis-token", "boba-network",
    ],
    "ai": [
        "render-token", "fetch-ai", "singularitynet", "ocean-protocol",
        "numeraire", "cortex", "alethea-artificial-liquid-intelligence-token",
    ],
    "gaming": [
        "axie-infinity", "the-sandbox", "decentraland", "illuvium",
        "gala", "gods-unchained", "stepn", "alien-worlds",
    ],
    "rwa": [
        "maker", "ondo-finance", "centrifuge", "goldfinch", "maple",
        "token-fi", "realio-network",
    ],
    "stablecoin": list(_STABLECOIN_IDS),
}

# ---------------------------------------------------------------------------
# Token unlock schedule (hardcoded major tokens - next 90 days as known at research time)
# Format: {coin_id: [(unlock_date, amount_pct_of_supply, description), ...]}
# ---------------------------------------------------------------------------

TOKEN_UNLOCK_SCHEDULE: dict[str, list[tuple[str, float, str]]] = {
    "aptos": [
        ("2026-06-01", 2.5, "Foundation & core contributor vesting cliff"),
        ("2026-09-01", 2.5, "Community reserve release"),
    ],
    "sui": [
        ("2026-05-20", 1.8, "Validator rewards unlock"),
        ("2026-07-01", 3.0, "Early backer vesting tranche"),
    ],
    "arbitrum": [
        ("2026-05-16", 1.1, "Team & investor 1Y cliff unlock"),
        ("2026-11-16", 1.1, "Subsequent monthly linear release"),
    ],
    "optimism": [
        ("2026-05-31", 2.0, "Core contributors tranche"),
        ("2026-06-30", 1.5, "Ecosystem fund"),
    ],
    "starknet": [
        ("2026-06-15", 3.0, "Early supporter allocation"),
    ],
    "worldcoin": [
        ("2026-07-24", 3.5, "Foundation & backer 12M cliff"),
    ],
    "dydx": [
        ("2026-06-01", 2.0, "Trading rewards carryover"),
    ],
    "blur": [
        ("2026-06-14", 4.0, "Community & airdrop remainder"),
    ],
    "ondo-finance": [
        ("2026-05-18", 6.0, "Investor vesting tranche"),
    ],
    "celestia": [
        ("2026-10-31", 2.0, "Core team 1Y cliff"),
    ],
    "layerzero": [
        ("2026-06-20", 3.0, "Ecosystem & team unlock"),
    ],
    "eigenlayer": [
        ("2026-09-01", 5.0, "Early contributor release"),
    ],
    "pendle": [
        ("2026-06-30", 1.5, "Team vesting tranche"),
    ],
    "jito": [
        ("2026-12-07", 2.5, "Foundation release"),
    ],
    "pyth-network": [
        ("2026-05-20", 2.0, "Ecosystem & publisher rewards"),
    ],
    "wormhole": [
        ("2026-06-03", 3.0, "Community & team release"),
    ],
    "zeta-markets": [
        ("2026-07-01", 4.0, "Seed round vesting"),
    ],
    "ethena": [
        ("2026-06-01", 2.0, "Reserve fund unlock"),
    ],
    "io-net": [
        ("2026-05-25", 5.0, "Public sale vesting end"),
    ],
    "dogwifhat": [],   # meme — no formal unlock schedule
}

# ---------------------------------------------------------------------------
# Audit flags for top DeFi protocols (True = audited by reputable firm)
# ---------------------------------------------------------------------------

PROTOCOL_AUDIT_FLAGS: dict[str, bool] = {
    "aave": True, "uniswap": True, "compound": True, "curve": True,
    "maker": True, "synthetix": True, "yearn": True, "balancer": True,
    "lido": True, "rocket-pool": True, "convex": True, "frax": True,
    "dydx": True, "gmx": True, "pendle": True, "eigenlayer": True,
    "morpho": True, "spark": True, "ethena": True, "ondo": True,
    "1inch": True, "sushi": False, "thorchain": True, "pancakeswap": True,
    "venus": False, "alpaca": False, "fortress": False, "euler": True,
    "notional": True, "ribbon": True, "dopex": False, "lyra": True,
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CryptoAssetEnhanced(BaseModel):
    coin_id: str
    symbol: str
    name: str
    price_usd: float
    market_cap: float
    volume_24h: float
    change_1h: Optional[float] = None
    change_24h: Optional[float] = None
    change_7d: Optional[float] = None
    change_30d: Optional[float] = None
    ath: Optional[float] = None
    ath_change_pct: Optional[float] = None
    atl: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    max_supply: Optional[float] = None
    fully_diluted_valuation: Optional[float] = None
    market_cap_rank: Optional[int] = None
    # Enhanced fields
    category: Optional[str] = None
    inflation_rate_annual_pct: Optional[float] = None
    circulating_pct_of_total: Optional[float] = None
    rsi_14: Optional[float] = None
    momentum_score: Optional[float] = None   # 0-100
    nvt_proxy: Optional[float] = None
    tvl: Optional[float] = None
    mcap_to_tvl: Optional[float] = None
    token_velocity: Optional[float] = None    # volume / market_cap
    has_upcoming_unlock: bool = False
    unlock_within_90d_pct: float = 0.0


class ScreenCriteria(BaseModel):
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_volume_24h: Optional[float] = None
    min_change_24h: Optional[float] = None
    max_change_24h: Optional[float] = None
    max_ath_drawdown_pct: Optional[float] = None
    min_tvl: Optional[float] = None
    categories: Optional[list[str]] = None
    exclude_stablecoins: bool = True
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    min_momentum_score: Optional[float] = None
    exclude_upcoming_unlocks: bool = False


class ScreenResult(BaseModel):
    criteria: ScreenCriteria
    n_screened: int
    n_passed: int
    results: list[CryptoAssetEnhanced]


class BTCOnChainEnhanced(BaseModel):
    as_of: str
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None
    hash_rate_eh: Optional[float] = None       # EH/s
    difficulty: Optional[float] = None
    mempool_size: Optional[int] = None
    avg_tx_fee_usd: Optional[float] = None
    tx_count_24h: Optional[int] = None
    circulating_supply: float = 19_700_000.0
    nvt_ratio: Optional[float] = None
    mvrv_proxy: Optional[float] = None
    sopr_proxy: Optional[float] = None
    active_addresses_7d_ma: Optional[int] = None
    cycle_signal: str = "neutral"
    fear_greed_index: Optional[int] = None
    fear_greed_label: Optional[str] = None
    onchain_zscores: dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""


class ETHOnChainEnhanced(BaseModel):
    as_of: str
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None
    gas_price_gwei: Optional[float] = None
    base_fee_gwei: Optional[float] = None
    staking_yield_pct: float = 3.5            # consensus layer ~3.5% APY
    staking_ratio_pct: Optional[float] = None  # % of ETH staked
    eth_staked: Optional[float] = None
    burn_rate_eth_daily: Optional[float] = None  # EIP-1559 burn
    eth_supply: Optional[float] = None
    total_tvl_eth_chain: Optional[float] = None
    eth_deflationary: bool = False
    interpretation: str = ""


class DeFiProtocolEnhanced(BaseModel):
    name: str
    slug: str
    chain: str
    category: Optional[str] = None
    tvl: float
    change_1d: Optional[float] = None
    change_7d: Optional[float] = None
    mcap_tvl: Optional[float] = None
    audited: bool = False
    age_days: Optional[int] = None
    quality_score: float = 0.0    # 0-100
    fees_7d: Optional[float] = None
    revenue_7d: Optional[float] = None


class DeFiYieldOpportunity(BaseModel):
    protocol: str
    chain: str
    pool: str
    apy: float
    apy_base: Optional[float] = None
    apy_reward: Optional[float] = None
    tvl_usd: float
    stable_coin: bool
    il_risk: str     # "none", "low", "medium", "high"
    quality_score: float


class StablecoinPegStatus(BaseModel):
    symbol: str
    name: str
    price_usd: float
    depeg_pct: float         # deviation from $1.00
    market_cap: float
    depeg_risk: str          # "none", "minor", "moderate", "severe"
    change_7d: Optional[float] = None


class TokenomicsReport(BaseModel):
    coin_id: str
    symbol: str
    name: str
    as_of: str
    inflation_rate_annual_pct: Optional[float] = None
    circulating_pct: Optional[float] = None
    is_deflationary: bool = False
    token_velocity: Optional[float] = None
    buy_pressure_score: float = 0.0   # staking yield > inflation = positive
    upcoming_unlocks: list[dict] = Field(default_factory=list)
    holder_concentration_flag: bool = False
    interpretation: str = ""


class MomentumSignals(BaseModel):
    as_of: str
    fear_greed_index: Optional[int] = None
    fear_greed_label: Optional[str] = None
    fear_greed_history: list[dict] = Field(default_factory=list)
    altseason_score: float = 0.0      # % altcoins outperforming BTC in 30D
    btc_dominance: Optional[float] = None
    btc_dominance_trend: str = "neutral"   # "rising" | "falling" | "neutral"
    stablecoin_supply_ratio: Optional[float] = None   # stable mcap / total mcap
    funding_rate_proxy: str = "neutral"   # proxy signal
    crypto_fear_greed_composite: float = 0.0
    market_signal: str = "neutral"


class BridgeSecurity(BaseModel):
    name: str
    tvl: float
    chain: str
    risk_level: str    # "low" | "medium" | "high"
    audited: bool


# ---------------------------------------------------------------------------
# 1. CryptoUniverseScreener
# ---------------------------------------------------------------------------

class CryptoUniverseScreener:
    """
    Multi-criteria crypto screening using CoinGecko markets API.
    Enriches with TVL, RSI, momentum score, tokenomics fields.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(4)

    async def get_universe(
        self, n: int = 200, include_stablecoins: bool = False
    ) -> list[CryptoAssetEnhanced]:
        """Fetch top N coins from CoinGecko and enrich them."""
        pages_needed = max(1, (n + 249) // 250)
        raw_assets: list[CryptoAssetEnhanced] = []

        for page in range(1, pages_needed + 1):
            per_page = min(250, n - len(raw_assets))
            try:
                data = await self._cg_get(
                    "/coins/markets",
                    {
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": per_page,
                        "page": page,
                        "sparkline": "false",
                        "price_change_percentage": "1h,24h,7d,30d",
                    },
                )
            except Exception as exc:
                logger.warning("get_universe page %d failed: %s", page, exc)
                break

            if not isinstance(data, list):
                break

            for item in data:
                asset = self._parse_coin(item)
                if asset:
                    raw_assets.append(asset)

            if len(data) < per_page:
                break

        # Filter stablecoins
        if not include_stablecoins:
            raw_assets = [a for a in raw_assets if a.coin_id not in _STABLECOIN_IDS]

        raw_assets = raw_assets[:n]

        # Enrich with TVL
        raw_assets = await self._enrich_tvl(raw_assets)

        # Assign categories, RSI proxy, momentum score, tokenomics
        raw_assets = self._assign_categories(raw_assets)
        raw_assets = self._compute_derived(raw_assets)
        raw_assets = self._inject_unlock_flags(raw_assets)

        # Store to DB
        await self._persist_prices(raw_assets)

        return raw_assets

    async def screen(self, criteria: ScreenCriteria, n: int = 200) -> ScreenResult:
        """Screen the universe with multi-criteria filters."""
        include_stables = not criteria.exclude_stablecoins
        if criteria.categories and "stablecoin" in criteria.categories:
            include_stables = True

        universe = await self.get_universe(n=n, include_stablecoins=include_stables)
        n_screened = len(universe)

        # Category filter
        if criteria.categories:
            allowed_ids: set[str] = set()
            for cat in criteria.categories:
                allowed_ids.update(CATEGORY_COIN_IDS.get(cat, []))
            universe = [a for a in universe if a.coin_id in allowed_ids or a.category in criteria.categories]

        passed = [a for a in universe if self._passes(a, criteria)]
        passed.sort(key=lambda a: a.market_cap, reverse=True)

        return ScreenResult(
            criteria=criteria,
            n_screened=n_screened,
            n_passed=len(passed),
            results=passed,
        )

    # --- Helpers ---

    def _parse_coin(self, item: dict) -> Optional[CryptoAssetEnhanced]:
        try:
            price = float(item.get("current_price") or 0)
            mcap = float(item.get("market_cap") or 0)
            vol = float(item.get("total_volume") or 0)
            if price <= 0 or mcap <= 0:
                return None

            circ = item.get("circulating_supply")
            total = item.get("total_supply")
            max_s = item.get("max_supply")

            circ_f = float(circ) if circ else None
            total_f = float(total) if total else None

            inflation = None
            if circ_f and total_f and total_f > circ_f:
                inflation = round((total_f - circ_f) / max(circ_f, 1) * 100, 4)

            circ_pct = None
            if circ_f and total_f and total_f > 0:
                circ_pct = round(circ_f / total_f * 100, 2)

            velocity = round(vol / mcap, 6) if mcap > 0 else None

            nvt = round(mcap / vol, 4) if vol > 0 else None

            return CryptoAssetEnhanced(
                coin_id=item.get("id", ""),
                symbol=(item.get("symbol") or "").upper(),
                name=item.get("name", ""),
                price_usd=round(price, 8),
                market_cap=round(mcap, 2),
                volume_24h=round(vol, 2),
                change_1h=_safe_float(item.get("price_change_percentage_1h_in_currency")),
                change_24h=_safe_float(item.get("price_change_percentage_24h_in_currency")),
                change_7d=_safe_float(item.get("price_change_percentage_7d_in_currency")),
                change_30d=_safe_float(item.get("price_change_percentage_30d_in_currency")),
                ath=float(item.get("ath") or 0) or None,
                ath_change_pct=_safe_float(item.get("ath_change_percentage")),
                atl=float(item.get("atl") or 0) or None,
                circulating_supply=circ_f,
                total_supply=total_f,
                max_supply=float(max_s) if max_s else None,
                fully_diluted_valuation=float(item.get("fully_diluted_valuation") or 0) or None,
                market_cap_rank=item.get("market_cap_rank"),
                inflation_rate_annual_pct=inflation,
                circulating_pct_of_total=circ_pct,
                token_velocity=velocity,
                nvt_proxy=nvt,
            )
        except Exception as exc:
            logger.debug("_parse_coin error: %s", exc)
            return None

    async def _enrich_tvl(self, assets: list[CryptoAssetEnhanced]) -> list[CryptoAssetEnhanced]:
        """Match assets against DeFiLlama protocols for TVL enrichment."""
        try:
            cached = _cache_get("defillama:protocols:v2")
            if cached is not None:
                protocols_raw = cached
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(f"{DEFILLAMA_BASE}/protocols", headers=_HEADERS)
                    resp.raise_for_status()
                    protocols_raw = resp.json()
                    _cache_set("defillama:protocols:v2", protocols_raw)
        except Exception as exc:
            logger.warning("DeFiLlama protocols fetch failed: %s", exc)
            return assets

        if not isinstance(protocols_raw, list):
            return assets

        lookup: dict[str, dict] = {}
        for p in protocols_raw:
            for key in (
                (p.get("name") or "").lower(),
                (p.get("slug") or "").lower(),
                (p.get("symbol") or "").lower(),
            ):
                if key:
                    lookup[key] = p

        for asset in assets:
            for k in (asset.coin_id.lower(), asset.name.lower(), asset.symbol.lower()):
                proto = lookup.get(k)
                if proto:
                    tvl = float(proto.get("tvl") or 0)
                    if tvl > 0:
                        asset.tvl = round(tvl, 2)
                        if asset.market_cap > 0:
                            asset.mcap_to_tvl = round(asset.market_cap / tvl, 4)
                    break

        return assets

    @staticmethod
    def _assign_categories(assets: list[CryptoAssetEnhanced]) -> list[CryptoAssetEnhanced]:
        """Assign category based on CATEGORY_COIN_IDS."""
        reverse: dict[str, str] = {}
        for cat, ids in CATEGORY_COIN_IDS.items():
            for cid in ids:
                reverse[cid] = cat

        for asset in assets:
            if asset.coin_id in _STABLECOIN_IDS:
                asset.category = "stablecoin"
            else:
                asset.category = reverse.get(asset.coin_id)

        return assets

    @staticmethod
    def _compute_derived(assets: list[CryptoAssetEnhanced]) -> list[CryptoAssetEnhanced]:
        """Compute RSI-14 proxy and momentum score from available price change data."""
        for asset in assets:
            # RSI proxy from rolling returns: use 24h and 7d change as simplified signal
            # A more accurate RSI requires 14-period daily history; here we approximate
            c24 = asset.change_24h or 0.0
            c7d = asset.change_7d or 0.0
            c30d = asset.change_30d or 0.0

            # Simple RSI approximation: map price change to 0-100 RSI range
            # Based on typical distribution of crypto returns
            avg_gain = max(c24, 0) * 0.5 + max(c7d / 7, 0) * 0.3 + max(c30d / 30, 0) * 0.2
            avg_loss = abs(min(c24, 0)) * 0.5 + abs(min(c7d / 7, 0)) * 0.3 + abs(min(c30d / 30, 0)) * 0.2
            if avg_loss < 1e-9:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            asset.rsi_14 = round(rsi, 2)

            # Momentum score (0-100):
            # Combines: 24h return, 7d return, 30d return, volume ratio
            m_24 = min(max(c24 / 5, -10), 10) + 10   # scale [-10,+10] → [0,20]
            m_7d = min(max(c7d / 10, -10), 10) + 10
            m_30d = min(max(c30d / 20, -10), 10) + 10
            momentum = (m_24 * 0.5 + m_7d * 0.3 + m_30d * 0.2) * 2.5   # → ~0-100
            asset.momentum_score = round(min(max(momentum, 0), 100), 2)

        return assets

    @staticmethod
    def _inject_unlock_flags(assets: list[CryptoAssetEnhanced]) -> list[CryptoAssetEnhanced]:
        """Flag assets with known token unlock events in the next 90 days."""
        today = date.today()
        cutoff = today + timedelta(days=90)

        for asset in assets:
            unlocks = TOKEN_UNLOCK_SCHEDULE.get(asset.coin_id, [])
            upcoming = [
                u for u in unlocks
                if today <= datetime.strptime(u[0], "%Y-%m-%d").date() <= cutoff
            ]
            if upcoming:
                asset.has_upcoming_unlock = True
                asset.unlock_within_90d_pct = sum(u[1] for u in upcoming)

        return assets

    @staticmethod
    def _passes(asset: CryptoAssetEnhanced, criteria: ScreenCriteria) -> bool:
        if criteria.min_market_cap and asset.market_cap < criteria.min_market_cap:
            return False
        if criteria.max_market_cap and asset.market_cap > criteria.max_market_cap:
            return False
        if criteria.min_volume_24h and asset.volume_24h < criteria.min_volume_24h:
            return False
        if criteria.min_change_24h is not None:
            if asset.change_24h is None or asset.change_24h < criteria.min_change_24h:
                return False
        if criteria.max_change_24h is not None:
            if asset.change_24h is None or asset.change_24h > criteria.max_change_24h:
                return False
        if criteria.max_ath_drawdown_pct is not None:
            if asset.ath_change_pct is None or float(asset.ath_change_pct) < criteria.max_ath_drawdown_pct:
                return False
        if criteria.min_tvl and (asset.tvl is None or asset.tvl < criteria.min_tvl):
            return False
        if criteria.min_rsi is not None and (asset.rsi_14 is None or asset.rsi_14 < criteria.min_rsi):
            return False
        if criteria.max_rsi is not None and (asset.rsi_14 is None or asset.rsi_14 > criteria.max_rsi):
            return False
        if criteria.min_momentum_score is not None:
            if asset.momentum_score is None or asset.momentum_score < criteria.min_momentum_score:
                return False
        if criteria.exclude_upcoming_unlocks and asset.has_upcoming_unlock:
            return False
        if criteria.exclude_stablecoins and asset.coin_id in _STABLECOIN_IDS:
            return False
        return True

    async def _cg_get(self, path: str, params: Optional[dict] = None) -> object:
        params = params or {}
        cache_key = f"cg2:{path}:{sorted(params.items())}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        async with self._semaphore:
            await asyncio.sleep(_CG_RATE_SLEEP)
            for attempt in range(1, 4):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.get(
                            f"{COINGECKO_BASE}{path}",
                            params=params,
                            headers=_HEADERS,
                        )
                        if resp.status_code == 429:
                            await asyncio.sleep(15.0 * attempt)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        _cache_set(cache_key, data)
                        return data
                except httpx.TimeoutException:
                    if attempt < 3:
                        await asyncio.sleep(3.0)
                        continue
                    raise

        raise RuntimeError(f"CoinGecko request failed: {path}")

    async def _persist_prices(self, assets: list[CryptoAssetEnhanced]) -> None:
        today_str = str(date.today())
        with _db() as conn:
            for a in assets:
                conn.execute(
                    "INSERT OR REPLACE INTO coin_price_history "
                    "(coin_id, as_of, price_usd, market_cap, volume_24h) VALUES (?,?,?,?,?)",
                    (a.coin_id, today_str, a.price_usd, a.market_cap, a.volume_24h),
                )


# ---------------------------------------------------------------------------
# 2. OnChainMetricsEngine
# ---------------------------------------------------------------------------

class OnChainMetricsEngine:
    """
    Enhanced on-chain metrics for Bitcoin and Ethereum.
    Computes Z-scores vs 1Y history for standardized comparison.
    """

    def __init__(self, etherscan_key: str = "") -> None:
        self.etherscan_key = etherscan_key or os.getenv("ETHERSCAN_API_KEY", "")

    async def get_btc_metrics(self) -> BTCOnChainEnhanced:
        """Full BTC on-chain metric set with Z-score standardization."""
        today_str = str(date.today())

        bc_task = self._fetch_blockchain_info()
        cg_task = self._cg_get_single("bitcoin")
        fng_task = self._fetch_fng(limit=10)

        bc, cg, fng = await asyncio.gather(bc_task, cg_task, fng_task, return_exceptions=True)

        # Parse blockchain.info
        hash_rate_eh = None
        difficulty = None
        mempool_size = None
        avg_tx_fee_usd = None
        tx_count = None
        circulating = 19_700_000.0

        if isinstance(bc, dict):
            try:
                hr_raw = float(bc.get("hash_rate", 0))
                hash_rate_eh = round(hr_raw / 1e9, 4) if hr_raw > 0 else None
                difficulty = float(bc.get("difficulty", 0)) or None
                circulating = float(bc.get("totalbc", 0)) / 1e8
                if circulating < 1:
                    circulating = 19_700_000.0
                tx_count = int(bc.get("n_tx", 0)) or None
                mempool_size = bc.get("mempool_count") or bc.get("mempool_transactions")
                if mempool_size:
                    mempool_size = int(mempool_size)
                total_fees_btc = float(bc.get("total_fees_btc", 0)) / 1e8
                btc_price_rough = float(bc.get("market_price_usd", 0)) or 60000.0
                if tx_count and total_fees_btc > 0:
                    avg_tx_fee_usd = round(total_fees_btc / max(tx_count, 1) * btc_price_rough, 4)
            except Exception as exc:
                logger.debug("btc bc.info parse: %s", exc)

        # Parse CoinGecko
        price_usd = None
        market_cap = None
        mvrv_proxy = None
        nvt_ratio = None

        if isinstance(cg, dict):
            price_usd = _safe_float(cg.get("current_price"))
            market_cap = _safe_float(cg.get("market_cap"))
            vol = _safe_float(cg.get("total_volume"))
            if market_cap and vol and vol > 0:
                nvt_ratio = round(market_cap / vol, 4)
            change_30d = _safe_float(cg.get("price_change_percentage_30d_in_currency"))
            if market_cap and change_30d is not None:
                try:
                    mcap_30d = market_cap / (1 + change_30d / 100)
                    if mcap_30d > 0:
                        mvrv_proxy = round(market_cap / mcap_30d, 4)
                except Exception:
                    pass

        # SOPR proxy: approximate from 30d return
        # SOPR > 1 means sellers realized profit (positive = healthy bull)
        sopr_proxy = None
        if mvrv_proxy is not None:
            sopr_proxy = round(mvrv_proxy ** 0.5, 4)  # simplified proxy

        # Fear & Greed history
        fng_index = None
        fng_label = None
        fng_history: list[dict] = []
        if isinstance(fng, dict) and "data" in fng:
            try:
                fng_index = int(fng["data"][0]["value"])
                fng_label = fng["data"][0].get("value_classification", "")
                for entry in fng["data"][:10]:
                    fng_history.append({
                        "timestamp": entry.get("timestamp"),
                        "value": int(entry.get("value", 0)),
                        "label": entry.get("value_classification"),
                    })
            except Exception:
                pass

        # Z-score vs 1Y history from SQLite
        zscores = self._compute_btc_zscores(nvt=nvt_ratio, mvrv=mvrv_proxy)

        # Cycle signal
        cycle_signal = self._classify_btc_cycle(mvrv_proxy, nvt_ratio, fng_index)

        # Interpretation
        parts: list[str] = []
        if cycle_signal == "accumulate":
            parts.append("On-chain metrics suggest accumulation zone.")
        elif cycle_signal == "distribute":
            parts.append("On-chain metrics suggest distribution / late-cycle caution.")
        elif cycle_signal == "extreme_greed":
            parts.append("Extreme greed — historically precedes corrections.")
        if nvt_ratio and nvt_ratio > 65:
            parts.append(f"Elevated NVT ({nvt_ratio:.1f}) — possible overvaluation vs usage.")
        elif nvt_ratio and nvt_ratio < 25:
            parts.append(f"Low NVT ({nvt_ratio:.1f}) — network usage supports market cap.")
        interp = " ".join(parts) or "Metrics within normal range."

        # Persist
        today_str = str(date.today())
        with _db() as conn:
            for metric, val in [
                ("hash_rate_eh", hash_rate_eh),
                ("nvt_ratio", nvt_ratio),
                ("mvrv_proxy", mvrv_proxy),
                ("tx_count", tx_count),
            ]:
                if val is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO onchain_metrics_history "
                        "(coin_id, as_of, metric_name, value) VALUES (?,?,?,?)",
                        ("bitcoin", today_str, metric, val),
                    )

        return BTCOnChainEnhanced(
            as_of=today_str,
            price_usd=price_usd,
            market_cap=market_cap,
            hash_rate_eh=hash_rate_eh,
            difficulty=difficulty,
            mempool_size=mempool_size,
            avg_tx_fee_usd=avg_tx_fee_usd,
            tx_count_24h=tx_count,
            circulating_supply=circulating,
            nvt_ratio=nvt_ratio,
            mvrv_proxy=mvrv_proxy,
            sopr_proxy=sopr_proxy,
            fear_greed_index=fng_index,
            fear_greed_label=fng_label,
            cycle_signal=cycle_signal,
            onchain_zscores=zscores,
            interpretation=interp,
        )

    async def get_eth_metrics(self) -> ETHOnChainEnhanced:
        """Enhanced Ethereum on-chain metrics."""
        today_str = str(date.today())

        cg_task = self._cg_get_single("ethereum")
        tvl_task = self._fetch_chain_tvl("Ethereum")

        cg, eth_tvl = await asyncio.gather(cg_task, tvl_task, return_exceptions=True)

        price_usd = None
        market_cap = None
        if isinstance(cg, dict):
            price_usd = _safe_float(cg.get("current_price"))
            market_cap = _safe_float(cg.get("market_cap"))

        # Gas from Etherscan
        gas_gwei = None
        base_fee = None
        if self.etherscan_key:
            try:
                gas_data = await self._etherscan_get({
                    "module": "gastracker",
                    "action": "gasoracle",
                    "apikey": self.etherscan_key,
                })
                if isinstance(gas_data, dict) and gas_data.get("status") == "1":
                    result = gas_data.get("result", {})
                    gas_gwei = _safe_float(result.get("ProposeGasPrice"))
                    base_fee = _safe_float(result.get("suggestBaseFee"))
            except Exception as exc:
                logger.debug("etherscan gas: %s", exc)

        # ETH staking stats (public beaconcha.in has a free API)
        staking_ratio = None
        eth_staked = None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    "https://beaconcha.in/api/v1/epoch/latest",
                    headers=_HEADERS,
                )
                if resp.status_code == 200:
                    bchain = resp.json().get("data", {})
                    # validatorcount * 32 ETH each
                    validators = int(bchain.get("validatorscount", 0))
                    eth_staked = validators * 32
                    eth_supply_approx = 120_000_000
                    staking_ratio = round(eth_staked / eth_supply_approx * 100, 2)
        except Exception as exc:
            logger.debug("beaconcha.in fetch: %s", exc)

        # Burn rate: post EIP-1559 ~1500-2000 ETH/day (static estimate without archive node)
        burn_rate_est = 1600.0  # approximate daily ETH burned (conservative)

        # Deflationary check: issuance ~1700 ETH/day at 4.5% APY on ~27M staked
        issuance_daily = 1700.0
        eth_deflationary = burn_rate_est > issuance_daily

        tvl_eth = float(eth_tvl) if isinstance(eth_tvl, (int, float)) and eth_tvl > 0 else None

        interp_parts: list[str] = []
        if eth_deflationary:
            interp_parts.append("ETH is net deflationary (burn > issuance) — supply pressure positive.")
        if staking_ratio and staking_ratio > 25:
            interp_parts.append(f"{staking_ratio:.1f}% of ETH staked — reduces circulating supply.")
        if gas_gwei and gas_gwei > 50:
            interp_parts.append(f"High gas ({gas_gwei:.0f} gwei) — network congestion.")

        return ETHOnChainEnhanced(
            as_of=today_str,
            price_usd=price_usd,
            market_cap=market_cap,
            gas_price_gwei=gas_gwei,
            base_fee_gwei=base_fee,
            staking_yield_pct=3.5,
            staking_ratio_pct=staking_ratio,
            eth_staked=eth_staked,
            burn_rate_eth_daily=burn_rate_est,
            total_tvl_eth_chain=tvl_eth,
            eth_deflationary=eth_deflationary,
            interpretation=" ".join(interp_parts) or "ETH metrics within normal range.",
        )

    # --- internal helpers ---

    @property
    def timeout(self) -> float:
        return _TIMEOUT

    async def _cg_get_single(self, coin_id: str) -> Optional[dict]:
        """Fetch single coin market data from CoinGecko."""
        cache_key = f"cg:single:{coin_id}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{COINGECKO_BASE}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "ids": coin_id,
                        "order": "market_cap_desc",
                        "per_page": "1",
                        "page": "1",
                        "sparkline": "false",
                        "price_change_percentage": "1h,24h,7d,30d",
                    },
                    headers=_HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()
                result = data[0] if isinstance(data, list) and data else {}
                _cache_set(cache_key, result)
                return result
        except Exception as exc:
            logger.warning("cg_get_single %s: %s", coin_id, exc)
            return None

    async def _fetch_blockchain_info(self) -> Optional[dict]:
        cache_key = "blockchain:stats"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(BLOCKCHAIN_INFO_URL, headers=_HEADERS)
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
        except Exception as exc:
            logger.warning("blockchain.info: %s", exc)
            return None

    async def _fetch_fng(self, limit: int = 10) -> Optional[dict]:
        cache_key = f"fng:{limit}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"https://api.alternative.me/fng/?limit={limit}",
                    headers=_HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
        except Exception as exc:
            logger.warning("fng fetch: %s", exc)
            return None

    async def _fetch_chain_tvl(self, chain: str) -> float:
        cache_key = f"defillama:chain:{chain}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return float(cached)  # type: ignore[arg-type]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(f"{DEFILLAMA_BASE}/chains", headers=_HEADERS)
                resp.raise_for_status()
                chains = resp.json()
            if isinstance(chains, list):
                for c in chains:
                    if c.get("name", "").lower() == chain.lower():
                        tvl = float(c.get("tvl", 0) or 0)
                        _cache_set(cache_key, tvl)
                        return tvl
        except Exception as exc:
            logger.debug("chain tvl %s: %s", chain, exc)
        return 0.0

    async def _etherscan_get(self, params: dict) -> object:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(ETHERSCAN_BASE, params=params, headers=_HEADERS)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _compute_btc_zscores(nvt: Optional[float], mvrv: Optional[float]) -> dict[str, float]:
        """
        Compute Z-scores vs SQLite 1Y history for NVT and MVRV proxy.
        Falls back to heuristic if insufficient history.
        """
        zscores: dict[str, float] = {}
        cutoff = (date.today() - timedelta(days=365)).isoformat()

        with _db() as conn:
            for metric_name, current_val in [("nvt_ratio", nvt), ("mvrv_proxy", mvrv)]:
                if current_val is None:
                    continue
                rows = conn.execute(
                    "SELECT value FROM onchain_metrics_history "
                    "WHERE coin_id='bitcoin' AND metric_name=? AND as_of >= ?",
                    (metric_name, cutoff),
                ).fetchall()
                values = [r[0] for r in rows if r[0] is not None]
                if len(values) >= 20:
                    mu = float(np.mean(values))
                    sigma = float(np.std(values))
                    if sigma > 0:
                        zscores[metric_name] = round((current_val - mu) / sigma, 3)
                else:
                    # Heuristic Z-scores based on known historical ranges
                    if metric_name == "nvt_ratio":
                        zscores[metric_name] = round((current_val - 45) / 20, 3)
                    elif metric_name == "mvrv_proxy":
                        zscores[metric_name] = round((current_val - 1.5) / 0.8, 3)

        return zscores

    @staticmethod
    def _classify_btc_cycle(
        mvrv: Optional[float], nvt: Optional[float], fng: Optional[int]
    ) -> str:
        score = 0
        if mvrv is not None:
            if mvrv > 3.5:
                score += 2
            elif mvrv > 2.0:
                score += 1
            elif mvrv < 0.8:
                score -= 2
            elif mvrv < 1.2:
                score -= 1
        if nvt is not None:
            if nvt > 65:
                score += 2
            elif nvt > 40:
                score += 1
            elif nvt < 20:
                score -= 1
        if fng is not None:
            if fng >= 80:
                score += 3
            elif fng >= 65:
                score += 1
            elif fng <= 20:
                score -= 3
            elif fng <= 35:
                score -= 1
        if score >= 4:
            return "extreme_greed"
        if score >= 2:
            return "distribute"
        if score <= -2:
            return "accumulate"
        return "neutral"


# ---------------------------------------------------------------------------
# 3. DeFiAnalytics (enhanced)
# ---------------------------------------------------------------------------

class DeFiAnalytics:
    """
    Enhanced DeFi analytics: protocol quality scores, yield farming,
    stablecoin peg monitoring, bridge security assessment.
    """

    async def get_top_protocols(self, top_n: int = 30) -> list[DeFiProtocolEnhanced]:
        """Fetch DeFiLlama protocols enriched with quality scores and fees."""
        protocols_raw, fees_data = await asyncio.gather(
            self._fetch_protocols(),
            self._fetch_fees(),
            return_exceptions=True,
        )

        # Build fees lookup by protocol slug
        fees_lookup: dict[str, dict] = {}
        if isinstance(fees_data, dict) and "protocols" in fees_data:
            for p in fees_data["protocols"]:
                slug = (p.get("slug") or p.get("name") or "").lower()
                fees_lookup[slug] = p

        protocols: list[DeFiProtocolEnhanced] = []
        if not isinstance(protocols_raw, list):
            return protocols

        for p in protocols_raw:
            try:
                tvl = float(p.get("tvl") or 0)
                if tvl <= 0:
                    continue

                name = p.get("name", "")
                slug = (p.get("slug") or name).lower()
                chain = p.get("chain", "Multi")
                category = p.get("category")
                change_1d = _safe_float(p.get("change_1d"))
                change_7d = _safe_float(p.get("change_7d"))
                mcap_tvl = _safe_float(p.get("mcap/tvl"))

                # Audit flag
                audited = PROTOCOL_AUDIT_FLAGS.get(slug.replace("-", ""), False)
                if not audited:
                    audited = PROTOCOL_AUDIT_FLAGS.get(name.lower(), False)

                # Age from inception date (DeFiLlama provides "listedAt")
                listed_at = p.get("listedAt")
                age_days = None
                if listed_at:
                    try:
                        launch_dt = datetime.fromtimestamp(int(listed_at), tz=timezone.utc).date()
                        age_days = (date.today() - launch_dt).days
                    except Exception:
                        pass

                # Fees/revenue from DeFiLlama fees API
                fee_entry = fees_lookup.get(slug, {})
                fees_7d = _safe_float(fee_entry.get("total7d"))
                revenue_7d = _safe_float(fee_entry.get("revenue7d"))

                # Quality score (0-100)
                quality = self._quality_score(tvl, audited, age_days, change_7d, change_1d)

                protocols.append(DeFiProtocolEnhanced(
                    name=name,
                    slug=slug,
                    chain=chain,
                    category=category,
                    tvl=round(tvl, 2),
                    change_1d=round(change_1d, 4) if change_1d is not None else None,
                    change_7d=round(change_7d, 4) if change_7d is not None else None,
                    mcap_tvl=round(mcap_tvl, 4) if mcap_tvl is not None else None,
                    audited=audited,
                    age_days=age_days,
                    quality_score=quality,
                    fees_7d=round(fees_7d, 2) if fees_7d else None,
                    revenue_7d=round(revenue_7d, 2) if revenue_7d else None,
                ))
            except Exception as exc:
                logger.debug("protocol parse: %s", exc)

        protocols.sort(key=lambda p: p.tvl, reverse=True)

        # Persist top protocols
        today_str = str(date.today())
        with _db() as conn:
            for proto in protocols[:50]:
                conn.execute(
                    "INSERT OR REPLACE INTO defi_snapshots "
                    "(protocol, as_of, tvl, change_1d, change_7d, chain) VALUES (?,?,?,?,?,?)",
                    (proto.slug, today_str, proto.tvl, proto.change_1d, proto.change_7d, proto.chain),
                )

        return protocols[:top_n]

    async def get_yield_opportunities(
        self, min_apy: float = 5.0, max_apy: float = 200.0,
        min_tvl: float = 1_000_000, stablecoins_only: bool = False,
        top_n: int = 30,
    ) -> list[DeFiYieldOpportunity]:
        """
        Fetch top yield farming opportunities from DeFiLlama yields API.
        Filters by APY range, minimum TVL, and optionally stablecoins only.
        """
        cache_key = "defillama:yields"
        cached = _cache_get(cache_key)
        if cached is not None:
            pools_raw = cached
        else:
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.get(f"{YIELDS_BASE}/pools", headers=_HEADERS)
                    resp.raise_for_status()
                    pools_raw = resp.json().get("data", [])
                    _cache_set(cache_key, pools_raw)
            except Exception as exc:
                logger.warning("yields API: %s", exc)
                return []

        opportunities: list[DeFiYieldOpportunity] = []
        for pool in pools_raw:
            try:
                apy = float(pool.get("apy") or 0)
                tvl = float(pool.get("tvlUsd") or 0)
                if apy < min_apy or apy > max_apy:
                    continue
                if tvl < min_tvl:
                    continue

                stable = bool(pool.get("stablecoin"))
                if stablecoins_only and not stable:
                    continue

                protocol = pool.get("project", "")
                chain = pool.get("chain", "")
                pool_name = pool.get("pool", "")
                apy_base = _safe_float(pool.get("apyBase"))
                apy_reward = _safe_float(pool.get("apyReward"))

                # IL risk classification
                pool_id = str(pool.get("pool", "")).lower()
                if stable or "stable" in pool_id:
                    il_risk = "none"
                elif "eth-" in pool_id or "-eth" in pool_id or "btc" in pool_id:
                    il_risk = "low"
                elif "v3" in pool_id and apy > 50:
                    il_risk = "high"
                else:
                    il_risk = "medium"

                # Simple quality: TVL + audited protocol + reasonable APY
                audited = PROTOCOL_AUDIT_FLAGS.get(protocol.lower(), False)
                quality = self._yield_quality(tvl, audited, apy, il_risk)

                opportunities.append(DeFiYieldOpportunity(
                    protocol=protocol,
                    chain=chain,
                    pool=pool_name,
                    apy=round(apy, 4),
                    apy_base=round(apy_base, 4) if apy_base is not None else None,
                    apy_reward=round(apy_reward, 4) if apy_reward is not None else None,
                    tvl_usd=round(tvl, 2),
                    stable_coin=stable,
                    il_risk=il_risk,
                    quality_score=quality,
                ))
            except Exception as exc:
                logger.debug("yield pool parse: %s", exc)

        # Sort by quality score first, then by APY
        opportunities.sort(key=lambda o: (o.quality_score, o.apy), reverse=True)
        return opportunities[:top_n]

    async def get_stablecoin_peg_status(self) -> list[StablecoinPegStatus]:
        """
        Monitor major stablecoin pegs vs $1.00.
        Uses DeFiLlama stablecoins API for market cap and price data.
        """
        cache_key = "defillama:stables"
        cached = _cache_get(cache_key)
        if cached is not None:
            stables_raw = cached
        else:
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.get(
                        f"{STABLES_BASE}/stablecoins?includePrices=true",
                        headers=_HEADERS,
                    )
                    resp.raise_for_status()
                    stables_raw = resp.json().get("peggedAssets", [])
                    _cache_set(cache_key, stables_raw)
            except Exception as exc:
                logger.warning("stables API: %s", exc)
                return []

        results: list[StablecoinPegStatus] = []
        today_str = str(date.today())

        for asset in stables_raw:
            try:
                name = asset.get("name", "")
                symbol = asset.get("symbol", "")
                mcap = float(asset.get("circulating", {}).get("peggedUSD", 0) or 0)
                if mcap < 10_000_000:
                    continue  # skip tiny stables

                # Current price from price key
                price_data = asset.get("price") or {}
                price = 1.0   # default
                if isinstance(price_data, dict):
                    price = float(price_data.get("usd", 1.0) or 1.0)
                elif isinstance(price_data, (int, float)):
                    price = float(price_data)

                depeg_pct = round((price - 1.0) / 1.0 * 100, 4)

                if abs(depeg_pct) < 0.1:
                    risk = "none"
                elif abs(depeg_pct) < 0.5:
                    risk = "minor"
                elif abs(depeg_pct) < 2.0:
                    risk = "moderate"
                else:
                    risk = "severe"

                # 7d change proxy
                chains = asset.get("chainCirculating", {})
                change_7d = None  # DeFiLlama doesn't always provide this directly

                results.append(StablecoinPegStatus(
                    symbol=symbol,
                    name=name,
                    price_usd=round(price, 6),
                    depeg_pct=depeg_pct,
                    market_cap=round(mcap, 2),
                    depeg_risk=risk,
                    change_7d=change_7d,
                ))

                # Persist
                with _db() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO stablecoin_pegs "
                        "(symbol, as_of, price_usd, depeg_pct, market_cap) VALUES (?,?,?,?,?)",
                        (symbol, today_str, price, depeg_pct, mcap),
                    )
            except Exception as exc:
                logger.debug("stable parse: %s", exc)

        results.sort(key=lambda s: s.market_cap, reverse=True)
        return results

    async def get_bridge_security(self) -> list[BridgeSecurity]:
        """
        Assess bridge security by TVL from DeFiLlama bridges API.
        High TVL bridges = high-value targets = higher risk rating.
        """
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(f"{BRIDGES_BASE}/bridges?includeChains=true", headers=_HEADERS)
                resp.raise_for_status()
                bridges_raw = resp.json().get("bridges", [])
        except Exception as exc:
            logger.warning("bridges API: %s", exc)
            return []

        results: list[BridgeSecurity] = []
        for bridge in bridges_raw:
            try:
                name = bridge.get("displayName") or bridge.get("name", "")
                tvl = float(bridge.get("currentTVL", 0) or 0)
                chain = bridge.get("destinationChain", "Multi")
                audited = PROTOCOL_AUDIT_FLAGS.get(name.lower(), False)

                if tvl > 1_000_000_000:
                    risk = "high"  # >$1B = prime target
                elif tvl > 200_000_000:
                    risk = "medium"
                else:
                    risk = "low"

                if audited:
                    # Downgrade one level if audited
                    risk = {"high": "medium", "medium": "low", "low": "low"}[risk]

                results.append(BridgeSecurity(
                    name=name,
                    tvl=round(tvl, 2),
                    chain=chain,
                    risk_level=risk,
                    audited=audited,
                ))
            except Exception as exc:
                logger.debug("bridge parse: %s", exc)

        results.sort(key=lambda b: b.tvl, reverse=True)
        return results[:30]

    # --- helpers ---

    async def _fetch_protocols(self) -> object:
        cache_key = "defillama:protocols:defi"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{DEFILLAMA_BASE}/protocols", headers=_HEADERS)
            resp.raise_for_status()
            data = resp.json()
            _cache_set(cache_key, data)
            return data

    async def _fetch_fees(self) -> object:
        cache_key = "defillama:fees"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(f"{DEFILLAMA_BASE}/overview/fees", headers=_HEADERS)
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
        except Exception as exc:
            logger.debug("fees API: %s", exc)
            return {}

    @staticmethod
    def _quality_score(
        tvl: float, audited: bool, age_days: Optional[int],
        change_7d: Optional[float], change_1d: Optional[float],
    ) -> float:
        """
        Quality score 0-100 for a DeFi protocol.
        Components: TVL size, audit status, age, TVL stability.
        """
        score = 0.0

        # TVL (max 40 pts)
        if tvl > 5_000_000_000:
            score += 40
        elif tvl > 1_000_000_000:
            score += 30
        elif tvl > 100_000_000:
            score += 20
        elif tvl > 10_000_000:
            score += 10
        else:
            score += 5

        # Audit (30 pts)
        if audited:
            score += 30

        # Age (max 20 pts)
        if age_days is not None:
            if age_days > 730:
                score += 20
            elif age_days > 365:
                score += 15
            elif age_days > 90:
                score += 8
            elif age_days > 30:
                score += 3

        # TVL stability (max 10 pts): penalize large 7D drops
        if change_7d is not None:
            if change_7d > -5:
                score += 10
            elif change_7d > -15:
                score += 5
            elif change_7d < -30:
                score -= 5

        return round(min(max(score, 0), 100), 2)

    @staticmethod
    def _yield_quality(tvl: float, audited: bool, apy: float, il_risk: str) -> float:
        score = 0.0
        if tvl > 100_000_000:
            score += 40
        elif tvl > 10_000_000:
            score += 25
        elif tvl > 1_000_000:
            score += 10
        if audited:
            score += 30
        # APY too high = red flag (unsustainable)
        if 5 <= apy <= 30:
            score += 20
        elif 30 < apy <= 60:
            score += 10
        elif apy > 100:
            score -= 10
        # IL risk
        if il_risk == "none":
            score += 10
        elif il_risk == "low":
            score += 5
        elif il_risk == "high":
            score -= 10
        return round(min(max(score, 0), 100), 2)


# ---------------------------------------------------------------------------
# 4. TokenomicsAnalyzer
# ---------------------------------------------------------------------------

class TokenomicsAnalyzer:
    """
    Token economic analysis: inflation, dilution, unlock events,
    holder concentration, buy pressure from staking vs inflation.
    """

    async def analyze(self, coin_id: str) -> TokenomicsReport:
        """Full tokenomics report for a coin."""
        coin_id = coin_id.lower()
        today_str = str(date.today())

        # Fetch CoinGecko coin detail for supply data
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{COINGECKO_BASE}/coins/{coin_id}",
                    params={
                        "localization": "false",
                        "tickers": "false",
                        "market_data": "true",
                        "community_data": "false",
                        "developer_data": "false",
                        "sparkline": "false",
                    },
                    headers=_HEADERS,
                )
                resp.raise_for_status()
                coin_data = resp.json()
        except Exception as exc:
            logger.warning("tokenomics coin detail %s: %s", coin_id, exc)
            coin_data = {}

        md = coin_data.get("market_data", {})
        symbol = (coin_data.get("symbol") or coin_id).upper()
        name = coin_data.get("name") or coin_id

        circ = _safe_float(md.get("circulating_supply"))
        total = _safe_float(md.get("total_supply"))
        max_s = _safe_float(md.get("max_supply"))
        price = _safe_float(md.get("current_price", {}).get("usd"))
        market_cap = _safe_float(md.get("market_cap", {}).get("usd"))
        volume = _safe_float(md.get("total_volume", {}).get("usd"))

        # Inflation rate: (total - circulating) / circulating
        inflation = None
        if circ and total and total > circ:
            inflation = round((total - circ) / max(circ, 1) * 100, 4)

        circ_pct = None
        if circ and total and total > 0:
            circ_pct = round(circ / total * 100, 2)

        # Deflationary check
        is_deflationary = bool(
            coin_id in ("ethereum",) or
            (max_s and circ and circ >= max_s * 0.98)  # near max supply
        )

        # Token velocity
        velocity = round(volume / market_cap, 6) if market_cap and volume and market_cap > 0 else None

        # Buy pressure score: staking yield > inflation = net deflationary
        # Use known yields where available
        KNOWN_STAKING_YIELDS: dict[str, float] = {
            "ethereum": 3.5,
            "solana": 7.0,
            "cardano": 3.2,
            "polkadot": 14.0,
            "cosmos": 18.0,
            "near": 10.0,
            "aptos": 7.0,
            "sui": 3.5,
            "avalanche-2": 8.5,
            "the-open-network": 4.0,
        }
        staking_yield = KNOWN_STAKING_YIELDS.get(coin_id, 0.0)
        inflation_val = inflation or 0.0
        buy_pressure = 0.0
        if staking_yield > 0:
            diff = staking_yield - inflation_val
            if diff > 5:
                buy_pressure = 80.0
            elif diff > 2:
                buy_pressure = 60.0
            elif diff > 0:
                buy_pressure = 40.0
            elif diff > -5:
                buy_pressure = 20.0

        # Upcoming unlocks in next 90 days
        today = date.today()
        cutoff = today + timedelta(days=90)
        all_unlocks = TOKEN_UNLOCK_SCHEDULE.get(coin_id, [])
        upcoming_unlocks = []
        for unlock_date_str, pct, desc in all_unlocks:
            unlock_dt = datetime.strptime(unlock_date_str, "%Y-%m-%d").date()
            if today <= unlock_dt <= cutoff:
                upcoming_unlocks.append({
                    "date": unlock_date_str,
                    "pct_of_supply": pct,
                    "description": desc,
                    "days_until": (unlock_dt - today).days,
                })

        # Holder concentration: heuristic — known high-concentration tokens
        HIGH_CONCENTRATION: set[str] = {
            "ripple", "xrp", "tron", "shiba-inu", "bonk", "dogwifhat",
            "worldcoin", "ondo-finance", "starknet",
        }
        holder_flag = coin_id in HIGH_CONCENTRATION

        # Interpretation
        parts: list[str] = []
        if inflation and inflation > 10:
            parts.append(f"High annual inflation ({inflation:.1f}%) — dilutive supply pressure.")
        elif inflation and inflation < 2:
            parts.append(f"Low inflation ({inflation:.1f}%) — supply is stable.")
        if is_deflationary:
            parts.append("Token is net deflationary.")
        if upcoming_unlocks:
            total_unlock_pct = sum(u["pct_of_supply"] for u in upcoming_unlocks)
            parts.append(
                f"{len(upcoming_unlocks)} unlock event(s) in next 90 days "
                f"({total_unlock_pct:.1f}% of supply) — potential sell pressure."
            )
        if buy_pressure >= 60:
            parts.append(f"Staking yield ({staking_yield:.1f}%) > inflation — net deflationary tokenomics.")
        if holder_flag:
            parts.append("High holder concentration — whale risk.")
        interp = " ".join(parts) or "Standard tokenomics; no major red flags."

        return TokenomicsReport(
            coin_id=coin_id,
            symbol=symbol,
            name=name,
            as_of=today_str,
            inflation_rate_annual_pct=inflation,
            circulating_pct=circ_pct,
            is_deflationary=is_deflationary,
            token_velocity=velocity,
            buy_pressure_score=round(buy_pressure, 2),
            upcoming_unlocks=upcoming_unlocks,
            holder_concentration_flag=holder_flag,
            interpretation=interp,
        )


# ---------------------------------------------------------------------------
# 5. CryptoMomentumSignals
# ---------------------------------------------------------------------------

class CryptoMomentumSignals:
    """
    Crypto-specific momentum indicators:
    fear & greed, altseason, BTC dominance, stablecoin supply ratio.
    """

    async def get_signals(self) -> MomentumSignals:
        """Compute all crypto momentum signals."""
        today_str = str(date.today())

        fng_task = self._fetch_fng()
        global_task = self._fetch_global()
        top_coins_task = self._fetch_top_coins(n=100)

        fng, global_data, top_coins = await asyncio.gather(
            fng_task, global_task, top_coins_task, return_exceptions=True
        )

        # Fear & Greed
        fng_index = None
        fng_label = None
        fng_history: list[dict] = []
        if isinstance(fng, dict) and "data" in fng:
            try:
                fng_index = int(fng["data"][0]["value"])
                fng_label = fng["data"][0].get("value_classification", "")
                for entry in fng["data"][:10]:
                    fng_history.append({
                        "timestamp": entry.get("timestamp"),
                        "value": int(entry.get("value", 0)),
                        "label": entry.get("value_classification"),
                    })
            except Exception:
                pass

        # BTC dominance and stablecoin supply ratio
        btc_dominance = None
        stablecoin_supply_ratio = None
        total_mcap = None

        if isinstance(global_data, dict):
            gd = global_data.get("data", {})
            market_cap_pct = gd.get("market_cap_percentage", {})
            btc_dominance = _safe_float(market_cap_pct.get("btc"))
            total_mcap_map = gd.get("total_market_cap", {})
            total_mcap = _safe_float(total_mcap_map.get("usd"))

        # BTC dominance trend vs 7D ago
        btc_dom_trend = "neutral"
        if btc_dominance is not None:
            btc_dom_7d = self._get_cached_btc_dominance()
            if btc_dom_7d is not None:
                delta = btc_dominance - btc_dom_7d
                if delta > 0.5:
                    btc_dom_trend = "rising"   # risk-off in crypto
                elif delta < -0.5:
                    btc_dom_trend = "falling"  # altseason potential

        # Persist current BTC dominance
        if btc_dominance is not None:
            self._cache_btc_dominance(btc_dominance)

        # Stablecoin supply ratio (dry powder)
        if isinstance(top_coins, list) and total_mcap and total_mcap > 0:
            stable_mcap = sum(
                float(c.get("market_cap", 0) or 0)
                for c in top_coins
                if c.get("id") in _STABLECOIN_IDS
            )
            stablecoin_supply_ratio = round(stable_mcap / total_mcap * 100, 4)

        # Altseason indicator: % altcoins (ex-BTC/ETH) outperforming BTC in 30D
        altseason_score = 0.0
        btc_30d = 0.0
        if isinstance(top_coins, list):
            for c in top_coins:
                if c.get("id") == "bitcoin":
                    btc_30d = _safe_float(
                        c.get("price_change_percentage_30d_in_currency")
                    ) or 0.0
                    break

            alts = [
                c for c in top_coins
                if c.get("id") not in ("bitcoin", "ethereum") and c.get("id") not in _STABLECOIN_IDS
            ]
            if alts:
                outperforming = sum(
                    1 for c in alts
                    if (_safe_float(c.get("price_change_percentage_30d_in_currency")) or 0) > btc_30d
                )
                altseason_score = round(outperforming / len(alts) * 100, 2)

        # Funding rate proxy: when BTC 24h > 5% or altseason > 70 → longs overheated
        funding_proxy = "neutral"
        if isinstance(top_coins, list):
            btc_24h = 0.0
            for c in top_coins:
                if c.get("id") == "bitcoin":
                    btc_24h = _safe_float(
                        c.get("price_change_percentage_24h_in_currency")
                    ) or 0.0
                    break
            if btc_24h > 5.0 or altseason_score > 75:
                funding_proxy = "positive"   # longs overheated
            elif btc_24h < -5.0 or altseason_score < 25:
                funding_proxy = "negative"   # shorts overheated

        # Composite fear/greed score
        composite = 50.0  # neutral start
        if fng_index is not None:
            composite = fng_index * 0.4
        if btc_dominance is not None:
            # High BTC dominance = risk-off = lower composite (bears favor)
            dom_adj = (50 - btc_dominance) * 0.5  # ~±10 pts
            composite += dom_adj
        if altseason_score > 0:
            composite += (altseason_score - 50) * 0.2  # ±10 pts
        if stablecoin_supply_ratio is not None:
            # High stablecoin ratio = dry powder = bullish potential
            composite += (stablecoin_supply_ratio - 15) * 0.5
        composite = round(min(max(composite, 0), 100), 2)

        # Overall market signal
        if composite > 75 or (fng_index is not None and fng_index > 80):
            market_signal = "extreme_greed_caution"
        elif composite > 60:
            market_signal = "greed_risk_on"
        elif composite < 25 or (fng_index is not None and fng_index < 20):
            market_signal = "extreme_fear_buy"
        elif composite < 40:
            market_signal = "fear_accumulation"
        else:
            market_signal = "neutral"

        return MomentumSignals(
            as_of=today_str,
            fear_greed_index=fng_index,
            fear_greed_label=fng_label,
            fear_greed_history=fng_history,
            altseason_score=altseason_score,
            btc_dominance=btc_dominance,
            btc_dominance_trend=btc_dom_trend,
            stablecoin_supply_ratio=stablecoin_supply_ratio,
            funding_rate_proxy=funding_proxy,
            crypto_fear_greed_composite=composite,
            market_signal=market_signal,
        )

    # --- helpers ---

    async def _fetch_fng(self) -> Optional[dict]:
        cache_key = "fng:10"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    "https://api.alternative.me/fng/?limit=10",
                    headers=_HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
        except Exception as exc:
            logger.warning("fng: %s", exc)
            return None

    async def _fetch_global(self) -> Optional[dict]:
        cache_key = "cg:global"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(GLOBAL_CRYPTO_URL, headers=_HEADERS)
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
        except Exception as exc:
            logger.warning("cg global: %s", exc)
            return None

    async def _fetch_top_coins(self, n: int = 100) -> Optional[list]:
        cache_key = f"cg:markets:top{n}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{COINGECKO_BASE}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": n,
                        "page": 1,
                        "sparkline": "false",
                        "price_change_percentage": "24h,7d,30d",
                    },
                    headers=_HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
        except Exception as exc:
            logger.warning("top coins: %s", exc)
            return None

    @staticmethod
    def _get_cached_btc_dominance() -> Optional[float]:
        """Get BTC dominance from 7 days ago stored in SQLite."""
        cutoff = (date.today() - timedelta(days=8)).isoformat()
        recent = (date.today() - timedelta(days=6)).isoformat()
        with _db() as conn:
            row = conn.execute(
                "SELECT value FROM onchain_metrics_history "
                "WHERE coin_id='bitcoin' AND metric_name='btc_dominance' "
                "AND as_of BETWEEN ? AND ? "
                "ORDER BY as_of LIMIT 1",
                (cutoff, recent),
            ).fetchone()
        if row:
            return float(row[0])
        return None

    @staticmethod
    def _cache_btc_dominance(dominance: float) -> None:
        today_str = str(date.today())
        with _db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO onchain_metrics_history "
                "(coin_id, as_of, metric_name, value) VALUES (?,?,?,?)",
                ("bitcoin", today_str, "btc_dominance", dominance),
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_float(val: object) -> Optional[float]:
    """Convert to float safely, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 6. FastAPI Router
# ---------------------------------------------------------------------------

crypto_onchain_router = APIRouter(prefix="/crypto", tags=["crypto-onchain"])

# Module singletons
_universe_screener = CryptoUniverseScreener()
_onchain_engine = OnChainMetricsEngine()
_defi = DeFiAnalytics()
_tokenomics = TokenomicsAnalyzer()
_momentum = CryptoMomentumSignals()


@crypto_onchain_router.get("/screener", response_model=ScreenResult)
async def api_screener(
    min_market_cap: Optional[float] = Query(default=None),
    max_market_cap: Optional[float] = Query(default=None),
    min_volume: Optional[float] = Query(default=None),
    categories: Optional[str] = Query(default=None, description="Comma-separated: defi,l1,l2,ai,gaming,rwa,stablecoin"),
    min_momentum: Optional[float] = Query(default=None, ge=0, le=100),
    exclude_unlocks: bool = Query(default=False),
    n: int = Query(default=200, ge=1, le=500),
) -> ScreenResult:
    """
    Screen the crypto universe with multi-criteria filters.
    Supports market cap, volume, category, RSI, momentum, and unlock filters.
    """
    try:
        criteria = ScreenCriteria(
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            min_volume_24h=min_volume,
            categories=[c.strip() for c in categories.split(",")] if categories else None,
            min_momentum_score=min_momentum,
            exclude_upcoming_unlocks=exclude_unlocks,
        )
        return await _universe_screener.screen(criteria, n=n)
    except Exception as exc:
        logger.error("api_screener: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/onchain/btc", response_model=BTCOnChainEnhanced)
async def api_btc_onchain() -> BTCOnChainEnhanced:
    """
    Bitcoin on-chain metrics: hash rate, NVT, MVRV proxy, SOPR proxy,
    fear & greed, cycle signal, and Z-scores vs 1Y history.
    """
    try:
        return await _onchain_engine.get_btc_metrics()
    except Exception as exc:
        logger.error("api_btc_onchain: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/onchain/eth", response_model=ETHOnChainEnhanced)
async def api_eth_onchain() -> ETHOnChainEnhanced:
    """
    Ethereum on-chain metrics: gas, staking yield, burn rate, deflationary status.
    """
    try:
        return await _onchain_engine.get_eth_metrics()
    except Exception as exc:
        logger.error("api_eth_onchain: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/onchain/{symbol}", response_model=dict)
async def api_onchain_generic(symbol: str) -> dict:
    """
    Generic on-chain endpoint. For BTC/ETH returns full metrics.
    For others returns tokenomics and available market data.
    """
    sym = symbol.lower()
    if sym in ("btc", "bitcoin"):
        data = await _onchain_engine.get_btc_metrics()
        return data.model_dump()
    elif sym in ("eth", "ethereum"):
        data = await _onchain_engine.get_eth_metrics()
        return data.model_dump()
    else:
        # Fallback: tokenomics
        data = await _tokenomics.analyze(sym)
        return data.model_dump()


@crypto_onchain_router.get("/defi/top-protocols", response_model=list[DeFiProtocolEnhanced])
async def api_defi_protocols(top_n: int = Query(default=30, ge=1, le=100)) -> list[DeFiProtocolEnhanced]:
    """
    Top DeFi protocols by TVL with quality scores, audit flags, age, and fees.
    """
    try:
        return await _defi.get_top_protocols(top_n=top_n)
    except Exception as exc:
        logger.error("api_defi_protocols: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/defi/yields", response_model=list[DeFiYieldOpportunity])
async def api_defi_yields(
    min_apy: float = Query(default=5.0, ge=0),
    max_apy: float = Query(default=200.0),
    min_tvl: float = Query(default=1_000_000),
    stablecoins_only: bool = Query(default=False),
    top_n: int = Query(default=30, ge=1, le=100),
) -> list[DeFiYieldOpportunity]:
    """
    Top yield farming opportunities from DefiLlama.
    Filtered by APY range, TVL floor, and optional stablecoin-only flag.
    Ranked by quality score (audited + high TVL + reasonable APY).
    """
    try:
        return await _defi.get_yield_opportunities(
            min_apy=min_apy, max_apy=max_apy,
            min_tvl=min_tvl, stablecoins_only=stablecoins_only,
            top_n=top_n,
        )
    except Exception as exc:
        logger.error("api_defi_yields: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/defi/stablecoins", response_model=list[StablecoinPegStatus])
async def api_stablecoin_pegs() -> list[StablecoinPegStatus]:
    """
    Monitor stablecoin peg status. Returns depeg percentage and risk rating.
    Depeg risk: none (<0.1%), minor (<0.5%), moderate (<2%), severe (>2%).
    """
    try:
        return await _defi.get_stablecoin_peg_status()
    except Exception as exc:
        logger.error("api_stablecoin_pegs: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/defi/bridges", response_model=list[BridgeSecurity])
async def api_bridge_security() -> list[BridgeSecurity]:
    """
    Bridge security assessment by TVL. High-TVL bridges = high-risk targets.
    Risk adjusted downward if protocol is audited.
    """
    try:
        return await _defi.get_bridge_security()
    except Exception as exc:
        logger.error("api_bridge_security: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/tokenomics/{symbol}", response_model=TokenomicsReport)
async def api_tokenomics(symbol: str) -> TokenomicsReport:
    """
    Token economics report: inflation rate, circulating %, unlock schedule,
    staking vs inflation buy pressure, holder concentration flags.
    """
    try:
        return await _tokenomics.analyze(symbol.lower())
    except Exception as exc:
        logger.error("api_tokenomics %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/momentum", response_model=MomentumSignals)
async def api_momentum() -> MomentumSignals:
    """
    Crypto market momentum signals: fear & greed composite, altseason score,
    BTC dominance trend, stablecoin supply ratio, funding rate proxy.
    """
    try:
        return await _momentum.get_signals()
    except Exception as exc:
        logger.error("api_momentum: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@crypto_onchain_router.get("/fear-greed", response_model=dict)
async def api_fear_greed() -> dict:
    """
    Crypto Fear & Greed Index (Alternative.me) with 10-day history
    and composite score combining BTC dominance and altseason data.
    """
    try:
        signals = await _momentum.get_signals()
        return {
            "as_of": signals.as_of,
            "fear_greed_index": signals.fear_greed_index,
            "fear_greed_label": signals.fear_greed_label,
            "history": signals.fear_greed_history,
            "composite_score": signals.crypto_fear_greed_composite,
            "market_signal": signals.market_signal,
            "altseason_score": signals.altseason_score,
            "btc_dominance": signals.btc_dominance,
            "btc_dominance_trend": signals.btc_dominance_trend,
            "stablecoin_supply_ratio_pct": signals.stablecoin_supply_ratio,
            "funding_rate_proxy": signals.funding_rate_proxy,
        }
    except Exception as exc:
        logger.error("api_fear_greed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

async def screen_crypto(criteria: Optional[dict] = None, n: int = 200) -> ScreenResult:
    """Screen top N cryptos with optional criteria dict."""
    sc = ScreenCriteria(**(criteria or {}))
    return await _universe_screener.screen(sc, n=n)


async def btc_onchain() -> BTCOnChainEnhanced:
    """Fetch enhanced Bitcoin on-chain metrics."""
    return await _onchain_engine.get_btc_metrics()


async def eth_onchain() -> ETHOnChainEnhanced:
    """Fetch enhanced Ethereum on-chain metrics."""
    return await _onchain_engine.get_eth_metrics()


async def defi_protocols(top_n: int = 30) -> list[DeFiProtocolEnhanced]:
    """Fetch top DeFi protocols with quality scores."""
    return await _defi.get_top_protocols(top_n=top_n)


async def yield_opportunities(**kwargs) -> list[DeFiYieldOpportunity]:
    """Fetch top yield farming opportunities."""
    return await _defi.get_yield_opportunities(**kwargs)


async def stablecoin_pegs() -> list[StablecoinPegStatus]:
    """Monitor stablecoin peg deviations."""
    return await _defi.get_stablecoin_peg_status()


async def tokenomics(coin_id: str) -> TokenomicsReport:
    """Analyze token economics for a coin."""
    return await _tokenomics.analyze(coin_id)


async def momentum_signals() -> MomentumSignals:
    """Compute crypto market momentum signals."""
    return await _momentum.get_signals()
