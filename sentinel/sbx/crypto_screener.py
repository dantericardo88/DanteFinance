"""
Comprehensive crypto asset screener combining on-chain and market data — dim_075.

Free data sources (no paid key required):
  CoinGecko   — markets, coin detail, historical, trending, DeFi global
  DeFiLlama   — protocol TVL, chain breakdown
  Blockchain.info — Bitcoin network stats
  Alternative.me  — Fear & Greed Index
  Etherscan   — ETH gas / network (optional key via ETHERSCAN_API_KEY env var)

Implements:
  CryptoScreener.get_top_cryptos()
  CryptoScreener.screen()
  CryptoScreener.get_trending()
  CryptoScreener.get_btc_onchain()
  CryptoScreener.get_eth_network()
  CryptoScreener.get_defi_overview()
  CryptoScreener.momentum_screen()
  CryptoScreener.value_screen()

Module-level helpers:
  top_cryptos(), screen_crypto(), btc_onchain(), eth_network(), defi_overview()
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFILLAMA_BASE = "https://api.llama.fi"
BLOCKCHAIN_INFO = "https://blockchain.info/stats?format=json"
ETHERSCAN_BASE = "https://api.etherscan.io/api"
FNG_URL = "https://api.alternative.me/fng/?limit=1"

_HEADERS = {"User-Agent": "SENTINEL financial-terminal/1.0"}
_TIMEOUT = 20.0
_CG_RATE_SLEEP = 1.3   # CoinGecko free tier: ~30 req/min; 1.3 s between calls is safe

# Known stablecoin CoinGecko IDs (to support exclude_stablecoins filter)
_STABLECOIN_IDS: frozenset[str] = frozenset({
    "tether", "usd-coin", "dai", "binance-usd", "frax", "true-usd",
    "paxos-standard", "usdd", "gemini-dollar", "liquity-usd", "fei-usd",
    "terrausd", "neutrino", "tribe-2", "celo-dollar", "usdk", "stasis-eurs",
    "usdt-avalanche-bridged-usdt-e", "bridged-usdc-polygon-pos-bridge",
})

# Screen category presets (CoinGecko coin IDs)
SCREEN_CATEGORIES: dict[str, list[str]] = {
    "layer1": [
        "bitcoin", "ethereum", "solana", "avalanche-2", "polkadot",
        "cardano", "near", "tron", "aptos", "sui", "the-open-network",
        "injective-protocol", "cosmos", "algorand",
    ],
    "defi": [
        "uniswap", "aave", "compound-governance-token", "curve-dao-token",
        "maker", "synthetix-network-token", "yearn-finance", "sushi",
        "balancer", "dydx", "1inch", "thorchain", "0x",
    ],
    "stablecoin": [
        "tether", "usd-coin", "dai", "frax", "binance-usd", "true-usd",
    ],
    "layer2": [
        "matic-network", "optimism", "arbitrum", "immutable-x",
        "starknet", "zksync", "loopring",
    ],
    "meme": [
        "dogecoin", "shiba-inu", "pepe", "floki", "bonk",
    ],
    "infrastructure": [
        "chainlink", "the-graph", "filecoin", "arweave", "helium",
        "livepeer", "storj", "render-token",
    ],
}

# Stress periods used in historical correlation / scenario analysis
# (referenced here for potential future use — not directly used in this file)
_STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "2018_crypto_winter": ("2017-12-18", "2018-12-15"),
    "covid_crash_2020":   ("2020-02-20", "2020-03-23"),
    "2022_bear":          ("2021-11-10", "2022-11-21"),
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CryptoAsset(BaseModel):
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
    # On-chain / DeFi
    tvl: Optional[float] = None
    protocol_revenue_24h: Optional[float] = None
    active_addresses_24h: Optional[int] = None
    transaction_count_24h: Optional[int] = None
    # Derived
    nvt_proxy: Optional[float] = None       # market_cap / volume_24h
    mcap_to_tvl: Optional[float] = None    # market_cap / TVL (DeFi valuation ratio)


class CryptoScreenCriteria(BaseModel):
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_volume_24h: Optional[float] = None
    min_change_24h: Optional[float] = None
    max_change_24h: Optional[float] = None
    max_ath_drawdown_pct: Optional[float] = None   # e.g. -50 means max 50% from ATH
    min_tvl: Optional[float] = None
    categories: Optional[list[str]] = None
    exclude_stablecoins: bool = True


class CryptoScreenResult(BaseModel):
    criteria: CryptoScreenCriteria
    n_screened: int
    n_passed: int
    results: list[CryptoAsset]


class BTCOnChainMetrics(BaseModel):
    as_of: date
    # Network health
    hash_rate: Optional[float] = None       # EH/s (exahashes per second)
    difficulty: Optional[float] = None
    mempool_size: Optional[int] = None      # number of unconfirmed transactions
    avg_tx_fee_usd: Optional[float] = None
    tx_count_24h: Optional[int] = None
    # Valuation
    market_cap: Optional[float] = None
    nvt_ratio: Optional[float] = None       # market_cap / on-chain tx volume proxy
    mvrv_proxy: Optional[float] = None      # market_cap / 30d-ago market_cap proxy
    # Supply
    circulating_supply: float
    # Cycle signals
    fear_greed_index: Optional[int] = None   # 0-100 from Alternative.me
    fear_greed_classification: Optional[str] = None
    # Derived signal
    cycle_signal: str = "neutral"   # "accumulate", "neutral", "distribute", "extreme_greed"


class ETHNetworkMetrics(BaseModel):
    as_of: date
    gas_price_gwei: Optional[float] = None
    base_fee_gwei: Optional[float] = None
    daily_transactions: Optional[int] = None
    eth_supply: Optional[float] = None
    staking_apy_proxy: Optional[float] = None   # estimate from public data
    # DeFi context
    total_value_locked: Optional[float] = None   # ETH chain TVL from DeFiLlama
    # Market
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None


class DeFiProtocol(BaseModel):
    name: str
    slug: str
    chain: str
    category: Optional[str] = None
    tvl: float
    change_1d: Optional[float] = None
    change_7d: Optional[float] = None
    mcap_tvl: Optional[float] = None


class DeFiOverview(BaseModel):
    as_of: date
    total_tvl: float
    total_tvl_change_1d: Optional[float] = None
    top_protocols: list[DeFiProtocol]
    chain_breakdown: dict[str, float]          # chain_name → TVL
    defi_dominance_pct: Optional[float] = None  # DeFi TVL / total crypto market cap
    total_crypto_market_cap: Optional[float] = None


# ---------------------------------------------------------------------------
# In-memory cache (TTL-based)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 300.0   # 5 minutes


def _cache_get(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if datetime.now(timezone.utc).timestamp() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value: object) -> None:
    _cache[key] = (datetime.now(timezone.utc).timestamp(), value)


# ---------------------------------------------------------------------------
# CryptoScreener
# ---------------------------------------------------------------------------

class CryptoScreener:
    """
    Comprehensive crypto screener combining CoinGecko market data with
    DeFiLlama TVL, Blockchain.info Bitcoin stats, and Etherscan ETH metrics.

    All I/O is async. Rate-limiting is handled via an internal semaphore
    and sleep-based throttling for the CoinGecko free tier.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout
        self._etherscan_key = os.getenv("ETHERSCAN_API_KEY", "")
        # CoinGecko free tier: max ~30 req/min, enforce max 5 concurrent
        self._rate_limiter = asyncio.Semaphore(5)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def get_top_cryptos(
        self, n: int = 100, include_stablecoins: bool = False
    ) -> list[CryptoAsset]:
        """
        Fetch the top N cryptos by market cap from CoinGecko /coins/markets.
        Enriches DeFi tokens with TVL from DeFiLlama.

        Args:
            n: Number of assets to return (max 250 per CoinGecko page).
            include_stablecoins: If False, filter out known stablecoins.

        Returns:
            List of CryptoAsset sorted by market_cap_rank.
        """
        pages_needed = max(1, (n + 249) // 250)
        assets: list[CryptoAsset] = []

        for page in range(1, pages_needed + 1):
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": min(250, n - len(assets)),
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "1h,24h,7d,30d",
            }
            try:
                data = await self._coingecko_get("/coins/markets", params)
            except Exception as exc:
                logger.warning("get_top_cryptos page %d failed: %s", page, exc)
                break

            if not isinstance(data, list):
                break

            for item in data:
                asset = self._parse_market_item(item)
                if asset:
                    assets.append(asset)

            if len(data) < params["per_page"]:
                break

        # Filter stablecoins
        if not include_stablecoins:
            assets = [a for a in assets if a.coin_id not in _STABLECOIN_IDS]

        # Trim to n
        assets = assets[:n]

        # Enrich with TVL
        assets = await self._enrich_with_tvl(assets)

        logger.info(
            "get_top_cryptos.complete",
            requested=n,
            returned=len(assets),
            include_stablecoins=include_stablecoins,
        )
        return assets

    async def screen(
        self,
        criteria: CryptoScreenCriteria,
        n: int = 200,
    ) -> CryptoScreenResult:
        """
        Fetch top N cryptos and apply screening criteria.

        Args:
            criteria: CryptoScreenCriteria filter specification.
            n: Universe size to pull before filtering.

        Returns:
            CryptoScreenResult with matching assets sorted by market cap descending.
        """
        include_stables = not criteria.exclude_stablecoins

        # Restrict universe to requested categories
        if criteria.categories:
            category_ids: set[str] = set()
            for cat in criteria.categories:
                category_ids.update(SCREEN_CATEGORIES.get(cat, []))
            # If user asked for stablecoin category, respect it
            if "stablecoin" in criteria.categories:
                include_stables = True

        universe = await self.get_top_cryptos(n=n, include_stablecoins=include_stables)
        n_screened = len(universe)

        # Further filter to category coin IDs if specified
        if criteria.categories:
            universe = [a for a in universe if a.coin_id in category_ids]

        passed = [a for a in universe if self._matches_criteria(a, criteria)]
        passed.sort(key=lambda a: a.market_cap, reverse=True)

        logger.info(
            "screen.complete",
            n_screened=n_screened,
            n_passed=len(passed),
        )
        return CryptoScreenResult(
            criteria=criteria,
            n_screened=n_screened,
            n_passed=len(passed),
            results=passed,
        )

    async def get_trending(self) -> list[CryptoAsset]:
        """
        Fetch CoinGecko trending coins (/search/trending) and return
        full CryptoAsset data for each trending coin.

        Returns:
            List of CryptoAsset for trending coins (typically 7 coins).
        """
        try:
            data = await self._coingecko_get("/search/trending")
        except Exception as exc:
            logger.warning("get_trending failed: %s", exc)
            return []

        if not isinstance(data, dict):
            return []

        coin_ids: list[str] = []
        for entry in data.get("coins", []):
            item = entry.get("item", {})
            cid = item.get("id")
            if cid:
                coin_ids.append(cid)

        if not coin_ids:
            return []

        # Fetch market data for trending coins
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": "50",
            "page": "1",
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d,30d",
        }
        try:
            market_data = await self._coingecko_get("/coins/markets", params)
        except Exception as exc:
            logger.warning("get_trending markets fetch failed: %s", exc)
            return []

        if not isinstance(market_data, list):
            return []

        assets = [self._parse_market_item(item) for item in market_data]
        result = [a for a in assets if a is not None]
        result = await self._enrich_with_tvl(result)

        logger.info("get_trending.complete", n=len(result))
        return result

    async def get_btc_onchain(self) -> BTCOnChainMetrics:
        """
        Fetch Bitcoin on-chain metrics from Blockchain.info stats API
        and CoinGecko market data. Computes NVT proxy and Fear & Greed index.

        Returns:
            BTCOnChainMetrics with network health, valuation, and cycle signals.
        """
        today = date.today()

        # Fetch concurrently: blockchain.info, CoinGecko market data, Fear & Greed
        bc_task = self._fetch_blockchain_info()
        cg_task = self._coingecko_get(
            "/coins/markets",
            {
                "vs_currency": "usd",
                "ids": "bitcoin",
                "order": "market_cap_desc",
                "per_page": "1",
                "page": "1",
                "sparkline": "false",
                "price_change_percentage": "30d",
            },
        )
        fng_task = self._fetch_fear_greed()

        bc_data, cg_data, fng_data = await asyncio.gather(
            bc_task, cg_task, fng_task, return_exceptions=True
        )

        # Parse blockchain.info
        hash_rate: Optional[float] = None
        difficulty: Optional[float] = None
        mempool_size: Optional[int] = None
        avg_tx_fee_usd: Optional[float] = None
        tx_count_24h: Optional[int] = None
        circulating_supply: float = 19_700_000.0   # fallback

        if isinstance(bc_data, dict):
            try:
                # hash_rate from blockchain.info is in GH/s
                hr_raw = float(bc_data.get("hash_rate", 0))
                hash_rate = round(hr_raw / 1e9, 4) if hr_raw > 0 else None   # convert to EH/s
                difficulty = bc_data.get("difficulty")
                if difficulty:
                    difficulty = float(difficulty)
                n_blocks = int(bc_data.get("n_blocks_total", 0))
                # Approximate circulating supply from block count
                # ~21M BTC; blocks * reward halved periodically
                circulating_supply = float(bc_data.get("totalbc", 0)) / 1e8
                if circulating_supply < 1:
                    circulating_supply = 19_700_000.0
                tx_count_24h = int(bc_data.get("n_tx", 0)) or None
                # mempool: blockchain.info calls it "mempool_size" as bytes; count in "mempool_transactions"
                mempool_size = bc_data.get("mempool_count") or bc_data.get("mempool_transactions")
                if mempool_size is not None:
                    mempool_size = int(mempool_size)
                # Fee in satoshis → USD
                total_fees_btc = float(bc_data.get("total_fees_btc", 0)) / 1e8
                btc_price_rough = float(bc_data.get("market_price_usd", 0)) or 40000.0
                if tx_count_24h and total_fees_btc > 0 and btc_price_rough > 0:
                    avg_tx_fee_usd = round(
                        (total_fees_btc / max(tx_count_24h, 1)) * btc_price_rough, 4
                    )
            except Exception as exc:
                logger.debug("btc blockchain.info parse error: %s", exc)

        # Parse CoinGecko market data
        market_cap: Optional[float] = None
        price_usd: Optional[float] = None
        mvrv_proxy: Optional[float] = None

        if isinstance(cg_data, list) and cg_data:
            btc = cg_data[0]
            market_cap = btc.get("market_cap")
            if market_cap:
                market_cap = float(market_cap)
            price_usd = btc.get("current_price")
            if price_usd:
                price_usd = float(price_usd)
            # MVRV proxy: current market cap / market cap 30d ago
            # CoinGecko returns price_change_percentage_30d_in_currency for the price,
            # so market cap 30d ago ≈ current_market_cap / (1 + change/100)
            change_30d = btc.get("price_change_percentage_30d_in_currency")
            if market_cap and change_30d is not None:
                try:
                    mcap_30d_ago = market_cap / (1 + float(change_30d) / 100)
                    if mcap_30d_ago > 0:
                        mvrv_proxy = round(market_cap / mcap_30d_ago, 4)
                except Exception:
                    pass

        # NVT proxy using blockchain.info tx count
        nvt_ratio: Optional[float] = None
        if market_cap and tx_count_24h and price_usd and tx_count_24h > 0:
            # Approximate on-chain volume: tx_count × median_tx_value_estimate
            # Use fee/volume proxy: for BTC, 1 tx ≈ $10k-$50k in typical value; we use volume from CG instead
            # Simpler: NVT = market_cap / (daily_volume_usd from CoinGecko)
            # We use CG volume when available
            pass   # will be set below via CG volume

        cg_volume: Optional[float] = None
        if isinstance(cg_data, list) and cg_data:
            cg_volume = cg_data[0].get("total_volume")
            if cg_volume:
                cg_volume = float(cg_volume)
        if market_cap and cg_volume and cg_volume > 0:
            nvt_ratio = round(market_cap / cg_volume, 4)

        # Fear & Greed
        fear_greed_index: Optional[int] = None
        fear_greed_classification: Optional[str] = None

        if isinstance(fng_data, dict) and "data" in fng_data:
            try:
                first = fng_data["data"][0]
                fear_greed_index = int(first.get("value", 0))
                fear_greed_classification = first.get("value_classification", "")
            except Exception:
                pass

        # Cycle signal
        cycle_signal = self._classify_btc_cycle(
            mvrv_proxy=mvrv_proxy,
            nvt_ratio=nvt_ratio,
            fear_greed=fear_greed_index,
        )

        return BTCOnChainMetrics(
            as_of=today,
            hash_rate=hash_rate,
            difficulty=difficulty,
            mempool_size=mempool_size,
            avg_tx_fee_usd=avg_tx_fee_usd,
            tx_count_24h=tx_count_24h,
            market_cap=market_cap,
            nvt_ratio=nvt_ratio,
            mvrv_proxy=mvrv_proxy,
            circulating_supply=circulating_supply,
            fear_greed_index=fear_greed_index,
            fear_greed_classification=fear_greed_classification,
            cycle_signal=cycle_signal,
        )

    async def get_eth_network(self) -> ETHNetworkMetrics:
        """
        Fetch Ethereum network metrics from Etherscan (gas, supply) and
        DeFiLlama (ETH TVL). Falls back gracefully if Etherscan key is absent.

        Returns:
            ETHNetworkMetrics with gas prices, TVL, and market context.
        """
        today = date.today()

        # Parallel: Etherscan gas, ETH market data from CoinGecko, DeFiLlama ETH TVL
        eth_cg_task = self._coingecko_get(
            "/coins/markets",
            {
                "vs_currency": "usd",
                "ids": "ethereum",
                "order": "market_cap_desc",
                "per_page": "1",
                "page": "1",
                "sparkline": "false",
            },
        )
        eth_tvl_task = self._fetch_defillama_chain_tvl("Ethereum")

        results = await asyncio.gather(eth_cg_task, eth_tvl_task, return_exceptions=True)
        cg_data, eth_tvl = results[0], results[1]

        # Gas price from Etherscan (optional key)
        gas_price_gwei: Optional[float] = None
        base_fee_gwei: Optional[float] = None

        if self._etherscan_key:
            try:
                gas_data = await self._etherscan_get(
                    params={
                        "module": "gastracker",
                        "action": "gasoracle",
                        "apikey": self._etherscan_key,
                    }
                )
                if isinstance(gas_data, dict) and gas_data.get("status") == "1":
                    result_data = gas_data.get("result", {})
                    gas_price_gwei = float(result_data.get("ProposeGasPrice", 0)) or None
                    base_fee_gwei = float(result_data.get("suggestBaseFee", 0)) or None
            except Exception as exc:
                logger.debug("etherscan gas fetch failed: %s", exc)
        else:
            # Without key: estimate from CoinGecko or leave None
            logger.debug("ETHERSCAN_API_KEY not set; gas price unavailable")

        # ETH supply from Etherscan (no key needed for basic stats)
        eth_supply: Optional[float] = None
        try:
            if self._etherscan_key:
                supply_data = await self._etherscan_get(
                    params={
                        "module": "stats",
                        "action": "ethsupply2",
                        "apikey": self._etherscan_key,
                    }
                )
                if isinstance(supply_data, dict) and supply_data.get("status") == "1":
                    result_data = supply_data.get("result", {})
                    eth_supply_wei = int(result_data.get("EthSupply", 0))
                    eth_supply = round(eth_supply_wei / 1e18, 2)
        except Exception as exc:
            logger.debug("etherscan supply fetch failed: %s", exc)

        # CoinGecko ETH data
        price_usd: Optional[float] = None
        market_cap: Optional[float] = None

        if isinstance(cg_data, list) and cg_data:
            eth = cg_data[0]
            price_usd = eth.get("current_price")
            if price_usd:
                price_usd = float(price_usd)
            market_cap = eth.get("market_cap")
            if market_cap:
                market_cap = float(market_cap)

        # TVL from DeFiLlama
        total_value_locked: Optional[float] = None
        if isinstance(eth_tvl, (int, float)) and eth_tvl > 0:
            total_value_locked = float(eth_tvl)

        # Staking APY proxy: ETH staking roughly 3-4% — could be fetched from
        # rated.network or beaconcha.in (free APIs), but to avoid extra deps
        # we use a static reasonable estimate updated rarely.
        # This is a known limitation; a future version can add beaconcha.in.
        staking_apy_proxy = 3.5   # approximate current ETH staking APY %

        logger.info(
            "get_eth_network.complete",
            gas_gwei=gas_price_gwei,
            tvl=total_value_locked,
            price=price_usd,
        )

        return ETHNetworkMetrics(
            as_of=today,
            gas_price_gwei=gas_price_gwei,
            base_fee_gwei=base_fee_gwei,
            eth_supply=eth_supply,
            staking_apy_proxy=staking_apy_proxy,
            total_value_locked=total_value_locked,
            price_usd=price_usd,
            market_cap=market_cap,
        )

    async def get_defi_overview(self) -> DeFiOverview:
        """
        Aggregate DeFi metrics from DeFiLlama protocols and CoinGecko global data.

        Returns:
            DeFiOverview with total_tvl, top_protocols (top 20), and chain_breakdown.
        """
        today = date.today()

        # Concurrent: DeFiLlama protocols, CoinGecko global
        protocols_task = self._fetch_defillama_protocols()
        cg_global_task = self._coingecko_get("/global")

        protocols_raw, cg_global = await asyncio.gather(
            protocols_task, cg_global_task, return_exceptions=True
        )

        # Parse DeFiLlama protocols
        all_protocols: list[DeFiProtocol] = []
        total_tvl: float = 0.0
        chain_breakdown: dict[str, float] = {}

        if isinstance(protocols_raw, list):
            for p in protocols_raw:
                try:
                    name = p.get("name", "")
                    slug = p.get("slug", "")
                    category = p.get("category")
                    chain = p.get("chain", "Multi")
                    tvl = float(p.get("tvl", 0) or 0)
                    change_1d = p.get("change_1d")
                    change_7d = p.get("change_7d")
                    mcap_tvl = p.get("mcap/tvl")

                    if tvl <= 0:
                        continue

                    all_protocols.append(DeFiProtocol(
                        name=name,
                        slug=slug,
                        chain=chain,
                        category=category,
                        tvl=round(tvl, 2),
                        change_1d=round(float(change_1d), 4) if change_1d is not None else None,
                        change_7d=round(float(change_7d), 4) if change_7d is not None else None,
                        mcap_tvl=round(float(mcap_tvl), 4) if mcap_tvl is not None else None,
                    ))

                    # Chain TVL aggregation
                    chain_key = chain if chain else "Other"
                    chain_breakdown[chain_key] = chain_breakdown.get(chain_key, 0.0) + tvl
                    total_tvl += tvl

                except Exception as exc:
                    logger.debug("defi_protocol parse error: %s", exc)

        # Sort protocols by TVL descending; take top 20
        all_protocols.sort(key=lambda p: p.tvl, reverse=True)
        top_protocols = all_protocols[:20]

        # Sort chain breakdown descending
        chain_breakdown = dict(
            sorted(chain_breakdown.items(), key=lambda kv: kv[1], reverse=True)
        )

        # CoinGecko global for market context
        total_crypto_market_cap: Optional[float] = None
        total_tvl_change_1d: Optional[float] = None
        defi_dominance_pct: Optional[float] = None

        if isinstance(cg_global, dict):
            gd = cg_global.get("data", {})
            mcap_map = gd.get("total_market_cap", {})
            total_crypto_market_cap = float(mcap_map.get("usd", 0)) or None

        if total_crypto_market_cap and total_tvl > 0:
            defi_dominance_pct = round(total_tvl / total_crypto_market_cap * 100, 4)

        logger.info(
            "get_defi_overview.complete",
            total_tvl=total_tvl,
            n_protocols=len(all_protocols),
            chains=len(chain_breakdown),
        )

        return DeFiOverview(
            as_of=today,
            total_tvl=round(total_tvl, 2),
            total_tvl_change_1d=total_tvl_change_1d,
            top_protocols=top_protocols,
            chain_breakdown={k: round(v, 2) for k, v in chain_breakdown.items()},
            defi_dominance_pct=defi_dominance_pct,
            total_crypto_market_cap=total_crypto_market_cap,
        )

    async def momentum_screen(
        self, lookback_days: int = 30, top_n: int = 20
    ) -> list[CryptoAsset]:
        """
        Screen for top momentum assets by lookback-period price change.

        Filters:
          - Min market cap $100M
          - Non-stablecoin
          - Sorted by change_{lookback}d descending

        Args:
            lookback_days: Lookback period. Supported: 7, 30 (maps to CoinGecko fields).
            top_n: Number of results to return.

        Returns:
            Top performing CryptoAsset list.
        """
        universe = await self.get_top_cryptos(n=250, include_stablecoins=False)

        # Filter by minimum market cap $100M
        universe = [a for a in universe if a.market_cap >= 100_000_000]

        # Select change field
        if lookback_days <= 7:
            key = "change_7d"
        else:
            key = "change_30d"

        # Filter out None change values and sort descending
        scored = [a for a in universe if getattr(a, key) is not None]
        scored.sort(key=lambda a: getattr(a, key) or float("-inf"), reverse=True)

        result = scored[:top_n]
        logger.info(
            "momentum_screen.complete",
            lookback_days=lookback_days,
            top_n=top_n,
            returned=len(result),
        )
        return result

    def compute_on_chain_momentum(
        self,
        active_addresses_history: list[int],
        window_days: int = 30,
        zscore_lookback: int = 90,
    ) -> dict:
        """On-chain address momentum.

        daily_growth_rate = active_address_30d_change / 30
        z_score           = z-score of daily_growth_rate over the past 90 days

        Args:
            active_addresses_history: Daily active address counts (oldest → newest).
                                      Must have at least ``window_days + 1`` entries.
            window_days: Rolling window to compute change (default 30).
            zscore_lookback: Days to use for z-score denominator (default 90).

        Returns:
            {
              "daily_growth_rate": float,
              "total_30d_change": int,
              "z_score": float | None,
              "momentum_label": str,
            }
        """
        if len(active_addresses_history) < window_days + 1:
            return {
                "daily_growth_rate": 0.0,
                "total_30d_change": 0,
                "z_score": None,
                "momentum_label": "insufficient_data",
            }

        series = np.array(active_addresses_history, dtype=float)

        # 30-day change and daily growth rate
        total_30d_change = int(series[-1] - series[-window_days - 1])
        daily_growth_rate = total_30d_change / window_days

        # Rolling 90-day daily changes for z-score
        daily_changes = np.diff(series)
        lookback_changes = daily_changes[-zscore_lookback:] if len(daily_changes) >= zscore_lookback else daily_changes
        mean_daily = float(np.mean(lookback_changes))
        std_daily = float(np.std(lookback_changes, ddof=1)) if len(lookback_changes) > 1 else 1.0

        z_score: float | None = None
        if std_daily > 0:
            z_score = round((daily_growth_rate - mean_daily) / std_daily, 4)

        if z_score is not None:
            if z_score > 2.0:
                label = "strong_growth"
            elif z_score > 0.5:
                label = "moderate_growth"
            elif z_score < -2.0:
                label = "strong_decline"
            elif z_score < -0.5:
                label = "moderate_decline"
            else:
                label = "neutral"
        else:
            label = "neutral"

        return {
            "daily_growth_rate": round(daily_growth_rate, 2),
            "total_30d_change": total_30d_change,
            "z_score": z_score,
            "momentum_label": label,
        }

    def screen_by_defi_metrics(
        self,
        protocols: list[DeFiProtocol],
        min_protocol_revenue_monthly: float = 1_000_000.0,
        min_tvl_growth_90d_pct: float = 20.0,
    ) -> list[DeFiProtocol]:
        """Filter DeFi protocols by quality revenue and TVL growth metrics.

        Passes when BOTH conditions hold:
          - protocol_revenue > $1M / month  (proxy: tvl × assumed_revenue_yield)
          - TVL growth 90d > 20%            (proxy: change_7d annualised ÷ 13)

        When protocol-level revenue data is unavailable, a TVL-based proxy is
        used: revenue ≈ TVL × 0.003 (30bps yield, conservative DeFi estimate).

        Args:
            protocols: List of DeFiProtocol objects from get_defi_overview().
            min_protocol_revenue_monthly: Revenue threshold in USD (default $1M).
            min_tvl_growth_90d_pct: TVL growth threshold in % (default 20%).

        Returns:
            Filtered list of DeFiProtocol objects sorted by TVL descending.
        """
        passed: list[DeFiProtocol] = []
        for p in protocols:
            # Revenue proxy: TVL × 0.003 monthly
            revenue_proxy = p.tvl * 0.003
            passes_revenue = revenue_proxy >= min_protocol_revenue_monthly

            # TVL growth proxy: change_7d (weekly %) annualised then converted to ~90d
            passes_growth = False
            if p.change_7d is not None:
                # Approximate 90d growth from 7d change:
                # weekly_pct → 13-week cumulative
                approx_90d_growth = ((1 + p.change_7d / 100) ** 13 - 1) * 100
                passes_growth = approx_90d_growth >= min_tvl_growth_90d_pct

            if passes_revenue and passes_growth:
                passed.append(p)

        passed.sort(key=lambda p: p.tvl, reverse=True)
        logger.info(
            "screen_by_defi_metrics.complete total=%d passed=%d",
            len(protocols), len(passed),
        )
        return passed

    def compute_crypto_fear_greed(
        self,
        price_momentum_score: float,
        volume_score: float,
        social_score: float,
        dominance_score: float,
        trends_score: float,
    ) -> dict:
        """Composite Crypto Fear & Greed index (5-component model).

        Component weights:
          price_momentum  25%
          volume          25%
          social          25%
          dominance       15%
          trends          10%

        Each input must be in [0, 100].
        Output is also in [0, 100]:
          0  = Extreme Fear
          100 = Extreme Greed

        Args:
            price_momentum_score: Price momentum component (0-100).
            volume_score: Volume component (0-100).
            social_score: Social sentiment component (0-100).
            dominance_score: BTC dominance component (0-100).
            trends_score: Google Trends component (0-100).

        Returns:
            {
              "composite": float  (0-100),
              "classification": str,
              "components": dict,
            }
        """
        weights = {
            "price_momentum": 0.25,
            "volume": 0.25,
            "social": 0.25,
            "dominance": 0.15,
            "trends": 0.10,
        }
        scores = {
            "price_momentum": price_momentum_score,
            "volume": volume_score,
            "social": social_score,
            "dominance": dominance_score,
            "trends": trends_score,
        }

        composite = sum(scores[k] * weights[k] for k in weights)
        composite = max(0.0, min(100.0, composite))

        if composite >= 75:
            classification = "Extreme Greed"
        elif composite >= 55:
            classification = "Greed"
        elif composite >= 45:
            classification = "Neutral"
        elif composite >= 25:
            classification = "Fear"
        else:
            classification = "Extreme Fear"

        return {
            "composite": round(composite, 2),
            "classification": classification,
            "components": {k: round(v, 2) for k, v in scores.items()},
            "weights": weights,
        }

    async def value_screen(self) -> list[CryptoAsset]:
        """
        Screen for potentially undervalued assets using:
          1. Low NVT proxy (market_cap / volume_24h < 20): undervalued by tx volume
          2. Low mcap_to_tvl ratio (< 1.5 for DeFi): undervalued by protocol usage

        Returns:
            CryptoAsset list for value candidates, sorted by market cap descending.
        """
        universe = await self.get_top_cryptos(n=200, include_stablecoins=False)

        # Min $50M market cap, exclude very small cap noise
        universe = [a for a in universe if a.market_cap >= 50_000_000]

        value_picks: list[CryptoAsset] = []

        for asset in universe:
            is_value = False

            # Low NVT: high relative transaction volume to market cap
            if asset.nvt_proxy is not None and asset.nvt_proxy < 20:
                is_value = True

            # Low mcap/TVL: DeFi protocol worth less than its locked capital
            if asset.mcap_to_tvl is not None and 0 < asset.mcap_to_tvl < 1.5:
                is_value = True

            if is_value:
                value_picks.append(asset)

        value_picks.sort(key=lambda a: a.market_cap, reverse=True)

        logger.info("value_screen.complete", n=len(value_picks))
        return value_picks

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    async def _enrich_with_tvl(
        self, assets: list[CryptoAsset]
    ) -> list[CryptoAsset]:
        """
        Match CoinGecko assets against DeFiLlama protocols by name/slug
        and inject tvl, protocol_revenue_24h, and mcap_to_tvl.
        """
        try:
            protocols_raw = await self._fetch_defillama_protocols()
        except Exception as exc:
            logger.warning("_enrich_with_tvl DeFiLlama fetch failed: %s", exc)
            return assets

        if not isinstance(protocols_raw, list):
            return assets

        # Build lookup: normalised name/slug → protocol entry
        protocol_lookup: dict[str, dict] = {}
        for p in protocols_raw:
            name = (p.get("name") or "").lower().strip()
            slug = (p.get("slug") or "").lower().strip()
            symbol = (p.get("symbol") or "").lower().strip()
            for key in (name, slug, symbol):
                if key:
                    protocol_lookup[key] = p

        for asset in assets:
            # Attempt to find a match by coin_id, name, or symbol
            match_keys = [
                asset.coin_id.lower(),
                asset.name.lower(),
                asset.symbol.lower(),
            ]
            protocol: Optional[dict] = None
            for key in match_keys:
                if key in protocol_lookup:
                    protocol = protocol_lookup[key]
                    break

            if protocol is None:
                continue

            tvl = protocol.get("tvl")
            if tvl and float(tvl) > 0:
                asset.tvl = round(float(tvl), 2)
                asset.nvt_proxy = (
                    round(asset.market_cap / asset.volume_24h, 4)
                    if asset.volume_24h > 0 else None
                )
                if asset.market_cap > 0:
                    asset.mcap_to_tvl = round(asset.market_cap / float(tvl), 4)

        return assets

    async def _coingecko_get(
        self, path: str, params: Optional[dict] = None
    ) -> object:
        """Rate-limited CoinGecko GET with in-memory caching and retry logic."""
        params = params or {}
        cache_key = f"cg:{path}:{sorted(params.items())}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        async with self._rate_limiter:
            await asyncio.sleep(_CG_RATE_SLEEP)
            for attempt in range(1, 4):
                try:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        resp = await client.get(
                            f"{COINGECKO_BASE}{path}",
                            params=params,
                            headers=_HEADERS,
                        )
                        if resp.status_code == 429:
                            wait = 12.0 * attempt
                            logger.warning(
                                "coingecko 429 on %s, sleeping %.0fs", path, wait
                            )
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        _cache_set(cache_key, data)
                        return data
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        await asyncio.sleep(12.0 * attempt)
                        continue
                    logger.error(
                        "coingecko HTTP %d on %s", exc.response.status_code, path
                    )
                    raise
                except httpx.TimeoutException:
                    logger.warning("coingecko timeout on %s (attempt %d)", path, attempt)
                    if attempt < 3:
                        await asyncio.sleep(2.0)
                        continue
                    raise
                except Exception:
                    raise

        raise RuntimeError(f"CoinGecko request failed after retries: {path}")

    async def _fetch_defillama_protocols(self) -> object:
        """Fetch all DeFiLlama protocols (cached 5 min)."""
        cache_key = "defillama:protocols"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{DEFILLAMA_BASE}/protocols", headers=_HEADERS)
            resp.raise_for_status()
            data = resp.json()
            _cache_set(cache_key, data)
            return data

    async def _fetch_defillama_chain_tvl(self, chain: str) -> float:
        """Fetch TVL for a single chain from DeFiLlama /chains endpoint."""
        cache_key = f"defillama:chain:{chain}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return float(cached)  # type: ignore[arg-type]

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
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
            logger.debug("defillama chain tvl failed for %s: %s", chain, exc)

        return 0.0

    async def _fetch_blockchain_info(self) -> object:
        """Fetch Bitcoin network stats from Blockchain.info."""
        cache_key = "blockchain_info:stats"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(BLOCKCHAIN_INFO, headers=_HEADERS)
            resp.raise_for_status()
            data = resp.json()
            _cache_set(cache_key, data)
            return data

    async def _fetch_fear_greed(self) -> object:
        """Fetch Fear & Greed Index from Alternative.me."""
        cache_key = "alternative_me:fng"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(FNG_URL, headers=_HEADERS)
            resp.raise_for_status()
            data = resp.json()
            _cache_set(cache_key, data)
            return data

    async def _etherscan_get(self, params: dict) -> object:
        """HTTP GET to Etherscan API."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(ETHERSCAN_BASE, params=params, headers=_HEADERS)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _parse_market_item(item: dict) -> Optional[CryptoAsset]:
        """Parse a single CoinGecko /coins/markets item into a CryptoAsset."""
        try:
            price_usd = float(item.get("current_price") or 0)
            market_cap = float(item.get("market_cap") or 0)
            volume_24h = float(item.get("total_volume") or 0)

            if price_usd <= 0 or market_cap <= 0:
                return None

            nvt_proxy: Optional[float] = None
            if volume_24h > 0:
                nvt_proxy = round(market_cap / volume_24h, 4)

            pcp = item.get("price_change_percentage_1h_in_currency")
            pcp24 = item.get("price_change_percentage_24h_in_currency")
            pcp7d = item.get("price_change_percentage_7d_in_currency")
            pcp30d = item.get("price_change_percentage_30d_in_currency")

            return CryptoAsset(
                coin_id=item.get("id", ""),
                symbol=(item.get("symbol") or "").upper(),
                name=item.get("name", ""),
                price_usd=round(price_usd, 8),
                market_cap=round(market_cap, 2),
                volume_24h=round(volume_24h, 2),
                change_1h=round(float(pcp), 4) if pcp is not None else None,
                change_24h=round(float(pcp24), 4) if pcp24 is not None else None,
                change_7d=round(float(pcp7d), 4) if pcp7d is not None else None,
                change_30d=round(float(pcp30d), 4) if pcp30d is not None else None,
                ath=float(item.get("ath") or 0) or None,
                ath_change_pct=item.get("ath_change_percentage"),
                atl=float(item.get("atl") or 0) or None,
                circulating_supply=item.get("circulating_supply"),
                total_supply=item.get("total_supply"),
                max_supply=item.get("max_supply"),
                fully_diluted_valuation=item.get("fully_diluted_valuation"),
                market_cap_rank=item.get("market_cap_rank"),
                nvt_proxy=nvt_proxy,
            )
        except Exception as exc:
            logger.debug("_parse_market_item error: %s", exc)
            return None

    @staticmethod
    def _matches_criteria(
        asset: CryptoAsset, criteria: CryptoScreenCriteria
    ) -> bool:
        """Return True if asset passes all non-None filter conditions."""
        if criteria.min_market_cap is not None:
            if asset.market_cap < criteria.min_market_cap:
                return False
        if criteria.max_market_cap is not None:
            if asset.market_cap > criteria.max_market_cap:
                return False
        if criteria.min_volume_24h is not None:
            if asset.volume_24h < criteria.min_volume_24h:
                return False
        if criteria.min_change_24h is not None:
            if asset.change_24h is None or asset.change_24h < criteria.min_change_24h:
                return False
        if criteria.max_change_24h is not None:
            if asset.change_24h is None or asset.change_24h > criteria.max_change_24h:
                return False
        if criteria.max_ath_drawdown_pct is not None:
            if asset.ath_change_pct is None:
                return False
            if float(asset.ath_change_pct) < criteria.max_ath_drawdown_pct:
                return False
        if criteria.min_tvl is not None:
            if asset.tvl is None or asset.tvl < criteria.min_tvl:
                return False
        if criteria.exclude_stablecoins:
            if asset.coin_id in _STABLECOIN_IDS:
                return False
        return True

    @staticmethod
    def _classify_btc_cycle(
        mvrv_proxy: Optional[float],
        nvt_ratio: Optional[float],
        fear_greed: Optional[int],
    ) -> str:
        """
        Simple rule-based BTC cycle signal.

        Returns one of: "extreme_greed", "distribute", "neutral", "accumulate".
        """
        score = 0

        # MVRV: high = overvalued, low = undervalued
        if mvrv_proxy is not None:
            if mvrv_proxy > 3.5:
                score += 2    # historically expensive
            elif mvrv_proxy > 2.0:
                score += 1
            elif mvrv_proxy < 0.8:
                score -= 2    # historically cheap
            elif mvrv_proxy < 1.2:
                score -= 1

        # NVT: high = market cap not supported by usage
        if nvt_ratio is not None:
            if nvt_ratio > 65:
                score += 2
            elif nvt_ratio > 40:
                score += 1
            elif nvt_ratio < 20:
                score -= 1

        # Fear & Greed
        if fear_greed is not None:
            if fear_greed >= 80:
                score += 3    # extreme greed
            elif fear_greed >= 65:
                score += 1
            elif fear_greed <= 20:
                score -= 3    # extreme fear → buy
            elif fear_greed <= 35:
                score -= 1

        if score >= 4:
            return "extreme_greed"
        if score >= 2:
            return "distribute"
        if score <= -3:
            return "accumulate"
        if score <= -1:
            return "accumulate"
        return "neutral"


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

async def top_cryptos(n: int = 50) -> list[CryptoAsset]:
    """Fetch top N cryptos by market cap (non-stablecoin)."""
    screener = CryptoScreener()
    return await screener.get_top_cryptos(n=n, include_stablecoins=False)


async def screen_crypto(criteria: dict) -> CryptoScreenResult:
    """Screen the top 200 cryptos by market cap against the given criteria dict."""
    screener = CryptoScreener()
    screen_criteria = CryptoScreenCriteria(**criteria)
    return await screener.screen(screen_criteria, n=200)


async def btc_onchain() -> BTCOnChainMetrics:
    """Fetch Bitcoin on-chain metrics."""
    screener = CryptoScreener()
    return await screener.get_btc_onchain()


async def eth_network() -> ETHNetworkMetrics:
    """Fetch Ethereum network metrics."""
    screener = CryptoScreener()
    return await screener.get_eth_network()


async def defi_overview() -> DeFiOverview:
    """Fetch DeFi market overview from DeFiLlama."""
    screener = CryptoScreener()
    return await screener.get_defi_overview()
