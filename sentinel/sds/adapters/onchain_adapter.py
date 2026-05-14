"""On-chain event monitoring and DEX/AMM liquidity analytics — Dimensions 109 & 110.

Data sources (all free-tier):
  - Etherscan API  (ETHERSCAN_API_KEY env var)   — token transfers, contract logs, gas oracle
  - DeFiLlama      (no key)                       — protocol TVL, yields, DEX volumes, stablecoins
  - The Graph      (no key)                       — Uniswap V3 subgraph, Aave V3 subgraph
  - Dune Analytics (DUNE_API_KEY env var)         — custom on-chain queries
  - CoinGecko      (no key, 30 calls/min)         — DeFi market stats, token prices

Usage:
    adapter = OnChainAdapter()
    gas   = await adapter.get_gas_oracle()
    pools = await adapter.get_uniswap_pools(min_tvl=5_000_000)
    yield_ops = await adapter.get_yield_opportunities(min_apy=8.0, stable_only=True)
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Any
import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Base URLs
# ---------------------------------------------------------------------------

ETHERSCAN_BASE = "https://api.etherscan.io/api"
DEFILLAMA_BASE = "https://api.llama.fi"
DEFI_YIELDS    = "https://yields.llama.fi"
STABLECOINS_API = "https://stablecoins.llama.fi"
GRAPH_UNISWAP  = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
GRAPH_AAVE     = "https://api.thegraph.com/subgraphs/name/aave/protocol-v3"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DUNE_BASE      = "https://api.dune.com/api/v1"

# ---------------------------------------------------------------------------
# Well-known contract addresses (all lowercase for comparison)
# ---------------------------------------------------------------------------

CONTRACTS = {
    "uniswap_v3_factory":  "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    "uniswap_v3_router":   "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    "aave_v3_pool":        "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
    "curve_3pool":         "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",
    "compound_v3_usdc":    "0xc3d688B66703497DAA19211EEdff47f25384cdc3",
    "lido_steth":          "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
}

# Well-known ERC-20 event topic signatures (keccak256 of ABI signature)
EVENT_TOPICS = {
    "Transfer": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "Swap":     "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",
    "Mint":     "0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f",
    "Burn":     "0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496",
    "Sync":     "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1",
}

# Known exchange / protocol address labels
KNOWN_LABELS: dict[str, str] = {
    "0xd551234ae421e3bcba99a0da6d736074f22192ff": "Binance Cold Wallet",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance Hot Wallet",
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance Exchange Wallet",
    "0x6262998ced04146fa42253a5c0af90ca02dfd2a3": "Crypto.com Hot Wallet",
    "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae": "Ethereum Foundation",
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": "Lido stETH",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC Token Contract",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT Token Contract",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH Token Contract",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "UNI Token Contract",
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave V2 Lending Pool",
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": "Aave V3 Lending Pool",
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Uniswap V3 Router",
    "0x1f98431c8ad98523631ae4a59f267346ea31f984": "Uniswap V3 Factory",
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7": "Curve 3pool",
    "0xc3d688b66703497daa19211eedff47f25384cdc3": "Compound V3 USDC",
    "0x00000000219ab540356cbb839cbe05303d7705fa": "Ethereum 2.0 Deposit Contract",
}

# Stablecoins for risk classification
STABLECOIN_SYMBOLS = frozenset({
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FRAX", "LUSD",
    "USDD", "USDP", "GUSD", "SUSD", "CRVUSD", "PYUSD", "FDUSD",
})

# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

_ETHERSCAN_WINDOW: list[float] = []   # timestamps of Etherscan calls
_COINGECKO_WINDOW: list[float] = []   # timestamps of CoinGecko calls


async def _etherscan_throttle() -> None:
    """Etherscan free tier: 5 calls/sec."""
    now = time.monotonic()
    window = 1.0
    _ETHERSCAN_WINDOW[:] = [t for t in _ETHERSCAN_WINDOW if now - t < window]
    if len(_ETHERSCAN_WINDOW) >= 5:
        sleep_for = window - (now - _ETHERSCAN_WINDOW[0]) + 0.05
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
    _ETHERSCAN_WINDOW.append(time.monotonic())


async def _coingecko_throttle() -> None:
    """CoinGecko free tier: 30 calls/min."""
    now = time.monotonic()
    window = 60.0
    _COINGECKO_WINDOW[:] = [t for t in _COINGECKO_WINDOW if now - t < window]
    if len(_COINGECKO_WINDOW) >= 28:
        sleep_for = window - (now - _COINGECKO_WINDOW[0]) + 0.5
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
    _COINGECKO_WINDOW.append(time.monotonic())


# ---------------------------------------------------------------------------
# Simple in-memory response cache (TTL 5 min by default)
# ---------------------------------------------------------------------------

_resp_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str, ttl: float = 300.0) -> Optional[Any]:
    entry = _resp_cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > ttl:
        del _resp_cache[key]
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _resp_cache[key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OnChainEvent(BaseModel):
    tx_hash: str
    block_number: int
    timestamp: datetime
    contract_address: str
    event_name: Optional[str] = None
    event_topic: Optional[str] = None
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    value_eth: Optional[float] = None
    value_usd: Optional[float] = None
    token_symbol: Optional[str] = None
    token_amount: Optional[float] = None
    chain: str = "ethereum"


class TokenTransfer(BaseModel):
    tx_hash: str
    block: int
    timestamp: datetime
    token_address: str
    token_name: str
    token_symbol: str
    from_address: str
    to_address: str
    amount: float
    amount_usd: Optional[float] = None


class WhaleAlert(BaseModel):
    timestamp: datetime
    token_symbol: str
    amount: float
    amount_usd: float
    from_label: Optional[str] = None
    to_label: Optional[str] = None
    tx_hash: str
    alert_type: str  # "exchange_inflow" | "exchange_outflow" | "wallet_accumulation" | "defi_interaction"


class GasOracle(BaseModel):
    timestamp: datetime
    safe_gas_gwei: float
    propose_gas_gwei: float
    fast_gas_gwei: float
    base_fee_gwei: float
    eth_price_usd: float
    tx_cost_usdc_transfer: float   # USD cost of a simple ERC-20 USDC transfer at safe gas


class DefiProtocol(BaseModel):
    name: str
    slug: str
    category: str
    chain: str
    tvl_usd: float
    tvl_change_24h: float
    tvl_change_7d: float
    revenue_24h: Optional[float] = None
    fees_24h: Optional[float] = None
    token_symbol: Optional[str] = None
    token_price: Optional[float] = None


class UniswapPool(BaseModel):
    pool_address: str
    token0: str
    token1: str
    fee_tier: int           # 100=0.01%, 500=0.05%, 3000=0.3%, 10000=1%
    tvl_usd: float
    volume_24h: float
    volume_7d: float
    fee_apr: float          # annualised: volume_7d * (feeTier/1e6) / tvl * 52
    liquidity: Optional[float] = None
    price_token0_in_token1: Optional[float] = None


class AMMSnapshot(BaseModel):
    as_of: datetime
    total_dex_volume_24h: float
    total_dex_tvl: float
    top_pools: list[UniswapPool]
    protocol_breakdown: list[dict]   # [{protocol, volume_24h, market_share_pct}]
    dominant_pairs: list[str]


class YieldOpportunity(BaseModel):
    protocol: str
    pool_name: str
    chain: str
    apy: float
    apy_with_rewards: float
    tvl_usd: float
    token_pair: Optional[str] = None
    risk_level: str         # "low" | "medium" | "high"
    pool_id: str
    stable_pool: bool


class StablecoinMetrics(BaseModel):
    symbol: str
    name: str
    peg_type: str           # "FIAT" | "CRYPTO" | "ALGO"
    circulating_usd: float
    peg_deviation_pct: float
    chain_distribution: dict[str, float]   # {chain: % of supply}
    depeg_risk: str         # "low" | "medium" | "high"


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class OnChainAdapter:
    """Unified on-chain monitoring and DeFi analytics adapter.

    Environment variables
    ---------------------
    ETHERSCAN_API_KEY   — required for Etherscan endpoints (register free at etherscan.io)
    DUNE_API_KEY        — optional; enables Dune Analytics queries
    """

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        self._etherscan_key = os.getenv("ETHERSCAN_API_KEY", "")
        self._dune_key = os.getenv("DUNE_API_KEY", "")
        if not self._etherscan_key:
            logger.warning(
                "ETHERSCAN_API_KEY not set — Etherscan calls will use public rate limits "
                "(1 call/5 sec). Set the env var for full 5 calls/sec."
            )

    # -----------------------------------------------------------------------
    # Internal HTTP helpers
    # -----------------------------------------------------------------------

    async def _get_json(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        cache_ttl: float = 300.0,
    ) -> Optional[Any]:
        cache_key = url + str(sorted((params or {}).items()))
        cached = _cache_get(cache_key, cache_ttl)
        if cached is not None:
            return cached

        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 429:
                    wait = 2.0 * attempt
                    logger.warning("onchain_adapter 429 on %s — sleeping %.1fs", url, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
            except httpx.TimeoutException:
                logger.warning("onchain_adapter timeout %s (attempt %d/3)", url, attempt)
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "onchain_adapter HTTP %d on %s", exc.response.status_code, url
                )
                return None
            except Exception as exc:
                logger.error("onchain_adapter error on %s: %s", url, exc)
                return None
            if attempt < 3:
                await asyncio.sleep(1.5 * attempt)
        return None

    async def _post_json(
        self,
        url: str,
        body: dict,
        headers: Optional[dict] = None,
        cache_ttl: float = 120.0,
    ) -> Optional[Any]:
        import json as _json
        cache_key = url + _json.dumps(body, sort_keys=True)
        cached = _cache_get(cache_key, cache_ttl)
        if cached is not None:
            return cached

        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=body, headers=headers or {})
                if resp.status_code == 429:
                    wait = 2.0 * attempt
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                _cache_set(cache_key, data)
                return data
            except httpx.TimeoutException:
                logger.warning("onchain_adapter POST timeout %s (attempt %d/3)", url, attempt)
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "onchain_adapter POST HTTP %d on %s", exc.response.status_code, url
                )
                return None
            except Exception as exc:
                logger.error("onchain_adapter POST error on %s: %s", url, exc)
                return None
            if attempt < 3:
                await asyncio.sleep(1.5 * attempt)
        return None

    # -----------------------------------------------------------------------
    # Etherscan / on-chain events
    # -----------------------------------------------------------------------

    async def get_gas_oracle(self) -> GasOracle:
        """Fetch live gas prices from Etherscan and ETH price from CoinGecko.

        Returns a GasOracle with safe/propose/fast gas in gwei, base fee,
        ETH price, and USD cost of a USDC transfer at safe gas.
        """
        await _etherscan_throttle()
        gas_data = await self._get_json(
            ETHERSCAN_BASE,
            params={
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": self._etherscan_key or "YourApiKeyToken",
            },
            cache_ttl=30.0,   # gas prices are very fresh
        )

        safe_gwei = propose_gwei = fast_gwei = base_fee = 0.0
        if isinstance(gas_data, dict) and gas_data.get("status") == "1":
            result = gas_data.get("result", {})
            safe_gwei    = float(result.get("SafeGasPrice", 0))
            propose_gwei = float(result.get("ProposeGasPrice", 0))
            fast_gwei    = float(result.get("FastGasPrice", 0))
            # suggestBaseFee is a float string like "12.345"
            base_fee = float(result.get("suggestBaseFee", 0))
        else:
            logger.warning("gas_oracle: unexpected Etherscan response: %s", gas_data)

        # ETH price from CoinGecko
        await _coingecko_throttle()
        price_data = await self._get_json(
            f"{COINGECKO_BASE}/simple/price",
            params={"ids": "ethereum", "vs_currencies": "usd"},
            cache_ttl=60.0,
        )
        eth_price = 0.0
        if isinstance(price_data, dict):
            eth_price = float(price_data.get("ethereum", {}).get("usd", 0))

        # ERC-20 transfer costs 65 000 gas (USDC: ~65k, native ETH: 21 000)
        usdc_gas_units = 65_000
        tx_cost_usd = (safe_gwei * 1e-9) * usdc_gas_units * eth_price

        return GasOracle(
            timestamp=datetime.now(timezone.utc),
            safe_gas_gwei=round(safe_gwei, 2),
            propose_gas_gwei=round(propose_gwei, 2),
            fast_gas_gwei=round(fast_gwei, 2),
            base_fee_gwei=round(base_fee, 4),
            eth_price_usd=round(eth_price, 2),
            tx_cost_usdc_transfer=round(tx_cost_usd, 4),
        )

    async def monitor_contract_events(
        self,
        contract_address: str,
        event_topic: str,
        from_block: int = -10000,
        limit: int = 100,
    ) -> list[OnChainEvent]:
        """Fetch contract event logs from Etherscan.

        Parameters
        ----------
        contract_address : str
            Ethereum contract address (e.g., Uniswap V3 pool).
        event_topic : str
            keccak256 of the event ABI signature, e.g.
            EVENT_TOPICS["Transfer"] for ERC-20 Transfer events.
        from_block : int
            Negative = relative to latest (default -10000 = last ~1.5 days).
            Positive = absolute block number.
        limit : int
            Maximum number of log entries to return (Etherscan max: 1000).
        """
        await _etherscan_throttle()

        # Resolve from_block
        if from_block < 0:
            latest_data = await self._get_json(
                ETHERSCAN_BASE,
                params={
                    "module": "proxy",
                    "action": "eth_blockNumber",
                    "apikey": self._etherscan_key or "YourApiKeyToken",
                },
                cache_ttl=15.0,
            )
            if isinstance(latest_data, dict):
                latest_hex = latest_data.get("result", "0x0")
                latest_block = int(latest_hex, 16)
                resolved_from = max(0, latest_block + from_block)
            else:
                resolved_from = 0
        else:
            resolved_from = from_block

        await _etherscan_throttle()
        data = await self._get_json(
            ETHERSCAN_BASE,
            params={
                "module": "logs",
                "action": "getLogs",
                "address": contract_address,
                "topic0": event_topic,
                "fromBlock": resolved_from,
                "toBlock": "latest",
                "page": 1,
                "offset": min(limit, 1000),
                "apikey": self._etherscan_key or "YourApiKeyToken",
            },
            cache_ttl=60.0,
        )

        events: list[OnChainEvent] = []
        if not isinstance(data, dict) or data.get("status") != "1":
            logger.debug(
                "monitor_contract_events: no results for %s topic %s",
                contract_address, event_topic,
            )
            return events

        event_name = next(
            (name for name, sig in EVENT_TOPICS.items() if sig == event_topic),
            None,
        )

        for raw in (data.get("result") or [])[:limit]:
            try:
                block_num = int(raw.get("blockNumber", "0x0"), 16)
                ts_unix   = int(raw.get("timeStamp", "0x0"), 16)
                ts        = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                tx_hash   = raw.get("transactionHash", "")
                address   = raw.get("address", "")
                topics    = raw.get("topics", [])

                from_addr = None
                to_addr   = None
                # For standard Transfer: topics[1] = from (padded), topics[2] = to
                if len(topics) >= 3 and event_name in ("Transfer", "Swap"):
                    from_addr = _unpad_address(topics[1]) if len(topics) > 1 else None
                    to_addr   = _unpad_address(topics[2]) if len(topics) > 2 else None

                # Parse value from data field (first 32 bytes = first uint256)
                data_hex = raw.get("data", "0x")
                value_int = _parse_hex_uint256(data_hex)
                value_eth = value_int / 1e18 if value_int else None

                events.append(OnChainEvent(
                    tx_hash=tx_hash,
                    block_number=block_num,
                    timestamp=ts,
                    contract_address=address.lower(),
                    event_name=event_name,
                    event_topic=event_topic,
                    from_address=from_addr,
                    to_address=to_addr,
                    value_eth=round(value_eth, 8) if value_eth else None,
                    chain="ethereum",
                ))
            except Exception as exc:
                logger.debug("monitor_contract_events parse error: %s", exc)

        return events

    async def get_whale_alerts(
        self,
        token_address: str,
        min_amount_usd: float = 1_000_000,
        hours_back: int = 24,
    ) -> list[WhaleAlert]:
        """Detect large token transfers (whale movements) via Etherscan tokentx.

        Classifies from/to addresses as exchange inflow/outflow, wallet
        accumulation, or DeFi interaction.
        """
        transfers = await self.get_token_transfers(
            address=token_address,
            token_address=token_address,
            limit=1000,
        )

        # Filter by time window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        recent = [t for t in transfers if t.timestamp >= cutoff]

        # Get token price to compute USD amounts
        symbol = recent[0].token_symbol if recent else "UNKNOWN"

        # Fetch token price via CoinGecko by contract address
        await _coingecko_throttle()
        price_data = await self._get_json(
            f"{COINGECKO_BASE}/simple/token_price/ethereum",
            params={
                "contract_addresses": token_address,
                "vs_currencies": "usd",
            },
            cache_ttl=120.0,
        )
        token_price = 0.0
        if isinstance(price_data, dict):
            for _, val in price_data.items():
                token_price = float(val.get("usd", 0))
                break

        alerts: list[WhaleAlert] = []
        for tx in recent:
            amount_usd = tx.amount * token_price if token_price else (tx.amount_usd or 0.0)
            if amount_usd < min_amount_usd:
                continue

            from_label = self._classify_address(tx.from_address)
            to_label   = self._classify_address(tx.to_address)

            # Classify alert type
            exchange_labels = {
                "Binance Cold Wallet", "Binance Hot Wallet",
                "Binance Exchange Wallet", "Crypto.com Hot Wallet",
            }
            defi_labels = {
                "Lido stETH", "Aave V2 Lending Pool", "Aave V3 Lending Pool",
                "Uniswap V3 Router", "Uniswap V3 Factory", "Curve 3pool",
                "Compound V3 USDC",
            }

            if to_label in exchange_labels:
                alert_type = "exchange_inflow"
            elif from_label in exchange_labels:
                alert_type = "exchange_outflow"
            elif to_label in defi_labels or from_label in defi_labels:
                alert_type = "defi_interaction"
            else:
                alert_type = "wallet_accumulation"

            alerts.append(WhaleAlert(
                timestamp=tx.timestamp,
                token_symbol=tx.token_symbol,
                amount=tx.amount,
                amount_usd=round(amount_usd, 2),
                from_label=from_label,
                to_label=to_label,
                tx_hash=tx.tx_hash,
                alert_type=alert_type,
            ))

        alerts.sort(key=lambda a: a.amount_usd, reverse=True)
        return alerts

    async def get_token_transfers(
        self,
        address: str,
        token_address: Optional[str] = None,
        limit: int = 100,
    ) -> list[TokenTransfer]:
        """Fetch ERC-20 token transfers for an address via Etherscan tokentx.

        Parameters
        ----------
        address : str
            Wallet or contract address to query.
        token_address : str, optional
            Filter by specific token contract address.
        limit : int
            Maximum number of transfers to return (max 10 000 via pagination).
        """
        await _etherscan_throttle()

        params: dict = {
            "module":     "account",
            "action":     "tokentx",
            "address":    address,
            "startblock": 0,
            "endblock":   "latest",
            "sort":       "desc",
            "page":       1,
            "offset":     min(limit, 10_000),
            "apikey":     self._etherscan_key or "YourApiKeyToken",
        }
        if token_address:
            params["contractaddress"] = token_address

        data = await self._get_json(ETHERSCAN_BASE, params=params, cache_ttl=120.0)

        transfers: list[TokenTransfer] = []
        if not isinstance(data, dict) or data.get("status") not in ("1", 1):
            logger.debug("get_token_transfers: no data for %s", address)
            return transfers

        for raw in (data.get("result") or [])[:limit]:
            try:
                decimals = int(raw.get("tokenDecimal", "18") or "18")
                raw_value = int(raw.get("value", "0") or "0")
                amount = raw_value / (10 ** decimals)
                ts = datetime.fromtimestamp(
                    int(raw.get("timeStamp", "0")), tz=timezone.utc
                )
                transfers.append(TokenTransfer(
                    tx_hash=raw.get("hash", ""),
                    block=int(raw.get("blockNumber", 0)),
                    timestamp=ts,
                    token_address=raw.get("contractAddress", "").lower(),
                    token_name=raw.get("tokenName", "Unknown"),
                    token_symbol=raw.get("tokenSymbol", "???"),
                    from_address=raw.get("from", "").lower(),
                    to_address=raw.get("to", "").lower(),
                    amount=round(amount, 8),
                    amount_usd=None,   # enriched later if price is known
                ))
            except Exception as exc:
                logger.debug("get_token_transfers parse error: %s", exc)

        return transfers

    # -----------------------------------------------------------------------
    # DeFiLlama
    # -----------------------------------------------------------------------

    async def get_defi_protocols(
        self,
        min_tvl: float = 100_000_000,
        category: Optional[str] = None,
    ) -> list[DefiProtocol]:
        """Fetch all DeFi protocols from DeFiLlama, filtered by TVL and category.

        DeFiLlama /protocols returns a flat list of all tracked protocols with
        TVL, category, chain, and fee/revenue data where available.
        """
        data = await self._get_json(
            f"{DEFILLAMA_BASE}/protocols",
            cache_ttl=300.0,
        )
        if not isinstance(data, list):
            logger.error("get_defi_protocols: unexpected response type %s", type(data))
            return []

        protocols: list[DefiProtocol] = []
        for raw in data:
            try:
                tvl = float(raw.get("tvl") or 0)
                if tvl < min_tvl:
                    continue

                cat = raw.get("category", "")
                if category and cat.lower() != category.lower():
                    continue

                # Primary chain: use the first chain in the chain list
                chains: list[str] = raw.get("chains", [])
                primary_chain = chains[0] if chains else raw.get("chain", "Ethereum")

                # TVL changes
                change_1d = float(raw.get("change_1d") or 0)
                change_7d = float(raw.get("change_7d") or 0)

                protocols.append(DefiProtocol(
                    name=raw.get("name", ""),
                    slug=raw.get("slug", ""),
                    category=cat,
                    chain=primary_chain,
                    tvl_usd=round(tvl, 2),
                    tvl_change_24h=round(change_1d, 4),
                    tvl_change_7d=round(change_7d, 4),
                    revenue_24h=float(raw.get("revenue24h") or 0) or None,
                    fees_24h=float(raw.get("fees24h") or 0) or None,
                    token_symbol=raw.get("symbol"),
                    token_price=float(raw.get("tokenBreakdowns", {}).get("price", 0) or 0) or None,
                ))
            except Exception as exc:
                logger.debug("get_defi_protocols parse error: %s", exc)

        protocols.sort(key=lambda p: p.tvl_usd, reverse=True)
        logger.info(
            "get_defi_protocols: %d protocols above %.0f TVL",
            len(protocols), min_tvl,
        )
        return protocols

    async def get_protocol_history(
        self,
        slug: str,
        days_back: int = 90,
    ) -> dict[str, float]:
        """Fetch TVL history for a specific protocol from DeFiLlama.

        Returns {date_iso: tvl_usd} for the last `days_back` calendar days.
        """
        data = await self._get_json(
            f"{DEFILLAMA_BASE}/protocol/{slug}",
            cache_ttl=600.0,
        )
        if not isinstance(data, dict):
            logger.error("get_protocol_history: bad response for %s", slug)
            return {}

        tvl_records: list[dict] = data.get("tvl", [])
        cutoff_ts = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).timestamp()

        history: dict[str, float] = {}
        for rec in tvl_records:
            try:
                ts = float(rec.get("date", 0))
                if ts < cutoff_ts:
                    continue
                date_str = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                history[date_str] = round(float(rec.get("totalLiquidityUSD", 0)), 2)
            except Exception as exc:
                logger.debug("get_protocol_history parse error: %s", exc)

        return dict(sorted(history.items()))

    async def get_chain_tvl_summary(self) -> dict[str, float]:
        """Fetch total TVL per blockchain from DeFiLlama.

        Returns {chain_name: tvl_usd} sorted by TVL descending.
        """
        data = await self._get_json(
            f"{DEFILLAMA_BASE}/v2/chains",
            cache_ttl=300.0,
        )
        if not isinstance(data, list):
            logger.error("get_chain_tvl_summary: unexpected response")
            return {}

        summary: dict[str, float] = {}
        for chain in data:
            try:
                name = chain.get("name", "Unknown")
                tvl  = float(chain.get("tvl", 0))
                summary[name] = round(tvl, 2)
            except Exception as exc:
                logger.debug("get_chain_tvl_summary parse error: %s", exc)

        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))

    async def get_stablecoin_metrics(self) -> list[StablecoinMetrics]:
        """Fetch stablecoin supply, peg, and chain distribution from DeFiLlama.

        Flags depeg risk when peg deviation > 0.5%.
        """
        data = await self._get_json(
            f"{STABLECOINS_API}/stablecoins",
            params={"includePrices": "true"},
            cache_ttl=300.0,
        )
        if not isinstance(data, dict):
            logger.error("get_stablecoin_metrics: unexpected response")
            return []

        pegged_assets: list[dict] = data.get("peggedAssets", [])
        result: list[StablecoinMetrics] = []

        for raw in pegged_assets:
            try:
                symbol     = raw.get("symbol", "???")
                name       = raw.get("name", symbol)
                peg_type   = raw.get("pegType", "FIAT")
                peg_mechanism = raw.get("pegMechanism", "")

                # Normalise peg type
                if "algo" in peg_mechanism.lower():
                    peg_type = "ALGO"
                elif "crypto" in peg_type.lower():
                    peg_type = "CRYPTO"
                else:
                    peg_type = "FIAT"

                # Total circulating in USD
                circulating_map: dict = raw.get("circulating", {})
                circulating_usd = float(
                    circulating_map.get("peggedUSD", 0) or
                    circulating_map.get("peggedEUR", 0) or 0
                )

                # Current price for peg deviation
                price_data: dict = raw.get("price") or {}
                if isinstance(price_data, dict):
                    current_price = float(price_data.get("peggedUSD", 1.0) or 1.0)
                elif isinstance(price_data, (int, float)):
                    current_price = float(price_data)
                else:
                    current_price = 1.0

                peg_deviation_pct = round(abs(current_price - 1.0) * 100, 4)

                # Chain distribution (% of total supply per chain)
                chain_balances: dict = raw.get("chainBalances", {})
                chain_dist: dict[str, float] = {}
                total_supply = sum(
                    float((v or {}).get("peggedUSD", 0) or 0)
                    for v in chain_balances.values()
                    if isinstance(v, dict)
                ) or 1.0

                for chain_name, chain_data in chain_balances.items():
                    if not isinstance(chain_data, dict):
                        continue
                    chain_usd = float(chain_data.get("peggedUSD", 0) or 0)
                    chain_pct = round(chain_usd / total_supply * 100, 2)
                    if chain_pct >= 0.1:   # only report chains with >= 0.1% share
                        chain_dist[chain_name] = chain_pct

                # Depeg risk classification
                if peg_deviation_pct > 2.0 or peg_type == "ALGO":
                    depeg_risk = "high"
                elif peg_deviation_pct > 0.5:
                    depeg_risk = "medium"
                else:
                    depeg_risk = "low"

                result.append(StablecoinMetrics(
                    symbol=symbol,
                    name=name,
                    peg_type=peg_type,
                    circulating_usd=round(circulating_usd, 2),
                    peg_deviation_pct=peg_deviation_pct,
                    chain_distribution=chain_dist,
                    depeg_risk=depeg_risk,
                ))
            except Exception as exc:
                logger.debug("get_stablecoin_metrics parse error: %s", exc)

        result.sort(key=lambda s: s.circulating_usd, reverse=True)
        return result

    # -----------------------------------------------------------------------
    # DEX analytics
    # -----------------------------------------------------------------------

    async def get_amm_snapshot(self) -> AMMSnapshot:
        """Comprehensive DEX/AMM state snapshot.

        Combines DeFiLlama DEX volume overview with Uniswap V3 top pools
        from The Graph to produce a unified liquidity and volume picture.
        """
        # Fetch in parallel to reduce latency
        dex_overview_task = asyncio.create_task(
            self._get_json(f"{DEFILLAMA_BASE}/overview/dexs", cache_ttl=300.0)
        )
        pools_task = asyncio.create_task(
            self.get_uniswap_pools(min_tvl=5_000_000, limit=20)
        )
        dex_data, top_pools = await asyncio.gather(dex_overview_task, pools_task)

        total_volume_24h = 0.0
        total_tvl        = 0.0
        protocol_breakdown: list[dict] = []

        if isinstance(dex_data, dict):
            total_volume_24h = float(dex_data.get("total24h") or 0)
            total_tvl        = float(dex_data.get("totalValueLocked") or 0)

            protocols_raw: list[dict] = dex_data.get("protocols", [])
            for p in sorted(
                protocols_raw,
                key=lambda x: float(x.get("total24h") or 0),
                reverse=True,
            )[:15]:
                vol = float(p.get("total24h") or 0)
                share = (vol / total_volume_24h * 100) if total_volume_24h else 0.0
                protocol_breakdown.append({
                    "protocol":         p.get("name", ""),
                    "volume_24h":       round(vol, 2),
                    "market_share_pct": round(share, 2),
                })

        # Identify dominant pairs from top pools
        pair_counts: dict[str, int] = {}
        for pool in top_pools:
            pair = f"{pool.token0}/{pool.token1}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        dominant_pairs = sorted(pair_counts, key=lambda k: pair_counts[k], reverse=True)[:10]

        return AMMSnapshot(
            as_of=datetime.now(timezone.utc),
            total_dex_volume_24h=round(total_volume_24h, 2),
            total_dex_tvl=round(total_tvl, 2),
            top_pools=top_pools[:10],
            protocol_breakdown=protocol_breakdown,
            dominant_pairs=dominant_pairs,
        )

    async def get_uniswap_pools(
        self,
        min_tvl: float = 1_000_000,
        limit: int = 50,
    ) -> list[UniswapPool]:
        """Query Uniswap V3 pools from The Graph subgraph.

        Computes fee APR = (volume_7d * feeTier/1e6) / tvl * 52
        to give an annualised fee yield for liquidity providers.
        """
        query_limit = min(limit * 2, 200)   # fetch extra to filter by min_tvl
        graphql_query = """
        {
          pools(
            first: %d,
            orderBy: totalValueLockedUSD,
            orderDirection: desc,
            where: { totalValueLockedUSD_gt: "%s" }
          ) {
            id
            token0 { symbol }
            token1 { symbol }
            feeTier
            totalValueLockedUSD
            volumeUSD
            poolDayData(first: 7, orderBy: date, orderDirection: desc) {
              volumeUSD
            }
            liquidity
            token0Price
          }
        }
        """ % (query_limit, int(min_tvl))

        data = await self._post_json(
            GRAPH_UNISWAP,
            body={"query": graphql_query},
            cache_ttl=120.0,
        )

        pools: list[UniswapPool] = []
        if not isinstance(data, dict):
            logger.error("get_uniswap_pools: unexpected response from The Graph")
            return pools

        raw_pools: list[dict] = (data.get("data") or {}).get("pools", [])

        for raw in raw_pools[:limit]:
            try:
                tvl = float(raw.get("totalValueLockedUSD", 0))
                if tvl < min_tvl:
                    continue

                fee_tier = int(raw.get("feeTier", 3000))

                # Sum 7-day volume from poolDayData
                day_data: list[dict] = raw.get("poolDayData", [])
                volume_7d = sum(float(d.get("volumeUSD", 0)) for d in day_data)
                # 24h volume: use the most recent day entry
                volume_24h = float(day_data[0].get("volumeUSD", 0)) if day_data else 0.0

                # Annualised fee APR for LPs
                fee_apr = 0.0
                if tvl > 0 and volume_7d > 0:
                    fee_rate = fee_tier / 1_000_000   # e.g., 3000 → 0.003 = 0.3%
                    fee_apr  = round((volume_7d * fee_rate) / tvl * 52 * 100, 2)

                liquidity_raw = raw.get("liquidity")
                liquidity = float(liquidity_raw) if liquidity_raw else None

                token0_price_raw = raw.get("token0Price")
                token0_price = float(token0_price_raw) if token0_price_raw else None

                pools.append(UniswapPool(
                    pool_address=raw.get("id", "").lower(),
                    token0=raw.get("token0", {}).get("symbol", "?"),
                    token1=raw.get("token1", {}).get("symbol", "?"),
                    fee_tier=fee_tier,
                    tvl_usd=round(tvl, 2),
                    volume_24h=round(volume_24h, 2),
                    volume_7d=round(volume_7d, 2),
                    fee_apr=fee_apr,
                    liquidity=liquidity,
                    price_token0_in_token1=round(token0_price, 8) if token0_price else None,
                ))
            except Exception as exc:
                logger.debug("get_uniswap_pools parse error: %s", exc)

        pools.sort(key=lambda p: p.tvl_usd, reverse=True)
        logger.info("get_uniswap_pools: %d pools found (min TVL %.0f)", len(pools), min_tvl)
        return pools

    async def get_yield_opportunities(
        self,
        min_apy: float = 5.0,
        max_tvl_risk: float = 10_000_000,
        stable_only: bool = False,
    ) -> list[YieldOpportunity]:
        """Fetch DeFi yield opportunities from DeFiLlama Yields API.

        Parameters
        ----------
        min_apy : float
            Minimum base APY threshold (percent).
        max_tvl_risk : float
            Pools with TVL below this are marked "high" risk.
        stable_only : bool
            If True, return only pools where both tokens are stablecoins.
        """
        data = await self._get_json(
            f"{DEFI_YIELDS}/pools",
            cache_ttl=300.0,
        )
        if not isinstance(data, dict):
            logger.error("get_yield_opportunities: unexpected response")
            return []

        raw_pools: list[dict] = data.get("data", [])
        opportunities: list[YieldOpportunity] = []

        for raw in raw_pools:
            try:
                apy_base    = float(raw.get("apy") or 0)
                apy_rewards = float(raw.get("apyReward") or 0)
                total_apy   = apy_base + apy_rewards

                if total_apy < min_apy:
                    continue

                tvl = float(raw.get("tvlUsd") or 0)
                if tvl <= 0:
                    continue

                symbol      = raw.get("symbol", "")
                chain       = raw.get("chain", "")
                project     = raw.get("project", "")
                pool_id     = raw.get("pool", "")

                # Determine if this is a stablecoin pool
                tokens = [t.strip().upper() for t in symbol.split("-")]
                is_stable = all(t in STABLECOIN_SYMBOLS for t in tokens if t)

                if stable_only and not is_stable:
                    continue

                # Risk classification
                if is_stable and tvl > 10_000_000:
                    risk = "low"
                elif tvl < max_tvl_risk or not is_stable:
                    risk = "high" if tvl < max_tvl_risk / 2 else "medium"
                else:
                    risk = "medium"

                # Token pair display
                token_pair = "-".join(tokens[:2]) if len(tokens) >= 2 else symbol

                opportunities.append(YieldOpportunity(
                    protocol=project,
                    pool_name=symbol,
                    chain=chain,
                    apy=round(apy_base, 4),
                    apy_with_rewards=round(total_apy, 4),
                    tvl_usd=round(tvl, 2),
                    token_pair=token_pair,
                    risk_level=risk,
                    pool_id=pool_id,
                    stable_pool=is_stable,
                ))
            except Exception as exc:
                logger.debug("get_yield_opportunities parse error: %s", exc)

        opportunities.sort(key=lambda o: o.apy_with_rewards, reverse=True)
        logger.info(
            "get_yield_opportunities: %d opportunities (min APY %.1f%%, stable_only=%s)",
            len(opportunities), min_apy, stable_only,
        )
        return opportunities

    async def get_aave_market_data(self) -> list[dict]:
        """Query Aave V3 market data from The Graph subgraph.

        Returns a list of reserve records with:
            asset, supply_apy, borrow_apy, utilization_pct, tvl_usd
        """
        graphql_query = """
        {
          reserves(
            first: 50,
            orderBy: totalLiquidityUSD,
            orderDirection: desc,
            where: { isActive: true }
          ) {
            id
            symbol
            name
            totalLiquidityUSD
            totalDepositBalanceUSD
            totalDebtBalanceUSD
            supplyAPY
            variableBorrowAPY
            stableBorrowAPY
            utilizationRate
            isActive
            isFrozen
            decimals
          }
        }
        """
        data = await self._post_json(
            GRAPH_AAVE,
            body={"query": graphql_query},
            cache_ttl=120.0,
        )

        markets: list[dict] = []
        if not isinstance(data, dict):
            logger.error("get_aave_market_data: unexpected response from The Graph")
            return markets

        reserves: list[dict] = (data.get("data") or {}).get("reserves", [])

        for raw in reserves:
            try:
                tvl     = float(raw.get("totalLiquidityUSD") or 0)
                deposits = float(raw.get("totalDepositBalanceUSD") or 0)
                borrows  = float(raw.get("totalDebtBalanceUSD") or 0)
                util_raw = float(raw.get("utilizationRate") or 0)
                util_pct = round(util_raw * 100, 2)   # The Graph returns 0–1

                # APY fields already in decimal (multiply by 100 for percent)
                supply_apy = round(float(raw.get("supplyAPY") or 0) * 100, 4)
                var_borrow = round(float(raw.get("variableBorrowAPY") or 0) * 100, 4)
                stb_borrow = round(float(raw.get("stableBorrowAPY") or 0) * 100, 4)

                markets.append({
                    "asset":            raw.get("symbol", "?"),
                    "name":             raw.get("name", ""),
                    "supply_apy":       supply_apy,
                    "borrow_apy_var":   var_borrow,
                    "borrow_apy_stable": stb_borrow,
                    "utilization_pct":  util_pct,
                    "tvl_usd":          round(tvl, 2),
                    "total_deposits":   round(deposits, 2),
                    "total_borrows":    round(borrows, 2),
                    "is_frozen":        raw.get("isFrozen", False),
                })
            except Exception as exc:
                logger.debug("get_aave_market_data parse error: %s", exc)

        markets.sort(key=lambda m: m["tvl_usd"], reverse=True)
        logger.info("get_aave_market_data: %d active Aave V3 markets", len(markets))
        return markets

    # -----------------------------------------------------------------------
    # Cross-chain flows
    # -----------------------------------------------------------------------

    async def get_cross_chain_flows(self) -> dict:
        """Fetch bridge volume data from DeFiLlama to measure cross-chain flows.

        Returns daily inflow/outflow aggregated across major bridges.
        """
        data = await self._get_json(
            f"{DEFILLAMA_BASE}/overview/bridges",
            cache_ttl=600.0,
        )
        if not isinstance(data, dict):
            logger.error("get_cross_chain_flows: unexpected response")
            return {}

        bridges_raw: list[dict] = data.get("protocols", [])
        total_24h = float(data.get("total24h") or 0)
        total_7d  = float(data.get("total7d") or 0)

        top_bridges = []
        for bridge in sorted(
            bridges_raw,
            key=lambda x: float(x.get("total24h") or 0),
            reverse=True,
        )[:10]:
            vol_24h = float(bridge.get("total24h") or 0)
            share   = (vol_24h / total_24h * 100) if total_24h else 0.0
            top_bridges.append({
                "name":           bridge.get("name", ""),
                "volume_24h_usd": round(vol_24h, 2),
                "volume_7d_usd":  round(float(bridge.get("total7d") or 0), 2),
                "market_share":   round(share, 2),
                "chains":         bridge.get("chains", []),
            })

        return {
            "as_of":             datetime.now(timezone.utc).isoformat(),
            "total_bridge_24h":  round(total_24h, 2),
            "total_bridge_7d":   round(total_7d, 2),
            "top_bridges":       top_bridges,
        }

    # -----------------------------------------------------------------------
    # Dune Analytics
    # -----------------------------------------------------------------------

    async def run_dune_query(
        self,
        query_id: int,
        query_params: Optional[dict] = None,
        max_wait_seconds: int = 60,
    ) -> Optional[list[dict]]:
        """Execute a Dune Analytics query and return result rows.

        Polls for completion up to `max_wait_seconds`.
        Requires DUNE_API_KEY environment variable.

        Parameters
        ----------
        query_id : int
            Dune query ID (visible in query URL on dune.com).
        query_params : dict, optional
            Named query parameters to override in the query.
        max_wait_seconds : int
            Maximum seconds to wait for the query execution to complete.
        """
        if not self._dune_key:
            logger.warning("run_dune_query: DUNE_API_KEY not set — skipping Dune query %d", query_id)
            return None

        headers = {"X-Dune-API-Key": self._dune_key, "Content-Type": "application/json"}
        body: dict = {}
        if query_params:
            body["query_parameters"] = query_params

        # Execute query
        exec_data = await self._post_json(
            f"{DUNE_BASE}/query/{query_id}/execute",
            body=body,
            headers=headers,
            cache_ttl=0,   # never cache executions
        )
        if not isinstance(exec_data, dict):
            logger.error("run_dune_query: failed to start execution for query %d", query_id)
            return None

        execution_id = exec_data.get("execution_id")
        if not execution_id:
            logger.error("run_dune_query: no execution_id returned for query %d", query_id)
            return None

        # Poll for results
        poll_url = f"{DUNE_BASE}/execution/{execution_id}/results"
        deadline = time.monotonic() + max_wait_seconds
        poll_interval = 3.0

        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            result_data = await self._get_json(
                poll_url,
                headers=headers,
                cache_ttl=0,
            )
            if not isinstance(result_data, dict):
                break

            state = result_data.get("state", "")
            if state == "QUERY_STATE_COMPLETED":
                rows: list[dict] = (
                    (result_data.get("result") or {})
                    .get("rows", [])
                )
                logger.info(
                    "run_dune_query: query %d completed — %d rows", query_id, len(rows)
                )
                return rows
            elif state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                logger.error(
                    "run_dune_query: query %d ended with state %s", query_id, state
                )
                return None
            # Still pending — continue polling
            poll_interval = min(poll_interval * 1.5, 15.0)

        logger.warning(
            "run_dune_query: query %d did not complete within %ds", query_id, max_wait_seconds
        )
        return None

    # -----------------------------------------------------------------------
    # Address classification
    # -----------------------------------------------------------------------

    def _classify_address(self, address: str) -> str:
        """Return a human-readable label for a known Ethereum address.

        Falls back to 'Unknown Wallet' for unlabelled addresses.
        """
        return KNOWN_LABELS.get(address.lower(), "Unknown Wallet")

    # -----------------------------------------------------------------------
    # CoinGecko DeFi overview
    # -----------------------------------------------------------------------

    async def get_defi_market_overview(self) -> dict:
        """Fetch DeFi market aggregate stats from CoinGecko.

        Returns total DeFi market cap, volume, ETH dominance, and DeFi/total ratio.
        """
        await _coingecko_throttle()
        data = await self._get_json(
            f"{COINGECKO_BASE}/global/decentralized_finance_defi",
            cache_ttl=300.0,
        )
        if not isinstance(data, dict):
            logger.error("get_defi_market_overview: unexpected response")
            return {}

        gd = data.get("data", {})
        return {
            "defi_market_cap_usd":   float(gd.get("defi_market_cap", 0)),
            "eth_market_cap_usd":    float(gd.get("eth_market_cap", 0)),
            "defi_to_eth_ratio":     float(gd.get("defi_to_eth_ratio", 0)),
            "trading_volume_24h":    float(gd.get("trading_volume_24h", 0)),
            "defi_dominance":        float(gd.get("defi_dominance", 0)),
            "top_coin_name":         gd.get("top_coin_name", ""),
            "top_coin_defi_dominance": float(gd.get("top_coin_defi_dominance", 0)),
        }


# ---------------------------------------------------------------------------
# Hex parsing helpers
# ---------------------------------------------------------------------------

def _unpad_address(hex_topic: str) -> str:
    """Convert a 32-byte zero-padded topic to a 20-byte Ethereum address."""
    raw = hex_topic.lstrip("0x").lstrip("0x")
    # Ethereum addresses are the last 40 hex characters (20 bytes) of a 64-char field
    cleaned = hex_topic.replace("0x", "").replace("0X", "")
    if len(cleaned) >= 40:
        return "0x" + cleaned[-40:].lower()
    return "0x" + cleaned.lower()


def _parse_hex_uint256(data_hex: str) -> Optional[int]:
    """Parse the first uint256 from a raw ABI-encoded data field.

    Etherscan returns the `data` field as a hex string with 32-byte words.
    The first word (bytes 0–31, chars 2–66) is the first uint256.
    """
    if not data_hex or data_hex in ("0x", "0X", ""):
        return None
    try:
        cleaned = data_hex.lstrip("0x").lstrip("0X")
        if not cleaned:
            return None
        # Take only the first 64 hex chars (32 bytes = one uint256)
        word = cleaned[:64]
        return int(word, 16)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_adapter: Optional[OnChainAdapter] = None


def _get_adapter() -> OnChainAdapter:
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = OnChainAdapter()
    return _default_adapter


async def gas_oracle() -> GasOracle:
    """Module-level helper: fetch live gas oracle."""
    return await _get_adapter().get_gas_oracle()


async def defi_protocols(min_tvl: float = 100_000_000) -> list[DefiProtocol]:
    """Module-level helper: fetch DeFi protocols above min TVL."""
    return await _get_adapter().get_defi_protocols(min_tvl=min_tvl)


async def amm_snapshot() -> AMMSnapshot:
    """Module-level helper: fetch comprehensive DEX/AMM snapshot."""
    return await _get_adapter().get_amm_snapshot()


async def yield_opportunities(min_apy: float = 5.0) -> list[YieldOpportunity]:
    """Module-level helper: fetch yield opportunities above min APY."""
    return await _get_adapter().get_yield_opportunities(min_apy=min_apy)


async def whale_alerts(
    token_address: str,
    min_amount_usd: float = 1_000_000,
) -> list[WhaleAlert]:
    """Module-level helper: detect whale movements for a token contract."""
    return await _get_adapter().get_whale_alerts(
        token_address=token_address,
        min_amount_usd=min_amount_usd,
    )
