"""
On-chain event monitoring: whale transactions, smart contract events,
protocol governance votes, token transfers, bridge movements.
Free data: Etherscan (free tier), blockchain.info, DeFiLlama events.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
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

_ETHERSCAN_BASE = "https://api.etherscan.io/api"
_BLOCKCHAIN_INFO_TX = "https://blockchain.info/rawtx/{}"
_BLOCKCHAIN_INFO_ADDRESS = "https://blockchain.info/rawaddr/{}"
_BLOCKCHAIN_INFO_UNCONFIRMED = "https://blockchain.info/unconfirmed-transactions?format=json"
_DEFILLAMA_BRIDGES = "https://bridges.llama.fi"
_SNAPSHOT_GRAPHQL = "https://hub.snapshot.org/graphql"
_TALLY_GRAPHQL = "https://api.tally.xyz/query"
_DEFILLAMA_EVENTS = "https://api.llama.fi/protocols"

_HEADERS = {"User-Agent": "SENTINEL-financial-terminal/2.0", "Accept": "application/json"}
_TIMEOUT = 25.0
_CACHE_TTL = 60  # 1-minute cache for on-chain data

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "onchain_events.db"


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS whale_txs (
                tx_hash         TEXT PRIMARY KEY,
                chain           TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                amount          REAL NOT NULL,
                amount_usd      REAL,
                from_addr       TEXT,
                to_addr         TEXT,
                direction       TEXT,
                classification  TEXT,
                ts              INTEGER,
                detected_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exchange_flows (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                chain           TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                exchange        TEXT NOT NULL,
                flow_type       TEXT NOT NULL,
                amount          REAL NOT NULL,
                amount_usd      REAL,
                tx_hash         TEXT,
                ts              INTEGER,
                recorded_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS governance_proposals (
                proposal_id     TEXT PRIMARY KEY,
                protocol        TEXT NOT NULL,
                title           TEXT,
                state           TEXT,
                start_ts        INTEGER,
                end_ts          INTEGER,
                votes_for       REAL,
                votes_against   REAL,
                quorum_pct      REAL,
                impact_score    REAL,
                fetched_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bridge_flows (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                bridge_name     TEXT NOT NULL,
                source_chain    TEXT NOT NULL,
                dest_chain      TEXT NOT NULL,
                volume_usd      REAL,
                net_flow_usd    REAL,
                recorded_date   TEXT NOT NULL,
                anomaly_score   REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS contract_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                contract        TEXT NOT NULL,
                protocol        TEXT,
                event_type      TEXT NOT NULL,
                tx_hash         TEXT,
                block_number    INTEGER,
                amount_usd      REAL,
                severity        TEXT,
                raw_data        TEXT,
                detected_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS onchain_signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                signal_type     TEXT NOT NULL,
                score           REAL NOT NULL,
                direction       TEXT,
                rationale       TEXT,
                computed_at     TEXT NOT NULL
            );
        """)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------

_mem_cache: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _mem_cache[key]
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _mem_cache[key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Known exchange hot wallets (major centralised exchanges)
# ---------------------------------------------------------------------------

KNOWN_EXCHANGE_WALLETS: Dict[str, str] = {
    # Binance
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
    "0xd551234ae421e3bcba99a0da6d736074f22192ff": "Binance",
    "0x564286362092d8e7936f0549571a803b203aaced": "Binance",
    # Coinbase
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e": "Coinbase",
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13": "Kraken",
    # FTX (historical)
    "0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2": "FTX",
    # Gemini
    "0xd24400ae8bfebb18ca49be86258a3c749cf46853": "Gemini",
    # Bitfinex
    "0x1151314c646ce4e0efd76d1af4760ae66a9fe30f": "Bitfinex",
    "0x742d35cc6634c0532925a3b844bc454e4438f44e": "Bitfinex (hot)",
    # Huobi
    "0xab5c66752a9e8167967685f1450532fb96d5d24f": "Huobi",
    "0x6748f50f686bfbca6fe8ad62b22228b87f31ff2b": "Huobi",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX",
    # KuCoin
    "0x2b5634c42055806a59e9107ed44d43c426e58258": "KuCoin",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "Bybit",
    # Crypto.com
    "0x72a53cdbbcc1b9efa39c834a540550e23463aacb": "Crypto.com",
    # BTC known exchange addresses
    "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s": "Binance BTC Cold",
    "3E1jkF7PQZiSHHMpGGXvYcfA1NX5YWXDSX": "Coinbase BTC",
}

# ---------------------------------------------------------------------------
# Known DeFi protocol contracts
# ---------------------------------------------------------------------------

KNOWN_CONTRACTS: Dict[str, Dict[str, str]] = {
    # Uniswap
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": {"protocol": "Uniswap", "type": "token"},
    "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f": {"protocol": "Uniswap V2", "type": "factory"},
    "0x1f98431c8ad98523631ae4a59f267346ea31f984": {"protocol": "Uniswap V3", "type": "factory"},
    # Aave
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": {"protocol": "Aave", "type": "token"},
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": {"protocol": "Aave V3", "type": "pool"},
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": {"protocol": "Aave V2", "type": "pool"},
    # Compound
    "0xc00e94cb662c3520282e6f5717214004a7f26888": {"protocol": "Compound", "type": "token"},
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": {"protocol": "Compound V2", "type": "comptroller"},
    # Curve
    "0xd533a949740bb3306d119cc777fa900ba034cd52": {"protocol": "Curve", "type": "token"},
    "0xbabe61887f1de2713c6f97e567623453d3c79f67": {"protocol": "Curve Factory", "type": "factory"},
    # MakerDAO / DAI
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": {"protocol": "MakerDAO", "type": "token"},
    "0x6b175474e89094c44da98b954eedeac495271d0f": {"protocol": "DAI", "type": "stablecoin"},
    "0x35d1b3f3d7966a1dfe207aa4514c12a259a0492b": {"protocol": "MakerDAO Vat", "type": "core"},
    # Lido
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": {"protocol": "Lido stETH", "type": "lst"},
    "0x5a98fcbea516cf06857215779fd812ca3bef1b32": {"protocol": "Lido DAO", "type": "governance"},
    # Chainlink
    "0x514910771af9ca656af840dff83e8264ecf986ca": {"protocol": "Chainlink", "type": "token"},
    # Synthetix
    "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f": {"protocol": "Synthetix", "type": "token"},
    # Balancer
    "0xba100000625a3754423978a60c9317c58a424e3d": {"protocol": "Balancer", "type": "token"},
    "0xba12222222228d8ba445958a75a0704d566bf2c8": {"protocol": "Balancer V2 Vault", "type": "vault"},
    # Convex
    "0x4e3fbd56cd56c3e72c1403e103b45db9da5b9d2b": {"protocol": "Convex", "type": "token"},
    # Rocket Pool
    "0xd33526068d116ce69f19a9ee46f0bd304f21a51f": {"protocol": "Rocket Pool RPL", "type": "token"},
    # Yearn
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e": {"protocol": "Yearn YFI", "type": "token"},
    # 1inch
    "0x111111111117dc0aa78b770fa6a738034120c302": {"protocol": "1inch", "type": "token"},
    # dYdX
    "0x92d6c1e31e14520e676a687f0a93788b716beff5": {"protocol": "dYdX", "type": "token"},
    # Frax
    "0x853d955acef822db058eb8505911ed77f175b99e": {"protocol": "Frax", "type": "stablecoin"},
    # GMX
    "0xfc5a1a6eb076a2c7ad06ed22c90d7e710e35ad0a": {"protocol": "GMX", "type": "token"},
    # Arbitrum
    "0x912ce59144191c1204e64559fe8253a0e49e6548": {"protocol": "Arbitrum ARB", "type": "token"},
    # Optimism
    "0x4200000000000000000000000000000000000042": {"protocol": "Optimism OP", "type": "token"},
    # Morpho
    "0x9994e35db50125e0df82e4c2dde62496ce330999": {"protocol": "Morpho", "type": "token"},
    # Spark
    "0xc13e21b648a5ee794902342038ff3adab66be987": {"protocol": "Spark", "type": "pool"},
    # Pendle
    "0x808507121b80c02388fad14726482e061b8da827": {"protocol": "Pendle", "type": "token"},
    # EigenLayer
    "0xec53bf9167f50cdeb3ae105f56099aaab9061f83": {"protocol": "EigenLayer EIGEN", "type": "token"},
    # Ethena
    "0x57e114b691db790c35207b2e685d4a43181e6061": {"protocol": "Ethena ENA", "type": "token"},
    # EtherFi
    "0xfe0c30065b384f05761f15d0cc899d4f9f9cc0eb": {"protocol": "EtherFi ETHFI", "type": "token"},
    # Usual
    "0xd3b86b0e1c0028f0f7b22aa9e14f2c19d7d73e0e": {"protocol": "Usual USD0", "type": "stablecoin"},
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WhaleTransaction(BaseModel):
    tx_hash: str
    chain: str
    symbol: str
    amount: float
    amount_usd: Optional[float]
    from_addr: Optional[str]
    to_addr: Optional[str]
    direction: str  # "exchange_deposit" | "exchange_withdrawal" | "whale_to_whale" | "unknown"
    classification: str
    ts: Optional[int]
    detected_at: str


class ExchangeFlow(BaseModel):
    chain: str
    symbol: str
    exchange: str
    flow_type: str  # "inflow" | "outflow"
    amount: float
    amount_usd: Optional[float]
    tx_hash: Optional[str]
    ts: Optional[int]
    recorded_at: str


class GovernanceProposal(BaseModel):
    proposal_id: str
    protocol: str
    title: str
    state: str  # "active" | "passed" | "rejected" | "pending"
    start_ts: Optional[int]
    end_ts: Optional[int]
    votes_for: float = 0.0
    votes_against: float = 0.0
    quorum_pct: float = 0.0
    impact_score: float = 0.0  # 0-10 estimated token-economic impact
    turnout_pct: float = 0.0
    fetched_at: str


class BridgeFlowSummary(BaseModel):
    bridge_name: str
    source_chain: str
    dest_chain: str
    volume_usd: float
    net_flow_usd: float
    recorded_date: str
    anomaly_score: float = 0.0  # z-score relative to 30-day history


class ContractEvent(BaseModel):
    contract: str
    protocol: str
    event_type: str  # "liquidation" | "large_swap" | "governance" | "pause" | "flash_loan"
    tx_hash: Optional[str]
    block_number: Optional[int]
    amount_usd: Optional[float]
    severity: str  # "low" | "medium" | "high" | "critical"
    detected_at: str


class OnChainSignal(BaseModel):
    symbol: str
    signal_type: str
    score: float  # -100 to +100
    direction: str  # "bullish" | "bearish" | "neutral"
    rationale: str
    computed_at: str


class OnChainCompositeScore(BaseModel):
    symbol: str
    exchange_netflow_score: float
    whale_accumulation_score: float
    tvl_delta_score: float
    governance_risk_score: float
    bridge_flow_score: float
    composite_score: float  # -100 to +100
    direction: str
    computed_at: str


# ---------------------------------------------------------------------------
# Helper: safe GET with retry
# ---------------------------------------------------------------------------


def _get(url: str, params: Optional[Dict] = None, retries: int = 2) -> Any:
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited at %s — sleeping %ss", url, wait)
                time.sleep(wait)
            else:
                raise
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            logger.warning("GET %s attempt %d failed: %s", url, attempt + 1, exc)
            time.sleep(1)
    return {}


def _post_graphql(url: str, query: str, variables: Optional[Dict] = None) -> Any:
    payload: Dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        resp = requests.post(url, json=payload, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphQL POST %s failed: %s", url, exc)
        return {}


def _etherscan_key() -> Optional[str]:
    return os.environ.get("ETHERSCAN_API_KEY")


# ---------------------------------------------------------------------------
# 1. EtherscanEventAdapter
# ---------------------------------------------------------------------------


class EtherscanEventAdapter:
    """Ethereum on-chain event queries via Etherscan free API."""

    # Known whale ETH addresses (top fund/protocol treasuries + whale wallets)
    KNOWN_WHALE_ADDRESSES: List[str] = [
        # Ethereum Foundation
        "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae",
        # Vitalik Buterin (public)
        "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        # Justin Sun
        "0x3ddfa8ec3052539b6c9549f12cea2c295cff5296",
        # Grayscale ETHE
        "0x9a36d8fd2e67f2b98bb35e43ef2cdcc27284a02f",
        # Lido Staking
        "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",
        # Binance: CEX cold wallets
        "0x28c6c06298d514db089934071355e5743bf21d60",
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549",
        # Coinbase cold
        "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",
        # Gemini cold
        "0x5f65f7b609678448494de4c87521cdf6cef1e932",
        # Kraken
        "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0",
        # Bitfinex
        "0x1151314c646ce4e0efd76d1af4760ae66a9fe30f",
        # Jump Trading
        "0x046340c200172c78e6e1a55b1df6a4e9cc25d19",
        # Alameda (historical)
        "0x4de76b3dfd38292ba71cf2dba5d0e4aa5b4f9ca2",
        # Paradigm
        "0x1dc4c1cefef38a777b15aa20260a54e584b16c48",
        # a16z
        "0x05e793ce0c6027323ac150f6d45c2344d28b6019",
        # Multicoin Capital
        "0xfc2ef3f3e6c52c2c0703ac52a0b89c5fedbba26b",
        # Three Arrows Capital (historical)
        "0x6a6d0cdc56e6f2a08b5de01ba1b81c36a3c8a3ac",
        # Cumberland DRW
        "0x6262998ced04146fa42253a5c0af90ca02dfd2a3",
        # Wintermute
        "0x4a137fd5e7a256ef08a7de531a17d0be0cc7b6b6",
    ]

    def __init__(self) -> None:
        self._api_key = _etherscan_key()

    def _build_params(self, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {"apikey": self._api_key or "YourApiKeyToken"}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params

    def get_token_transfers(
        self,
        contract_address: str,
        address: Optional[str] = None,
        start_block: int = 0,
        end_block: int = 99999999,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch ERC-20 token transfer events."""
        cache_key = f"erc20_transfers:{contract_address}:{address}:{start_block}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        params = self._build_params(
            module="account",
            action="tokentx",
            contractaddress=contract_address,
            address=address,
            startblock=start_block,
            endblock=end_block,
            page=1,
            offset=limit,
            sort="desc",
        )
        try:
            data = _get(_ETHERSCAN_BASE, params)
            result = data.get("result", [])
            if isinstance(result, list):
                df = pd.DataFrame(result)
                _cache_set(cache_key, df)
                return df
        except Exception as exc:  # noqa: BLE001
            logger.error("Etherscan token transfers failed: %s", exc)
        return pd.DataFrame()

    def get_internal_transactions(
        self, address: str, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch internal (contract-to-contract) transactions."""
        params = self._build_params(
            module="account",
            action="txlistinternal",
            address=address,
            startblock=0,
            endblock=99999999,
            page=1,
            offset=limit,
            sort="desc",
        )
        try:
            data = _get(_ETHERSCAN_BASE, params)
            result = data.get("result", [])
            if isinstance(result, list):
                return pd.DataFrame(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Etherscan internal txs failed: %s", exc)
        return pd.DataFrame()

    def get_eth_balance(self, address: str) -> float:
        """Get ETH balance for an address (in ETH)."""
        params = self._build_params(module="account", action="balance", address=address, tag="latest")
        try:
            data = _get(_ETHERSCAN_BASE, params)
            wei = int(data.get("result", 0))
            return wei / 1e18
        except Exception as exc:  # noqa: BLE001
            logger.error("ETH balance fetch failed for %s: %s", address, exc)
            return 0.0

    def get_events(
        self,
        contract: str,
        topic0: str,
        from_block: int = 0,
        to_block: int = 99999999,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch raw event logs by contract address and topic0 (event signature hash)."""
        params = self._build_params(
            module="logs",
            action="getLogs",
            address=contract,
            topic0=topic0,
            fromBlock=from_block,
            toBlock=to_block,
            page=1,
            offset=limit,
        )
        try:
            data = _get(_ETHERSCAN_BASE, params)
            result = data.get("result", [])
            if isinstance(result, list):
                return pd.DataFrame(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Etherscan getLogs failed: %s", exc)
        return pd.DataFrame()

    def get_normal_transactions(
        self, address: str, limit: int = 100, min_block: int = 0
    ) -> pd.DataFrame:
        """Fetch normal ETH transactions for an address."""
        params = self._build_params(
            module="account",
            action="txlist",
            address=address,
            startblock=min_block,
            endblock=99999999,
            page=1,
            offset=limit,
            sort="desc",
        )
        try:
            data = _get(_ETHERSCAN_BASE, params)
            result = data.get("result", [])
            if isinstance(result, list):
                return pd.DataFrame(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Etherscan txlist failed for %s: %s", address, exc)
        return pd.DataFrame()

    def get_whale_balances(self) -> pd.DataFrame:
        """Query ETH balances for all known whale addresses."""
        rows = []
        for addr in self.KNOWN_WHALE_ADDRESSES[:10]:  # limit to avoid rate limiting
            bal = self.get_eth_balance(addr)
            rows.append({"address": addr, "eth_balance": bal, "checked_at": datetime.now(timezone.utc).isoformat()})
            time.sleep(0.25)  # Etherscan free: 5 req/sec
        return pd.DataFrame(rows)

    def decode_event_topic(self, topic0: str) -> Optional[str]:
        """Map known topic0 hashes to human-readable event names."""
        # keccak256 hashes of common event signatures
        _KNOWN_EVENTS: Dict[str, str] = {
            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef": "Transfer(address,address,uint256)",
            "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925": "Approval(address,address,uint256)",
            "0xe1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c": "Deposit(address,uint256)",
            "0x7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65": "Withdrawal(address,uint256)",
            "0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f": "Mint(address,uint256,uint256)",
            "0xd6d4f5681c246c9f42c203e287975af1601f8df8035a9251f79aab5c8f09e2f8": "LiquidationCall(...)",
            "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822": "Swap(address,uint256,uint256,uint256,uint256,address)",
        }
        return _KNOWN_EVENTS.get(topic0.lower())


# ---------------------------------------------------------------------------
# 2. WhaleTransactionMonitor
# ---------------------------------------------------------------------------


_BTC_WHALE_THRESHOLD = 100.0   # BTC (lowered from 1000 for better coverage)
_ETH_WHALE_THRESHOLD = 500.0   # ETH
_TOKEN_WHALE_USD = 500_000.0   # $500K


class WhaleTransactionMonitor:
    """Detect and classify large on-chain transactions."""

    def __init__(self) -> None:
        self._etherscan = EtherscanEventAdapter()

    def fetch_btc_large_txs(self, min_btc: float = _BTC_WHALE_THRESHOLD) -> pd.DataFrame:
        """Fetch recent BTC transactions above threshold via blockchain.info."""
        cache_key = f"btc_whales:{min_btc}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            data = _get("https://blockchain.info/unconfirmed-transactions?format=json")
            txs = data.get("txs", [])
            rows = []
            for tx in txs:
                out_val_btc = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
                if out_val_btc >= min_btc:
                    rows.append({
                        "tx_hash": tx.get("hash", ""),
                        "chain": "BTC",
                        "symbol": "BTC",
                        "amount": out_val_btc,
                        "amount_usd": None,  # price not available here
                        "from_addr": None,
                        "to_addr": None,
                        "ts": tx.get("time"),
                    })
            df = pd.DataFrame(rows)
            _cache_set(cache_key, df)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("BTC whale fetch failed: %s", exc)
            return pd.DataFrame()

    def fetch_eth_large_txs(
        self, address: Optional[str] = None, min_eth: float = _ETH_WHALE_THRESHOLD
    ) -> pd.DataFrame:
        """Fetch large ETH transactions via Etherscan."""
        target_addresses = [address] if address else self._etherscan.KNOWN_WHALE_ADDRESSES[:5]
        all_rows = []
        for addr in target_addresses:
            df = self._etherscan.get_normal_transactions(addr, limit=50)
            if df.empty:
                continue
            if "value" in df.columns:
                df["eth_value"] = df["value"].astype(float) / 1e18
                large = df[df["eth_value"] >= min_eth].copy()
                large["chain"] = "ETH"
                large["symbol"] = "ETH"
                large["amount"] = large["eth_value"]
                large["amount_usd"] = None
                all_rows.append(large)
            time.sleep(0.2)  # rate limit respect
        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    def classify_transaction(self, tx: Dict[str, Any]) -> str:
        """
        Classify a transaction into:
        - exchange_deposit: whale → exchange
        - exchange_withdrawal: exchange → whale
        - whale_to_whale: large wallet → large wallet
        - unknown
        """
        from_addr = str(tx.get("from_addr") or tx.get("from") or "").lower()
        to_addr = str(tx.get("to_addr") or tx.get("to") or "").lower()

        exchange_wallets_lower = {k.lower(): v for k, v in KNOWN_EXCHANGE_WALLETS.items()}
        whale_set = {a.lower() for a in EtherscanEventAdapter.KNOWN_WHALE_ADDRESSES}

        from_is_exchange = from_addr in exchange_wallets_lower
        to_is_exchange = to_addr in exchange_wallets_lower
        from_is_whale = from_addr in whale_set
        to_is_whale = to_addr in whale_set

        if to_is_exchange:
            return "exchange_deposit"
        if from_is_exchange:
            return "exchange_withdrawal"
        if from_is_whale and to_is_whale:
            return "whale_to_whale"
        return "unknown"

    def monitor_exchange_flows(self, symbol: str = "ETH") -> List[ExchangeFlow]:
        """
        Derive net exchange inflow/outflow signals by inspecting
        transactions to/from known exchange wallets.
        """
        flows: List[ExchangeFlow] = []
        ex_wallets = list(KNOWN_EXCHANGE_WALLETS.items())[:5]  # limit to 5 to avoid rate limits

        for addr, exchange_name in ex_wallets:
            if not addr.startswith("0x"):
                continue  # skip BTC addresses for ETH analysis
            df = self._etherscan.get_normal_transactions(addr, limit=20)
            if df.empty:
                time.sleep(0.25)
                continue

            for _, row in df.iterrows():
                to_addr = str(row.get("to", "")).lower()
                from_addr = str(row.get("from", "")).lower()
                val_eth = float(row.get("value", 0)) / 1e18

                if val_eth < 10:  # skip dust
                    continue

                if to_addr == addr.lower():
                    flow_type = "inflow"
                elif from_addr == addr.lower():
                    flow_type = "outflow"
                else:
                    continue

                flow = ExchangeFlow(
                    chain="ETH",
                    symbol=symbol,
                    exchange=exchange_name,
                    flow_type=flow_type,
                    amount=val_eth,
                    amount_usd=None,
                    tx_hash=str(row.get("hash", "")),
                    ts=int(row.get("timeStamp", 0)) or None,
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                )
                flows.append(flow)

                self._persist_flow(flow)

            time.sleep(0.25)

        return flows

    def _persist_flow(self, flow: ExchangeFlow) -> None:
        with _db() as conn:
            conn.execute(
                """INSERT INTO exchange_flows
                   (chain, symbol, exchange, flow_type, amount, amount_usd,
                    tx_hash, ts, recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (flow.chain, flow.symbol, flow.exchange, flow.flow_type,
                 flow.amount, flow.amount_usd, flow.tx_hash, flow.ts, flow.recorded_at),
            )

    def persist_whale_tx(self, wtx: WhaleTransaction) -> None:
        with _db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO whale_txs
                   (tx_hash, chain, symbol, amount, amount_usd, from_addr, to_addr,
                    direction, classification, ts, detected_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (wtx.tx_hash, wtx.chain, wtx.symbol, wtx.amount, wtx.amount_usd,
                 wtx.from_addr, wtx.to_addr, wtx.direction, wtx.classification,
                 wtx.ts, wtx.detected_at),
            )

    def get_recent_whale_txs(self, symbol: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        with _db() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM whale_txs WHERE symbol=? ORDER BY detected_at DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM whale_txs ORDER BY detected_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def compute_exchange_netflow(self, symbol: str = "ETH", hours: int = 24) -> Dict[str, float]:
        """Net inflow/outflow across all tracked exchanges in the last N hours."""
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with _db() as conn:
            rows = conn.execute(
                """SELECT flow_type, SUM(amount) as total
                   FROM exchange_flows
                   WHERE symbol=? AND recorded_at>=?
                   GROUP BY flow_type""",
                (symbol, since),
            ).fetchall()
        totals: Dict[str, float] = {"inflow": 0.0, "outflow": 0.0}
        for r in rows:
            totals[r["flow_type"]] = float(r["total"] or 0)
        totals["net"] = totals["inflow"] - totals["outflow"]
        return totals


# ---------------------------------------------------------------------------
# 3. SmartContractMonitor
# ---------------------------------------------------------------------------

# Common event topic0 hashes
_TOPIC_LIQUIDATION = "0xd6d4f5681c246c9f42c203e287975af1601f8df8035a9251f79aab5c8f09e2f8"
_TOPIC_SWAP = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
_TOPIC_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
_TOPIC_PAUSED = "0x62e78cea01bee320cd4e420270b5ea74000d11b0c9f74754ebdbfc544b05a258"


class SmartContractMonitor:
    """Monitor key DeFi protocol contracts for significant events."""

    LARGE_SWAP_USD = 100_000.0  # $100K threshold

    def __init__(self) -> None:
        self._etherscan = EtherscanEventAdapter()

    def monitor_liquidations(
        self,
        protocol: str = "aave",
        lookback_blocks: int = 1000,
    ) -> List[ContractEvent]:
        """Monitor Aave/Compound for liquidation events."""
        contract_map = {
            "aave": "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",
            "aave_v2": "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",
            "compound": "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",
        }
        contract = contract_map.get(protocol.lower())
        if not contract:
            return []

        events: List[ContractEvent] = []
        df = self._etherscan.get_events(
            contract, _TOPIC_LIQUIDATION, limit=50
        )
        if df.empty:
            return events

        for _, row in df.iterrows():
            evt = ContractEvent(
                contract=contract,
                protocol=protocol.capitalize(),
                event_type="liquidation",
                tx_hash=str(row.get("transactionHash", "")),
                block_number=int(row.get("blockNumber", 0), 16) if row.get("blockNumber") else None,
                amount_usd=None,  # requires price feed decode
                severity="high",
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
            events.append(evt)
            self._persist_event(evt)

        return events

    def monitor_large_swaps(
        self,
        protocol: str = "uniswap_v3",
        min_amount_usd: float = LARGE_SWAP_USD,
    ) -> List[ContractEvent]:
        """Monitor Uniswap for large swap events."""
        contract_map = {
            "uniswap_v3": "0x1f98431c8ad98523631ae4a59f267346ea31f984",
            "uniswap_v2": "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
            "balancer": "0xba12222222228d8ba445958a75a0704d566bf2c8",
            "curve": "0xbabe61887f1de2713c6f97e567623453d3c79f67",
        }
        contract = contract_map.get(protocol.lower())
        if not contract:
            return []

        events: List[ContractEvent] = []
        df = self._etherscan.get_events(contract, _TOPIC_SWAP, limit=50)
        if df.empty:
            return events

        for _, row in df.iterrows():
            evt = ContractEvent(
                contract=contract,
                protocol=protocol.replace("_", " ").title(),
                event_type="large_swap",
                tx_hash=str(row.get("transactionHash", "")),
                block_number=int(row.get("blockNumber", 0), 16) if row.get("blockNumber") else None,
                amount_usd=None,
                severity="medium",
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
            events.append(evt)
            self._persist_event(evt)

        return events

    def scan_protocol_pause(self, protocol: str = "aave") -> List[ContractEvent]:
        """Detect protocol pause/emergency events — critical severity."""
        contract_map = {
            "aave": "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",
            "compound": "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",
        }
        contract = contract_map.get(protocol.lower())
        if not contract:
            return []

        events: List[ContractEvent] = []
        df = self._etherscan.get_events(contract, _TOPIC_PAUSED, limit=10)
        if df.empty:
            return events

        for _, row in df.iterrows():
            evt = ContractEvent(
                contract=contract,
                protocol=protocol.capitalize(),
                event_type="pause",
                tx_hash=str(row.get("transactionHash", "")),
                block_number=int(row.get("blockNumber", 0), 16) if row.get("blockNumber") else None,
                amount_usd=None,
                severity="critical",
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
            events.append(evt)
            self._persist_event(evt)

        return events

    def scan_all_contracts(self) -> List[ContractEvent]:
        """Run a quick scan across all known contracts."""
        events: List[ContractEvent] = []
        # Scan major lending protocols for liquidations
        for proto in ["aave", "compound"]:
            try:
                events.extend(self.monitor_liquidations(proto))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Liquidation scan failed for %s: %s", proto, exc)

        # Scan DEXes for large swaps
        for proto in ["uniswap_v3", "balancer"]:
            try:
                events.extend(self.monitor_large_swaps(proto))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Swap scan failed for %s: %s", proto, exc)

        return events

    def _persist_event(self, evt: ContractEvent) -> None:
        with _db() as conn:
            conn.execute(
                """INSERT INTO contract_events
                   (contract, protocol, event_type, tx_hash, block_number,
                    amount_usd, severity, detected_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (evt.contract, evt.protocol, evt.event_type, evt.tx_hash,
                 evt.block_number, evt.amount_usd, evt.severity, evt.detected_at),
            )

    def get_recent_events(
        self, event_type: Optional[str] = None, limit: int = 50
    ) -> pd.DataFrame:
        with _db() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM contract_events WHERE event_type=? ORDER BY detected_at DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM contract_events ORDER BY detected_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_contract_info(self, address: str) -> Optional[Dict[str, str]]:
        return KNOWN_CONTRACTS.get(address.lower())


# ---------------------------------------------------------------------------
# 4. GovernanceVoteTracker
# ---------------------------------------------------------------------------

_SNAPSHOT_QUERY = """
query GetProposals($protocols: [String!], $state: String, $first: Int) {
  proposals(
    first: $first
    where: { space_in: $protocols, state: $state }
    orderBy: "created"
    orderDirection: desc
  ) {
    id
    title
    body
    state
    start
    end
    scores
    scores_total
    quorum
    space { id name }
  }
}
"""

_SNAPSHOT_SPACES = [
    "uniswapgovernance.eth",
    "aave.eth",
    "compound-governance.eth",
    "curve.eth",
    "balancer.eth",
    "lido-snapshot.eth",
    "gitcoindao.eth",
    "sushigov.eth",
    "dydxgov.eth",
    "frax.eth",
    "gmx.eth",
    "arbitrumfoundation.eth",
    "opcollective.eth",
]

# Token-economic impact keywords → score boost
_IMPACT_KEYWORDS: Dict[str, float] = {
    "token": 3.0, "emission": 4.0, "inflation": 4.0, "supply": 3.0,
    "burn": 5.0, "buyback": 5.0, "treasury": 4.0, "fee": 3.0,
    "parameter": 2.0, "upgrade": 3.0, "emergency": 8.0, "pause": 8.0,
    "liquidation": 5.0, "collateral": 4.0, "interest": 3.0, "halting": 9.0,
}


class GovernanceVoteTracker:
    """Track on-chain and off-chain governance proposals across major DeFi protocols."""

    def __init__(self) -> None:
        pass

    def fetch_snapshot_proposals(
        self,
        spaces: Optional[List[str]] = None,
        state: str = "active",
        limit: int = 20,
    ) -> List[GovernanceProposal]:
        """Fetch proposals from Snapshot (off-chain governance)."""
        cache_key = f"snapshot:{state}:{limit}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        target_spaces = spaces or _SNAPSHOT_SPACES
        variables = {"protocols": target_spaces, "state": state, "first": limit}
        data = _post_graphql(_SNAPSHOT_GRAPHQL, _SNAPSHOT_QUERY, variables)

        proposals: List[GovernanceProposal] = []
        for p in data.get("data", {}).get("proposals", []):
            scores = p.get("scores", [])
            scores_total = float(p.get("scores_total") or 0)
            votes_for = float(scores[0]) if scores else 0.0
            votes_against = float(scores[1]) if len(scores) > 1 else 0.0
            quorum = float(p.get("quorum") or 0)
            quorum_pct = (scores_total / quorum * 100) if quorum > 0 else 0.0
            turnout_pct = (scores_total / quorum * 100) if quorum > 0 else 0.0
            impact = self._score_impact(p.get("title", "") + " " + p.get("body", "")[:500])

            prop = GovernanceProposal(
                proposal_id=p.get("id", ""),
                protocol=p.get("space", {}).get("name", ""),
                title=p.get("title", ""),
                state=p.get("state", ""),
                start_ts=p.get("start"),
                end_ts=p.get("end"),
                votes_for=votes_for,
                votes_against=votes_against,
                quorum_pct=min(quorum_pct, 100.0),
                impact_score=impact,
                turnout_pct=min(turnout_pct, 100.0),
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
            proposals.append(prop)
            self._persist_proposal(prop)

        _cache_set(cache_key, proposals)
        return proposals

    def fetch_all_proposals(self, limit: int = 50) -> List[GovernanceProposal]:
        """Fetch both active and closed proposals from Snapshot."""
        active = self.fetch_snapshot_proposals(state="active", limit=limit // 2)
        closed = self.fetch_snapshot_proposals(state="closed", limit=limit // 2)
        return active + closed

    def _score_impact(self, text: str) -> float:
        """Heuristic impact score (0–10) based on proposal text keywords."""
        text_lower = text.lower()
        score = 1.0
        for kw, weight in _IMPACT_KEYWORDS.items():
            if kw in text_lower:
                score += weight
        return min(score, 10.0)

    def get_governance_risk(self, protocol: str) -> Dict[str, Any]:
        """
        Compute governance risk score for a protocol:
        - Low voter turnout → higher risk
        - Contested votes (close margins) → higher risk
        - High-impact proposals pending → higher risk
        """
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM governance_proposals WHERE LOWER(protocol) LIKE ? ORDER BY fetched_at DESC LIMIT 20",
                (f"%{protocol.lower()}%",),
            ).fetchall()

        if not rows:
            return {"protocol": protocol, "risk_score": 5.0, "reason": "no data"}

        df = pd.DataFrame([dict(r) for r in rows])
        avg_turnout = df["quorum_pct"].mean()
        avg_impact = df["impact_score"].mean()
        contested = df.apply(
            lambda r: abs(r["votes_for"] - r["votes_against"]) / (r["votes_for"] + r["votes_against"] + 1) < 0.1,
            axis=1,
        ).mean()

        # Risk = low turnout + high impact + contested = high risk
        risk = (
            (100 - avg_turnout) / 100 * 3  # turnout component
            + avg_impact / 10 * 4           # impact component
            + contested * 3                 # contestedness
        )
        risk = min(10.0, max(0.0, risk))

        return {
            "protocol": protocol,
            "risk_score": round(risk, 2),
            "avg_turnout_pct": round(avg_turnout, 1),
            "avg_impact_score": round(avg_impact, 2),
            "contested_pct": round(contested * 100, 1),
        }

    def _persist_proposal(self, prop: GovernanceProposal) -> None:
        with _db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO governance_proposals
                   (proposal_id, protocol, title, state, start_ts, end_ts,
                    votes_for, votes_against, quorum_pct, impact_score, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (prop.proposal_id, prop.protocol, prop.title, prop.state,
                 prop.start_ts, prop.end_ts, prop.votes_for, prop.votes_against,
                 prop.quorum_pct, prop.impact_score, prop.fetched_at),
            )

    def get_recent_proposals(self, limit: int = 50) -> pd.DataFrame:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM governance_proposals ORDER BY fetched_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. BridgeFlowMonitor
# ---------------------------------------------------------------------------

_BRIDGE_VOLUME_ENDPOINT = f"{_DEFILLAMA_BRIDGES}/bridgevolume/{{chain}}?id={{bridge_id}}"
_MAJOR_BRIDGES = {
    "arbitrum": {"bridge_ids": [1, 2], "chains": ["Ethereum", "Arbitrum"]},
    "optimism": {"bridge_ids": [3], "chains": ["Ethereum", "Optimism"]},
    "polygon": {"bridge_ids": [4, 5], "chains": ["Ethereum", "Polygon"]},
    "bsc": {"bridge_ids": [6], "chains": ["Ethereum", "BSC"]},
    "avalanche": {"bridge_ids": [7], "chains": ["Ethereum", "Avalanche"]},
    "base": {"bridge_ids": [8], "chains": ["Ethereum", "Base"]},
    "zksync": {"bridge_ids": [9], "chains": ["Ethereum", "zkSync Era"]},
    "starknet": {"bridge_ids": [10], "chains": ["Ethereum", "Starknet"]},
}


class BridgeFlowMonitor:
    """Monitor cross-chain bridge flows via DeFiLlama bridges API."""

    def __init__(self) -> None:
        pass

    def fetch_bridge_list(self) -> pd.DataFrame:
        """Fetch all bridges tracked by DeFiLlama."""
        cache_key = "defillama_bridges"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            data = _get(f"{_DEFILLAMA_BRIDGES}/bridges")
            bridges = data.get("bridges", [])
            df = pd.DataFrame(bridges)
            _cache_set(cache_key, df)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("DeFiLlama bridges fetch failed: %s", exc)
            return pd.DataFrame()

    def fetch_bridge_volume(self, bridge_id: int, chain: str = "Ethereum") -> pd.DataFrame:
        """Fetch historical bridge volume for a specific bridge."""
        cache_key = f"bridge_vol:{bridge_id}:{chain}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        url = f"{_DEFILLAMA_BRIDGES}/bridgevolume/{chain}?id={bridge_id}"
        try:
            data = _get(url)
            if isinstance(data, list):
                df = pd.DataFrame(data)
                _cache_set(cache_key, df)
                return df
        except Exception as exc:  # noqa: BLE001
            logger.error("Bridge volume fetch failed (id=%d, chain=%s): %s", bridge_id, chain, exc)
        return pd.DataFrame()

    def fetch_bridge_day_data(self, bridge_id: int) -> Dict[str, Any]:
        """Fetch daily bridge data from DeFiLlama."""
        url = f"{_DEFILLAMA_BRIDGES}/bridge/{bridge_id}"
        try:
            return _get(url)
        except Exception as exc:  # noqa: BLE001
            logger.error("Bridge day data failed: %s", exc)
            return {}

    def compute_net_flows(
        self,
        chain: str = "Arbitrum",
        days: int = 7,
    ) -> List[BridgeFlowSummary]:
        """Compute net flows into/out of a chain across tracked bridges."""
        bridges_df = self.fetch_bridge_list()
        if bridges_df.empty:
            return []

        summaries: List[BridgeFlowSummary] = []
        for _, bridge in bridges_df.head(10).iterrows():
            bridge_id = bridge.get("id")
            bridge_name = bridge.get("displayName") or bridge.get("name", "Unknown")
            if bridge_id is None:
                continue

            vol_df = self.fetch_bridge_volume(int(bridge_id), chain)
            if vol_df.empty:
                continue

            if "depositUSD" in vol_df.columns and "withdrawUSD" in vol_df.columns:
                recent = vol_df.tail(days)
                total_dep = float(recent["depositUSD"].sum())
                total_with = float(recent["withdrawUSD"].sum())
                net = total_dep - total_with
                anomaly = self._compute_anomaly_score(vol_df, "depositUSD")

                summary = BridgeFlowSummary(
                    bridge_name=str(bridge_name),
                    source_chain="Ethereum",
                    dest_chain=chain,
                    volume_usd=total_dep + total_with,
                    net_flow_usd=net,
                    recorded_date=datetime.now(timezone.utc).date().isoformat(),
                    anomaly_score=anomaly,
                )
                summaries.append(summary)
                self._persist_bridge_flow(summary)

        return summaries

    def _compute_anomaly_score(self, df: pd.DataFrame, col: str) -> float:
        """Z-score of the most recent value vs 30-day history."""
        if col not in df.columns or len(df) < 5:
            return 0.0
        series = df[col].astype(float).dropna()
        if len(series) < 3:
            return 0.0
        mean = series.mean()
        std = series.std()
        if std == 0:
            return 0.0
        last_val = series.iloc[-1]
        return round(float((last_val - mean) / std), 2)

    def detect_bridge_anomalies(self, z_threshold: float = 2.5) -> List[BridgeFlowSummary]:
        """Return bridge flow summaries with anomalous volume (potential exploits)."""
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM bridge_flows WHERE ABS(anomaly_score) >= ? ORDER BY anomaly_score DESC",
                (z_threshold,),
            ).fetchall()
        if not rows:
            return []
        results = []
        for r in rows:
            results.append(BridgeFlowSummary(
                bridge_name=r["bridge_name"],
                source_chain=r["source_chain"],
                dest_chain=r["dest_chain"],
                volume_usd=r["volume_usd"],
                net_flow_usd=r["net_flow_usd"],
                recorded_date=r["recorded_date"],
                anomaly_score=r["anomaly_score"],
            ))
        return results

    def _persist_bridge_flow(self, flow: BridgeFlowSummary) -> None:
        with _db() as conn:
            conn.execute(
                """INSERT INTO bridge_flows
                   (bridge_name, source_chain, dest_chain, volume_usd,
                    net_flow_usd, recorded_date, anomaly_score)
                   VALUES (?,?,?,?,?,?,?)""",
                (flow.bridge_name, flow.source_chain, flow.dest_chain,
                 flow.volume_usd, flow.net_flow_usd, flow.recorded_date, flow.anomaly_score),
            )

    def get_bridge_history(self, bridge_name: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        with _db() as conn:
            if bridge_name:
                rows = conn.execute(
                    "SELECT * FROM bridge_flows WHERE bridge_name=? ORDER BY recorded_date DESC LIMIT ?",
                    (bridge_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM bridge_flows ORDER BY recorded_date DESC LIMIT ?", (limit,)
                ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. OnChainSignalEngine
# ---------------------------------------------------------------------------


class OnChainSignalEngine:
    """Combine on-chain data into tradeable signals scored -100 to +100."""

    def __init__(self) -> None:
        self._whale_monitor = WhaleTransactionMonitor()
        self._bridge_monitor = BridgeFlowMonitor()
        self._governance = GovernanceVoteTracker()

    def exchange_netflow_signal(self, symbol: str = "ETH") -> OnChainSignal:
        """
        Exchange netflow: large inflows = selling pressure (bearish),
        large outflows = accumulation (bullish).
        """
        flows = self._whale_monitor.compute_exchange_netflow(symbol, hours=24)
        net = flows.get("net", 0.0)
        inflow = flows.get("inflow", 0.0)
        outflow = flows.get("outflow", 0.0)
        total = inflow + outflow

        if total == 0:
            score = 0.0
            direction = "neutral"
            rationale = "No exchange flow data available"
        else:
            netflow_ratio = net / total  # -1 (all outflow) to +1 (all inflow)
            # Inflow is bearish (-), outflow is bullish (+)
            score = -netflow_ratio * 80
            direction = "bullish" if score > 10 else "bearish" if score < -10 else "neutral"
            rationale = (
                f"Net exchange flow: {net:+.1f} {symbol} over 24h. "
                f"Inflow {inflow:.1f} / Outflow {outflow:.1f}. "
                f"{'Selling pressure' if net > 0 else 'Accumulation'} signal."
            )

        sig = OnChainSignal(
            symbol=symbol,
            signal_type="exchange_netflow",
            score=round(score, 1),
            direction=direction,
            rationale=rationale,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._persist_signal(sig)
        return sig

    def whale_accumulation_signal(self, symbol: str = "BTC") -> OnChainSignal:
        """
        Track large wallet behavior:
        - Recent inflows to whale wallets from exchanges = accumulation (bullish)
        - Recent outflows to exchanges = distribution (bearish)
        """
        recent = self._whale_monitor.get_recent_whale_txs(symbol, limit=50)
        if recent.empty:
            return OnChainSignal(
                symbol=symbol, signal_type="whale_accumulation", score=0.0,
                direction="neutral", rationale="No whale TX data in DB",
                computed_at=datetime.now(timezone.utc).isoformat(),
            )

        deposit_count = (recent["direction"] == "exchange_deposit").sum()
        withdrawal_count = (recent["direction"] == "exchange_withdrawal").sum()
        whale_to_whale = (recent["direction"] == "whale_to_whale").sum()
        total = len(recent)

        # Deposits = distribution = bearish; withdrawals = accumulation = bullish
        net_score = ((withdrawal_count - deposit_count) / total) * 100 if total > 0 else 0
        direction = "bullish" if net_score > 15 else "bearish" if net_score < -15 else "neutral"
        rationale = (
            f"Whale TXs (last {total}): {deposit_count} deposits, "
            f"{withdrawal_count} withdrawals, {whale_to_whale} whale-to-whale."
        )

        sig = OnChainSignal(
            symbol=symbol,
            signal_type="whale_accumulation",
            score=round(net_score, 1),
            direction=direction,
            rationale=rationale,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._persist_signal(sig)
        return sig

    def defi_tvl_signal(self, protocol_slugs: Optional[List[str]] = None) -> OnChainSignal:
        """
        DeFi TVL trend: rising TVL = increasing protocol health (bullish for DeFi tokens).
        Uses DeFiLlama protocol data.
        """
        try:
            data = _get("https://api.llama.fi/v2/historicalChainTvl")
            if isinstance(data, list) and len(data) >= 14:
                recent_7d = np.mean([d.get("tvl", 0) for d in data[-7:]])
                prior_7d = np.mean([d.get("tvl", 0) for d in data[-14:-7]])
                if prior_7d > 0:
                    pct_change = (recent_7d - prior_7d) / prior_7d * 100
                    score = min(max(pct_change * 5, -100), 100)  # scale 20% = 100
                    direction = "bullish" if score > 10 else "bearish" if score < -10 else "neutral"
                    rationale = f"DeFi TVL 7d change: {pct_change:+.1f}%. Recent avg: ${recent_7d/1e9:.1f}B"
                else:
                    score, direction, rationale = 0.0, "neutral", "Insufficient TVL history"
            else:
                score, direction, rationale = 0.0, "neutral", "TVL data unavailable"
        except Exception as exc:  # noqa: BLE001
            logger.error("TVL signal computation failed: %s", exc)
            score, direction, rationale = 0.0, "neutral", f"TVL fetch error: {exc}"

        sig = OnChainSignal(
            symbol="ETH",
            signal_type="defi_tvl_delta",
            score=round(score, 1),
            direction=direction,
            rationale=rationale,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._persist_signal(sig)
        return sig

    def governance_risk_signal(self, protocol: str = "uniswap") -> OnChainSignal:
        """
        Governance risk: high-impact contested proposal = elevated risk (bearish signal).
        """
        risk = self._governance.get_governance_risk(protocol)
        risk_score = float(risk.get("risk_score", 5.0))
        # Map risk 0-10 → signal -100 to 0 (higher risk = more bearish)
        score = -(risk_score / 10) * 60
        direction = "bearish" if score < -20 else "neutral"
        rationale = (
            f"Governance risk score {risk_score:.1f}/10 for {protocol}. "
            f"Avg turnout {risk.get('avg_turnout_pct', 0):.0f}%, "
            f"contested votes {risk.get('contested_pct', 0):.0f}%."
        )

        sig = OnChainSignal(
            symbol=protocol.upper(),
            signal_type="governance_risk",
            score=round(score, 1),
            direction=direction,
            rationale=rationale,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._persist_signal(sig)
        return sig

    def bridge_flow_signal(self, chain: str = "Arbitrum") -> OnChainSignal:
        """
        Bridge flow: net capital movement into a chain signals ecosystem growth (bullish).
        Anomalous outflows may signal exploits (bearish).
        """
        anomalies = self._bridge_monitor.detect_bridge_anomalies(z_threshold=2.0)
        if anomalies:
            max_anomaly = max(abs(a.anomaly_score) for a in anomalies)
            score = -min(max_anomaly * 20, 100) if any(a.anomaly_score < 0 for a in anomalies) else 0
            direction = "bearish" if score < -20 else "neutral"
            rationale = f"Bridge anomaly detected: {len(anomalies)} bridges with z-score > 2.0"
        else:
            # Check net flows
            summaries = self._bridge_monitor.get_bridge_history(limit=30)
            if not summaries.empty and "net_flow_usd" in summaries.columns:
                net = float(summaries["net_flow_usd"].sum())
                score = min(max(net / 1_000_000, -100), 100)  # scale: $1M = 1 point
                direction = "bullish" if score > 10 else "bearish" if score < -10 else "neutral"
                rationale = f"Net bridge flows to {chain}: ${net/1e6:+.1f}M"
            else:
                score, direction, rationale = 0.0, "neutral", "No bridge flow data available"

        sig = OnChainSignal(
            symbol=chain,
            signal_type="bridge_flow",
            score=round(score, 1),
            direction=direction,
            rationale=rationale,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._persist_signal(sig)
        return sig

    def compute_composite_score(self, symbol: str = "ETH") -> OnChainCompositeScore:
        """
        Aggregate all on-chain signals into a single composite score (-100 to +100).
        Weights:
          exchange_netflow  : 25%
          whale_accumulation: 30%
          defi_tvl_delta    : 20%
          governance_risk   : 10%
          bridge_flow       : 15%
        """
        try:
            netflow = self.exchange_netflow_signal(symbol)
        except Exception:  # noqa: BLE001
            netflow = OnChainSignal(symbol=symbol, signal_type="exchange_netflow", score=0.0,
                                   direction="neutral", rationale="error",
                                   computed_at=datetime.now(timezone.utc).isoformat())
        try:
            whale = self.whale_accumulation_signal(symbol)
        except Exception:  # noqa: BLE001
            whale = OnChainSignal(symbol=symbol, signal_type="whale_accumulation", score=0.0,
                                  direction="neutral", rationale="error",
                                  computed_at=datetime.now(timezone.utc).isoformat())
        try:
            tvl = self.defi_tvl_signal()
        except Exception:  # noqa: BLE001
            tvl = OnChainSignal(symbol=symbol, signal_type="defi_tvl_delta", score=0.0,
                                direction="neutral", rationale="error",
                                computed_at=datetime.now(timezone.utc).isoformat())
        try:
            gov = self.governance_risk_signal()
        except Exception:  # noqa: BLE001
            gov = OnChainSignal(symbol=symbol, signal_type="governance_risk", score=0.0,
                                direction="neutral", rationale="error",
                                computed_at=datetime.now(timezone.utc).isoformat())
        try:
            bridge = self.bridge_flow_signal()
        except Exception:  # noqa: BLE001
            bridge = OnChainSignal(symbol=symbol, signal_type="bridge_flow", score=0.0,
                                   direction="neutral", rationale="error",
                                   computed_at=datetime.now(timezone.utc).isoformat())

        composite = (
            netflow.score * 0.25
            + whale.score * 0.30
            + tvl.score * 0.20
            + gov.score * 0.10
            + bridge.score * 0.15
        )
        composite = round(min(max(composite, -100), 100), 1)
        direction = "bullish" if composite > 15 else "bearish" if composite < -15 else "neutral"

        return OnChainCompositeScore(
            symbol=symbol,
            exchange_netflow_score=netflow.score,
            whale_accumulation_score=whale.score,
            tvl_delta_score=tvl.score,
            governance_risk_score=gov.score,
            bridge_flow_score=bridge.score,
            composite_score=composite,
            direction=direction,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _persist_signal(self, sig: OnChainSignal) -> None:
        with _db() as conn:
            conn.execute(
                """INSERT INTO onchain_signals
                   (symbol, signal_type, score, direction, rationale, computed_at)
                   VALUES (?,?,?,?,?,?)""",
                (sig.symbol, sig.signal_type, sig.score,
                 sig.direction, sig.rationale, sig.computed_at),
            )

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        with _db() as conn:
            conditions: List[str] = []
            args: List[Any] = []
            if symbol:
                conditions.append("symbol=?")
                args.append(symbol)
            if signal_type:
                conditions.append("signal_type=?")
                args.append(signal_type)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = conn.execute(
                f"SELECT * FROM onchain_signals {where} ORDER BY computed_at DESC LIMIT ?",
                args + [limit],
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

onchain_events_router = APIRouter(prefix="/onchain", tags=["onchain-events"])

_whale_monitor = WhaleTransactionMonitor()
_contract_monitor = SmartContractMonitor()
_governance_tracker = GovernanceVoteTracker()
_bridge_monitor = BridgeFlowMonitor()
_signal_engine = OnChainSignalEngine()


@onchain_events_router.get("/whale-txs")
def get_whale_transactions(
    symbol: Optional[str] = Query(None),
    chain: str = Query("ETH"),
    limit: int = Query(50),
    live: bool = Query(False),
):
    """
    Return recent whale transactions. Use ?live=true to fetch fresh data
    from blockchain.info / Etherscan (may be slow).
    """
    if live:
        if chain.upper() == "BTC":
            df = _whale_monitor.fetch_btc_large_txs()
        else:
            df = _whale_monitor.fetch_eth_large_txs()
        records = df.head(limit).to_dict("records") if not df.empty else []
    else:
        df = _whale_monitor.get_recent_whale_txs(symbol, limit)
        records = df.to_dict("records") if not df.empty else []

    return {
        "whale_txs": records,
        "total": len(records),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@onchain_events_router.get("/exchange-flows")
def get_exchange_flows(
    symbol: str = Query("ETH"),
    hours: int = Query(24),
    live: bool = Query(False),
):
    """Net exchange inflows/outflows for a symbol over the last N hours."""
    if live:
        flows = _whale_monitor.monitor_exchange_flows(symbol)
        flow_records = [f.model_dump() for f in flows]
    else:
        flow_records = []

    netflow = _whale_monitor.compute_exchange_netflow(symbol, hours)
    return {
        "symbol": symbol,
        "hours": hours,
        "netflow": netflow,
        "live_flows": flow_records,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@onchain_events_router.get("/governance")
def get_governance(
    state: str = Query("active"),
    protocol: Optional[str] = Query(None),
    limit: int = Query(20),
):
    """Fetch governance proposals from Snapshot.org."""
    spaces = None
    if protocol:
        # Try to match protocol name to a Snapshot space
        matched = [s for s in _SNAPSHOT_SPACES if protocol.lower() in s.lower()]
        spaces = matched if matched else _SNAPSHOT_SPACES

    proposals = _governance_tracker.fetch_snapshot_proposals(spaces=spaces, state=state, limit=limit)
    return {
        "proposals": [p.model_dump() for p in proposals],
        "total": len(proposals),
        "state": state,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@onchain_events_router.get("/bridge-flows")
def get_bridge_flows(
    chain: str = Query("Arbitrum"),
    days: int = Query(7),
    anomaly_only: bool = Query(False),
):
    """Fetch bridge flow data from DeFiLlama."""
    if anomaly_only:
        summaries = _bridge_monitor.detect_bridge_anomalies()
        return {
            "anomalies": [s.model_dump() for s in summaries],
            "total": len(summaries),
        }
    summaries = _bridge_monitor.compute_net_flows(chain=chain, days=days)
    return {
        "bridge_flows": [s.model_dump() for s in summaries],
        "chain": chain,
        "days": days,
        "total": len(summaries),
    }


@onchain_events_router.get("/signals")
def get_onchain_signals(
    symbol: str = Query("ETH"),
    composite: bool = Query(True),
):
    """Compute on-chain signals. Use ?composite=true for full score."""
    if composite:
        score = _signal_engine.compute_composite_score(symbol)
        return score.model_dump()

    try:
        netflow = _signal_engine.exchange_netflow_signal(symbol)
        whale = _signal_engine.whale_accumulation_signal(symbol)
        tvl = _signal_engine.defi_tvl_signal()
    except Exception as exc:
        raise HTTPException(500, f"Signal computation error: {exc}")

    return {
        "symbol": symbol,
        "signals": [netflow.model_dump(), whale.model_dump(), tvl.model_dump()],
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@onchain_events_router.get("/liquidations")
def get_liquidations(
    protocol: str = Query("aave"),
    limit: int = Query(50),
):
    """Fetch recent liquidation events from Aave/Compound via Etherscan."""
    events = _contract_monitor.monitor_liquidations(protocol)
    db_events = _contract_monitor.get_recent_events("liquidation", limit)
    return {
        "protocol": protocol,
        "live_events": [e.model_dump() for e in events],
        "historical": db_events.to_dict("records"),
        "total_live": len(events),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@onchain_events_router.get("/events/{contract}")
def get_contract_events(
    contract: str,
    event_type: Optional[str] = Query(None),
    limit: int = Query(50),
):
    """Fetch recent events for a known contract address."""
    contract_lower = contract.lower()
    contract_info = _contract_monitor.get_contract_info(contract_lower)

    db_events = _contract_monitor.get_recent_events(event_type, limit)
    if not db_events.empty and "contract" in db_events.columns:
        filtered = db_events[db_events["contract"].str.lower() == contract_lower]
    else:
        filtered = db_events

    return {
        "contract": contract,
        "protocol_info": contract_info,
        "events": filtered.to_dict("records"),
        "total": len(filtered),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@onchain_events_router.get("/signal-history")
def get_signal_history(
    symbol: Optional[str] = Query(None),
    signal_type: Optional[str] = Query(None),
    limit: int = Query(100),
):
    """Retrieve historical on-chain signal scores from the database."""
    df = _signal_engine.get_signal_history(symbol, signal_type, limit)
    return {
        "signals": df.to_dict("records"),
        "total": len(df),
    }


@onchain_events_router.get("/governance-risk/{protocol}")
def get_governance_risk(protocol: str):
    """Compute governance risk score for a protocol."""
    risk = _governance_tracker.get_governance_risk(protocol)
    return risk


@onchain_events_router.get("/known-contracts")
def list_known_contracts(protocol: Optional[str] = Query(None)):
    """List known DeFi protocol contracts tracked by SENTINEL."""
    contracts = KNOWN_CONTRACTS
    if protocol:
        contracts = {
            addr: info for addr, info in KNOWN_CONTRACTS.items()
            if protocol.lower() in info.get("protocol", "").lower()
        }
    return {"contracts": contracts, "total": len(contracts)}


@onchain_events_router.get("/known-wallets")
def list_known_wallets():
    """List tracked exchange and whale wallet addresses."""
    return {
        "exchange_wallets": KNOWN_EXCHANGE_WALLETS,
        "whale_addresses": EtherscanEventAdapter.KNOWN_WHALE_ADDRESSES,
        "total_exchange": len(KNOWN_EXCHANGE_WALLETS),
        "total_whale": len(EtherscanEventAdapter.KNOWN_WHALE_ADDRESSES),
    }
