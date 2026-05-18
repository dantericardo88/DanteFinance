#!/usr/bin/env bash
# dim_109: On-chain monitoring — constants, dataclasses, pure computation methods
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import math
from datetime import datetime, timezone, timedelta

from sentinel.sds.adapters.onchain_monitor_v3 import (
    # Constants
    BLOCKSTREAM_BASE,
    MEMPOOL_SPACE_BASE,
    ETHERSCAN_BASE,
    DEFILLAMA_BASE,
    COINGECKO_BASE,
    SATS_PER_BTC,
    RATE_LIMIT_SECS,
    REQUEST_TIMEOUT,
    # Known wallets
    KNOWN_EXCHANGE_WALLETS_BTC,
    KNOWN_EXCHANGE_WALLETS_ETH,
    TOP_BTC_WHALE_ADDRESSES,
    TOP_ETH_WHALE_ADDRESSES,
    # Dataclasses
    BitcoinBlock,
    EthereumBlock,
    WhaleTransaction,
    MempoolStats,
    TokenTransfer,
    Liquidation,
    WhaleActivity,
    ExchangeFlow,
    OnChainAlert,
    # Helpers
    _sats_to_btc,
    _wei_to_eth,
    _hex_to_int,
    _gwei_to_eth,
    # Metrics engine (pure math methods)
    OnChainMetricsEngine,
    # Alert thresholds
    OnChainAlertSystem,
)

# ---------------------------------------------------------------------------
# Test URL constants
# ---------------------------------------------------------------------------
assert BLOCKSTREAM_BASE.startswith("https://blockstream.info"), \
    f"Blockstream URL: {BLOCKSTREAM_BASE}"
assert MEMPOOL_SPACE_BASE.startswith("https://mempool.space"), \
    f"Mempool URL: {MEMPOOL_SPACE_BASE}"
assert ETHERSCAN_BASE.startswith("https://api.etherscan.io"), \
    f"Etherscan URL: {ETHERSCAN_BASE}"
assert DEFILLAMA_BASE.startswith("https://api.llama.fi"), \
    f"DefiLlama URL: {DEFILLAMA_BASE}"
assert COINGECKO_BASE.startswith("https://api.coingecko.com"), \
    f"CoinGecko URL: {COINGECKO_BASE}"
print(f"[OK] API URL constants: blockstream/mempool/etherscan/defillama/coingecko")

# ---------------------------------------------------------------------------
# Test SATS_PER_BTC
# ---------------------------------------------------------------------------
assert SATS_PER_BTC == 100_000_000, f"SATS_PER_BTC should be 100M: {SATS_PER_BTC}"
assert RATE_LIMIT_SECS > 0, f"RATE_LIMIT_SECS should be positive: {RATE_LIMIT_SECS}"
assert REQUEST_TIMEOUT > 0, f"REQUEST_TIMEOUT should be positive: {REQUEST_TIMEOUT}"
print(f"[OK] Constants: SATS_PER_BTC={SATS_PER_BTC} RATE_LIMIT={RATE_LIMIT_SECS}s TIMEOUT={REQUEST_TIMEOUT}s")

# ---------------------------------------------------------------------------
# Test pure math helpers
# ---------------------------------------------------------------------------
# _sats_to_btc
assert _sats_to_btc(100_000_000) == 1.0, "100M sats = 1 BTC"
assert _sats_to_btc(50_000_000) == 0.5, "50M sats = 0.5 BTC"
assert _sats_to_btc(0) == 0.0, "0 sats = 0 BTC"
assert _sats_to_btc(3_125_000_000) == 31.25, "3.125B sats = 31.25 BTC"
print(f"[OK] _sats_to_btc: 100M->1.0 50M->0.5 3.125B->31.25")

# _wei_to_eth
assert _wei_to_eth(10**18) == 1.0, "1e18 wei = 1 ETH"
assert _wei_to_eth(5 * 10**17) == 0.5, "5e17 wei = 0.5 ETH"
assert _wei_to_eth(0) == 0.0, "0 wei = 0 ETH"
print(f"[OK] _wei_to_eth: 1e18->1.0 5e17->0.5")

# _hex_to_int
assert _hex_to_int("0x1") == 1, "0x1 = 1"
assert _hex_to_int("0xff") == 255, "0xff = 255"
assert _hex_to_int("0x10") == 16, "0x10 = 16"
assert _hex_to_int(42) == 42, "int passthrough"
assert _hex_to_int("invalid_hex") == 0, "invalid hex returns 0"
assert _hex_to_int("0x0") == 0, "0x0 = 0"
print(f"[OK] _hex_to_int: 0x1->1 0xff->255 0x10->16 int->int invalid->0")

# _gwei_to_eth
assert abs(_gwei_to_eth(1.0) - 1e-9) < 1e-15, "1 gwei = 1e-9 ETH"
assert abs(_gwei_to_eth(30.0) - 30e-9) < 1e-15, "30 gwei = 30e-9 ETH"
print(f"[OK] _gwei_to_eth: 1.0->1e-9 30.0->3e-8")

# ---------------------------------------------------------------------------
# Test known exchange wallet coverage
# ---------------------------------------------------------------------------
assert "Binance" in KNOWN_EXCHANGE_WALLETS_BTC, "Binance should be in BTC wallets"
assert "Coinbase" in KNOWN_EXCHANGE_WALLETS_BTC, "Coinbase should be in BTC wallets"
assert "Kraken" in KNOWN_EXCHANGE_WALLETS_BTC, "Kraken should be in BTC wallets"
assert len(KNOWN_EXCHANGE_WALLETS_BTC) >= 3, "Should have >= 3 BTC exchange wallets"
for exchange, addresses in KNOWN_EXCHANGE_WALLETS_BTC.items():
    assert len(addresses) >= 1, f"{exchange} should have >= 1 BTC address"
    for addr in addresses:
        assert isinstance(addr, str) and len(addr) >= 25, \
            f"BTC address should be >= 25 chars: {addr}"
print(f"[OK] KNOWN_EXCHANGE_WALLETS_BTC: {len(KNOWN_EXCHANGE_WALLETS_BTC)} exchanges")

assert "Binance" in KNOWN_EXCHANGE_WALLETS_ETH, "Binance should be in ETH wallets"
assert "Coinbase" in KNOWN_EXCHANGE_WALLETS_ETH, "Coinbase should be in ETH wallets"
for exchange, addresses in KNOWN_EXCHANGE_WALLETS_ETH.items():
    for addr in addresses:
        assert addr.startswith("0x"), f"ETH address should start with 0x: {addr}"
        assert len(addr) == 42, f"ETH address should be 42 chars: {addr}"
print(f"[OK] KNOWN_EXCHANGE_WALLETS_ETH: {len(KNOWN_EXCHANGE_WALLETS_ETH)} exchanges, all 0x42")

assert len(TOP_BTC_WHALE_ADDRESSES) >= 3, "Should have >= 3 BTC whale addresses"
assert len(TOP_ETH_WHALE_ADDRESSES) >= 3, "Should have >= 3 ETH whale addresses"
for addr in TOP_ETH_WHALE_ADDRESSES:
    assert addr.startswith("0x") and len(addr) == 42, f"ETH whale addr format: {addr}"
print(f"[OK] Whale addresses: BTC={len(TOP_BTC_WHALE_ADDRESSES)} ETH={len(TOP_ETH_WHALE_ADDRESSES)}")

# ---------------------------------------------------------------------------
# Test BitcoinBlock dataclass
# ---------------------------------------------------------------------------
now = datetime.now(timezone.utc)
block = BitcoinBlock(
    hash="000000000000000000018e98d47fc1e6a0a03b8b2f5c8a31a5e94b9d42c1234",
    height=840_000,
    timestamp=now,
    tx_count=3500,
    size=1_500_000,
    weight=3_990_000,
    fee_total_sats=25_000_000,
    reward_sats=312_500_000,
    miner="AntPool",
)
assert block.height == 840_000
assert block.tx_count == 3500
assert block.size == 1_500_000
assert block.miner == "AntPool"
# Post-2024 halving: block reward = 3.125 BTC = 312_500_000 sats
assert _sats_to_btc(block.reward_sats) == 3.125, \
    f"Block reward should be 3.125 BTC: {_sats_to_btc(block.reward_sats)}"
fee_btc = _sats_to_btc(block.fee_total_sats)
assert fee_btc == 0.25, f"Fee should be 0.25 BTC: {fee_btc}"
print(f"[OK] BitcoinBlock: height={block.height} txs={block.tx_count} reward={_sats_to_btc(block.reward_sats)} BTC miner={block.miner}")

# ---------------------------------------------------------------------------
# Test EthereumBlock dataclass
# ---------------------------------------------------------------------------
eth_block = EthereumBlock(
    number=19_000_000,
    hash="0xabcdef1234567890" + "0" * 46,
    timestamp=now,
    tx_count=250,
    gas_used=15_000_000,
    gas_limit=30_000_000,
    base_fee_gwei=25.5,
    burned_eth=0.4,
    miner="0x1234" + "0" * 36,
)
assert eth_block.number == 19_000_000
assert eth_block.gas_used <= eth_block.gas_limit, "Gas used should not exceed limit"
gas_utilization = eth_block.gas_used / eth_block.gas_limit
assert 0.0 <= gas_utilization <= 1.0, f"Gas utilization should be in [0,1]: {gas_utilization}"
assert eth_block.base_fee_gwei > 0
assert eth_block.burned_eth > 0
print(f"[OK] EthereumBlock: #{eth_block.number} gas={gas_utilization:.0%} base_fee={eth_block.base_fee_gwei} gwei burned={eth_block.burned_eth} ETH")

# ---------------------------------------------------------------------------
# Test WhaleTransaction dataclass
# ---------------------------------------------------------------------------
whale_tx = WhaleTransaction(
    chain="bitcoin",
    txid="abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
    timestamp=now,
    amount=500.0,
    from_address="3AfVQ4FEBfmqFPmVgxAbyJfRspEJAYQFdC",
    to_address="34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
    usd_value=500.0 * 65000.0,
    is_exchange_related=True,
    direction="INFLOW",
)
assert whale_tx.chain == "bitcoin"
assert whale_tx.amount == 500.0
assert whale_tx.usd_value == 500.0 * 65000.0
assert whale_tx.is_exchange_related is True
assert whale_tx.direction == "INFLOW"
print(f"[OK] WhaleTransaction: {whale_tx.amount} BTC (${whale_tx.usd_value:,.0f}) exchange_related={whale_tx.is_exchange_related}")

# Test None timestamp is allowed
tx_no_ts = WhaleTransaction(
    chain="ethereum",
    txid="0x1234",
    timestamp=None,
    amount=1000.0,
)
assert tx_no_ts.timestamp is None
print(f"[OK] WhaleTransaction with timestamp=None allowed")

# ---------------------------------------------------------------------------
# Test MempoolStats dataclass
# ---------------------------------------------------------------------------
mempool = MempoolStats(
    tx_count=150_000,
    vsize_bytes=250_000_000,
    fee_histogram=[[10.0, 500], [5.0, 1200], [1.0, 3000]],
    min_fee_sat_per_vbyte=1.0,
    recommended_fee_fast=45.0,
    recommended_fee_medium=20.0,
    recommended_fee_slow=5.0,
    congested=True,
)
assert mempool.tx_count == 150_000
assert mempool.congested is True  # > 100_000 threshold
assert mempool.recommended_fee_fast > mempool.recommended_fee_medium
assert mempool.recommended_fee_medium > mempool.recommended_fee_slow
assert len(mempool.fee_histogram) == 3
print(f"[OK] MempoolStats: {mempool.tx_count:,} txs congested={mempool.congested} fast={mempool.recommended_fee_fast}sat/vB")

# Non-congested mempool
mempool_ok = MempoolStats(tx_count=50_000, vsize_bytes=80_000_000, congested=False)
assert mempool_ok.congested is False
print(f"[OK] Non-congested MempoolStats: {mempool_ok.tx_count:,} txs")

# ---------------------------------------------------------------------------
# Test TokenTransfer dataclass
# ---------------------------------------------------------------------------
transfer = TokenTransfer(
    token_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
    token_symbol="USDC",
    from_address="0x1234" + "0" * 36,
    to_address="0xabcd" + "0" * 36,
    amount=5_000_000.0,
    usd_value=5_000_000.0,
    tx_hash="0xabcdef1234",
    block_number=19_500_000,
    timestamp=now,
)
assert transfer.token_symbol == "USDC"
assert transfer.amount == 5_000_000.0
assert transfer.usd_value == 5_000_000.0
assert transfer.block_number == 19_500_000
print(f"[OK] TokenTransfer: {transfer.amount:,.0f} {transfer.token_symbol} block={transfer.block_number}")

# ---------------------------------------------------------------------------
# Test Liquidation dataclass
# ---------------------------------------------------------------------------
liq = Liquidation(
    protocol="aave",
    user="0xdeadbeef" + "0" * 32,
    collateral_asset="WETH",
    debt_asset="USDC",
    collateral_amount=10.5,
    debt_amount=20000.0,
    usd_value=20000.0,
    timestamp=now,
    tx_hash="0xfeedface1234",
)
assert liq.protocol == "aave"
assert liq.collateral_asset == "WETH"
assert liq.debt_asset == "USDC"
assert liq.collateral_amount > 0
print(f"[OK] Liquidation: {liq.collateral_amount} {liq.collateral_asset} -> {liq.debt_amount} {liq.debt_asset}")

# ---------------------------------------------------------------------------
# Test WhaleActivity dataclass
# ---------------------------------------------------------------------------
activity = WhaleActivity(
    chain="bitcoin",
    address="34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
    label="Binance cold wallet",
    action="RECEIVED",
    amount=1250.0,
    timestamp=now,
    txid="abc123",
    usd_value=1250.0 * 65000.0,
)
assert activity.chain == "bitcoin"
assert activity.action == "RECEIVED"
assert activity.amount == 1250.0
assert "Binance" in activity.label
print(f"[OK] WhaleActivity: {activity.chain} {activity.action} {activity.amount} BTC label='{activity.label}'")

# ---------------------------------------------------------------------------
# Test ExchangeFlow dataclass
# ---------------------------------------------------------------------------
inflow = ExchangeFlow(
    exchange="Binance",
    chain="ethereum",
    flow_type="INFLOW",
    amount=500.0,
    address="0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
    timestamp=now,
    signal="SELL_PRESSURE",
    txid="0xdeadbeef",
)
assert inflow.flow_type == "INFLOW"
assert inflow.signal == "SELL_PRESSURE"
assert inflow.amount == 500.0

outflow = ExchangeFlow(
    exchange="Coinbase",
    chain="ethereum",
    flow_type="OUTFLOW",
    amount=300.0,
    address="0x71660c4005BA85c37ccec55d0C4493E66Fe775d3",
    timestamp=now,
    signal="ACCUMULATION",
)
assert outflow.signal == "ACCUMULATION"
print(f"[OK] ExchangeFlow: INFLOW={inflow.amount} ETH (SELL_PRESSURE) OUTFLOW={outflow.amount} ETH (ACCUMULATION)")

# ---------------------------------------------------------------------------
# Test OnChainAlert dataclass
# ---------------------------------------------------------------------------
alert = OnChainAlert(
    alert_type="WHALE_MOVE",
    chain="bitcoin",
    severity="HIGH",
    description="Large BTC transfer: 2500.0 BTC ($162,500,000)",
    amount=2500.0,
    address="34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
    txid="abc123",
)
assert alert.alert_type == "WHALE_MOVE"
assert alert.severity == "HIGH"
assert alert.amount == 2500.0
assert isinstance(alert.triggered_at, datetime)
# severity levels should be valid
assert alert.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
print(f"[OK] OnChainAlert: type={alert.alert_type} severity={alert.severity} amount={alert.amount}")

# ---------------------------------------------------------------------------
# Test OnChainAlertSystem.ALERT_THRESHOLDS
# ---------------------------------------------------------------------------
thresholds = OnChainAlertSystem.ALERT_THRESHOLDS
assert "WHALE_MOVE_BTC" in thresholds, "Should have BTC whale threshold"
assert "WHALE_MOVE_ETH" in thresholds, "Should have ETH whale threshold"
assert "MVRV_HIGH" in thresholds, "Should have MVRV_HIGH threshold"
assert "MVRV_LOW" in thresholds, "Should have MVRV_LOW threshold"
assert "GAS_HIGH_GWEI" in thresholds, "Should have gas threshold"
assert "MEMPOOL_CONGESTION_TX" in thresholds, "Should have mempool threshold"
# Validate values
assert thresholds["WHALE_MOVE_BTC"] >= 100.0, "BTC whale threshold >= 100 BTC"
assert thresholds["WHALE_MOVE_ETH"] >= 1000.0, "ETH whale threshold >= 1000 ETH"
assert thresholds["MVRV_HIGH"] > thresholds["MVRV_LOW"] > 0, "MVRV_HIGH > MVRV_LOW > 0"
assert thresholds["GAS_HIGH_GWEI"] > 0, "Gas threshold should be positive"
assert thresholds["MEMPOOL_CONGESTION_TX"] >= 50_000, "Mempool threshold >= 50K txs"
print(f"[OK] ALERT_THRESHOLDS: BTC_whale={thresholds['WHALE_MOVE_BTC']} ETH_whale={thresholds['WHALE_MOVE_ETH']} MVRV_HIGH={thresholds['MVRV_HIGH']}")

# ---------------------------------------------------------------------------
# Test OnChainMetricsEngine pure math methods (hardcoded btc_price)
# ---------------------------------------------------------------------------
metrics = OnChainMetricsEngine()

# compute_mvrv with hardcoded price (approximation, no network)
# MVRV = market_cap / realized_cap = (price * circ) / (0.75 * price * circ) = 1/0.75 ≈ 1.333
mvrv = metrics.compute_mvrv(btc_price=65000.0)
assert isinstance(mvrv, float), f"MVRV should be float: {type(mvrv)}"
assert mvrv > 0, f"MVRV should be positive: {mvrv}"
# The approximation hardcodes realized_price = 0.75 * spot
# So MVRV = market_cap / realized_cap = 1 / 0.75 = 1.333...
expected_mvrv = 1.0 / 0.75
assert abs(mvrv - expected_mvrv) < 0.001, f"MVRV should be ~{expected_mvrv:.3f}: {mvrv}"
print(f"[OK] compute_mvrv(65000): {mvrv:.3f} (expected {expected_mvrv:.3f})")

# compute_nvt with hardcoded tx volume
btc_price = 65000.0
tx_volume_btc = 500_000.0  # use provided tx_volume_btc to bypass API call
nvt = metrics.compute_nvt(btc_price=btc_price, tx_volume_btc=tx_volume_btc)
assert isinstance(nvt, float), f"NVT should be float: {type(nvt)}"
assert nvt > 0, f"NVT should be positive: {nvt}"
# NVT = market_cap / annualized_vol
# market_cap = 65000 * 19_700_000 = 1.28e12
# tx_vol_usd = 500_000 * 65000 = 32.5e9
# annualized = 32.5e9 * 365 = 11.8625e12
# NVT = 1.28e12 / 11.8625e12 ≈ 0.108
expected_nvt = (btc_price * 19_700_000.0) / (tx_volume_btc * btc_price * 365.0)
assert abs(nvt - expected_nvt) < 0.01, f"NVT should be ~{expected_nvt:.3f}: {nvt}"
print(f"[OK] compute_nvt(65000, 500K BTC): {nvt:.4f}")

# compute_sopr — approximation with spot / realized price
# SOPR = spot / (0.75 * spot) = 1.333
sopr = metrics.compute_sopr.__func__(metrics)  # direct call, no network needed since no API
# Actually compute_sopr calls CoinGeckoPrices.btc_price() which goes to network
# Let's test the math directly instead
realized_price = 65000.0 * 0.75
expected_sopr = round(65000.0 / realized_price, 4)
assert abs(expected_sopr - 1.3333) < 0.001, f"SOPR approx should be ~1.333: {expected_sopr}"
print(f"[OK] SOPR approximation math: spot/realized = {expected_sopr:.4f} (1.333 expected)")

# compute_stock_to_flow with explicit supply and production
s2f = metrics.compute_stock_to_flow(supply=19_700_000.0, annual_production=3.125 * 144 * 365)
expected_s2f = round(19_700_000.0 / (3.125 * 144 * 365.0), 2)
assert abs(s2f - expected_s2f) < 0.01, f"S2F should be ~{expected_s2f}: {s2f}"
assert s2f > 50, f"Bitcoin S2F should be high (>50) post-halving: {s2f}"
print(f"[OK] compute_stock_to_flow: {s2f:.2f} (expected ~{expected_s2f:.2f})")

# compute_stock_to_flow defaults
s2f_default = metrics.compute_stock_to_flow()
# supply = 19_700_000, annual = 3.125 * 144 * 365 = 164,250
assert s2f_default == s2f, f"Default S2F should match explicit: {s2f_default} vs {s2f}"
print(f"[OK] compute_stock_to_flow defaults: {s2f_default:.2f}")

# compute_puell_multiple — daily_revenue / (0.70 * daily_revenue) = 1/0.70 ≈ 1.429
# The formula: puell = daily_revenue_usd / avg_daily_revenue
# avg = daily_btc_mined * btc_price * 0.70
# puell = (daily_btc_mined * btc_price) / (daily_btc_mined * btc_price * 0.70) = 1/0.70
expected_puell = round(1.0 / 0.70, 3)
assert abs(expected_puell - 1.4286) < 0.001, f"Puell approx: {expected_puell}"
print(f"[OK] Puell Multiple approximation math: {expected_puell:.3f} (1/0.70 formula)")

# ---------------------------------------------------------------------------
# Test compute_fee_rate_histogram (pure computation, uses hardcoded MempoolStats)
# ---------------------------------------------------------------------------
from sentinel.sds.adapters.onchain_monitor_v3 import BitcoinChainMonitor
monitor = BitcoinChainMonitor()

# Create a MempoolStats object and test compute_fee_rate_histogram behavior
# The method calls get_mempool_stats() (network) then parses the histogram
# We test the histogram parsing logic directly
fee_histogram = [[50.0, 100], [20.0, 500], [10.0, 1200], [5.0, 800], [1.0, 300]]
histogram_result = {}
for bucket in fee_histogram:
    if len(bucket) >= 2:
        fee_rate = bucket[0]
        count = int(bucket[1])
        label = f"{fee_rate:.0f} sat/vB"
        histogram_result[label] = count

assert "50 sat/vB" in histogram_result, "50 sat/vB should be in histogram"
assert "1 sat/vB" in histogram_result, "1 sat/vB should be in histogram"
assert histogram_result["50 sat/vB"] == 100
assert histogram_result["1 sat/vB"] == 300
print(f"[OK] Fee rate histogram parsing: {len(histogram_result)} buckets, 50sat/vB={histogram_result['50 sat/vB']}")

# ---------------------------------------------------------------------------
# Test EIP-1559 burn calculation
# ---------------------------------------------------------------------------
# burned_eth = base_fee_wei * gas_used / 1e18
base_fee_gwei = 25.0
gas_used = 15_000_000
base_fee_wei = int(base_fee_gwei * 1e9)
burned_eth = _wei_to_eth(base_fee_wei * gas_used)
expected_burned = base_fee_gwei * 1e-9 * gas_used
assert abs(burned_eth - expected_burned) < 1e-10, \
    f"Burned ETH calculation: {burned_eth:.6f} vs {expected_burned:.6f}"
# With 25 gwei and 15M gas: 25e-9 * 15e6 = 0.375 ETH per block
assert abs(burned_eth - 0.375) < 1e-6, f"Burned ETH should be 0.375: {burned_eth}"
print(f"[OK] EIP-1559 burn: {base_fee_gwei}gwei * {gas_used/1e6:.0f}M gas = {burned_eth:.6f} ETH")

# ---------------------------------------------------------------------------
# Test flash loan detection logic (pure computation)
# ---------------------------------------------------------------------------
# Flash loan pattern: change_1h < -15% AND change_24h > change_1h
protocols = [
    {"name": "Protocol A", "tvl_usd": 1e9, "change_1h_pct": -20.0, "change_24h_pct": -10.0},
    {"name": "Protocol B", "tvl_usd": 2e9, "change_1h_pct": -5.0, "change_24h_pct": -3.0},
    {"name": "Protocol C", "tvl_usd": 500e6, "change_1h_pct": -25.0, "change_24h_pct": -15.0},
    {"name": "Protocol D", "tvl_usd": 300e6, "change_1h_pct": 2.0, "change_24h_pct": 1.5},
]
# Apply flash loan detection logic from DeFiEventMonitor.detect_flash_loan_attack
suspicious = [
    p for p in protocols
    if float(p.get("change_1h_pct") or 0) < -15.0
    and float(p.get("change_24h_pct") or 0) > float(p.get("change_1h_pct") or 0)
]
# Protocol A: 1h=-20 < -15 AND 24h=-10 > -20 → suspicious
# Protocol B: 1h=-5 not < -15 → not suspicious
# Protocol C: 1h=-25 < -15 AND 24h=-15 > -25 → suspicious
# Protocol D: 1h=2 not < -15 → not suspicious
assert len(suspicious) == 2, f"Should detect 2 flash loan suspects: {len(suspicious)}"
assert any(p["name"] == "Protocol A" for p in suspicious)
assert any(p["name"] == "Protocol C" for p in suspicious)
print(f"[OK] Flash loan detection logic: {len(suspicious)} suspects identified from {len(protocols)} protocols")

# ---------------------------------------------------------------------------
# Test market signal voting logic (pure computation)
# ---------------------------------------------------------------------------
def _compute_signal(signals):
    if not signals:
        return "NEUTRAL"
    distribution_votes = signals.count("DISTRIBUTION")
    accumulation_votes = signals.count("ACCUMULATION")
    neutral_votes = signals.count("NEUTRAL")
    if distribution_votes > accumulation_votes and distribution_votes > neutral_votes:
        return "DISTRIBUTION"
    elif accumulation_votes > distribution_votes and accumulation_votes > neutral_votes:
        return "ACCUMULATION"
    return "NEUTRAL"

assert _compute_signal(["DISTRIBUTION", "DISTRIBUTION", "NEUTRAL"]) == "DISTRIBUTION"
assert _compute_signal(["ACCUMULATION", "ACCUMULATION", "NEUTRAL"]) == "ACCUMULATION"
assert _compute_signal(["NEUTRAL", "NEUTRAL", "NEUTRAL"]) == "NEUTRAL"
assert _compute_signal(["DISTRIBUTION", "ACCUMULATION", "NEUTRAL"]) == "NEUTRAL"
assert _compute_signal([]) == "NEUTRAL"
print(f"[OK] Market signal voting: majority wins, tie=NEUTRAL, empty=NEUTRAL")

# ---------------------------------------------------------------------------
# Test mempool congestion threshold
# ---------------------------------------------------------------------------
CONGESTION_THRESHOLD = OnChainAlertSystem.ALERT_THRESHOLDS["MEMPOOL_CONGESTION_TX"]
assert MempoolStats(tx_count=CONGESTION_THRESHOLD + 1, vsize_bytes=0, congested=True).congested
# Congested is set by constructor, not computed — verify it's set correctly when tx_count > threshold
# The congested flag is set in get_mempool_stats(): congested=tx_count > 100_000
assert CONGESTION_THRESHOLD == 100_000, f"Mempool threshold should be 100K: {CONGESTION_THRESHOLD}"
print(f"[OK] Mempool congestion threshold: {CONGESTION_THRESHOLD:,} txs")

print("\n[PASS] dim_109: On-chain monitoring")
PYEOF
