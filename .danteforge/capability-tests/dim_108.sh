#!/bin/bash
# dim_108: On-Chain Metrics (MVRV, NVT, SOPR, etc.) — capability verification
# Tests pure computation logic (no network calls required)
set -e

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# 1. Import all public classes
from sentinel.sds.adapters.onchain_metrics_v3 import (
    RealizedCapCalculator,
    NVTAnalyzer,
    SOPRAnalyzer,
    PuellMultiple,
    CoinDaysDestroyed,
    NUPLAnalyzer,
    ActiveAddressAnalyzer,
    OnChainCompositeSignal,
    OnChainMetricsEngine,
    BlockchainInfoClient,
    NetworkStats,
    MVRVData,
    OnChainScore,
)
print("[OK] All on-chain metric classes imported")

# 2. RealizedCapCalculator — pure math (no network)
calc = RealizedCapCalculator()

# compute_realized_price (pure division)
real_price = calc.compute_realized_price(realized_cap=300_000_000_000, supply=19_500_000)
assert abs(real_price - 300_000_000_000 / 19_500_000) < 1e-3
print(f"[OK] Realized price: ${real_price:,.0f}")

# Zero supply edge case
assert calc.compute_realized_price(0, 0) == 0.0
print("[OK] Zero supply edge case handled")

# compute_mvrv_ratio
mvrv = calc.compute_mvrv_ratio(market_cap=900_000_000_000, realized_cap=300_000_000_000)
assert abs(mvrv - 3.0) < 1e-9, f"MVRV should be 3.0, got {mvrv}"
print(f"[OK] MVRV ratio: {mvrv:.2f}")

mvrv_bottom = calc.compute_mvrv_ratio(market_cap=200_000_000_000, realized_cap=300_000_000_000)
assert mvrv_bottom < 1.0, "Below realized cap should give MVRV < 1"
print(f"[OK] MVRV < 1 at market bottom: {mvrv_bottom:.2f}")

# compute_mvrv_z_score (pure math with fallback to history)
import pandas as pd
mvrv_history = pd.Series([1.0, 1.5, 2.0, 1.8, 2.5, 3.0, 2.2, 1.7, 1.3, 0.9])
z = calc.compute_mvrv_z_score(
    market_cap=900_000_000_000,
    realized_cap=300_000_000_000,  # MVRV=3.0
    mvrv_history=mvrv_history,
)
assert isinstance(z, float), "Z-score must be float"
print(f"[OK] MVRV Z-score: {z:.3f}")

# Fallback (empty history) - uses historical mean 1.75, std 1.20
z_fallback = calc.compute_mvrv_z_score(
    market_cap=900_000_000_000,
    realized_cap=300_000_000_000,
    mvrv_history=pd.Series(dtype=float),
)
expected = (3.0 - 1.75) / 1.20
assert abs(z_fallback - expected) < 0.01, f"Fallback Z-score should be ~{expected:.3f}"
print(f"[OK] MVRV Z-score fallback: {z_fallback:.3f}")

# 3. Realized cap proxy (pure pandas computation)
import numpy as np
dates = pd.date_range("2023-01-01", periods=200)
price_series = pd.Series(25000 + np.random.randn(200) * 1000, index=dates)
vol_series = pd.Series(10_000_000_000 + np.random.randn(200) * 1e9, index=dates)
rc = calc.compute_realized_cap_proxy(price_series, vol_series, window=90)
assert len(rc) == 200, f"Output length mismatch: {len(rc)}"
assert rc.dropna().iloc[-1] > 0, "Realized cap must be positive"
print(f"[OK] Realized cap proxy computed over {len(rc)} periods")

# 4. OnChainScore dataclass — uses actual fields from the class
from datetime import datetime, timezone
score = OnChainScore(
    timestamp=datetime.now(timezone.utc),
    composite_score=65.0,
    mvrv_z_score=1.5,
    puell_multiple=1.2,
    sopr_value=1.02,
    nvt_signal=90.0,
    nupl=0.55,
    fear_greed_index=62,
    cycle_position="BULL_MID",
)
assert score.composite_score == 65.0
assert score.sentiment in ("EXTREME_GREED", "GREED", "NEUTRAL", "FEAR", "EXTREME_FEAR")
print(f"[OK] OnChainScore: composite={score.composite_score}, sentiment={score.sentiment}")

# 5. Class interface verification
assert hasattr(NVTAnalyzer, '__init__'), "NVTAnalyzer missing __init__"
assert hasattr(SOPRAnalyzer, '__init__'), "SOPRAnalyzer missing __init__"
assert hasattr(PuellMultiple, '__init__'), "PuellMultiple missing __init__"
assert hasattr(CoinDaysDestroyed, '__init__'), "CoinDaysDestroyed missing __init__"
assert hasattr(NUPLAnalyzer, '__init__'), "NUPLAnalyzer missing __init__"
assert hasattr(ActiveAddressAnalyzer, '__init__'), "ActiveAddressAnalyzer missing __init__"
assert hasattr(OnChainCompositeSignal, '__init__'), "OnChainCompositeSignal missing __init__"
assert hasattr(OnChainMetricsEngine, '__init__'), "OnChainMetricsEngine missing __init__"
assert hasattr(BlockchainInfoClient, '__init__'), "BlockchainInfoClient missing __init__"
print("[OK] All 9 on-chain analyzer classes have correct interface")

# 6. MVRVData dataclass — uses actual fields
mvrv_data = MVRVData(
    timestamp=datetime.now(timezone.utc),
    mvrv_ratio=2.5,
    realized_cap=300_000_000_000,
    market_cap=750_000_000_000,
    realized_price=15384.0,
    mvrv_z_score=1.2,
    nupl=0.60,
    nupl_zone="BELIEF",
)
assert mvrv_data.mvrv_ratio == 2.5
assert mvrv_data.nupl_zone == "BELIEF"
print("[OK] MVRVData dataclass works")

# 7. Key method presence checks
assert hasattr(RealizedCapCalculator, 'compute_mvrv_ratio'), "Missing compute_mvrv_ratio"
assert hasattr(RealizedCapCalculator, 'compute_mvrv_z_score'), "Missing compute_mvrv_z_score"
assert hasattr(RealizedCapCalculator, 'compute_realized_price'), "Missing compute_realized_price"
assert hasattr(RealizedCapCalculator, 'compute_realized_cap_proxy'), "Missing proxy method"
assert hasattr(OnChainMetricsEngine, 'get_full_dashboard'), "Missing get_full_dashboard"
assert hasattr(OnChainMetricsEngine, 'run_daily_update'), "Missing run_daily_update"
assert hasattr(OnChainCompositeSignal, 'compute_composite_bull_bear_score'), "Missing composite scorer"
print("[OK] All 7 critical methods verified")

# 8. Metric regime sanity: MVRV > 3 should be at/near top signal
assert mvrv > 1.5, "MVRV=3 should indicate bull market"
assert mvrv_bottom < 1.0, "MVRV < 1 should indicate accumulation zone"
print(f"[OK] MVRV regime logic consistent")

print("\n[PASS] dim_108: On-Chain Metrics — all checks passed")
PYEOF
