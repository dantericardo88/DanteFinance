#!/bin/bash
# dim_046: cftc_cot_v3 — CFTC COT commitment of traders positioning
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.cftc_cot_v3 import (
    _CFTC_BASE,
    _CFTC_LATEST_DISAGG,
    _CFTC_LATEST_LEGACY,
    _ALL_MARKETS,
    _GRAINS,
    _ENERGY,
    _METALS,
    _EQUITY,
    _RATES,
    _FX,
    PositionChange,
    COTSignal,
    COTDataDownloader,
    COTMarketCoverage,
    COTSignalEngine,
    COTEngine,
    _find_col,
)
import pandas as pd

# --- constants ---
assert "cftc.gov" in _CFTC_BASE
assert "cftc.gov" in _CFTC_LATEST_DISAGG
assert "cftc.gov" in _CFTC_LATEST_LEGACY
print("[OK] CFTC URL constants present")

# --- market catalogs ---
assert "Corn" in _GRAINS
assert "Crude Oil WTI" in _ENERGY
assert "Gold" in _METALS
assert "S&P 500 E-Mini" in _EQUITY
assert "10-Year T-Note" in _RATES
assert "Euro FX" in _FX
assert len(_ALL_MARKETS) >= 30
print(f"[OK] Market catalogs: _ALL_MARKETS has {len(_ALL_MARKETS)} markets")

# --- _find_col ---
df = pd.DataFrame(columns=["M_Money_Positions_Long_All", "Date", "Market_and_Exchange_Names"])
col = _find_col(df, "M_Money_Positions_Long_All", "Managed Money Long")
assert col == "M_Money_Positions_Long_All"
col2 = _find_col(df, "Nonexistent_Col")
assert col2 is None
print("[OK] _find_col finds correct column, returns None for missing")

# --- PositionChange dataclass ---
pc = PositionChange(
    market="Gold",
    trader_type="Managed Money",
    prior_net=50000.0,
    current_net=65000.0,
    change_pct=30.0,
    is_significant=True,
    direction="increasing_long",
    as_of="2024-03-15",
    alert_text="Large position increase detected",
)
assert pc.market == "Gold"
assert pc.is_significant is True
assert pc.direction == "increasing_long"
print("[OK] PositionChange dataclass created")

# --- COTSignal dataclass ---
sig = COTSignal(
    market="S&P 500 E-Mini",
    cftc_code="13874+",
    as_of="2024-03-12",
    report_type="disaggregated",
    mm_net=250000.0,
    comm_net=-200000.0,
    nonrep_net=-50000.0,
    open_interest=2_500_000.0,
    cot_index=85.0,
    extreme_signal="EXTREME_LONG",
    commercial_signal="BEARISH",
    speculator_signal="CROWDED_LONG",
    contrarian_score=85.0,
    crowding_score=90.0,
    narrative="Managed money extremely long; commercials heavily short — contrarian signal.",
)
assert sig.market == "S&P 500 E-Mini"
assert sig.extreme_signal == "EXTREME_LONG"
assert sig.contrarian_score == 85.0
print("[OK] COTSignal dataclass created")

# --- class structure ---
assert hasattr(COTDataDownloader, "__init__")
assert hasattr(COTDataDownloader, "fetch_latest_cot")
assert hasattr(COTMarketCoverage, "get_market_code")
assert hasattr(COTSignalEngine, "build_signal") or hasattr(COTSignalEngine, "compute_cot_index")
assert hasattr(COTEngine, "__init__")
print("[OK] COTDataDownloader, COTMarketCoverage, COTSignalEngine, COTEngine structures present")

print("\n[PASS] dim_046: cftc_cot_v3 -- all checks passed")
PYEOF
