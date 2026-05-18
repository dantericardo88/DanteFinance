#!/bin/bash
# dim_046: cftc_cot_v3 — CFTC COT positioning + portfolio-level analytics
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
PYTHONIOENCODING=utf-8 python - <<'PYEOF'
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
    COTPortfolioAnalyzer,
    COTEngine,
    _find_col,
)
import pandas as pd
import numpy as np

# ── 1. constants ──────────────────────────────────────────────────────────────
assert "cftc.gov" in _CFTC_BASE
assert "cftc.gov" in _CFTC_LATEST_DISAGG
assert "cftc.gov" in _CFTC_LATEST_LEGACY
print("[OK] CFTC URL constants present")

# ── 2. market catalogs ────────────────────────────────────────────────────────
assert "Corn" in _GRAINS
assert "Crude Oil WTI" in _ENERGY
assert "Gold" in _METALS
assert "S&P 500 E-Mini" in _EQUITY
assert "10-Year T-Note" in _RATES
assert "Euro FX" in _FX
assert len(_ALL_MARKETS) >= 30
print(f"[OK] Market catalogs: _ALL_MARKETS has {len(_ALL_MARKETS)} markets")

# ── 3. _find_col ──────────────────────────────────────────────────────────────
df_test = pd.DataFrame(columns=["M_Money_Positions_Long_All", "Date", "Market_and_Exchange_Names"])
col = _find_col(df_test, "M_Money_Positions_Long_All", "Managed Money Long")
assert col == "M_Money_Positions_Long_All"
col2 = _find_col(df_test, "Nonexistent_Col")
assert col2 is None
print("[OK] _find_col finds correct column, returns None for missing")

# ── 4. PositionChange dataclass ───────────────────────────────────────────────
pc = PositionChange(
    market="Gold", trader_type="Managed Money",
    prior_net=50000.0, current_net=65000.0,
    change_pct=30.0, is_significant=True,
    direction="increasing_long", as_of="2024-03-15",
    alert_text="Large position increase detected",
)
assert pc.market == "Gold"
assert pc.is_significant is True
assert pc.direction == "increasing_long"
print("[OK] PositionChange dataclass created")

# ── 5. COTSignal dataclass ────────────────────────────────────────────────────
sig = COTSignal(
    market="S&P 500 E-Mini", cftc_code="13874+",
    as_of="2024-03-12", report_type="disaggregated",
    mm_net=250000.0, comm_net=-200000.0, nonrep_net=-50000.0,
    open_interest=2_500_000.0, cot_index=85.0,
    extreme_signal="EXTREME_LONG", commercial_signal="BEARISH",
    speculator_signal="CROWDED_LONG", contrarian_score=85.0,
    crowding_score=90.0,
    narrative="Managed money extremely long; commercials heavily short.",
)
assert sig.market == "S&P 500 E-Mini"
assert sig.extreme_signal == "EXTREME_LONG"
assert sig.contrarian_score == 85.0
print("[OK] COTSignal dataclass created")

# ── 6. class structure ────────────────────────────────────────────────────────
assert hasattr(COTDataDownloader, "__init__")
assert hasattr(COTDataDownloader, "fetch_latest_cot")
assert hasattr(COTMarketCoverage, "get_market_code")
assert hasattr(COTSignalEngine, "compute_cot_index")
assert hasattr(COTPortfolioAnalyzer, "compute_portfolio_cot_signal")
assert hasattr(COTPortfolioAnalyzer, "detect_cot_reversal")
assert hasattr(COTPortfolioAnalyzer, "aggregate_by_sector")
assert hasattr(COTPortfolioAnalyzer, "backtest_extreme_positioning")
assert hasattr(COTPortfolioAnalyzer, "parse_disaggregated_report")
assert hasattr(COTEngine, "__init__")
print("[OK] All class structures present (including 5 new portfolio methods)")

# ── build a COTPortfolioAnalyzer with synthetic history ──────────────────────
# Synthetic history: 60 weeks of data for Crude Oil WTI and Gold

n_weeks = 60
dates   = pd.date_range("2024-01-01", periods=n_weeks, freq="W")

crude_code = _ENERGY["Crude Oil WTI"]   # "067651"
gold_code  = _METALS["Gold"]            # "088691"

# Crude: net positions oscillate 0→50000 (COT index ramps from 0→100)
crude_net = np.linspace(0, 50000, n_weeks)
# Gold: net positions oscillate 10000→30000 (more neutral)
gold_net  = np.linspace(10000, 30000, n_weeks)

rows = []
for i, d in enumerate(dates):
    rows.append({"as_of_date": d, "cftc_code": crude_code,
                 "mm_net": crude_net[i], "comm_net": -crude_net[i],
                 "open_interest": 100000.0, "report_type": "disaggregated"})
    rows.append({"as_of_date": d, "cftc_code": gold_code,
                 "mm_net": gold_net[i], "comm_net": -gold_net[i],
                 "open_interest": 50000.0, "report_type": "disaggregated"})

history_df = pd.DataFrame(rows)
engine     = COTSignalEngine(history_df=history_df)
analyzer   = COTPortfolioAnalyzer(engine)

# ── 7. Portfolio-level COT signal ─────────────────────────────────────────────
# Use a mock engine where we can inject known COT indexes

class _MockEngine(COTSignalEngine):
    """Override compute_cot_index to return fixed values for unit testing."""
    _FIXED = {"crude": 75.0, "gold": 45.0}
    def __init__(self):
        super().__init__(history_df=None)
    def compute_cot_index(self, market, lookback_weeks=52, trader_type="mm"):
        return self._FIXED.get(market, float("nan"))

mock_engine   = _MockEngine()
mock_analyzer = COTPortfolioAnalyzer(mock_engine)

portfolio = [("crude", 0.30), ("gold", 0.70)]
port_cot  = mock_analyzer.compute_portfolio_cot_signal(portfolio)
expected  = 0.30 * 75.0 + 0.70 * 45.0   # = 54.0
assert abs(port_cot - expected) < 0.001, (
    f"Portfolio COT expected {expected:.2f}, got {port_cot:.2f}"
)
print(f"[OK] compute_portfolio_cot_signal: 0.3×75 + 0.7×45 = {port_cot:.2f} (expected {expected:.2f})")

# zero-weight entries are ignored
portfolio2 = [("crude", 0.30), ("gold", 0.70), ("corn", 0.0)]
port_cot2  = mock_analyzer.compute_portfolio_cot_signal(portfolio2)
assert abs(port_cot2 - expected) < 0.001
print("[OK] compute_portfolio_cot_signal: zero-weight market ignored correctly")

# ── 8. COT reversal detector ──────────────────────────────────────────────────
# Series: 82 → 78 → 72 → went from extreme long (>80) and dropped ≥5 → REVERSAL_LONG
series_long_reversal = [82.0, 78.0, 72.0]
result = analyzer.detect_cot_reversal(series_long_reversal, extreme_threshold=20.0, reversal_drop=5.0)
assert result == "REVERSAL_FROM_EXTREME_LONG", f"Expected REVERSAL_FROM_EXTREME_LONG, got '{result}'"
print(f"[OK] detect_cot_reversal: 82→78→72 → '{result}'")

# Short reversal: 12 → 15 → 22 → was below 20, now > trough+5
series_short_reversal = [12.0, 15.0, 22.0]
result_short = analyzer.detect_cot_reversal(series_short_reversal, extreme_threshold=20.0, reversal_drop=5.0)
assert result_short == "REVERSAL_FROM_EXTREME_SHORT", f"Expected REVERSAL_FROM_EXTREME_SHORT, got '{result_short}'"
print(f"[OK] detect_cot_reversal: 12→15→22 → '{result_short}'")

# No reversal: stays in neutral range
series_neutral = [50.0, 55.0, 52.0]
result_neutral = analyzer.detect_cot_reversal(series_neutral)
assert result_neutral == "NO_REVERSAL", f"Expected NO_REVERSAL, got '{result_neutral}'"
print(f"[OK] detect_cot_reversal: 50→55→52 → '{result_neutral}'")

# Insufficient data
result_insuff = analyzer.detect_cot_reversal([80.0, 75.0])
assert result_insuff == "INSUFFICIENT_DATA"
print("[OK] detect_cot_reversal: <3 points → INSUFFICIENT_DATA")

# ── 9. Sector aggregation ─────────────────────────────────────────────────────
# Use mock engine with fixed COT values for specific markets
class _SectorMockEngine(COTSignalEngine):
    _FIXED = {
        "Crude Oil WTI": 70.0,
        "Natural Gas":   60.0,
        "Gold":          45.0,
        "Silver":        50.0,
        "10-Year T-Note": 30.0,
    }
    def __init__(self):
        super().__init__(history_df=None)
    def compute_cot_index(self, market, lookback_weeks=52, trader_type="mm"):
        return self._FIXED.get(market, float("nan"))

sector_engine   = _SectorMockEngine()
sector_analyzer = COTPortfolioAnalyzer(sector_engine)

test_markets = ["Crude Oil WTI", "Natural Gas", "Gold", "Silver", "10-Year T-Note"]
sectors = sector_analyzer.aggregate_by_sector(markets=test_markets)

# Crude and Natural Gas → energy
assert "energy" in sectors, f"Expected 'energy' sector, got keys: {list(sectors.keys())}"
assert sectors["energy"]["market_count"] == 2
energy_avg = (70.0 + 60.0) / 2   # = 65.0
assert abs(sectors["energy"]["avg_cot_index"] - energy_avg) < 0.1, (
    f"Energy avg expected {energy_avg:.1f}, got {sectors['energy']['avg_cot_index']}"
)
print(f"[OK] aggregate_by_sector: Crude+NatGas → energy, avg={sectors['energy']['avg_cot_index']:.1f}")

# Gold and Silver → metals
assert "metals" in sectors
assert sectors["metals"]["market_count"] == 2
metals_avg = (45.0 + 50.0) / 2   # = 47.5
assert abs(sectors["metals"]["avg_cot_index"] - metals_avg) < 0.1
print(f"[OK] aggregate_by_sector: Gold+Silver → metals, avg={sectors['metals']['avg_cot_index']:.1f}")

# 10-Year T-Note → financials
assert "financials" in sectors
assert "10-Year T-Note" in sectors["financials"]["markets"]
print(f"[OK] aggregate_by_sector: 10-Year T-Note → financials")

# ── 10. Historical backtest ───────────────────────────────────────────────────
# Synthetic: always extreme long (COT=95), price always falls 10% after 4 weeks
# → 100% win rate for contrarian short

bt_n = 80
bt_dates  = pd.date_range("2020-01-01", periods=bt_n, freq="W")

# mm_net: starts near max so COT index always ~95+
bt_mm_net = np.full(bt_n, 50000.0)         # constant high → always extreme long
bt_mm_net[:52] = np.linspace(0, 50000, 52) # ramp for lookback window

bt_rows = [{"as_of_date": bt_dates[i], "cftc_code": "TEST001",
            "mm_net": bt_mm_net[i], "open_interest": 100000.0}
           for i in range(bt_n)]
bt_cot_df = pd.DataFrame(bt_rows)

# Price falls 10% each 4 weeks — so contrarian shorts always win
bt_prices = pd.Series(
    [100.0 * (0.90 ** (i / 4)) for i in range(bt_n * 2)],
    index=pd.date_range("2020-01-01", periods=bt_n * 2, freq="W"),
)

bt_result = analyzer.backtest_extreme_positioning(
    market="Test Market",
    cot_history=bt_cot_df,
    price_history=bt_prices,
    extreme_threshold=20.0,
    price_move_pct=5.0,
    forward_weeks=4,
    trader_type="mm",
)

assert bt_result["backtest_market"] == "Test Market"
assert bt_result["total_signals"] > 0, "Should have detected at least one extreme signal"
assert bt_result["long_signals"] > 0,  "Should have extreme-long signals (COT near 100)"
assert bt_result["win_rate_pct"] >= 80.0, (
    f"Win rate should be ≥80% with falling prices, got {bt_result['win_rate_pct']:.1f}%"
)
print(f"[OK] backtest_extreme_positioning: {bt_result['total_signals']} signals, "
      f"win_rate={bt_result['win_rate_pct']:.1f}%")

# Empty history → safe return
bt_empty = analyzer.backtest_extreme_positioning("X", pd.DataFrame(), pd.Series(dtype=float))
assert bt_empty["total_signals"] == 0
print("[OK] backtest_extreme_positioning: empty inputs return zero signals safely")

# ── 11. Disaggregated CSV parser ─────────────────────────────────────────────
sample_csv = (
    "Market_and_Exchange_Names,As_of_Date_In_Form_YYYY-MM-DD,CFTC_Commodity_Code,"
    "Open_Interest_All,"
    "Prod_Merc_Positions_Long_All,Prod_Merc_Positions_Short_All,"
    "Swap_Positions_Long_All,Swap_Positions_Short_All,Swap__Positions_Spread_All,"
    "M_Money_Positions_Long_All,M_Money_Positions_Short_All,M_Money_Positions_Spread_All,"
    "Other_Rept_Positions_Long_All,Other_Rept_Positions_Short_All,Other_Rept_Positions_Spread_All,"
    "NonRept_Positions_Long_All,NonRept_Positions_Short_All,"
    "Change_in_Open_Interest_All,Change_in_M_Money_Long_All,Change_in_M_Money_Short_All\n"
    "CRUDE OIL WTI - NYMEX,2024-03-12,067651,"
    "500000,"
    "100000,80000,"
    "50000,40000,5000,"
    "120000,70000,10000,"
    "30000,25000,3000,"
    "20000,15000,"
    "1000,5000,-2000\n"
)

parsed = analyzer.parse_disaggregated_report(sample_csv)
assert not parsed.empty, "Parsed DataFrame should not be empty"
assert "mm_net"   in parsed.columns, "mm_net column missing"
assert "comm_net" in parsed.columns, "comm_net column missing"
assert "swap_net" in parsed.columns, "swap_net column missing"

row = parsed.iloc[0]
# mm_net = 120000 - 70000 = 50000
assert abs(row["mm_net"]   - 50000.0) < 0.1, f"mm_net expected 50000, got {row['mm_net']}"
# comm_net = 100000 - 80000 = 20000
assert abs(row["comm_net"] - 20000.0) < 0.1, f"comm_net expected 20000, got {row['comm_net']}"
# swap_net = 50000 - 40000 = 10000
assert abs(row["swap_net"] - 10000.0) < 0.1, f"swap_net expected 10000, got {row['swap_net']}"
print(f"[OK] parse_disaggregated_report: mm_net={row['mm_net']:.0f}, "
      f"comm_net={row['comm_net']:.0f}, swap_net={row['swap_net']:.0f}")

print("\n[PASS] dim_046: cftc_cot_v3 -- all checks passed")
PYEOF
