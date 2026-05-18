#!/bin/bash
# dim_032: nport_analytics_v3 — N-PORT fund holdings intelligence
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import pandas as pd

from sentinel.sfe.nport_analytics_v3 import (
    _EFTS_BASE,
    _EDGAR_ARCHIVE,
    ASSET_CAT_MAP,
    FF5_SECTOR_LOADINGS,
    FUND_UNIVERSE,
    _SMART_MONEY_MIN_FUNDS,
    _strip_ns,
    _float,
    NPortXMLParser,
    NPortAnalyticsEngine,
)

# --- constants ---
assert "efts.sec.gov" in _EFTS_BASE
assert "sec.gov" in _EDGAR_ARCHIVE
print("[OK] EDGAR URL constants present")

# --- ASSET_CAT_MAP ---
assert "EC" in ASSET_CAT_MAP and ASSET_CAT_MAP["EC"] == "US Equity (Common)"
assert "DB" in ASSET_CAT_MAP
assert len(ASSET_CAT_MAP) >= 8
print("[OK] ASSET_CAT_MAP has expected entries")

# --- FF5_SECTOR_LOADINGS ---
assert "Technology" in FF5_SECTOR_LOADINGS
tech = FF5_SECTOR_LOADINGS["Technology"]
assert "mkt" in tech and "smb" in tech and "hml" in tech
assert tech["mkt"] > 1.0
print("[OK] FF5_SECTOR_LOADINGS: Technology loadings correct")

# --- FUND_UNIVERSE ---
assert len(FUND_UNIVERSE) >= 10
assert any("VOO" == v.get("ticker") for v in FUND_UNIVERSE.values())
assert _SMART_MONEY_MIN_FUNDS == 10
print(f"[OK] FUND_UNIVERSE has {len(FUND_UNIVERSE)} funds, _SMART_MONEY_MIN_FUNDS=10")

# --- _strip_ns ---
tag = "{http://www.sec.gov/edgar/nportXbrl}invstOrSec"
assert _strip_ns(tag) == "invstOrSec"
assert _strip_ns("noNamespace") == "noNamespace"
print("[OK] _strip_ns removes namespace prefix correctly")

# --- _float ---
assert _float("1,234.56") == 1234.56
assert _float(None) == 0.0
assert _float("invalid") == 0.0
print("[OK] _float helper handles various inputs")

# --- concentration_metrics ---
df = pd.DataFrame({
    "name": ["Apple Inc", "Microsoft Corp", "Amazon.com Inc", "NVIDIA Corp", "Alphabet Inc"],
    "ticker": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"],
    "pct_val": [25.0, 20.0, 15.0, 10.0, 8.0],
    "val_usd": [25_000_000, 20_000_000, 15_000_000, 10_000_000, 8_000_000],
})
metrics = NPortAnalyticsEngine.concentration_metrics(df)
assert "hhi" in metrics and "top10_pct" in metrics and "n_holdings" in metrics
expected_hhi = 0.25**2 + 0.20**2 + 0.15**2 + 0.10**2 + 0.08**2
assert abs(metrics["hhi"] - expected_hhi) < 0.001, f"HHI mismatch: {metrics['hhi']}"
assert metrics["n_holdings"] == 5
print(f"[OK] concentration_metrics: HHI={metrics['hhi']:.4f}, n_holdings={metrics['n_holdings']}")

# --- jaccard_overlap ---
df_a = pd.DataFrame({"ticker": ["AAPL", "MSFT", "AMZN", "NVDA"], "pct_val": [30.0, 25.0, 25.0, 20.0]})
df_b = pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOGL", "META"], "pct_val": [35.0, 30.0, 20.0, 15.0]})
overlap = NPortAnalyticsEngine.jaccard_overlap(df_a, df_b)
assert "jaccard" in overlap
assert abs(overlap["jaccard"] - 2/6) < 0.01, f"Jaccard={overlap['jaccard']}"
print(f"[OK] jaccard_overlap = {overlap['jaccard']:.3f} (expected 0.333)")

print("\n[PASS] dim_032: nport_analytics_v3 -- all checks passed")
PYEOF
