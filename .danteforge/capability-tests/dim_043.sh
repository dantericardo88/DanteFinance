#!/bin/bash
# dim_043: fred_macro_v3 — FRED macro time series adapter
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sds.adapters.fred_macro_v3 import (
    FRED_CSV_BASE,
    FRED_API_BASE,
    SeriesInfo,
    MacroSnapshot,
    RevisionRecord,
    FREDAPIClient,
    MacroSeriesLibrary,
    FREDMacroEngine,
)
from datetime import date

# --- constants ---
assert "fred.stlouisfed.org" in FRED_CSV_BASE
assert "stlouisfed.org" in FRED_API_BASE
print("[OK] FRED_CSV_BASE and FRED_API_BASE constants present")

# --- SeriesInfo dataclass ---
si = SeriesInfo(
    series_id="FEDFUNDS",
    title="Federal Funds Effective Rate",
    frequency="Monthly",
    units="Percent",
    seasonal_adjustment="NSA",
    category="Interest Rates",
)
assert si.series_id == "FEDFUNDS"
assert si.units == "Percent"
print("[OK] SeriesInfo dataclass created")

# --- MacroSnapshot dataclass ---
snap = MacroSnapshot(
    indicator="Federal Funds Rate",
    series_id="FEDFUNDS",
    as_of=date(2024, 3, 1),
    latest_value=5.33,
    previous_value=5.33,
    change=0.0,
    change_pct=0.0,
    frequency="Monthly",
    units="Percent",
    trend="STABLE",
)
assert snap.latest_value == 5.33
assert snap.trend == "STABLE"
print("[OK] MacroSnapshot dataclass created")

# --- RevisionRecord dataclass ---
rev = RevisionRecord(
    series_id="GDPC1",
    reference_period="2024-Q1",
    first_release=2.5,
    latest_value=2.8,
    revision=0.3,
    revision_pct=12.0,
    is_large=False,
)
assert rev.revision == 0.3
assert rev.series_id == "GDPC1"
print("[OK] RevisionRecord dataclass created")

# --- FREDAPIClient instantiation (no network calls) ---
client = FREDAPIClient(api_key=None)
assert hasattr(client, "fetch_series")
assert hasattr(client, "_cache")
assert client._api_key is None or isinstance(client._api_key, str)
print("[OK] FREDAPIClient instantiates, has fetch_series and cache")

# --- MacroSeriesLibrary class structure ---
lib = MacroSeriesLibrary()
assert hasattr(lib, "get_all_series")
all_series = lib.get_all_series()
assert isinstance(all_series, dict)
assert "GDPC1" in all_series or "UNRATE" in all_series
print(f"[OK] MacroSeriesLibrary.get_all_series() returns {len(all_series)} series")

# --- FREDMacroEngine class structure ---
engine = FREDMacroEngine(api_key=None)
assert hasattr(engine, "_client")
assert hasattr(engine, "_library")
assert hasattr(engine, "get_full_dashboard")
print("[OK] FREDMacroEngine instantiates with _client, _library, get_full_dashboard")

print("\n[PASS] dim_043: fred_macro_v3 -- all checks passed")
PYEOF
