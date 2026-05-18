#!/bin/bash
# dim_049: macro_cross_country — cross-country macro comparison engine
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sma.macro_cross_country import (
    WB_BASE,
    IMF_BASE,
    WB_INDICATORS,
    IMF_INDICATORS,
    MAJOR_ECONOMIES,
    _WB_CODE_MAP,
    WorldBankMacroAdapter,
    IMFDataAdapter,
    CountryMacroProfile,
    MacroCrossCountryEngine,
)

# --- constants ---
assert "worldbank.org" in WB_BASE
assert "imf.org" in IMF_BASE
print("[OK] WB_BASE and IMF_BASE constants present")

# --- WB_INDICATORS ---
assert "NY.GDP.MKTP.KD.ZG" in WB_INDICATORS
assert "FP.CPI.TOTL.ZG" in WB_INDICATORS
assert len(WB_INDICATORS) >= 5
print(f"[OK] WB_INDICATORS has {len(WB_INDICATORS)} indicators")

# --- IMF_INDICATORS ---
assert "NGDP_RPCH" in IMF_INDICATORS    # real GDP growth
assert "PCPIPCH" in IMF_INDICATORS      # inflation
assert len(IMF_INDICATORS) >= 4
print(f"[OK] IMF_INDICATORS has {len(IMF_INDICATORS)} indicators")

# --- MAJOR_ECONOMIES ---
assert "US" in MAJOR_ECONOMIES
assert "DE" in MAJOR_ECONOMIES
assert "JP" in MAJOR_ECONOMIES
assert "CN" in MAJOR_ECONOMIES
assert len(MAJOR_ECONOMIES) >= 20
print(f"[OK] MAJOR_ECONOMIES covers {len(MAJOR_ECONOMIES)} countries")

# --- _WB_CODE_MAP ---
assert _WB_CODE_MAP["GB"] == "GBR"
assert _WB_CODE_MAP["DE"] == "DEU"
assert _WB_CODE_MAP["US"] == "USA"
print("[OK] _WB_CODE_MAP ISO-2 to WB-3 code mapping correct")

# --- WorldBankMacroAdapter class structure ---
wb = WorldBankMacroAdapter()
assert hasattr(wb, "get_indicator")
assert hasattr(wb, "_wb_code")
assert wb._wb_code("GB") == "GBR"
assert wb._wb_code("US") == "USA"
print("[OK] WorldBankMacroAdapter instantiates, _wb_code works")

# --- IMFDataAdapter class structure ---
imf = IMFDataAdapter()
assert hasattr(imf, "get_imf_indicator")
print("[OK] IMFDataAdapter has get_imf_indicator method")

# --- CountryMacroProfile dataclass ---
profile_us = CountryMacroProfile(
    country="US",
    gdp_growth=2.5,
    inflation=3.2,
    unemployment=3.7,
    fiscal_balance=-6.5,
    current_account=-3.0,
    debt_pct_gdp=122.0,
    gdp_per_capita_ppp=65000.0,
    cycle_phase="expansion",
)
assert profile_us.country == "US"
assert profile_us.gdp_growth == 2.5
print("[OK] CountryMacroProfile dataclass created")

profile_de = CountryMacroProfile(
    country="DE",
    gdp_growth=1.5,
    inflation=2.1,
    unemployment=3.0,
    fiscal_balance=-2.5,
    current_account=5.0,
)
assert profile_de.current_account == 5.0
print("[OK] Two country profiles created successfully")

# --- MacroCrossCountryEngine class structure ---
engine = MacroCrossCountryEngine()
assert hasattr(engine, "wb")
assert hasattr(engine, "imf")
assert isinstance(engine.wb, WorldBankMacroAdapter)
assert isinstance(engine.imf, IMFDataAdapter)
print("[OK] MacroCrossCountryEngine instantiates with wb and imf adapters")

print("\n[PASS] dim_049: macro_cross_country -- all checks passed")
PYEOF
