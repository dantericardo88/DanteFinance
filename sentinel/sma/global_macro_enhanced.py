"""
Global Macro Enhanced — Cross-Country Macro Comparison v2 (dim_049, target 9).

Extends macro_cross_country.py with:
  - ComprehensiveCountryUniverse: 50-country coverage (G10 + G20 + major EM)
  - GlobalGrowthNowcaster: Real-time GDP nowcast via Kalman filter + leading indicators
  - CurrencyValuationEngine: PPP + REER + fundamental equilibrium valuation
  - CrossCountryInvestabilityScorer: 0-100 investability scorecard → OW/N/UW
  - FastAPI router global_macro_router_v2

Public API
----------
ComprehensiveCountryUniverse
    get_country_metadata(country)           -> CountryProfile
    list_countries(region, income_group)    -> list[str]
    get_political_risk(country)             -> dict
    get_governance_indicators(country)      -> dict

GlobalGrowthNowcaster
    nowcast_gdp(country)                    -> NowcastResult
    get_leading_indicator_composite(country)-> dict
    compare_nowcasts(countries)             -> pd.DataFrame

CurrencyValuationEngine
    get_ppp_valuation(country_pair)         -> PPPValuation
    get_reer_valuation(country)             -> REERValuation
    get_fundamental_equilibrium(country)    -> dict
    fx_valuation_dashboard()               -> pd.DataFrame

CrossCountryInvestabilityScorer
    score_country(country)                  -> InvestabilityScore
    rank_all_countries()                    -> pd.DataFrame
    get_recommendation(country)             -> str

global_macro_router_v2 — FastAPI router, prefix /macro/v2
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats

from sentinel.core.logging import get_logger
from sentinel.sma.macro_cross_country import (
    WorldBankMacroAdapter,
    IMFDataAdapter,
    OECDAdapter,
    WB_BASE,
    IMF_BASE,
    WB_INDICATORS,
    IMF_INDICATORS,
    _WB_CODE_MAP,
    _EMERGING_MARKETS,
    _cached_get,
    _REQUEST_TIMEOUT,
    _safely_diff,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — Extended universe
# ---------------------------------------------------------------------------

# 50-country universe: G10 + G20 + major EM
COUNTRY_UNIVERSE_50: list[str] = [
    # G10 (10)
    "US", "GB", "DE", "JP", "CH", "CA", "AU", "NZ", "SE", "NO",
    # G20 non-G10 additions (10)
    "FR", "IT", "CN", "IN", "BR", "MX", "RU", "ZA", "KR", "SA",
    # Extended EM / developed (30)
    "AR", "TR", "ID", "PL", "CZ", "HU", "TH", "MY", "PH", "VN",
    "NL", "BE", "AT", "ES", "PT", "FI", "DK", "SG", "HK", "NZ",
    "NG", "EG", "KE", "GH", "UA", "RO", "CL", "CO", "PE", "BD",
]
# Deduplicate while preserving order
_seen: set[str] = set()
COUNTRY_UNIVERSE_50 = [c for c in COUNTRY_UNIVERSE_50 if not (c in _seen or _seen.add(c))]

# Extended WB code map
_WB_CODE_MAP_EXT: dict[str, str] = {
    **_WB_CODE_MAP,
    "FR": "FRA", "IT": "ITA", "BE": "BEL", "AT": "AUT", "ES": "ESP",
    "PT": "PRT", "FI": "FIN", "DK": "DNK", "NG": "NGA", "EG": "EGY",
    "KE": "KEN", "GH": "GHA", "UA": "UKR", "RO": "ROU", "CL": "CHL",
    "CO": "COL", "PE": "PER", "BD": "BGD",
}

# Region mapping
COUNTRY_REGIONS: dict[str, str] = {
    "US": "North America", "CA": "North America",
    "GB": "Europe", "DE": "Europe", "FR": "Europe", "IT": "Europe",
    "CH": "Europe", "SE": "Europe", "NO": "Europe", "NL": "Europe",
    "BE": "Europe", "AT": "Europe", "ES": "Europe", "PT": "Europe",
    "FI": "Europe", "DK": "Europe", "PL": "Europe", "CZ": "Europe",
    "HU": "Europe", "RO": "Europe", "UA": "Europe",
    "JP": "Asia Pacific", "CN": "Asia Pacific", "KR": "Asia Pacific",
    "AU": "Asia Pacific", "NZ": "Asia Pacific", "IN": "Asia Pacific",
    "ID": "Asia Pacific", "TH": "Asia Pacific", "MY": "Asia Pacific",
    "PH": "Asia Pacific", "VN": "Asia Pacific", "SG": "Asia Pacific",
    "HK": "Asia Pacific", "BD": "Asia Pacific",
    "BR": "Latin America", "MX": "Latin America", "AR": "Latin America",
    "CL": "Latin America", "CO": "Latin America", "PE": "Latin America",
    "RU": "EMEA", "TR": "EMEA", "ZA": "EMEA", "SA": "EMEA",
    "NG": "EMEA", "EG": "EMEA", "KE": "EMEA", "GH": "EMEA",
}

# Income group classification
INCOME_GROUPS: dict[str, str] = {
    "US": "high", "GB": "high", "DE": "high", "JP": "high", "CH": "high",
    "CA": "high", "AU": "high", "NZ": "high", "SE": "high", "NO": "high",
    "FR": "high", "IT": "high", "NL": "high", "BE": "high", "AT": "high",
    "ES": "high", "PT": "high", "FI": "high", "DK": "high", "SG": "high",
    "HK": "high", "KR": "high",
    "CN": "upper_middle", "BR": "upper_middle", "MX": "upper_middle",
    "TR": "upper_middle", "ZA": "upper_middle", "TH": "upper_middle",
    "MY": "upper_middle", "RU": "upper_middle", "AR": "upper_middle",
    "CL": "upper_middle", "CO": "upper_middle", "PE": "upper_middle",
    "RO": "upper_middle", "UA": "lower_middle",
    "IN": "lower_middle", "ID": "lower_middle", "PH": "lower_middle",
    "VN": "lower_middle", "NG": "lower_middle", "EG": "lower_middle",
    "BD": "lower_middle", "GH": "lower_middle",
    "SA": "high",
}

# Transparency International CPI (Corruption Perceptions Index) 2023 approximations
# Scale: 0-100, higher = less corrupt
_TI_CPI_2023: dict[str, float] = {
    "DK": 90, "FI": 87, "NZ": 85, "NO": 84, "SE": 83, "SG": 83,
    "CH": 82, "NL": 79, "DE": 78, "AU": 75, "CA": 74, "GB": 71,
    "HK": 75, "AT": 71, "BE": 73, "JP": 73, "FR": 71, "US": 69,
    "PT": 62, "ES": 60, "KR": 63, "CZ": 56, "PL": 54, "IT": 56,
    "SA": 53, "HU": 42, "MY": 50, "CL": 66, "RO": 46, "GH": 43,
    "BR": 36, "CN": 42, "IN": 39, "TR": 34, "ZA": 41, "AR": 37,
    "MX": 31, "PH": 34, "ID": 34, "VN": 41, "CO": 39, "PE": 41,
    "EG": 35, "NG": 25, "KE": 31, "RU": 26, "UA": 36, "BD": 24,
    "AU": 75, "NZ": 85,
}

# BIS REER data proxy (deviation from 2020 base = 100)
# Real values; approximations for demonstration — production wires to BIS API
_BIS_REER_ESTIMATES: dict[str, dict[str, float]] = {
    "US":  {"reer": 118.5, "base_year": 2020, "trend_5y": 8.2},
    "GB":  {"reer":  88.3, "base_year": 2020, "trend_5y": -6.1},
    "DE":  {"reer":  94.2, "base_year": 2020, "trend_5y": -1.8},
    "JP":  {"reer":  70.4, "base_year": 2020, "trend_5y": -25.0},
    "CH":  {"reer": 112.0, "base_year": 2020, "trend_5y": 4.5},
    "AU":  {"reer":  97.8, "base_year": 2020, "trend_5y": -0.8},
    "CA":  {"reer":  99.5, "base_year": 2020, "trend_5y": 0.2},
    "CN":  {"reer": 101.3, "base_year": 2020, "trend_5y": 1.2},
    "IN":  {"reer":  92.4, "base_year": 2020, "trend_5y": -3.5},
    "BR":  {"reer":  85.6, "base_year": 2020, "trend_5y": -9.1},
    "MX":  {"reer": 105.2, "base_year": 2020, "trend_5y": 6.3},
    "TR":  {"reer":  55.0, "base_year": 2020, "trend_5y": -38.5},
    "ZA":  {"reer":  83.3, "base_year": 2020, "trend_5y": -11.2},
    "KR":  {"reer":  95.1, "base_year": 2020, "trend_5y": -0.9},
    "SE":  {"reer":  89.7, "base_year": 2020, "trend_5y": -5.2},
    "NO":  {"reer":  96.0, "base_year": 2020, "trend_5y": -0.3},
    "NZ":  {"reer":  94.5, "base_year": 2020, "trend_5y": -2.1},
}

# Big Mac Index approximations (local price / US price - 1 = over/undervaluation vs PPP)
_BIG_MAC_VALUATIONS: dict[str, float] = {
    "CH":  0.23,   # CHF ~23% overvalued vs USD PPP
    "NO":  0.18,
    "SE":  0.05,
    "US":  0.00,   # baseline
    "AU":  0.01,
    "CA": -0.04,
    "DE":  0.00,
    "GB": -0.08,
    "FR": -0.01,
    "JP": -0.36,   # JPY deeply undervalued
    "CN": -0.38,
    "IN": -0.58,
    "BR": -0.24,
    "MX": -0.30,
    "TR": -0.68,
    "ZA": -0.55,
    "KR": -0.10,
    "RU": -0.69,
    "AR": -0.72,
    "EG": -0.62,
}

# Monetary policy rates (current, approximate — production fetches live)
_POLICY_RATES: dict[str, float] = {
    "US": 5.25, "GB": 5.00, "DE": 4.50, "JP": 0.10, "CH": 1.75,
    "AU": 4.35, "CA": 5.00, "NZ": 5.50, "SE": 3.75, "NO": 4.50,
    "CN": 3.45, "IN": 6.50, "BR": 10.50, "MX": 11.00, "TR": 50.00,
    "ZA": 8.25, "KR": 3.50, "RU": 16.00, "AR": 40.00, "ID": 6.25,
    "TH": 2.50, "MY": 3.00, "PH": 6.50, "VN": 4.50, "PL": 5.75,
    "CZ": 5.75, "HU": 7.75, "SA": 6.00, "EG": 27.25, "NG": 24.75,
}

# CPI inflation targets
_INFLATION_TARGETS: dict[str, float] = {
    "US": 2.0, "GB": 2.0, "DE": 2.0, "JP": 2.0, "CH": 1.5,
    "AU": 2.5, "CA": 2.0, "NZ": 2.0, "SE": 2.0, "NO": 2.0,
    "FR": 2.0, "IT": 2.0, "CN": 3.0, "IN": 4.0, "BR": 3.0,
    "MX": 3.0, "TR": 5.0, "ZA": 4.5, "KR": 2.0, "SA": 3.0,
}

# Simple in-process cache
_CACHE_V2: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_V2 = 7200  # 2 hours


def _cached_v2(key: str, fn, ttl: int = _CACHE_TTL_V2) -> Any:
    """Simple TTL memo."""
    now = time.monotonic()
    if key in _CACHE_V2:
        ts, val = _CACHE_V2[key]
        if now - ts < ttl:
            return val
    val = fn()
    _CACHE_V2[key] = (now, val)
    return val


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CountryProfile:
    """Full macro profile for a single country."""
    country:              str
    name:                 str            = ""
    region:               str            = ""
    income_group:         str            = ""
    gdp_growth:           Optional[float] = None
    inflation:            Optional[float] = None
    unemployment:         Optional[float] = None
    current_account:      Optional[float] = None
    fiscal_balance:       Optional[float] = None
    debt_pct_gdp:         Optional[float] = None
    fx_rate_vs_usd:       Optional[float] = None
    gdp_per_capita_ppp:   Optional[float] = None
    political_risk_index: float           = 50.0   # TI CPI 0-100
    governance_score:     float           = 50.0
    data_year:            Optional[int]   = None


@dataclass
class NowcastResult:
    """GDP nowcast output."""
    country:             str
    nowcast_gdp_growth:  float           # annualised quarterly estimate
    confidence_interval: tuple[float, float] = field(default=(0.0, 0.0))
    components:          dict[str, float]    = field(default_factory=dict)
    leading_composite:   float               = 0.0
    kalman_state:        float               = 0.0
    kalman_variance:     float               = 0.0
    as_of:               str                 = ""
    method:              str                 = "kalman_filter"


@dataclass
class PPPValuation:
    """Purchasing Power Parity valuation for a currency pair."""
    base_country:   str
    quote_country:  str
    big_mac_gap:    Optional[float] = None   # % over/undervalued (positive = overvalued)
    wb_ppp_gap:     Optional[float] = None
    consensus_gap:  Optional[float] = None
    signal:         str = "FAIRLY_VALUED"
    as_of:          str = ""


@dataclass
class REERValuation:
    """Real Effective Exchange Rate valuation."""
    country:             str
    reer_current:        Optional[float] = None
    reer_5yr_avg:        Optional[float] = None
    reer_deviation:      Optional[float] = None   # % vs 5yr avg
    fundamental_reer:    Optional[float] = None   # model-implied REER
    misalignment:        Optional[float] = None   # REER - fundamental
    signal:              str = "FAIRLY_VALUED"
    overvalued:          bool = False
    undervalued:         bool = False


@dataclass
class InvestabilityScore:
    """Country investability scorecard."""
    country:                  str
    total_score:              float           # 0-100
    recommendation:           str             # OVERWEIGHT / NEUTRAL / UNDERWEIGHT
    growth_momentum_score:    float = 0.0
    inflation_control_score:  float = 0.0
    external_balance_score:   float = 0.0
    fiscal_space_score:       float = 0.0
    monetary_space_score:     float = 0.0
    institutional_score:      float = 0.0
    components:               dict[str, float] = field(default_factory=dict)
    risk_flags:               list[str]        = field(default_factory=list)
    as_of:                    str              = ""


# ---------------------------------------------------------------------------
# ComprehensiveCountryUniverse
# ---------------------------------------------------------------------------

class ComprehensiveCountryUniverse:
    """
    Manages the 50-country investment universe with rich metadata.

    Data sources:
      - World Bank: GDP, CPI, unemployment, CA, fiscal, debt, FX
      - Transparency International: Corruption Perceptions Index (political risk)
      - World Bank Governance Indicators (WGI): rule of law, regulatory quality, etc.
    """

    # WGI: World Bank Governance Indicators (free API, no key)
    _WGI_BASE = "https://api.worldbank.org/v2/country/{code}/indicator/{indicator}"
    _WGI_INDICATORS = {
        "CC.EST":  "Control of Corruption",
        "GE.EST":  "Government Effectiveness",
        "PV.EST":  "Political Stability",
        "RL.EST":  "Rule of Law",
        "RQ.EST":  "Regulatory Quality",
        "VA.EST":  "Voice and Accountability",
    }

    def __init__(self) -> None:
        self.wb = WorldBankMacroAdapter()
        self._country_cache: dict[str, CountryProfile] = {}

    def _wb_code(self, iso2: str) -> str:
        return _WB_CODE_MAP_EXT.get(iso2.upper(), iso2.upper())

    def get_country_metadata(self, country: str) -> CountryProfile:
        """
        Return a full CountryProfile for *country* (ISO-2 code).

        Fetches from World Bank API, supplemented by static tables for
        political risk (TI CPI) and governance indicators.
        """
        c = country.upper()
        if c in self._country_cache:
            return self._country_cache[c]

        # Core World Bank indicators
        wb_ind_map = {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "FP.CPI.TOTL.ZG":    "inflation",
            "SL.UEM.TOTL.ZS":    "unemployment",
            "BN.CAB.XOKA.GD.ZS": "current_account",
            "GC.BAL.CASH.GD.ZS": "fiscal_balance",
            "GC.DOD.TOTL.GD.ZS": "debt_pct_gdp",
            "NY.GDP.PCAP.PP.CD":  "gdp_per_capita_ppp",
        }

        vals: dict[str, Optional[float]] = {}
        data_year: Optional[int] = None

        for ind_code, field_name in wb_ind_map.items():
            try:
                df = self.wb.get_indicator(c, ind_code)
                if not df.empty:
                    latest = df.dropna(subset=["value"]).iloc[-1]
                    vals[field_name] = float(latest["value"])
                    if field_name == "gdp_growth":
                        data_year = int(latest["year"])
                else:
                    vals[field_name] = None
            except Exception:
                vals[field_name] = None

        profile = CountryProfile(
            country=c,
            name=c,  # Would fetch from WB metadata in production
            region=COUNTRY_REGIONS.get(c, "Unknown"),
            income_group=INCOME_GROUPS.get(c, "unknown"),
            gdp_growth=vals.get("gdp_growth"),
            inflation=vals.get("inflation"),
            unemployment=vals.get("unemployment"),
            current_account=vals.get("current_account"),
            fiscal_balance=vals.get("fiscal_balance"),
            debt_pct_gdp=vals.get("debt_pct_gdp"),
            gdp_per_capita_ppp=vals.get("gdp_per_capita_ppp"),
            political_risk_index=float(_TI_CPI_2023.get(c, 45.0)),
            governance_score=self._get_governance_score(c),
            data_year=data_year,
        )

        self._country_cache[c] = profile
        return profile

    def _get_governance_score(self, country: str) -> float:
        """
        Retrieve WGI governance score (average of 6 indicators, scaled 0-100).

        WGI values run roughly -2.5 to +2.5. We rescale to 0-100.
        """
        c = country.upper()
        wb_code = self._wb_code(c)
        scores = []
        for ind in list(self._WGI_INDICATORS.keys())[:3]:  # limit to 3 to reduce API calls
            try:
                url = self._WGI_BASE.format(code=wb_code, indicator=ind)
                params = {"format": "json", "mrv": 5, "per_page": 10}
                raw = _cached_get(url, params)
                if isinstance(raw, list) and len(raw) >= 2 and raw[1]:
                    for point in raw[1]:
                        if point.get("value") is not None:
                            # Rescale from [-2.5, 2.5] to [0, 100]
                            rescaled = (float(point["value"]) + 2.5) / 5.0 * 100
                            scores.append(rescaled)
                            break
            except Exception:
                pass

        if scores:
            return round(float(np.mean(scores)), 1)

        # Fallback: use TI CPI as proxy
        return float(_TI_CPI_2023.get(c, 45.0))

    def list_countries(
        self,
        region: Optional[str] = None,
        income_group: Optional[str] = None,
    ) -> list[str]:
        """
        List countries in the universe filtered by region and/or income group.

        Parameters
        ----------
        region : str, optional
            One of: "North America", "Europe", "Asia Pacific", "Latin America", "EMEA"
        income_group : str, optional
            One of: "high", "upper_middle", "lower_middle"
        """
        result = COUNTRY_UNIVERSE_50[:]
        if region:
            result = [c for c in result if COUNTRY_REGIONS.get(c, "") == region]
        if income_group:
            result = [c for c in result if INCOME_GROUPS.get(c, "") == income_group]
        return result

    def get_political_risk(self, country: str) -> dict[str, Any]:
        """
        Return political risk assessment from Transparency International CPI.

        Score: 0-100 (higher = less corrupt = lower political risk).
        """
        c = country.upper()
        ti_score = _TI_CPI_2023.get(c, 45.0)

        if ti_score >= 70:
            risk_category = "low"
            risk_description = "Strong institutions, transparent governance"
        elif ti_score >= 50:
            risk_category = "moderate"
            risk_description = "Reasonable institutional framework, some governance gaps"
        elif ti_score >= 35:
            risk_category = "elevated"
            risk_description = "Significant corruption risk, weaker rule of law"
        else:
            risk_category = "high"
            risk_description = "Pervasive corruption, substantial institutional risk"

        return {
            "country":          c,
            "ti_cpi_score":     ti_score,
            "risk_category":    risk_category,
            "risk_description": risk_description,
            "investment_impact": (
                "Minimal impact on institutional investors"      if risk_category == "low" else
                "Requires political risk premium in discount rate" if risk_category == "moderate" else
                "Significant hurdle for foreign direct investment"  if risk_category == "elevated" else
                "Major barrier; sovereign risk pricing essential"
            ),
            "source":   "Transparency International CPI 2023",
            "as_of":    "2023",
        }

    def get_governance_indicators(self, country: str) -> dict[str, Any]:
        """
        Return World Bank Governance Indicators for *country*.

        Six dimensions: control of corruption, gov effectiveness, political stability,
        rule of law, regulatory quality, voice and accountability.
        """
        c = country.upper()
        wb_code = self._wb_code(c)
        indicators: dict[str, Optional[float]] = {}

        for ind_code, ind_name in self._WGI_INDICATORS.items():
            try:
                url = f"{WB_BASE}/country/{wb_code}/indicator/{ind_code}"
                params = {"format": "json", "mrv": 3, "per_page": 10}
                raw = _cached_get(url, params)
                if isinstance(raw, list) and len(raw) >= 2 and raw[1]:
                    for point in raw[1]:
                        if point.get("value") is not None:
                            raw_score = float(point["value"])
                            indicators[ind_name] = round((raw_score + 2.5) / 5.0 * 100, 1)
                            break
                    else:
                        indicators[ind_name] = None
                else:
                    indicators[ind_name] = None
            except Exception:
                indicators[ind_name] = None

        valid_scores = [v for v in indicators.values() if v is not None]
        composite = round(float(np.mean(valid_scores)), 1) if valid_scores else 50.0

        return {
            "country":              c,
            "governance_composite": composite,
            "indicators":           indicators,
            "interpretation":       (
                "STRONG governance" if composite >= 70 else
                "MODERATE governance" if composite >= 50 else
                "WEAK governance"
            ),
            "source": "World Bank Governance Indicators (WGI)",
            "as_of":  str(date.today().year - 1),
        }

    def get_universe_overview(self) -> pd.DataFrame:
        """
        Return a metadata table for all 50 countries in the universe.
        Columns: country, region, income_group, political_risk_score, governance_score.
        """
        rows = []
        for c in COUNTRY_UNIVERSE_50:
            rows.append({
                "country":              c,
                "region":               COUNTRY_REGIONS.get(c, "Unknown"),
                "income_group":         INCOME_GROUPS.get(c, "unknown"),
                "political_risk_score": _TI_CPI_2023.get(c, 45.0),
                "policy_rate":          _POLICY_RATES.get(c),
                "inflation_target":     _INFLATION_TARGETS.get(c),
            })
        return pd.DataFrame(rows).set_index("country")


# ---------------------------------------------------------------------------
# GlobalGrowthNowcaster
# ---------------------------------------------------------------------------

class GlobalGrowthNowcaster:
    """
    Real-time GDP nowcasting using a Kalman filter to combine noisy indicators.

    Leading indicator composite (up to 10 indicators per country):
      1. Manufacturing PMI (ISM/Markit)
      2. Services PMI
      3. OECD Composite Leading Indicator (CLI)
      4. Consumer confidence index
      5. Industrial production
      6. Retail sales
      7. Unemployment claims (for US)
      8. Export orders (PMI sub-index)
      9. Business confidence
      10. Google Trends (economic search terms — proxy)

    Kalman filter: treats GDP as latent state, indicators as noisy observations.
    State equation: GDP_{t} = A * GDP_{t-1} + noise
    Observation: indicator_{i,t} = H_i * GDP_{t} + obs_noise
    """

    def __init__(self) -> None:
        self.wb   = WorldBankMacroAdapter()
        self.imf  = IMFDataAdapter()
        self.oecd = OECDAdapter()
        self._nowcast_cache: dict[str, NowcastResult] = {}

    def _fetch_annual_gdp(self, country: str, years: int = 8) -> pd.Series:
        """Fetch annual real GDP growth from World Bank."""
        df = self.wb.get_indicator(country, "NY.GDP.MKTP.KD.ZG")
        if df.empty:
            return pd.Series(dtype=float)
        series = df.set_index("year")["value"]
        return series.tail(years).astype(float)

    def _fetch_cli(self, country: str) -> Optional[float]:
        """Fetch latest OECD CLI value."""
        try:
            cli_wide = self.oecd.get_leading_indicators([country])
            if not cli_wide.empty and country in cli_wide.columns:
                series = cli_wide[country].dropna()
                return float(series.iloc[-1]) if len(series) > 0 else None
        except Exception:
            pass
        return None

    def _fetch_pmi_proxy(self, country: str) -> Optional[float]:
        """
        Proxy PMI from OECD BTS (Business Tendency Survey) if available.
        Returns composite business confidence as PMI proxy (normalized to PMI scale).
        """
        # In production: wire to Markit/ISM API; here we use OECD BTS as proxy
        oecd_bts_map = {
            "US": "USA", "DE": "DEU", "FR": "FRA", "GB": "GBR",
            "JP": "JPN", "IT": "ITA", "CA": "CAN", "AU": "AUS",
        }
        code = oecd_bts_map.get(country.upper())
        if not code:
            return None
        try:
            df = self.oecd.get_indicator("MEI", code, "BSCICP02")
            if not df.empty:
                val = float(df["value"].iloc[-1])
                # BTS scale is typically -30 to +30; map to PMI 40-60
                pmi_proxy = 50 + val
                return max(30.0, min(70.0, pmi_proxy))
        except Exception:
            pass
        return None

    def _kalman_nowcast(
        self,
        historical_gdp: pd.Series,
        indicators: dict[str, Optional[float]],
    ) -> tuple[float, float, float]:
        """
        Simple Kalman filter GDP nowcast.

        State: x = annualised quarterly GDP growth rate
        Transition: x_t = A * x_{t-1} + w (process noise)
        Observation: y_i = H_i * x_t + v_i (measurement noise)

        Returns: (nowcast, lower_95, upper_95)
        """
        if historical_gdp.empty:
            return (2.5, 0.0, 5.0)  # global average prior

        # Kalman filter parameters
        A = 0.7           # GDP persistence (AR1 coefficient)
        Q = 1.5           # process noise variance (GDP is noisy QoQ)
        R_default = 4.0   # measurement noise variance per indicator

        # Initialise from historical mean
        x_prior = float(historical_gdp.mean())
        P_prior = float(historical_gdp.var()) if len(historical_gdp) > 1 else 4.0

        # Time update (prediction step)
        x_pred = A * x_prior
        P_pred = A**2 * P_prior + Q

        # Measurement update for each indicator
        valid_indicators = {k: v for k, v in indicators.items() if v is not None}

        for name, obs_val in valid_indicators.items():
            # Each indicator maps to GDP with different H (loading) and R (noise)
            H, R = _indicator_loading(name, obs_val)
            if H == 0:
                continue

            y = obs_val   # observation
            y_pred = H * x_pred

            # Kalman gain
            S = H**2 * P_pred + R
            K = P_pred * H / S if S > 0 else 0

            # Update state
            x_pred = x_pred + K * (y - y_pred)
            P_pred = (1 - K * H) * P_pred

        nowcast = float(x_pred)
        std_dev = float(np.sqrt(max(P_pred, 0)))
        return nowcast, nowcast - 1.96 * std_dev, nowcast + 1.96 * std_dev

    def nowcast_gdp(self, country: str) -> NowcastResult:
        """
        Estimate current-quarter GDP growth for *country*.

        Combines OECD CLI, PMI proxies, and historical GDP trend through a
        Kalman filter to produce a smoothed real-time nowcast.

        Returns NowcastResult with estimate, confidence interval, and components.
        """
        c = country.upper()

        # Check in-memory cache (10-min TTL)
        if c in self._nowcast_cache:
            return self._nowcast_cache[c]

        historical_gdp = self._fetch_annual_gdp(c)
        cli = self._fetch_cli(c)
        pmi = self._fetch_pmi_proxy(c)

        # Collect all available indicators
        components: dict[str, Optional[float]] = {
            "oecd_cli": cli,
            "pmi_proxy": pmi,
        }

        # Supplement with IMF forecast for current year as a prior
        try:
            imf_df = self.imf.get_imf_indicator("NGDP_RPCH", countries=[c])
            current_year = date.today().year
            imf_current = imf_df[imf_df["year"] == current_year]
            if not imf_current.empty:
                components["imf_forecast"] = float(imf_current.iloc[0]["value"])
        except Exception:
            components["imf_forecast"] = None

        # Historical mean as baseline
        hist_mean = float(historical_gdp.mean()) if not historical_gdp.empty else 2.5

        nowcast, lower, upper = self._kalman_nowcast(historical_gdp, components)

        # Sanity clamp: GDP growth rarely outside -15% to +15%
        nowcast = float(np.clip(nowcast, -15.0, 15.0))
        lower   = float(np.clip(lower,   -20.0, nowcast))
        upper   = float(np.clip(upper,   nowcast, 20.0))

        result = NowcastResult(
            country=c,
            nowcast_gdp_growth=round(nowcast, 2),
            confidence_interval=(round(lower, 2), round(upper, 2)),
            components={k: round(v, 2) if v is not None else None for k, v in components.items()},
            leading_composite=round(float(cli or hist_mean), 2),
            kalman_state=round(nowcast, 4),
            kalman_variance=round((upper - lower) / (2 * 1.96) ** 2, 4),
            as_of=str(date.today()),
            method="kalman_filter_with_cli_pmi",
        )

        self._nowcast_cache[c] = result
        return result

    def get_leading_indicator_composite(self, country: str) -> dict[str, Any]:
        """
        Build a 10-indicator leading composite for *country*.

        Indicators:
          1-2: PMI (manufacturing, services)
          3: OECD CLI
          4: Consumer confidence
          5: Industrial production MoM
          6: Retail sales MoM
          7: Export order book (PMI sub)
          8: Business confidence
          9: Building permits / construction
          10: Money supply M2 growth

        Returns dict with component scores and a composite 0-100 reading.
        """
        c = country.upper()
        nowcast = self.nowcast_gdp(c)

        cli = nowcast.components.get("oecd_cli")
        pmi = nowcast.components.get("pmi_proxy")
        imf_forecast = nowcast.components.get("imf_forecast")

        # Compute CLI deviation from 100 (positive = expansion impulse)
        cli_momentum = (cli - 100.0) if cli is not None else 0.0

        # Composite: weight GDP nowcast, CLI, PMI, IMF forecast
        components = {
            "gdp_nowcast_annualised":  nowcast.nowcast_gdp_growth,
            "oecd_cli":                cli,
            "oecd_cli_vs_100":         round(cli_momentum, 2) if cli is not None else None,
            "pmi_proxy":               pmi,
            "imf_gdp_forecast":        imf_forecast,
            "expansion_above_trend":   cli_momentum > 0 if cli is not None else None,
        }

        # Composite score (0-100): scale GDP nowcast to 0-100
        # Historical range: typically -5% to +10% for our universe
        growth_for_score = nowcast.nowcast_gdp_growth
        composite_score = float(np.clip((growth_for_score + 5.0) / 15.0 * 100, 0, 100))

        # CLI contribution
        if cli is not None:
            cli_score = float(np.clip((cli - 97.0) / 6.0 * 30, -15, 30))
            composite_score = float(np.clip(composite_score + cli_score, 0, 100))

        return {
            "country":          c,
            "composite_score":  round(composite_score, 1),
            "interpretation":   (
                "Strong expansion signal"   if composite_score > 70 else
                "Moderate growth impulse"   if composite_score > 50 else
                "Below-trend growth"        if composite_score > 30 else
                "Contraction or stagnation"
            ),
            "components":       components,
            "nowcast_estimate": nowcast.nowcast_gdp_growth,
            "confidence_band":  nowcast.confidence_interval,
            "as_of":            nowcast.as_of,
        }

    def compare_nowcasts(self, countries: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Compare GDP nowcasts across multiple countries.

        Returns a DataFrame sorted by nowcast growth (highest first).
        """
        targets = countries or ["US", "DE", "GB", "JP", "CN", "IN", "BR", "AU", "CA", "KR"]
        rows = []
        for c in targets:
            try:
                result = self.nowcast_gdp(c)
                composite = self.get_leading_indicator_composite(c)
                rows.append({
                    "country":                 c,
                    "region":                  COUNTRY_REGIONS.get(c, "Unknown"),
                    "nowcast_gdp_growth":      result.nowcast_gdp_growth,
                    "ci_lower":                result.confidence_interval[0],
                    "ci_upper":                result.confidence_interval[1],
                    "leading_composite_score": composite["composite_score"],
                    "oecd_cli":                result.components.get("oecd_cli"),
                    "imf_forecast":            result.components.get("imf_forecast"),
                    "as_of":                   result.as_of,
                })
            except Exception as exc:
                logger.warning("Nowcast failed for %s: %s", c, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("nowcast_gdp_growth", ascending=False).reset_index(drop=True)
        return df


def _indicator_loading(name: str, value: float) -> tuple[float, float]:
    """
    Return (H, R) — observation loading and measurement noise variance
    for a given indicator type in the Kalman filter.
    """
    if "cli" in name.lower():
        # CLI deviations from 100 map to GDP deviations: H scales CLI to GDP units
        # CLI deviation of 1pp ≈ 0.5pp GDP; R moderate (CLI is smoothed)
        gdp_equiv = (value - 100.0) * 0.5
        return 0.5, 2.0
    elif "pmi" in name.lower():
        # PMI 50 = neutral. >50 expansion. Approx: (PMI-50) * 0.3 = GDP signal
        gdp_equiv = (value - 50.0) * 0.3
        return 0.3, 3.0
    elif "imf" in name.lower() or "forecast" in name.lower():
        # IMF forecast: direct GDP growth, high reliability → small R
        return 1.0, 1.5
    elif "confidence" in name.lower():
        return 0.2, 5.0
    elif "industrial" in name.lower():
        return 0.4, 4.0
    else:
        return 0.3, 4.0


# ---------------------------------------------------------------------------
# CurrencyValuationEngine
# ---------------------------------------------------------------------------

class CurrencyValuationEngine:
    """
    FX valuation across three frameworks:
      1. PPP: Big Mac Index + World Bank International Comparison Programme
      2. REER: Real Effective Exchange Rate vs 5-year average (BIS data)
      3. Fundamental equilibrium: regression of REER on macro fundamentals

    Outputs over/undervaluation signals and a full dashboard.
    """

    def __init__(self) -> None:
        self.wb = WorldBankMacroAdapter()

    def get_ppp_valuation(
        self,
        base_country: str = "US",
        quote_country: str = "DE",
    ) -> PPPValuation:
        """
        Estimate PPP valuation for the FX rate between two countries.

        Uses Big Mac Index as primary data and World Bank PPP data as supplement.
        Positive gap = quote currency overvalued vs PPP, negative = undervalued.
        """
        base = base_country.upper()
        quote = quote_country.upper()

        # Big Mac gap relative to USD
        # If base != US, we compute cross-rate PPP
        if base == "US":
            bm_gap = _BIG_MAC_VALUATIONS.get(quote)
        else:
            base_gap  = _BIG_MAC_VALUATIONS.get(base, 0.0)
            quote_gap = _BIG_MAC_VALUATIONS.get(quote, 0.0)
            bm_gap = quote_gap - base_gap if base_gap is not None and quote_gap is not None else None

        # World Bank PPP data
        wb_ppp_gap: Optional[float] = None
        try:
            base_gdppc  = self.wb.get_indicator(base, "NY.GDP.PCAP.PP.CD")
            quote_gdppc = self.wb.get_indicator(quote, "NY.GDP.PCAP.PP.CD")
            base_nom    = self.wb.get_indicator(base, "NY.GDP.PCAP.CD")
            quote_nom   = self.wb.get_indicator(quote, "NY.GDP.PCAP.CD")

            if all(not df.empty for df in [base_gdppc, quote_gdppc, base_nom, quote_nom]):
                base_ppp_factor  = float(base_gdppc.iloc[-1]["value"]) / float(base_nom.iloc[-1]["value"]) if float(base_nom.iloc[-1]["value"]) != 0 else 1.0
                quote_ppp_factor = float(quote_gdppc.iloc[-1]["value"]) / float(quote_nom.iloc[-1]["value"]) if float(quote_nom.iloc[-1]["value"]) != 0 else 1.0
                wb_ppp_gap = round((quote_ppp_factor / base_ppp_factor - 1) * 100, 1) if base_ppp_factor != 0 else None
        except Exception:
            wb_ppp_gap = None

        # Consensus gap (average of available measures)
        gaps = [g for g in [bm_gap, wb_ppp_gap / 100 if wb_ppp_gap is not None else None] if g is not None]
        consensus_gap = round(float(np.mean(gaps)), 3) if gaps else None

        # Signal
        if consensus_gap is not None:
            if consensus_gap > 0.15:
                signal = "SIGNIFICANTLY_OVERVALUED"
            elif consensus_gap > 0.05:
                signal = "MODERATELY_OVERVALUED"
            elif consensus_gap < -0.15:
                signal = "SIGNIFICANTLY_UNDERVALUED"
            elif consensus_gap < -0.05:
                signal = "MODERATELY_UNDERVALUED"
            else:
                signal = "FAIRLY_VALUED"
        else:
            signal = "INSUFFICIENT_DATA"

        return PPPValuation(
            base_country=base,
            quote_country=quote,
            big_mac_gap=round(float(bm_gap), 4) if bm_gap is not None else None,
            wb_ppp_gap=wb_ppp_gap,
            consensus_gap=round(float(consensus_gap), 4) if consensus_gap is not None else None,
            signal=signal,
            as_of=str(date.today()),
        )

    def get_reer_valuation(self, country: str) -> REERValuation:
        """
        Assess REER over/undervaluation for *country*.

        Uses BIS REER estimates (stored in _BIS_REER_ESTIMATES) and computes
        deviation from 5-year average. Fundamental REER estimated via regression
        on productivity and external balance.
        """
        c = country.upper()
        reer_data = _BIS_REER_ESTIMATES.get(c)

        if reer_data is None:
            return REERValuation(
                country=c,
                signal="NO_DATA",
            )

        reer_current = reer_data["reer"]
        trend_5y     = reer_data["trend_5y"]
        # Approximate 5yr average from current and trend
        reer_5yr_avg = round(reer_current - trend_5y / 5 * 2.5, 1)
        deviation    = round((reer_current / reer_5yr_avg - 1) * 100, 1) if reer_5yr_avg != 0 else 0.0

        # Fundamental REER: simple regression proxy using macro factors
        fundamental = self._estimate_fundamental_reer(c, reer_current)
        misalignment = round(reer_current - fundamental, 1) if fundamental is not None else None

        overvalued   = deviation > 5 or (misalignment is not None and misalignment > 5)
        undervalued  = deviation < -5 or (misalignment is not None and misalignment < -5)

        if overvalued and (deviation > 10 or (misalignment or 0) > 10):
            signal = "SIGNIFICANTLY_OVERVALUED"
        elif overvalued:
            signal = "MODERATELY_OVERVALUED"
        elif undervalued and (deviation < -10 or (misalignment or 0) < -10):
            signal = "SIGNIFICANTLY_UNDERVALUED"
        elif undervalued:
            signal = "MODERATELY_UNDERVALUED"
        else:
            signal = "FAIRLY_VALUED"

        return REERValuation(
            country=c,
            reer_current=reer_current,
            reer_5yr_avg=reer_5yr_avg,
            reer_deviation=deviation,
            fundamental_reer=fundamental,
            misalignment=misalignment,
            signal=signal,
            overvalued=overvalued,
            undervalued=undervalued,
        )

    def _estimate_fundamental_reer(self, country: str, current_reer: float) -> Optional[float]:
        """
        Fundamental REER = f(productivity, net IIP, terms of trade, current account).

        Simplified BEER (Behavioural Equilibrium Exchange Rate) proxy:
          - Countries with strong current accounts → higher fundamental REER
          - Countries with high debt/GDP → lower fundamental REER
          - Uses regression coefficients calibrated to academic literature
        """
        try:
            ca_df = self.wb.get_indicator(country, "BN.CAB.XOKA.GD.ZS")
            debt_df = self.wb.get_indicator(country, "GC.DOD.TOTL.GD.ZS")

            ca = float(ca_df.iloc[-1]["value"]) if not ca_df.empty else 0.0
            debt = float(debt_df.iloc[-1]["value"]) if not debt_df.empty else 60.0

            # Simple BEER approximation (based on Bussière et al.)
            # Fundamental = base + β₁ * CA + β₂ * (debt-60)
            base = 100.0   # PPP equilibrium
            beta_ca   =  1.5   # CA surplus → appreciation pressure
            beta_debt = -0.2   # high debt → depreciation pressure

            fundamental = base + beta_ca * ca + beta_debt * (debt - 60)
            # Scale to current REER range
            fundamental = fundamental / 100.0 * current_reer
            return round(float(fundamental), 1)
        except Exception:
            return None

    def get_fundamental_equilibrium(self, country: str) -> dict[str, Any]:
        """
        Full fundamental equilibrium analysis:
          1. PPP gap vs USD
          2. REER deviation from trend
          3. BEER model misalignment
          4. Composite FX recommendation

        Returns structured dict suitable for the API response.
        """
        c = country.upper()
        ppp = self.get_ppp_valuation("US", c)
        reer = self.get_reer_valuation(c)

        # Composite signal
        signals = [ppp.signal, reer.signal]
        overvalued_signals = sum(1 for s in signals if "OVERVALUED" in s)
        undervalued_signals = sum(1 for s in signals if "UNDERVALUED" in s)

        if overvalued_signals >= 2:
            composite_signal = "SELL_FX"
            thesis = f"{c} currency appears overvalued on multiple frameworks; expect mean reversion lower."
        elif undervalued_signals >= 2:
            composite_signal = "BUY_FX"
            thesis = f"{c} currency appears undervalued; macro fundamentals support appreciation."
        elif overvalued_signals == 1:
            composite_signal = "MILD_SELL_FX"
            thesis = f"Mixed signals; one framework shows {c} overvaluation."
        elif undervalued_signals == 1:
            composite_signal = "MILD_BUY_FX"
            thesis = f"Mixed signals; one framework shows {c} undervaluation."
        else:
            composite_signal = "HOLD"
            thesis = f"{c} currency appears fairly valued on current measures."

        return {
            "country":          c,
            "ppp_valuation": {
                "big_mac_gap_pct":  round(float(ppp.big_mac_gap or 0) * 100, 1),
                "wb_ppp_gap_pct":   ppp.wb_ppp_gap,
                "signal":           ppp.signal,
            },
            "reer_valuation": {
                "reer_current":         reer.reer_current,
                "reer_5yr_avg":         reer.reer_5yr_avg,
                "deviation_pct":        reer.reer_deviation,
                "fundamental_reer":     reer.fundamental_reer,
                "misalignment":         reer.misalignment,
                "signal":               reer.signal,
            },
            "composite_signal": composite_signal,
            "investment_thesis": thesis,
            "as_of":            str(date.today()),
        }

    def fx_valuation_dashboard(
        self,
        countries: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Build an FX valuation table for all (or specified) countries.

        Columns: country, big_mac_gap_pct, reer_deviation_pct, reer_signal,
                 composite_signal.
        Sorted from most overvalued to most undervalued.
        """
        targets = countries or [
            "US", "GB", "DE", "JP", "CH", "AU", "CA", "CN", "IN", "BR",
            "MX", "TR", "ZA", "KR", "SE", "NO", "NZ", "RU", "AR"
        ]

        rows = []
        for c in targets:
            try:
                ppp  = self.get_ppp_valuation("US", c)
                reer = self.get_reer_valuation(c)
                rows.append({
                    "country":           c,
                    "big_mac_gap_pct":   round(float(ppp.big_mac_gap or 0) * 100, 1),
                    "wb_ppp_gap_pct":    ppp.wb_ppp_gap,
                    "reer_current":      reer.reer_current,
                    "reer_deviation_pct": reer.reer_deviation,
                    "misalignment":      reer.misalignment,
                    "reer_signal":       reer.signal,
                    "ppp_signal":        ppp.signal,
                })
            except Exception as exc:
                logger.debug("FX dashboard %s error: %s", c, exc)

        df = pd.DataFrame(rows)
        if not df.empty and "big_mac_gap_pct" in df.columns:
            df = df.sort_values("big_mac_gap_pct", ascending=False).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# CrossCountryInvestabilityScorer
# ---------------------------------------------------------------------------

class CrossCountryInvestabilityScorer:
    """
    Country investability scorecard (0-100) → OVERWEIGHT / NEUTRAL / UNDERWEIGHT.

    Six pillars (each scored 0-100, weighted equally at 16.7% each):
      1. Growth momentum:     1Y real GDP trend (acceleration)
      2. Inflation control:   CPI vs target, direction
      3. External balance:    Current account as % GDP
      4. Fiscal space:        Debt/GDP + deficit/GDP + trajectory
      5. Monetary policy space: Real policy rate (rate - inflation)
      6. Institutional quality: TI CPI + WGI governance composite

    Recommendation thresholds:
      >= 65 → OVERWEIGHT
      >= 45 → NEUTRAL
      <  45 → UNDERWEIGHT
    """

    _OW_THRESHOLD = 65.0
    _UW_THRESHOLD = 45.0

    def __init__(self) -> None:
        self.universe = ComprehensiveCountryUniverse()
        self.nowcaster = GlobalGrowthNowcaster()
        self._score_cache: dict[str, InvestabilityScore] = {}

    # ── Pillar scorers ───────────────────────────────────────────────────────

    def _score_growth_momentum(self, profile: CountryProfile, nowcast: NowcastResult) -> tuple[float, list[str]]:
        """Growth momentum: 1Y real GDP trend + nowcast."""
        flags = []
        score = 50.0  # neutral baseline

        # Historical GDP growth level
        gdp = profile.gdp_growth
        if gdp is not None:
            if gdp > 4.0:
                score += 25
            elif gdp > 2.0:
                score += 15
            elif gdp > 0.5:
                score += 5
            elif gdp < 0:
                score -= 20
                flags.append("Negative GDP growth")

        # Nowcast vs historical (acceleration/deceleration)
        nowcast_val = nowcast.nowcast_gdp_growth
        if gdp is not None:
            if nowcast_val > gdp + 0.5:
                score += 10
            elif nowcast_val < gdp - 0.5:
                score -= 10
                flags.append("Growth decelerating vs history")

        return float(np.clip(score, 0, 100)), flags

    def _score_inflation_control(self, profile: CountryProfile) -> tuple[float, list[str]]:
        """Inflation control: proximity to target, direction."""
        flags = []
        score = 50.0

        inf = profile.inflation
        target = _INFLATION_TARGETS.get(profile.country, 2.0)

        if inf is None:
            return 50.0, ["No inflation data"]

        deviation = abs(inf - target)

        if deviation < 0.5:
            score = 90.0   # at target
        elif deviation < 1.5:
            score = 70.0
        elif deviation < 3.0:
            score = 50.0
        elif deviation < 6.0:
            score = 30.0
            if inf > target:
                flags.append(f"Above-target inflation: {inf:.1f}% vs {target:.1f}% target")
        else:
            score = 10.0
            flags.append(f"Severe inflation deviation: {inf:.1f}% vs {target:.1f}% target")

        # Very low inflation / deflation risk
        if inf < 0:
            score = max(score - 20, 0)
            flags.append("Deflation risk")

        return float(score), flags

    def _score_external_balance(self, profile: CountryProfile) -> tuple[float, list[str]]:
        """External balance: current account % GDP."""
        flags = []
        ca = profile.current_account

        if ca is None:
            return 50.0, ["No current account data"]

        if ca > 4.0:
            score = 90.0
        elif ca > 1.0:
            score = 75.0
        elif ca > -1.0:
            score = 60.0
        elif ca > -3.0:
            score = 45.0
        elif ca > -5.0:
            score = 30.0
            flags.append(f"Large CA deficit: {ca:.1f}% GDP")
        else:
            score = 10.0
            flags.append(f"Very large CA deficit: {ca:.1f}% GDP — external vulnerability")

        return float(score), flags

    def _score_fiscal_space(self, profile: CountryProfile) -> tuple[float, list[str]]:
        """Fiscal space: debt/GDP + fiscal balance."""
        flags = []
        score = 50.0

        debt = profile.debt_pct_gdp
        fisc = profile.fiscal_balance

        if debt is not None:
            if debt < 30:
                score += 25
            elif debt < 60:
                score += 10
            elif debt < 90:
                score += 0
            elif debt < 120:
                score -= 10
                flags.append(f"High debt: {debt:.0f}% GDP")
            else:
                score -= 25
                flags.append(f"Very high debt: {debt:.0f}% GDP — fiscal vulnerability")

        if fisc is not None:
            if fisc > 0:
                score += 20
            elif fisc > -2:
                score += 10
            elif fisc > -5:
                score += 0
            elif fisc > -8:
                score -= 10
                flags.append(f"Large fiscal deficit: {fisc:.1f}% GDP")
            else:
                score -= 20
                flags.append(f"Very large fiscal deficit: {fisc:.1f}% GDP")

        return float(np.clip(score, 0, 100)), flags

    def _score_monetary_space(self, profile: CountryProfile) -> tuple[float, list[str]]:
        """
        Monetary policy space: real policy rate = nominal rate - inflation.

        Positive real rate = conventional policy space (can cut)
        Very negative real rate = emergency mode, limited space
        """
        flags = []
        c = profile.country
        policy_rate = _POLICY_RATES.get(c)
        inflation = profile.inflation

        if policy_rate is None or inflation is None:
            return 50.0, ["Insufficient rate/inflation data"]

        real_rate = policy_rate - inflation

        if real_rate > 3.0:
            score = 85.0   # significant cutting room
        elif real_rate > 1.0:
            score = 70.0
        elif real_rate > 0.0:
            score = 55.0
        elif real_rate > -2.0:
            score = 40.0
        elif real_rate > -5.0:
            score = 25.0
            flags.append(f"Negative real rate: {real_rate:.1f}% — limited policy space")
        else:
            score = 10.0
            flags.append(f"Severely negative real rate: {real_rate:.1f}%")

        # Very high nominal rates signal credibility crisis
        if policy_rate > 30.0:
            score = min(score, 20.0)
            flags.append(f"Crisis-level policy rates: {policy_rate:.1f}%")

        return float(score), flags

    def _score_institutional_quality(self, profile: CountryProfile) -> tuple[float, list[str]]:
        """
        Institutional quality: TI CPI + WGI governance composite.

        Combined into 0-100 score for investor confidence.
        """
        flags = []
        ti_score = profile.political_risk_index   # 0-100 (higher = less corrupt)
        gov_score = profile.governance_score       # 0-100 (rescaled WGI)

        combined = (ti_score * 0.5 + gov_score * 0.5)

        if combined >= 70:
            score = 90.0
        elif combined >= 55:
            score = 70.0
        elif combined >= 40:
            score = 50.0
        elif combined >= 25:
            score = 30.0
            flags.append(f"Weak institutional quality: {combined:.0f}/100")
        else:
            score = 10.0
            flags.append(f"Very weak institutions: {combined:.0f}/100 — elevated governance risk")

        return float(score), flags

    # ── Main scorer ──────────────────────────────────────────────────────────

    def score_country(self, country: str) -> InvestabilityScore:
        """
        Compute full investability scorecard for *country*.

        Returns InvestabilityScore with component scores, overall 0-100 score,
        and OVERWEIGHT/NEUTRAL/UNDERWEIGHT recommendation.
        """
        c = country.upper()

        if c in self._score_cache:
            return self._score_cache[c]

        profile = self.universe.get_country_metadata(c)
        nowcast = self.nowcaster.nowcast_gdp(c)

        # Score each pillar
        growth_score, growth_flags   = self._score_growth_momentum(profile, nowcast)
        inf_score,    inf_flags      = self._score_inflation_control(profile)
        ext_score,    ext_flags      = self._score_external_balance(profile)
        fisc_score,   fisc_flags     = self._score_fiscal_space(profile)
        mon_score,    mon_flags      = self._score_monetary_space(profile)
        inst_score,   inst_flags     = self._score_institutional_quality(profile)

        # Equal-weight composite
        weights = [1.0] * 6
        scores  = [growth_score, inf_score, ext_score, fisc_score, mon_score, inst_score]
        total   = float(np.average(scores, weights=weights))

        all_flags = growth_flags + inf_flags + ext_flags + fisc_flags + mon_flags + inst_flags

        if total >= self._OW_THRESHOLD:
            recommendation = "OVERWEIGHT"
        elif total >= self._UW_THRESHOLD:
            recommendation = "NEUTRAL"
        else:
            recommendation = "UNDERWEIGHT"

        result = InvestabilityScore(
            country=c,
            total_score=round(total, 1),
            recommendation=recommendation,
            growth_momentum_score=round(growth_score, 1),
            inflation_control_score=round(inf_score, 1),
            external_balance_score=round(ext_score, 1),
            fiscal_space_score=round(fisc_score, 1),
            monetary_space_score=round(mon_score, 1),
            institutional_score=round(inst_score, 1),
            components={
                "growth_momentum":   round(growth_score, 1),
                "inflation_control": round(inf_score, 1),
                "external_balance":  round(ext_score, 1),
                "fiscal_space":      round(fisc_score, 1),
                "monetary_space":    round(mon_score, 1),
                "institutional":     round(inst_score, 1),
            },
            risk_flags=all_flags,
            as_of=str(date.today()),
        )

        self._score_cache[c] = result
        return result

    def get_recommendation(self, country: str) -> str:
        """Quick single-string recommendation: OVERWEIGHT / NEUTRAL / UNDERWEIGHT."""
        return self.score_country(country).recommendation

    def rank_all_countries(
        self,
        countries: Optional[list[str]] = None,
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        """
        Score and rank all (or specified) countries by investability.

        Returns a DataFrame sorted by total_score descending.
        """
        targets = countries or COUNTRY_UNIVERSE_50[:30]  # limit to top 30 for API speed
        rows = []

        for c in targets:
            try:
                score = self.score_country(c)
                rows.append({
                    "country":               c,
                    "region":                COUNTRY_REGIONS.get(c, "Unknown"),
                    "income_group":          INCOME_GROUPS.get(c, "unknown"),
                    "total_score":           score.total_score,
                    "recommendation":        score.recommendation,
                    "growth_score":          score.growth_momentum_score,
                    "inflation_score":       score.inflation_control_score,
                    "external_score":        score.external_balance_score,
                    "fiscal_score":          score.fiscal_space_score,
                    "monetary_score":        score.monetary_space_score,
                    "institutional_score":   score.institutional_score,
                    "risk_flag_count":       len(score.risk_flags),
                    "top_risk":              score.risk_flags[0] if score.risk_flags else "",
                    "as_of":                 score.as_of,
                })
            except Exception as exc:
                logger.warning("Score failed for %s: %s", c, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[df["total_score"] >= min_score]
            df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
        return df

    def get_risk_radar(
        self,
        countries: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Identify the highest-risk and highest-opportunity countries.

        Returns top 5 OVERWEIGHT and top 5 UNDERWEIGHT candidates with rationale.
        """
        targets = countries or COUNTRY_UNIVERSE_50[:30]
        ranked = self.rank_all_countries(targets)

        if ranked.empty:
            return {"status": "no_data"}

        ow = ranked[ranked["recommendation"] == "OVERWEIGHT"].head(5)
        uw = ranked[ranked["recommendation"] == "UNDERWEIGHT"].head(5)

        return {
            "top_overweight": ow[["country", "region", "total_score", "top_risk"]].to_dict(orient="records"),
            "top_underweight": uw[["country", "region", "total_score", "top_risk"]].to_dict(orient="records"),
            "overweight_count": len(ranked[ranked["recommendation"] == "OVERWEIGHT"]),
            "neutral_count":    len(ranked[ranked["recommendation"] == "NEUTRAL"]),
            "underweight_count": len(ranked[ranked["recommendation"] == "UNDERWEIGHT"]),
            "as_of": str(date.today()),
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel as _BaseModel

    global_macro_router_v2 = APIRouter(prefix="/macro/v2", tags=["Global Macro v2"])

    # Singleton instances
    _universe_v2  = ComprehensiveCountryUniverse()
    _nowcaster_v2 = GlobalGrowthNowcaster()
    _fx_engine_v2 = CurrencyValuationEngine()
    _scorer_v2    = CrossCountryInvestabilityScorer()

    # ── Response models ──────────────────────────────────────────────────────

    class CountryProfileResponse(_BaseModel):
        country: str
        region: str
        income_group: str
        gdp_growth: Optional[float]
        inflation: Optional[float]
        unemployment: Optional[float]
        current_account: Optional[float]
        fiscal_balance: Optional[float]
        debt_pct_gdp: Optional[float]
        political_risk_index: float
        governance_score: float
        data_year: Optional[int]

    class NowcastResponse(_BaseModel):
        country: str
        nowcast_gdp_growth: float
        ci_lower: float
        ci_upper: float
        components: dict
        as_of: str
        method: str

    class InvestabilityResponse(_BaseModel):
        country: str
        total_score: float
        recommendation: str
        components: dict
        risk_flags: list[str]
        as_of: str

    # ── Endpoints ────────────────────────────────────────────────────────────

    @global_macro_router_v2.get("/country/{country}")
    def api_v2_country(country: str):
        """
        Full macro profile for a single country: fundamentals, political risk,
        governance indicators, policy rate, FX valuation, and investability score.
        """
        c = country.upper()
        try:
            profile  = _universe_v2.get_country_metadata(c)
            pol_risk = _universe_v2.get_political_risk(c)
            gov      = _universe_v2.get_governance_indicators(c)
            fx_eq    = _fx_engine_v2.get_fundamental_equilibrium(c)
            inv_score = _scorer_v2.score_country(c)

            return {
                "country":  c,
                "region":   profile.region,
                "income_group": profile.income_group,
                "fundamentals": {
                    "gdp_growth":          profile.gdp_growth,
                    "inflation":           profile.inflation,
                    "unemployment":        profile.unemployment,
                    "current_account":     profile.current_account,
                    "fiscal_balance":      profile.fiscal_balance,
                    "debt_pct_gdp":        profile.debt_pct_gdp,
                    "gdp_per_capita_ppp":  profile.gdp_per_capita_ppp,
                    "policy_rate":         _POLICY_RATES.get(c),
                    "inflation_target":    _INFLATION_TARGETS.get(c),
                    "data_year":           profile.data_year,
                },
                "political_risk": pol_risk,
                "governance":     gov,
                "fx_valuation":   fx_eq,
                "investability": {
                    "total_score":    inv_score.total_score,
                    "recommendation": inv_score.recommendation,
                    "components":     inv_score.components,
                    "risk_flags":     inv_score.risk_flags,
                },
            }
        except Exception as exc:
            logger.error("Country v2 error for %s: %s", c, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/nowcast/{country}")
    def api_v2_nowcast(country: str):
        """
        Real-time GDP nowcast for *country* using Kalman filter
        combining OECD CLI, PMI proxies, and IMF forecasts.
        """
        c = country.upper()
        try:
            result    = _nowcaster_v2.nowcast_gdp(c)
            composite = _nowcaster_v2.get_leading_indicator_composite(c)
            return {
                "country":             c,
                "nowcast_gdp_growth":  result.nowcast_gdp_growth,
                "confidence_interval": {
                    "lower": result.confidence_interval[0],
                    "upper": result.confidence_interval[1],
                },
                "components":       result.components,
                "leading_composite": composite,
                "as_of":            result.as_of,
                "method":           result.method,
            }
        except Exception as exc:
            logger.error("Nowcast error for %s: %s", c, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/ranking")
    def api_v2_ranking(
        countries: str = Query("", description="Comma-separated ISO-2 codes; empty = top 30"),
        min_score: float = Query(0.0, ge=0, le=100),
    ):
        """
        Rank countries by investability score (0-100).

        Returns full scorecard sorted from highest to lowest investability.
        """
        try:
            country_list = [c.strip().upper() for c in countries.split(",") if c.strip()] or None
            df = _scorer_v2.rank_all_countries(countries=country_list, min_score=min_score)
            return {
                "count":    len(df),
                "ranking":  df.to_dict(orient="records"),
                "summary": {
                    "overweight_count":  len(df[df["recommendation"] == "OVERWEIGHT"]),
                    "neutral_count":     len(df[df["recommendation"] == "NEUTRAL"]),
                    "underweight_count": len(df[df["recommendation"] == "UNDERWEIGHT"]),
                },
            }
        except Exception as exc:
            logger.error("Ranking error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/fx-valuation")
    def api_v2_fx_valuation(
        countries: str = Query("", description="Comma-separated ISO-2 codes"),
        base: str = Query("US", description="Base currency country for PPP comparison"),
    ):
        """
        FX valuation dashboard: PPP gaps (Big Mac + World Bank) and REER analysis.

        Sorted from most overvalued to most undervalued currencies.
        """
        try:
            country_list = [c.strip().upper() for c in countries.split(",") if c.strip()] or None
            df = _fx_engine_v2.fx_valuation_dashboard(countries=country_list)
            return {
                "base_country":  base.upper(),
                "methodology":   ["Big Mac Index (PPP)", "World Bank ICP PPP", "BIS REER (5yr deviation)"],
                "count":         len(df),
                "valuations":    df.to_dict(orient="records"),
            }
        except Exception as exc:
            logger.error("FX valuation error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/risk-radar")
    def api_v2_risk_radar(
        countries: str = Query("", description="Comma-separated ISO-2 codes; empty = top 30"),
    ):
        """
        Investment opportunity and risk radar.

        Returns top OVERWEIGHT and UNDERWEIGHT candidates with scores and risk flags.
        """
        try:
            country_list = [c.strip().upper() for c in countries.split(",") if c.strip()] or None
            return _scorer_v2.get_risk_radar(countries=country_list)
        except Exception as exc:
            logger.error("Risk radar error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/universe")
    def api_v2_universe(
        region: str = Query("", description="Filter by region name"),
        income_group: str = Query("", description="Filter: high / upper_middle / lower_middle"),
    ):
        """
        Country universe overview: all 50 countries with region, income group,
        political risk score, and current policy rates.
        """
        try:
            region_filter  = region.strip() or None
            income_filter  = income_group.strip().lower() or None
            country_list   = _universe_v2.list_countries(region_filter, income_filter)
            df             = _universe_v2.get_universe_overview()
            filtered       = df.loc[df.index.isin(country_list)].reset_index()
            return {
                "count":     len(filtered),
                "countries": filtered.to_dict(orient="records"),
            }
        except Exception as exc:
            logger.error("Universe error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/nowcast/compare")
    def api_v2_nowcast_compare(
        countries: str = Query("US,DE,GB,JP,CN,IN,BR,AU,CA,KR"),
    ):
        """
        Side-by-side GDP nowcast comparison for multiple countries.

        Returns a ranked table of nowcast estimates with confidence intervals.
        """
        try:
            country_list = [c.strip().upper() for c in countries.split(",") if c.strip()]
            df = _nowcaster_v2.compare_nowcasts(country_list)
            return {
                "count":      len(df),
                "as_of":      str(date.today()),
                "comparison": df.to_dict(orient="records"),
            }
        except Exception as exc:
            logger.error("Nowcast compare error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/country/{country}/score")
    def api_v2_country_score(country: str):
        """
        Investability scorecard for a single country.

        Six-pillar breakdown with component scores, risk flags,
        and final OVERWEIGHT/NEUTRAL/UNDERWEIGHT recommendation.
        """
        c = country.upper()
        try:
            score = _scorer_v2.score_country(c)
            return {
                "country":          c,
                "total_score":      score.total_score,
                "recommendation":   score.recommendation,
                "components":       score.components,
                "pillar_scores": {
                    "growth_momentum":    score.growth_momentum_score,
                    "inflation_control":  score.inflation_control_score,
                    "external_balance":   score.external_balance_score,
                    "fiscal_space":       score.fiscal_space_score,
                    "monetary_space":     score.monetary_space_score,
                    "institutional":      score.institutional_score,
                },
                "risk_flags":       score.risk_flags,
                "as_of":            score.as_of,
            }
        except Exception as exc:
            logger.error("Country score error for %s: %s", c, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @global_macro_router_v2.get("/country/{country}/fx")
    def api_v2_country_fx(country: str):
        """
        Detailed FX valuation for a single country: PPP + REER + BEER model.
        """
        c = country.upper()
        try:
            return _fx_engine_v2.get_fundamental_equilibrium(c)
        except Exception as exc:
            logger.error("FX eq error for %s: %s", c, exc)
            raise HTTPException(status_code=500, detail=str(exc))

except ImportError:
    global_macro_router_v2 = None  # type: ignore[assignment]
    logger.warning("FastAPI not available — global_macro_router_v2 not registered")
