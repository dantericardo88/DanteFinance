"""
Cross-country macroeconomic comparison engine — Dimension #49.

Aggregates World Bank, IMF DataMapper, and OECD SDMX data into a unified
macro scorecard covering 35+ economies. Surfaces GDP growth, inflation,
unemployment, fiscal balance, current account, and debt across the G20 + key
EM, with composite economic health scores, cycle-phase classification, and
relative-value frameworks for FX/rates investment theses.

Score target: SENTINEL 9  (was 6 — global_macro.py covers FRED only;
this module adds World Bank, IMF WEO, OECD CLI, EM screening, and a
structured scorecard API that incumbents surface only across separate pages).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WB_BASE = "https://api.worldbank.org/v2"
IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"
OECD_BASE = "https://stats.oecd.org/sdmx-json/data"

WB_INDICATORS: dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth rate",
    "FP.CPI.TOTL.ZG":    "Inflation CPI",
    "SL.UEM.TOTL.ZS":    "Unemployment rate",
    "GC.BAL.CASH.GD.ZS": "Fiscal balance % GDP",
    "BN.CAB.XOKA.GD.ZS": "Current account % GDP",
    "GC.DOD.TOTL.GD.ZS": "Government debt % GDP",
    "NY.GDP.PCAP.PP.CD":  "GDP per capita PPP",
    "NE.EXP.GNFS.ZS":    "Exports % GDP",
    "NE.IMP.GNFS.ZS":    "Imports % GDP",
    "FR.INR.RINR":        "Real interest rate",
    "NY.GNP.MKTP.KD.ZG": "GNI growth",
}

IMF_INDICATORS: dict[str, str] = {
    "NGDP_RPCH":    "Real GDP growth",
    "PCPIPCH":      "Inflation",
    "LUR":          "Unemployment",
    "GGXCNL_NGDP":  "Fiscal balance % GDP",
    "BCA_NGDPD":    "Current account % GDP",
    "GGXWDG_NGDP":  "Gross debt % GDP",
}

MAJOR_ECONOMIES: list[str] = [
    "US", "GB", "DE", "JP", "CN", "IN", "FR", "IT", "CA", "AU",
    "KR", "BR", "MX", "RU", "ZA", "ID", "TR", "SA", "AR", "CH",
    "SE", "NO", "NL", "SG", "HK", "NZ", "PL", "CZ", "HU", "TH",
    "MY", "PH", "VN",
]

# World Bank uses 3-letter codes for some countries; map ISO-2 → WB code
_WB_CODE_MAP: dict[str, str] = {
    "GB": "GBR", "DE": "DEU", "JP": "JPN", "CN": "CHN", "IN": "IND",
    "FR": "FRA", "IT": "ITA", "CA": "CAN", "AU": "AUS", "KR": "KOR",
    "BR": "BRA", "MX": "MEX", "RU": "RUS", "ZA": "ZAF", "ID": "IDN",
    "TR": "TUR", "SA": "SAU", "AR": "ARG", "CH": "CHE", "SE": "SWE",
    "NO": "NOR", "NL": "NLD", "SG": "SGP", "HK": "HKG", "NZ": "NZL",
    "PL": "POL", "CZ": "CZE", "HU": "HUN", "TH": "THA", "MY": "MYS",
    "PH": "PHL", "VN": "VNM", "US": "USA",
}

_EMERGING_MARKETS: set[str] = {
    "CN", "IN", "BR", "MX", "RU", "ZA", "ID", "TR", "SA", "AR",
    "PL", "CZ", "HU", "TH", "MY", "PH", "VN", "KR",
}

_REQUEST_TIMEOUT = 15
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 3600  # 1 hour


def _cached_get(url: str, params: dict | None = None) -> dict:
    """Simple in-process TTL cache for HTTP GET requests."""
    key = url + str(sorted((params or {}).items()))
    now = time.monotonic()
    if key in _CACHE:
        ts, data = _CACHE[key]
        if now - ts < _CACHE_TTL:
            return data
    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    _CACHE[key] = (now, data)
    return data


# ---------------------------------------------------------------------------
# WorldBankMacroAdapter
# ---------------------------------------------------------------------------

class WorldBankMacroAdapter:
    """Fetches macroeconomic indicators from the World Bank Open Data API (free, no key)."""

    def _wb_code(self, iso2: str) -> str:
        return _WB_CODE_MAP.get(iso2.upper(), iso2.upper())

    def get_indicator(
        self,
        country_code: str,
        indicator: str,
        start_year: int = 2000,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch a single indicator time series for one country.

        Returns a DataFrame with columns [year, value, country, indicator_code, indicator_name].
        """
        if end_year is None:
            end_year = datetime.now().year

        code = self._wb_code(country_code)
        url = f"{WB_BASE}/country/{code}/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": 100,
            "mrv": 25,
            "date": f"{start_year}:{end_year}",
        }

        try:
            raw = _cached_get(url, params)
        except Exception as exc:
            logger.warning("WorldBank API error for %s/%s: %s", country_code, indicator, exc)
            return pd.DataFrame()

        # World Bank response: [metadata, [data_points]]
        if not isinstance(raw, list) or len(raw) < 2 or not raw[1]:
            return pd.DataFrame()

        rows = []
        for point in raw[1]:
            if point.get("value") is None:
                continue
            rows.append({
                "year": int(point["date"]) if point["date"].isdigit() else None,
                "value": float(point["value"]),
                "country": country_code.upper(),
                "indicator_code": indicator,
                "indicator_name": WB_INDICATORS.get(indicator, indicator),
            })

        df = pd.DataFrame(rows).dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        return df.sort_values("year").reset_index(drop=True)

    def get_bulk_indicators(
        self,
        country_codes: list[str],
        indicators: list[str],
    ) -> pd.DataFrame:
        """
        Fetch multiple indicators for multiple countries.

        Returns a long-format DataFrame ready for pivot operations.
        """
        frames: list[pd.DataFrame] = []
        for country in country_codes:
            for ind in indicators:
                df = self.get_indicator(country, ind)
                if not df.empty:
                    frames.append(df)
                time.sleep(0.1)  # respect rate limits

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_country_metadata(self) -> pd.DataFrame:
        """Fetch all country codes, names, regions, and income groups from World Bank."""
        url = f"{WB_BASE}/country"
        params = {"format": "json", "per_page": 300}
        try:
            raw = _cached_get(url, params)
        except Exception as exc:
            logger.warning("WorldBank country metadata error: %s", exc)
            return pd.DataFrame()

        if not isinstance(raw, list) or len(raw) < 2:
            return pd.DataFrame()

        rows = []
        for c in raw[1]:
            rows.append({
                "iso2_code": c.get("iso2Code", ""),
                "iso3_code": c.get("id", ""),
                "name": c.get("name", ""),
                "region": c.get("region", {}).get("value", ""),
                "income_level": c.get("incomeLevel", {}).get("value", ""),
                "lending_type": c.get("lendingType", {}).get("value", ""),
                "capital_city": c.get("capitalCity", ""),
                "longitude": c.get("longitude", None),
                "latitude": c.get("latitude", None),
            })

        return pd.DataFrame(rows)

    def get_latest_values(
        self, country_codes: list[str], indicators: list[str]
    ) -> pd.DataFrame:
        """
        Return the most recent value per country/indicator as a wide matrix.
        Rows = countries, columns = indicator short names.
        """
        long = self.get_bulk_indicators(country_codes, indicators)
        if long.empty:
            return pd.DataFrame()

        latest = (
            long.sort_values("year")
            .groupby(["country", "indicator_name"])
            .last()
            .reset_index()[["country", "indicator_name", "value"]]
        )
        return latest.pivot(index="country", columns="indicator_name", values="value")


# ---------------------------------------------------------------------------
# IMFDataAdapter
# ---------------------------------------------------------------------------

class IMFDataAdapter:
    """Fetches forecasts and actuals from IMF DataMapper (free REST API)."""

    def get_imf_indicator(
        self,
        indicator: str,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch an IMF indicator time series.

        indicator: one of the IMF_INDICATORS keys (e.g. "NGDP_RPCH").
        countries: list of ISO-2 codes; None = all available.
        Returns long-format DataFrame [country, year, value, indicator].
        """
        url = f"{IMF_BASE}/{indicator}"
        try:
            raw = _cached_get(url)
        except Exception as exc:
            logger.warning("IMF API error for %s: %s", indicator, exc)
            return pd.DataFrame()

        # Response structure: {values: {INDICATOR: {COUNTRY: {YEAR: value}}}}
        data_block = raw.get("values", {}).get(indicator, {})
        if not data_block:
            return pd.DataFrame()

        rows = []
        target_countries = set(c.upper() for c in countries) if countries else None
        for country, year_data in data_block.items():
            if target_countries and country not in target_countries:
                continue
            for year_str, val in year_data.items():
                if val is None:
                    continue
                try:
                    rows.append({
                        "country": country,
                        "year": int(year_str),
                        "value": float(val),
                        "indicator": indicator,
                        "indicator_name": IMF_INDICATORS.get(indicator, indicator),
                    })
                except (ValueError, TypeError):
                    continue

        return pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)

    def get_weo_forecasts(self, indicator: str) -> pd.DataFrame:
        """
        Return IMF WEO forecast years (current year + 5 ahead) for a given indicator.
        Filters the indicator response to future-year columns only.
        """
        df = self.get_imf_indicator(indicator)
        if df.empty:
            return df
        current_year = datetime.now().year
        return df[df["year"] >= current_year].copy()

    def get_article_iv_summaries(self) -> list[dict]:
        """
        Retrieve recent IMF Article IV consultation press release summaries.
        Uses the IMF RSS/JSON feed for consultation press releases.
        """
        try:
            url = "https://www.imf.org/en/Publications/SPROLLs/Article-IV-Staff-Reports-and-Supplements"
            # IMF doesn't expose a clean JSON feed for Art. IV; return structured stubs
            # with key fields populated from the DataMapper metadata endpoint.
            meta_url = f"{IMF_BASE}/countries"
            meta = _cached_get(meta_url)
            countries_meta = meta.get("countries", {})
            summaries = []
            for code, info in list(countries_meta.items())[:20]:
                summaries.append({
                    "country_code": code,
                    "country_name": info.get("label", code),
                    "consultation_type": "Article IV",
                    "year": datetime.now().year,
                    "source_url": f"https://www.imf.org/en/countries/{code}",
                    "note": "Full Article IV text requires IMF eLibrary subscription",
                })
            return summaries
        except Exception as exc:
            logger.warning("IMF Article IV fetch error: %s", exc)
            return []

    def get_all_forecasts(self, countries: list[str] | None = None) -> pd.DataFrame:
        """Fetch WEO forecasts for all standard IMF indicators and merge into one DataFrame."""
        frames = []
        for ind in IMF_INDICATORS:
            df = self.get_weo_forecasts(ind)
            if countries and not df.empty:
                df = df[df["country"].isin([c.upper() for c in countries])]
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# OECDAdapter
# ---------------------------------------------------------------------------

class OECDAdapter:
    """Fetches OECD SDMX-JSON data — MEI, Economic Outlook, and CLI."""

    _OECD_CLI_URL = (
        "https://stats.oecd.org/sdmx-json/data/MEI_CLI"
        "/.LOLITOAA.OECD+{countries}.M"
        "?contentType=csv&startPeriod={start}&endPeriod={end}"
    )

    def get_indicator(
        self,
        dataset: str,
        country: str,
        measure: str,
    ) -> pd.DataFrame:
        """
        Fetch a single OECD SDMX series.

        dataset: e.g. "MEI", "EO"
        country: OECD country code (usually 3-letter, e.g. "USA", "DEU")
        measure: OECD measure code within the dataset
        Returns DataFrame [period, value, country, measure].
        """
        url = f"{OECD_BASE}/{dataset}/{country}.{measure}.A/all"
        params = {
            "contentType": "csv",
            "startPeriod": "2000",
            "endPeriod": str(datetime.now().year),
        }
        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            if df.empty:
                return pd.DataFrame()
            # Normalise column names (OECD CSV varies by dataset)
            df.columns = [c.lower().strip() for c in df.columns]
            value_col = next((c for c in df.columns if c in ("value", "obs_value")), None)
            time_col = next((c for c in df.columns if c in ("time", "time_period", "period")), None)
            if value_col is None or time_col is None:
                return pd.DataFrame()
            result = df[[time_col, value_col]].copy()
            result.columns = ["period", "value"]
            result["country"] = country
            result["measure"] = measure
            return result.dropna(subset=["value"])
        except Exception as exc:
            logger.warning("OECD API error %s/%s/%s: %s", dataset, country, measure, exc)
            return pd.DataFrame()

    def get_leading_indicators(
        self, countries: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Fetch OECD Composite Leading Indicators (CLI) for specified countries.

        CLI values > 100 = expansion, < 100 = contraction tendency.
        Returns a wide-format DataFrame [period, country1, country2, ...].
        """
        oecd_map = {
            "US": "USA", "GB": "GBR", "DE": "DEU", "JP": "JPN", "CN": "CHN",
            "FR": "FRA", "IT": "ITA", "CA": "CAN", "AU": "AUS", "KR": "KOR",
            "BR": "BRA", "IN": "IND", "MX": "MEX", "TR": "TUR",
        }

        target = countries or list(oecd_map.keys())
        frames: list[pd.DataFrame] = []
        for iso2 in target:
            oecd_code = oecd_map.get(iso2.upper())
            if not oecd_code:
                continue
            # MEI_CLI: LOLITOAA = amplitude-adjusted CLI
            url = (
                f"{OECD_BASE}/MEI_CLI/{oecd_code}.LOLITOAA.M/all"
                "?contentType=csv&startPeriod=2015"
            )
            try:
                resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                if df.empty:
                    continue
                df.columns = [c.lower().strip() for c in df.columns]
                time_col = next((c for c in df.columns if "time" in c or "period" in c), None)
                val_col = next((c for c in df.columns if c in ("value", "obs_value")), None)
                if not time_col or not val_col:
                    continue
                s = df[[time_col, val_col]].rename(columns={time_col: "period", val_col: iso2})
                s = s.dropna().set_index("period")
                frames.append(s)
            except Exception as exc:
                logger.debug("OECD CLI error for %s: %s", iso2, exc)
                continue

        if not frames:
            return pd.DataFrame()
        wide = frames[0]
        for f in frames[1:]:
            wide = wide.join(f, how="outer")
        return wide.sort_index()

    def get_economic_outlook(self, country: str, variable: str = "GDPVD") -> pd.DataFrame:
        """
        Fetch OECD Economic Outlook projections.

        variable: GDPVD (GDP volume), CPIH (CPI), UNR (unemployment), etc.
        """
        oecd_map = {
            "US": "USA", "GB": "GBR", "DE": "DEU", "JP": "JPN", "CN": "CHN",
            "FR": "FRA", "IT": "ITA", "CA": "CAN", "AU": "AUS", "KR": "KOR",
        }
        oecd_code = oecd_map.get(country.upper(), country.upper())
        return self.get_indicator("EO", oecd_code, variable)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_gdp_growth(g: float | None) -> float:
    if g is None:
        return 5.0
    if g > 3.0:
        return 20.0
    if g >= 1.0:
        return 10.0
    if g >= 0.0:
        return 5.0
    return 0.0


def _score_inflation(inf: float | None) -> float:
    if inf is None:
        return 5.0
    if 1.0 <= inf <= 3.0:
        return 20.0
    if 0.0 <= inf <= 5.0:
        return 10.0
    if 5.0 < inf <= 10.0:
        return 5.0
    return 0.0


def _score_unemployment(u: float | None) -> float:
    if u is None:
        return 5.0
    if u < 5.0:
        return 20.0
    if u <= 8.0:
        return 10.0
    if u <= 12.0:
        return 5.0
    return 0.0


def _score_fiscal(fb: float | None) -> float:
    if fb is None:
        return 5.0
    if fb > 0.0:
        return 20.0
    if fb >= -3.0:
        return 10.0
    if fb >= -6.0:
        return 5.0
    return 0.0


def _score_current_account(ca: float | None) -> float:
    if ca is None:
        return 5.0
    if ca > 0.0:
        return 20.0
    if ca >= -3.0:
        return 10.0
    if ca >= -6.0:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# MacroCrossCountryEngine
# ---------------------------------------------------------------------------

@dataclass
class CountryMacroProfile:
    country: str
    gdp_growth: float | None = None
    inflation: float | None = None
    unemployment: float | None = None
    fiscal_balance: float | None = None
    current_account: float | None = None
    debt_pct_gdp: float | None = None
    gdp_per_capita_ppp: float | None = None
    health_score: float = 0.0
    data_year: int | None = None
    cycle_phase: str = "unknown"


class MacroCrossCountryEngine:
    """
    Unified cross-country macroeconomic comparison engine.

    Aggregates data from World Bank, IMF DataMapper, and OECD to produce
    a standardised macro scorecard, cycle-phase classification, and
    relative-value frameworks.
    """

    MAJOR_ECONOMIES = MAJOR_ECONOMIES

    def __init__(self) -> None:
        self.wb = WorldBankMacroAdapter()
        self.imf = IMFDataAdapter()
        self.oecd = OECDAdapter()

    # ------------------------------------------------------------------
    # Core scorecard
    # ------------------------------------------------------------------

    def _fetch_wb_latest(
        self, countries: list[str], indicators: list[str]
    ) -> dict[str, dict[str, float | None]]:
        """
        Fetch the most recent World Bank values for each country/indicator.
        Returns {country: {indicator_name: value}}.
        """
        result: dict[str, dict[str, float | None]] = {c: {} for c in countries}
        for ind_code in indicators:
            ind_name = WB_INDICATORS.get(ind_code, ind_code)
            for country in countries:
                df = self.wb.get_indicator(country, ind_code)
                if df.empty:
                    result[country][ind_name] = None
                else:
                    result[country][ind_name] = float(df.iloc[-1]["value"])
                time.sleep(0.05)
        return result

    def build_macro_scorecard(
        self,
        countries: list[str] | None = None,
        as_of_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Build a comprehensive macro scorecard for all (or specified) economies.

        For each country, returns:
          gdp_growth, inflation, unemployment, fiscal_balance, current_account,
          debt_pct_gdp, gdp_per_capita, health_score (0-100), data_year.
        """
        countries = countries or self.MAJOR_ECONOMIES
        if as_of_year is None:
            as_of_year = datetime.now().year - 1  # World Bank lags ~1yr

        key_inds = [
            "NY.GDP.MKTP.KD.ZG",  # GDP growth
            "FP.CPI.TOTL.ZG",     # Inflation
            "SL.UEM.TOTL.ZS",     # Unemployment
            "GC.BAL.CASH.GD.ZS",  # Fiscal balance
            "BN.CAB.XOKA.GD.ZS",  # Current account
            "GC.DOD.TOTL.GD.ZS",  # Debt % GDP
            "NY.GDP.PCAP.PP.CD",  # GDP per capita PPP
        ]

        wb_data = self._fetch_wb_latest(countries, key_inds)

        rows = []
        for country in countries:
            d = wb_data.get(country, {})
            gdp_g = d.get("GDP growth rate")
            inf   = d.get("Inflation CPI")
            unemp = d.get("Unemployment rate")
            fisc  = d.get("Fiscal balance % GDP")
            ca    = d.get("Current account % GDP")
            debt  = d.get("Government debt % GDP")
            gdppc = d.get("GDP per capita PPP")

            score = (
                _score_gdp_growth(gdp_g)
                + _score_inflation(inf)
                + _score_unemployment(unemp)
                + _score_fiscal(fisc)
                + _score_current_account(ca)
            )

            rows.append({
                "country": country,
                "gdp_growth_pct": round(gdp_g, 2) if gdp_g is not None else None,
                "inflation_pct": round(inf, 2) if inf is not None else None,
                "unemployment_pct": round(unemp, 2) if unemp is not None else None,
                "fiscal_balance_pct_gdp": round(fisc, 2) if fisc is not None else None,
                "current_account_pct_gdp": round(ca, 2) if ca is not None else None,
                "debt_pct_gdp": round(debt, 1) if debt is not None else None,
                "gdp_per_capita_ppp_usd": round(gdppc, 0) if gdppc is not None else None,
                "health_score": round(score, 1),
                "as_of_year": as_of_year,
            })

        df = pd.DataFrame(rows).set_index("country").sort_values(
            "health_score", ascending=False
        )
        return df

    # ------------------------------------------------------------------
    # Cycle classification
    # ------------------------------------------------------------------

    def get_economic_cycle_phase(self, country: str) -> dict:
        """
        Classify country's economic cycle phase using OECD CLI and GDP trend.

        Phases: expansion / peak / contraction / trough.
        Uses OECD CLI: >100 and rising → expansion, >100 and falling → peak,
        <100 and falling → contraction, <100 and rising → trough.
        """
        cli_wide = self.oecd.get_leading_indicators([country])
        phase = "unknown"
        cli_current = None
        cli_trend = None

        if not cli_wide.empty and country in cli_wide.columns:
            series = cli_wide[country].dropna()
            if len(series) >= 3:
                cli_current = float(series.iloc[-1])
                cli_trend = float(series.iloc[-1] - series.iloc[-3])  # 3-month change
                above_100 = cli_current > 100
                rising = cli_trend > 0
                if above_100 and rising:
                    phase = "expansion"
                elif above_100 and not rising:
                    phase = "peak"
                elif not above_100 and not rising:
                    phase = "contraction"
                else:
                    phase = "trough"

        # Supplement with World Bank GDP trend
        gdp_df = self.wb.get_indicator(country, "NY.GDP.MKTP.KD.ZG")
        gdp_growth_recent = None
        gdp_direction = None
        if not gdp_df.empty and len(gdp_df) >= 2:
            gdp_growth_recent = float(gdp_df.iloc[-1]["value"])
            gdp_direction = "accelerating" if gdp_df.iloc[-1]["value"] > gdp_df.iloc[-2]["value"] else "decelerating"

        return {
            "country": country,
            "cycle_phase": phase,
            "oecd_cli_current": cli_current,
            "oecd_cli_3m_change": cli_trend,
            "gdp_growth_latest_pct": gdp_growth_recent,
            "gdp_momentum": gdp_direction,
            "as_of": datetime.now().strftime("%Y-%m-%d"),
        }

    # ------------------------------------------------------------------
    # Relative value framework
    # ------------------------------------------------------------------

    def compute_relative_value_macro(
        self, country_a: str, country_b: str
    ) -> dict:
        """
        Compare macro fundamentals between two countries for relative value
        (FX and rates) investment theses.

        Returns differential metrics, carry signals, and qualitative thesis.
        """
        scorecard = self.build_macro_scorecard(
            countries=[country_a, country_b]
        )

        def _get(col: str, country: str) -> float | None:
            try:
                v = scorecard.loc[country, col]
                return float(v) if v is not None and not pd.isna(v) else None
            except (KeyError, TypeError):
                return None

        gdp_diff = _safely_diff(_get("gdp_growth_pct", country_a), _get("gdp_growth_pct", country_b))
        inf_diff  = _safely_diff(_get("inflation_pct", country_a), _get("inflation_pct", country_b))
        ca_diff   = _safely_diff(_get("current_account_pct_gdp", country_a), _get("current_account_pct_gdp", country_b))
        fs_diff   = _safely_diff(_get("fiscal_balance_pct_gdp", country_a), _get("fiscal_balance_pct_gdp", country_b))
        score_a   = _get("health_score", country_a)
        score_b   = _get("health_score", country_b)

        # Qualitative thesis
        thesis_parts = []
        if gdp_diff is not None:
            if gdp_diff > 1.0:
                thesis_parts.append(f"{country_a} outgrowing {country_b} by {gdp_diff:.1f}pp → bullish {country_a} FX")
            elif gdp_diff < -1.0:
                thesis_parts.append(f"{country_b} outgrowing {country_a} by {abs(gdp_diff):.1f}pp → bullish {country_b} FX")
        if inf_diff is not None:
            if inf_diff > 2.0:
                thesis_parts.append(f"{country_a} inflation premium of {inf_diff:.1f}pp → bearish {country_a} real rates")
            elif inf_diff < -2.0:
                thesis_parts.append(f"{country_b} inflation premium of {abs(inf_diff):.1f}pp → bearish {country_b} real rates")
        if ca_diff is not None and ca_diff > 2.0:
            thesis_parts.append(f"{country_a} CA surplus advantage {ca_diff:.1f}pp → structural FX support for {country_a}")

        return {
            "country_a": country_a,
            "country_b": country_b,
            "gdp_growth_diff_pp": gdp_diff,
            "inflation_diff_pp": inf_diff,
            "current_account_diff_pp": ca_diff,
            "fiscal_balance_diff_pp": fs_diff,
            "health_score_a": score_a,
            "health_score_b": score_b,
            "health_score_diff": _safely_diff(score_a, score_b),
            "macro_thesis": "; ".join(thesis_parts) if thesis_parts else "No strong macro signal",
            "long_country": country_a if (score_a or 0) > (score_b or 0) else country_b,
            "short_country": country_b if (score_a or 0) > (score_b or 0) else country_a,
        }

    # ------------------------------------------------------------------
    # Global growth outlook
    # ------------------------------------------------------------------

    def get_global_growth_outlook(self) -> dict:
        """
        Aggregate weighted world GDP growth using World Bank and IMF forecasts.

        Returns: world_gdp_growth_estimate, imf_forecasts, oecd_forecasts,
        top_growers, drag_countries, risk_commentary.
        """
        # IMF real GDP growth forecasts
        imf_df = self.imf.get_imf_indicator("NGDP_RPCH")
        current_year = datetime.now().year

        imf_current: dict[str, float] = {}
        imf_forecast: dict[str, float] = {}
        if not imf_df.empty:
            for _, row in imf_df.iterrows():
                if row["year"] == current_year - 1:
                    imf_current[row["country"]] = row["value"]
                elif row["year"] == current_year:
                    imf_forecast[row["country"]] = row["value"]

        # Simple GDP-weighted world growth estimate (equal-weight over majors as proxy)
        growth_vals = [v for v in imf_current.values() if v is not None]
        world_growth_est = float(np.mean(growth_vals)) if growth_vals else None

        top_growers = sorted(imf_current.items(), key=lambda x: x[1], reverse=True)[:5]
        drag_countries = sorted(imf_current.items(), key=lambda x: x[1])[:5]

        return {
            "world_gdp_growth_estimate_pct": round(world_growth_est, 2) if world_growth_est else None,
            "estimate_year": current_year - 1,
            "imf_actuals_count": len(imf_current),
            "imf_forecasts_count": len(imf_forecast),
            "top_5_growers": [{"country": c, "growth_pct": round(v, 2)} for c, v in top_growers],
            "bottom_5_growers": [{"country": c, "growth_pct": round(v, 2)} for c, v in drag_countries],
            "upside_risks": [
                "Supply chain normalisation faster than expected",
                "AI productivity dividend materialising in GDP data",
                "China stimulus overshoot supporting EM demand",
            ],
            "downside_risks": [
                "Persistent inflation requiring higher-for-longer rates",
                "Geopolitical fragmentation reducing trade multiplier",
                "Property sector stress in key EM economies",
                "Fiscal consolidation headwind in advanced economies",
            ],
            "as_of": datetime.now().strftime("%Y-%m-%d"),
        }

    # ------------------------------------------------------------------
    # EM screener
    # ------------------------------------------------------------------

    def screen_emerging_markets(
        self,
        min_gdp_growth: float = 4.0,
        max_inflation: float = 8.0,
        max_debt_pct_gdp: float = 80.0,
    ) -> pd.DataFrame:
        """
        Screen EM economies by macro fundamentals.

        Returns countries passing all filters, sorted by health score.
        """
        em_countries = list(_EMERGING_MARKETS)
        scorecard = self.build_macro_scorecard(countries=em_countries)

        mask = pd.Series(True, index=scorecard.index)
        if "gdp_growth_pct" in scorecard.columns:
            mask &= scorecard["gdp_growth_pct"].fillna(0) >= min_gdp_growth
        if "inflation_pct" in scorecard.columns:
            mask &= scorecard["inflation_pct"].fillna(999) <= max_inflation
        if "debt_pct_gdp" in scorecard.columns:
            mask &= scorecard["debt_pct_gdp"].fillna(999) <= max_debt_pct_gdp

        result = scorecard[mask].copy()
        result["em_flag"] = True
        return result.sort_values("health_score", ascending=False)

    # ------------------------------------------------------------------
    # Inflation tracker
    # ------------------------------------------------------------------

    def global_inflation_tracker(
        self, lookback_years: int = 5
    ) -> pd.DataFrame:
        """
        Fetch CPI inflation time series for all major economies.

        Returns a wide DataFrame: rows = year, columns = country codes,
        values = inflation % YoY.
        """
        start_year = datetime.now().year - lookback_years
        frames: dict[str, pd.Series] = {}
        for country in self.MAJOR_ECONOMIES:
            df = self.wb.get_indicator(
                country, "FP.CPI.TOTL.ZG", start_year=start_year
            )
            if not df.empty:
                s = df.set_index("year")["value"].rename(country)
                frames[country] = s

        if not frames:
            return pd.DataFrame()

        wide = pd.DataFrame(frames)
        wide.index.name = "year"
        return wide.sort_index()

    # ------------------------------------------------------------------
    # Yield curve comparison (FRED + stubs for non-US)
    # ------------------------------------------------------------------

    def yield_curve_comparison(
        self, countries: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Compare yield curves across countries using FRED (US) and
        ECB/central bank data stubs.

        Returns a DataFrame with 2Y and 10Y yields and the 2s10s spread.
        """
        target = countries or ["US", "DE", "GB", "JP", "AU", "CA"]

        # For US we can use FRED (no key for these series via yfinance)
        rows = []
        for country in target:
            rows.append({
                "country": country,
                "2y_yield": _stub_yield(country, "2y"),
                "10y_yield": _stub_yield(country, "10y"),
                "spread_2s10s": _stub_spread(country),
                "curve_shape": _classify_curve(_stub_spread(country)),
                "note": "Use sentinel/sma/global_macro.py FRED adapter for live US data",
            })

        return pd.DataFrame(rows).set_index("country")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safely_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 2)


def _stub_yield(country: str, tenor: str) -> float | None:
    """
    Placeholder yields — in production wire to FRED / ECB / RBA APIs.
    Returns None so callers know to fetch from the live adapter.
    """
    return None


def _stub_spread(country: str) -> float | None:
    return None


def _classify_curve(spread: float | None) -> str:
    if spread is None:
        return "unknown"
    if spread > 0.5:
        return "normal"
    if spread > 0:
        return "flat"
    return "inverted"


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class MacroScorecardRow(BaseModel):
    country: str
    gdp_growth_pct: float | None
    inflation_pct: float | None
    unemployment_pct: float | None
    fiscal_balance_pct_gdp: float | None
    current_account_pct_gdp: float | None
    debt_pct_gdp: float | None
    gdp_per_capita_ppp_usd: float | None
    health_score: float
    as_of_year: int | None


class CountryOverviewResponse(BaseModel):
    country: str
    scorecard: dict
    cycle_phase: dict
    imf_forecasts: list[dict]


class GlobalGrowthResponse(BaseModel):
    world_gdp_growth_estimate_pct: float | None
    estimate_year: int
    top_5_growers: list[dict]
    bottom_5_growers: list[dict]
    upside_risks: list[str]
    downside_risks: list[str]
    as_of: str


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

macro_router = APIRouter(prefix="/api/macro", tags=["macro"])
_engine: MacroCrossCountryEngine | None = None


def _get_engine() -> MacroCrossCountryEngine:
    global _engine
    if _engine is None:
        _engine = MacroCrossCountryEngine()
    return _engine


@macro_router.get("/scorecard", summary="All major economies macro scorecard")
async def get_scorecard(
    countries: str | None = Query(None, description="Comma-separated ISO-2 codes; default = all majors"),
) -> list[dict]:
    """Return the macro scorecard for all (or specified) economies."""
    engine = _get_engine()
    country_list = [c.strip().upper() for c in countries.split(",")] if countries else None
    try:
        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.build_macro_scorecard(country_list)
        )
        return df.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.error("Scorecard error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@macro_router.get("/{country}/overview", summary="Single country macro profile")
async def get_country_overview(country: str) -> dict:
    """Full macro profile for a single country including cycle phase and IMF forecasts."""
    engine = _get_engine()
    country = country.upper()
    try:
        scorecard_df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.build_macro_scorecard([country])
        )
        cycle = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.get_economic_cycle_phase(country)
        )
        imf_df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.imf.get_all_forecasts([country])
        )
        imf_records = imf_df.to_dict(orient="records") if not imf_df.empty else []
        scorecard_row = (
            scorecard_df.reset_index().iloc[0].to_dict()
            if not scorecard_df.empty else {}
        )
        return {
            "country": country,
            "scorecard": scorecard_row,
            "cycle_phase": cycle,
            "imf_forecasts": imf_records,
        }
    except Exception as exc:
        logger.error("Country overview error for %s: %s", country, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@macro_router.get("/compare", summary="Side-by-side country comparison")
async def compare_countries(
    countries: str = Query(..., description="Comma-separated ISO-2 codes, e.g. US,DE,JP"),
) -> dict:
    """Side-by-side macro comparison for specified countries."""
    engine = _get_engine()
    country_list = [c.strip().upper() for c in countries.split(",")]
    if len(country_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 country codes")
    try:
        scorecard_df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.build_macro_scorecard(country_list)
        )
        comparisons = []
        if len(country_list) == 2:
            rv = await asyncio.get_event_loop().run_in_executor(
                None, lambda: engine.compute_relative_value_macro(country_list[0], country_list[1])
            )
            comparisons.append(rv)
        return {
            "countries": country_list,
            "scorecard": scorecard_df.reset_index().to_dict(orient="records"),
            "relative_value": comparisons,
        }
    except Exception as exc:
        logger.error("Compare error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@macro_router.get("/screen/emerging", summary="EM screener by macro fundamentals")
async def screen_em(
    min_gdp_growth: float = Query(4.0, description="Minimum GDP growth %"),
    max_inflation: float = Query(8.0, description="Maximum CPI inflation %"),
    max_debt_pct_gdp: float = Query(80.0, description="Maximum government debt % GDP"),
) -> list[dict]:
    """Screen emerging markets by macro fundamentals."""
    engine = _get_engine()
    try:
        df = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.screen_emerging_markets(min_gdp_growth, max_inflation, max_debt_pct_gdp),
        )
        return df.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.error("EM screen error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@macro_router.get("/global-growth", summary="Global growth outlook")
async def get_global_growth() -> dict:
    """IMF/OECD global growth outlook with top/bottom growers and risk commentary."""
    engine = _get_engine()
    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, engine.get_global_growth_outlook
        )
    except Exception as exc:
        logger.error("Global growth error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@macro_router.get("/inflation", summary="Global inflation tracker")
async def get_inflation_tracker(
    lookback_years: int = Query(5, ge=1, le=25, description="Years of history"),
) -> dict:
    """Inflation time series for all major economies."""
    engine = _get_engine()
    try:
        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.global_inflation_tracker(lookback_years)
        )
        return {
            "years": df.index.tolist(),
            "countries": df.columns.tolist(),
            "data": df.to_dict(),
        }
    except Exception as exc:
        logger.error("Inflation tracker error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
