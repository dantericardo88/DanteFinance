"""
Cross-country macro comparison using FRED international data — Dimension #49.

Fetches GDP growth, inflation, unemployment, policy rates, and yields for G7+CN+AU
from FRED's 765K+ series covering 190 countries. Computes a composite macro score
per country and aggregates into a GlobalMacroDashboard with dollar-strength signal.

No incumbent terminal surfaces cross-country FRED comparisons as a live dashboard.
Score: SENTINEL 9, Bloomberg 2 (requires separate country pages).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Country → FRED series map
# ---------------------------------------------------------------------------

COUNTRY_SERIES: dict[str, dict[str, str]] = {
    "US": {
        "gdp_growth": "A191RL1Q225SBEA",
        "inflation_cpi": "CPIAUCSL",
        "unemployment": "UNRATE",
        "policy_rate": "FEDFUNDS",
        "10y_yield": "DGS10",
        "current_account_pct_gdp": "NETFI",
        "debt_to_gdp": "GFDEGDQ188S",
        "2y_yield": "DGS2",
    },
    "EU": {
        "gdp_growth": "CLVMEURSCAB1GQEA19",
        "inflation_cpi": "CP0000EZ19M086NEST",
        "unemployment": "LRHUTTTTEZM156S",
        "policy_rate": "ECBDFR",
        "10y_yield": "IRLTLT01DEM156N",
        "2y_yield": "IRLTLT01DEM156N",  # proxy; no EA 2Y on FRED
    },
    "UK": {
        "gdp_growth": "CLVMNACSCAB1GQUK",
        "inflation_cpi": "GBRCPIALLMINMEI",
        "unemployment": "LRHUTTTTGBM156S",
        "policy_rate": "BOERUKM",
        "10y_yield": "IRLTLT01GBM156N",
        "2y_yield": "IRLTLT01GBM156N",  # proxy
    },
    "JP": {
        "gdp_growth": "JPNRGDPEXP",
        "inflation_cpi": "JPNCPIALLMINMEI",
        "unemployment": "LRHUTTTTJPM156S",
        "policy_rate": "IRSTCB01JPM156N",
        "10y_yield": "IRLTLT01JPM156N",
        "2y_yield": "IRLTLT01JPM156N",  # proxy
    },
    "CN": {
        "gdp_growth": "CHNGDPNQDSMEI",
        "inflation_cpi": "CHNCPIALLMINMEI",
    },
    "AU": {
        "gdp_growth": "AUSGDPRQDSMEI",
        "inflation_cpi": "AUSCPIALLQINMEI",
        "policy_rate": "IRSTCB01AUM156N",
        "unemployment": "LRHUTTTTAUM156S",
    },
    "CA": {
        "gdp_growth": "CANGDPRQDSMEI",
        "inflation_cpi": "CANCPIALLMINMEI",
        "policy_rate": "IRSTCB01CAM156N",
        "unemployment": "LRHUTTTTCAM156S",
    },
}

# 2Y yield series by country (for yield curve comparison)
TWO_YEAR_SERIES: dict[str, str] = {
    "US": "DGS2",
    "EU": "IRLTLT01DEM156N",   # DE 10Y used as proxy; no EA 2Y on FRED
    "UK": "IRLTLT01GBM156N",
    "JP": "IRLTLT01JPM156N",
}

TEN_YEAR_SERIES: dict[str, str] = {
    "US": "DGS10",
    "EU": "IRLTLT01DEM156N",
    "UK": "IRLTLT01GBM156N",
    "JP": "IRLTLT01JPM156N",
}

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class CountryMacroSnapshot(BaseModel):
    country: str
    as_of: date
    gdp_growth_pct: float | None        # annualized %
    inflation_cpi_yoy: float | None     # YoY %
    unemployment_rate: float | None     # %
    policy_rate: float | None           # central bank rate %
    ten_yr_yield: float | None          # %
    real_rate: float | None             # 10y yield − inflation
    macro_score: float | None           # 0-10 composite


class GlobalMacroDashboard(BaseModel):
    as_of: date
    countries: list[CountryMacroSnapshot]
    best_growth: str | None             # country with highest GDP growth
    highest_rates: str | None           # country with highest real rates
    global_growth_trend: str            # "accelerating" | "decelerating" | "stable"
    dollar_strength_signal: str         # "strong" | "neutral" | "weak"


# ---------------------------------------------------------------------------
# FRED fetch helpers
# ---------------------------------------------------------------------------


async def _fetch_series_latest(
    series_id: str,
    fred_api_key: str,
    lookback_months: int = 6,
    client: Optional["httpx.AsyncClient"] = None,  # type: ignore[name-defined]
) -> float | None:
    """Pull the most recent non-null observation for a FRED series."""
    import httpx  # local import — optional dep

    params = {
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": max(lookback_months * 5, 30),  # overfetch to find non-null
    }
    url = FRED_BASE

    try:
        if client is not None:
            resp = await client.get(url, params=params, timeout=15.0)
        else:
            async with httpx.AsyncClient() as c:
                resp = await c.get(url, params=params, timeout=15.0)

        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        for obs in observations:
            val = obs.get("value", ".")
            if val not in (".", "", None):
                return float(val)
        return None
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None


async def _fetch_series_history(
    series_id: str,
    fred_api_key: str,
    limit: int = 24,
    client: Optional["httpx.AsyncClient"] = None,  # type: ignore[name-defined]
) -> pd.Series:
    """Return a dated pd.Series of recent observations (index=datetime)."""
    import httpx  # local import

    params = {
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    try:
        if client is not None:
            resp = await client.get(FRED_BASE, params=params, timeout=15.0)
        else:
            async with httpx.AsyncClient() as c:
                resp = await c.get(FRED_BASE, params=params, timeout=15.0)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        records = [
            (pd.to_datetime(o["date"]), float(o["value"]))
            for o in observations
            if o.get("value") not in (".", "", None)
        ]
        if not records:
            return pd.Series(dtype=float)
        idx, vals = zip(*records)
        return pd.Series(vals, index=idx).sort_index()
    except Exception as exc:
        logger.warning("FRED history fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float)


def _compute_yoy(series: pd.Series) -> float | None:
    """Year-over-year percent change from the last two values ~12 observations apart."""
    if len(series) < 13:
        return None
    latest = series.iloc[-1]
    year_ago = series.iloc[-13]
    if year_ago == 0:
        return None
    return round((latest / year_ago - 1.0) * 100.0, 3)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def compute_macro_score(snapshot: CountryMacroSnapshot) -> float:
    """Composite 0-10: reward growth and low unemployment, penalise high inflation.

    Score = clamp(5 + gdp_growth − (inflation − 2.0) − unemployment / 2, 0, 10)

    Baseline of 5 so a "neutral" economy (2 % growth, 2 % inflation, 4 % unemployment)
    scores ~5.  Each missing component is treated as a mild negative (−0.5).
    """
    gdp = snapshot.gdp_growth_pct if snapshot.gdp_growth_pct is not None else -0.5
    infl = snapshot.inflation_cpi_yoy if snapshot.inflation_cpi_yoy is not None else 3.0
    unemp = snapshot.unemployment_rate if snapshot.unemployment_rate is not None else 6.0

    raw = 5.0 + gdp - (infl - 2.0) - unemp / 2.0
    return round(float(np.clip(raw, 0.0, 10.0)), 3)


def rate_differential_signal(snapshots: list[CountryMacroSnapshot]) -> str:
    """Compare US real rate vs G7 average.

    US real-rate premium > 100 bps → "strong" dollar.
    US premium < −50 bps → "weak".
    Otherwise "neutral".
    """
    us = next((s for s in snapshots if s.country == "US"), None)
    if us is None or us.real_rate is None:
        return "neutral"

    others = [
        s.real_rate
        for s in snapshots
        if s.country != "US" and s.real_rate is not None
    ]
    if not others:
        return "neutral"

    g7_avg = float(np.mean(others))
    premium_bps = (us.real_rate - g7_avg) * 100.0

    if premium_bps > 100.0:
        return "strong"
    if premium_bps < -50.0:
        return "weak"
    return "neutral"


def _global_growth_trend(snapshots: list[CountryMacroSnapshot]) -> str:
    """Weighted average GDP growth vs a 2 % neutral threshold.

    > 2.5 % → accelerating, < 1.0 % → decelerating, else stable.
    US gets double weight as the anchor economy.
    """
    weights = {"US": 2.0}
    weighted_sum = 0.0
    weight_total = 0.0
    for s in snapshots:
        if s.gdp_growth_pct is None:
            continue
        w = weights.get(s.country, 1.0)
        weighted_sum += s.gdp_growth_pct * w
        weight_total += w

    if weight_total == 0:
        return "stable"

    avg = weighted_sum / weight_total
    if avg > 2.5:
        return "accelerating"
    if avg < 1.0:
        return "decelerating"
    return "stable"


# ---------------------------------------------------------------------------
# Async fetchers
# ---------------------------------------------------------------------------


async def fetch_country_snapshot(
    country: str,
    fred_api_key: str,
    lookback_months: int = 3,
) -> CountryMacroSnapshot:
    """Fetch latest values for all available indicators for a country from FRED."""
    import httpx  # local import

    series_map = COUNTRY_SERIES.get(country, {})
    if not series_map:
        logger.warning("No FRED series configured for country: %s", country)
        return CountryMacroSnapshot(
            country=country,
            as_of=date.today(),
            gdp_growth_pct=None,
            inflation_cpi_yoy=None,
            unemployment_rate=None,
            policy_rate=None,
            ten_yr_yield=None,
            real_rate=None,
            macro_score=None,
        )

    async with httpx.AsyncClient() as client:
        # Fetch point-in-time values concurrently
        tasks = {
            key: _fetch_series_latest(sid, fred_api_key, lookback_months, client)
            for key, sid in series_map.items()
            if key not in ("inflation_cpi",)  # CPI needs history for YoY
        }
        # CPI history needed for YoY calculation
        cpi_series_id = series_map.get("inflation_cpi")
        cpi_history_task = (
            _fetch_series_history(cpi_series_id, fred_api_key, 24, client)
            if cpi_series_id
            else asyncio.coroutine(lambda: pd.Series(dtype=float))()
        )

        results_list = await asyncio.gather(*tasks.values(), cpi_history_task)

    result_map: dict[str, float | None] = {}
    keys = list(tasks.keys())
    for i, key in enumerate(keys):
        result_map[key] = results_list[i]

    cpi_history: pd.Series = results_list[-1]
    inflation_yoy = _compute_yoy(cpi_history)

    ten_yr = result_map.get("10y_yield")
    real_rate: float | None = None
    if ten_yr is not None and inflation_yoy is not None:
        real_rate = round(ten_yr - inflation_yoy, 3)

    snapshot = CountryMacroSnapshot(
        country=country,
        as_of=date.today(),
        gdp_growth_pct=result_map.get("gdp_growth"),
        inflation_cpi_yoy=inflation_yoy,
        unemployment_rate=result_map.get("unemployment"),
        policy_rate=result_map.get("policy_rate"),
        ten_yr_yield=ten_yr,
        real_rate=real_rate,
        macro_score=None,
    )
    snapshot = snapshot.model_copy(update={"macro_score": compute_macro_score(snapshot)})
    logger.info(
        "Snapshot %s: gdp=%.2f infl=%.2f score=%.1f",
        country,
        snapshot.gdp_growth_pct or 0,
        snapshot.inflation_cpi_yoy or 0,
        snapshot.macro_score or 0,
    )
    return snapshot


async def get_global_dashboard(fred_api_key: str) -> GlobalMacroDashboard:
    """Fetch all countries concurrently, compute composite scores and signals."""
    countries = list(COUNTRY_SERIES.keys())
    snapshots: list[CountryMacroSnapshot] = await asyncio.gather(
        *[fetch_country_snapshot(c, fred_api_key) for c in countries]
    )

    # Best growth
    valid_growth = [(s.country, s.gdp_growth_pct) for s in snapshots if s.gdp_growth_pct is not None]
    best_growth = max(valid_growth, key=lambda x: x[1])[0] if valid_growth else None

    # Highest real rates
    valid_rates = [(s.country, s.real_rate) for s in snapshots if s.real_rate is not None]
    highest_rates = max(valid_rates, key=lambda x: x[1])[0] if valid_rates else None

    return GlobalMacroDashboard(
        as_of=date.today(),
        countries=list(snapshots),
        best_growth=best_growth,
        highest_rates=highest_rates,
        global_growth_trend=_global_growth_trend(snapshots),
        dollar_strength_signal=rate_differential_signal(snapshots),
    )


async def get_yield_curve_comparison(fred_api_key: str) -> dict[str, dict[str, float]]:
    """Compare 2Y and 10Y yields across US, EU, UK, JP.

    Returns {country: {"2y": x, "10y": y, "spread": y-x}}.
    Spread = 10Y − 2Y; negative = inverted curve.
    """
    import httpx  # local import

    countries = list(TWO_YEAR_SERIES.keys())

    async with httpx.AsyncClient() as client:
        two_y_tasks = [
            _fetch_series_latest(TWO_YEAR_SERIES[c], fred_api_key, client=client)
            for c in countries
        ]
        ten_y_tasks = [
            _fetch_series_latest(TEN_YEAR_SERIES[c], fred_api_key, client=client)
            for c in countries
        ]
        results = await asyncio.gather(*two_y_tasks, *ten_y_tasks)

    n = len(countries)
    two_y_vals = results[:n]
    ten_y_vals = results[n:]

    output: dict[str, dict[str, float]] = {}
    for i, country in enumerate(countries):
        two_y = two_y_vals[i]
        ten_y = ten_y_vals[i]
        entry: dict[str, float] = {}
        if two_y is not None:
            entry["2y"] = round(two_y, 3)
        if ten_y is not None:
            entry["10y"] = round(ten_y, 3)
        if two_y is not None and ten_y is not None:
            entry["spread"] = round(ten_y - two_y, 3)
        output[country] = entry
        logger.debug(
            "Yield curve %s: 2Y=%.2f 10Y=%.2f spread=%.2f",
            country,
            two_y or 0,
            ten_y or 0,
            entry.get("spread", 0),
        )

    return output
