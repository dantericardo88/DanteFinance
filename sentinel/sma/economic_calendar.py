"""
Economic calendar and release intelligence — upcoming macro data releases with importance scoring.

Data sources (all free):
  - FRED release calendar: https://api.stlouisfed.org/fred/releases/dates
  - FRED releases metadata: https://api.stlouisfed.org/fred/releases

Consensus estimates are NOT available on the free tier. The `consensus` field will always be None.
Prior values are not directly available from FRED release calendar endpoints; fetch via
series observation endpoints if needed (see fetch_prior_value).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"

# ── Importance tiers ──────────────────────────────────────────────────────────

HIGH_IMPACT_RELEASES: set[str] = {
    "fomc rate decision",
    "nonfarm payrolls",
    "cpi",
    "consumer price index",
    "ppi",
    "producer price index",
    "gdp",
    "gross domestic product",
    "retail sales",
    "ism manufacturing",
    "ism services",
    "jolts",
    "job openings",
    "initial jobless claims",
    "pce",
    "personal consumption expenditures",
    "core pce",
    "durable goods",
    "housing starts",
    "consumer confidence",
    "trade balance",
    "industrial production",
    "personal income",
    "michigan sentiment",
    "consumer sentiment",
    "adp employment",
    "adp national employment",
}

MEDIUM_IMPACT_RELEASES: set[str] = {
    "existing home sales",
    "new home sales",
    "building permits",
    "capacity utilization",
    "chicago pmi",
    "philadelphia fed",
    "empire state",
    "factory orders",
    "wholesale inventories",
    "business inventories",
    "current account",
    "import price",
    "export price",
    "leading indicators",
    "beige book",
    "non-farm productivity",
    "unit labor costs",
}

# Hardcoded release-to-FRED-series mapping for key releases
RELEASE_SERIES_MAP: dict[str, list[str]] = {
    "Consumer Price Index": ["CPIAUCSL", "CPILFESL"],
    "Producer Price Index": ["PPIACO", "PPIFID"],
    "Gross Domestic Product": ["GDP", "GDPC1", "GDPDEF"],
    "Advance Monthly Sales for Retail and Food Services": ["RSAFS", "RSXFS"],
    "Retail and Food Services Sales": ["RSAFS", "RSXFS"],
    "ISM Manufacturing": ["MANEMP", "NAPM"],
    "ISM Services": ["NMFCI"],
    "Job Openings and Labor Turnover Survey": ["JTSJOL", "JTSHIR"],
    "Unemployment Insurance Weekly Claims Report": ["ICSA", "CCSA"],
    "Personal Income and Outlays": ["PI", "PCE", "PCEPI", "PCEPILFE"],
    "Manufacturers' Shipments, Inventories, and Orders": ["DGORDER"],
    "New Residential Construction": ["HOUST", "PERMIT"],
    "The Conference Board Consumer Confidence Index": ["CONCCONF"],
    "U.S. International Trade in Goods and Services": ["BOPGSTB"],
    "Industrial Production and Capacity Utilization": ["INDPRO", "TCU"],
    "University of Michigan Consumer Sentiment": ["UMCSENT"],
    "ADP National Employment Report": ["PAYEMS"],
    "Nonfarm Payroll Employment": ["PAYEMS", "UNRATE", "CES0500000003"],
    "Employment Situation": ["PAYEMS", "UNRATE", "CIVPART"],
}

# Known release times (ET) for major releases
RELEASE_TIME_MAP: dict[str, str] = {
    "Employment Situation": "08:30 ET",
    "Nonfarm Payroll": "08:30 ET",
    "Consumer Price Index": "08:30 ET",
    "Producer Price Index": "08:30 ET",
    "Gross Domestic Product": "08:30 ET",
    "Personal Income and Outlays": "08:30 ET",
    "Advance Monthly Sales": "08:30 ET",
    "Retail and Food Services": "08:30 ET",
    "Manufacturers' Shipments": "10:00 ET",
    "Durable Goods": "08:30 ET",
    "New Residential Construction": "08:30 ET",
    "Existing-Home Sales": "10:00 ET",
    "New Home Sales": "10:00 ET",
    "Industrial Production": "09:15 ET",
    "University of Michigan": "10:00 ET",
    "Consumer Confidence": "10:00 ET",
    "ISM Manufacturing": "10:00 ET",
    "ISM Services": "10:00 ET",
    "Trade in Goods and Services": "08:30 ET",
    "Unemployment Insurance Weekly": "08:30 ET",
    "Job Openings": "10:00 ET",
    "ADP National Employment": "08:15 ET",
    "FOMC": "14:00 ET",
    "Beige Book": "14:00 ET",
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class EconomicRelease(BaseModel):
    model_config = ConfigDict(frozen=True)

    release_id: str
    name: str
    release_date: date
    release_time: Optional[str]         # "08:30 ET" where known
    frequency: str                      # "Monthly" | "Quarterly" | "Weekly" etc
    importance: str                     # "high" | "medium" | "low"
    fred_series_ids: list[str]          # key FRED series updated by this release
    consensus: Optional[float]          # Always None — not available on free tier
    prior: Optional[float]              # prior release value if fetchable
    notes: str = ""


class EconomicCalendar(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    releases: list[EconomicRelease]
    next_high_impact: Optional[EconomicRelease]
    releases_this_week: list[EconomicRelease]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_importance(release_name: str) -> str:
    """Return 'high' | 'medium' | 'low' based on name matching against impact lists."""
    name_lower = release_name.lower()
    for keyword in HIGH_IMPACT_RELEASES:
        if keyword in name_lower:
            return "high"
    for keyword in MEDIUM_IMPACT_RELEASES:
        if keyword in name_lower:
            return "medium"
    return "low"


def _lookup_release_time(release_name: str) -> Optional[str]:
    """Return known ET release time for a release name, or None."""
    for key, time_str in RELEASE_TIME_MAP.items():
        if key.lower() in release_name.lower():
            return time_str
    return None


def _lookup_series_ids(release_name: str) -> list[str]:
    """Return known FRED series IDs associated with a release name."""
    for key, series in RELEASE_SERIES_MAP.items():
        if key.lower() in release_name.lower() or release_name.lower() in key.lower():
            return series
    return []


def _this_week_window() -> tuple[date, date]:
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


# ── FRED HTTP helpers ─────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _get_fred(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    """GET a FRED API endpoint with retry."""
    resp = await client.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


async def _fetch_all_releases(api_key: str) -> dict[str, dict]:
    """
    Fetch all FRED release metadata (id → {name, link, frequency}).
    Returns a dict keyed by release_id string.
    """
    url = f"{FRED_BASE}/releases"
    params = {"api_key": api_key, "file_type": "json", "limit": 1000}
    releases: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient() as client:
            data = await _get_fred(client, url, params)
        for r in data.get("releases", []):
            releases[str(r["id"])] = {
                "name": r.get("name", ""),
                "link": r.get("link", ""),
                "frequency": _normalize_frequency(r.get("frequency", "")),
            }
        logger.info("FRED releases metadata loaded", count=len(releases))
    except Exception as exc:
        logger.error("FRED releases metadata failed", error=str(exc))
    return releases


async def _fetch_release_dates(
    api_key: str,
    start: date,
    end: date,
) -> list[dict]:
    """
    Fetch scheduled FRED release dates between start and end.
    Returns list of {release_id, release_date} dicts.
    """
    url = f"{FRED_BASE}/releases/dates"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": start.isoformat(),
        "realtime_end": end.isoformat(),
        "include_release_dates_with_no_data": "true",
        "limit": 1000,
    }
    release_dates: list[dict] = []
    try:
        async with httpx.AsyncClient() as client:
            data = await _get_fred(client, url, params)
        release_dates = data.get("release_dates", [])
        logger.info(
            "FRED release dates fetched",
            start=start.isoformat(),
            end=end.isoformat(),
            count=len(release_dates),
        )
    except Exception as exc:
        logger.error("FRED release dates failed", error=str(exc))
    return release_dates


def _normalize_frequency(raw: str) -> str:
    """Map FRED frequency strings to clean display values."""
    mapping = {
        "Annual": "Annual",
        "Semiannual": "Semiannual",
        "Quarterly": "Quarterly",
        "Monthly": "Monthly",
        "Biweekly": "Biweekly",
        "Weekly": "Weekly",
        "Daily": "Daily",
        "Weekly, Ending Friday": "Weekly",
        "Weekly, Ending Thursday": "Weekly",
        "Weekly, Ending Wednesday": "Weekly",
        "Weekly, Ending Tuesday": "Weekly",
        "Weekly, Ending Monday": "Weekly",
        "Weekly, Ending Saturday": "Weekly",
        "Weekly, Ending Sunday": "Weekly",
        "Bimonthly": "Bimonthly",
        "Irregular": "Irregular",
        "Not Applicable": "N/A",
    }
    for key, val in mapping.items():
        if key.lower() in raw.lower():
            return val
    return raw or "Unknown"


async def fetch_prior_value(api_key: str, series_id: str) -> Optional[float]:
    """
    Fetch the most recent observation value for a FRED series (prior release value).
    Returns None on failure.
    """
    url = f"{FRED_BASE}/series/observations"
    params = {
        "api_key": api_key,
        "series_id": series_id,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 2,
    }
    try:
        async with httpx.AsyncClient() as client:
            data = await _get_fred(client, url, params)
        observations = data.get("observations", [])
        # observations[0] = most recent (current), observations[1] = prior
        for obs in observations[1:]:
            val = obs.get("value", ".")
            if val and val != ".":
                return float(val)
    except Exception as exc:
        logger.warning("Prior value fetch failed", series_id=series_id, error=str(exc))
    return None


# ── Main fetch function ───────────────────────────────────────────────────────

async def fetch_fred_releases(api_key: str, days_ahead: int = 30) -> list[EconomicRelease]:
    """
    Fetch upcoming FRED release dates and map to EconomicRelease objects.
    Fetches metadata and scheduled dates in parallel, then joins them.
    """
    today = date.today()
    end = today + timedelta(days=days_ahead)

    # Parallel fetch: metadata + release dates
    meta_task = asyncio.create_task(_fetch_all_releases(api_key))
    dates_task = asyncio.create_task(_fetch_release_dates(api_key, today, end))
    release_meta, release_dates = await asyncio.gather(meta_task, dates_task)

    # Deduplicate: one EconomicRelease per (release_id, release_date)
    seen: set[tuple[str, str]] = set()
    releases: list[EconomicRelease] = []

    for entry in release_dates:
        rid = str(entry.get("release_id", ""))
        rd_str = entry.get("date", "")
        if not rid or not rd_str:
            continue
        key = (rid, rd_str)
        if key in seen:
            continue
        seen.add(key)

        try:
            rd = date.fromisoformat(rd_str)
        except ValueError:
            continue

        meta = release_meta.get(rid, {})
        name = meta.get("name", f"Release {rid}")
        freq = meta.get("frequency", "Unknown")
        importance = score_importance(name)
        series_ids = _lookup_series_ids(name)
        release_time = _lookup_release_time(name)

        releases.append(
            EconomicRelease(
                release_id=rid,
                name=name,
                release_date=rd,
                release_time=release_time,
                frequency=freq,
                importance=importance,
                fred_series_ids=series_ids,
                consensus=None,  # Not available on free tier
                prior=None,      # Populated separately via fetch_prior_value if needed
                notes=(
                    "Consensus estimate not available on free tier. "
                    "Use a paid data provider (Bloomberg, Refinitiv) for consensus data."
                    if importance == "high" else ""
                ),
            )
        )

    releases.sort(key=lambda r: r.release_date)
    logger.info("Economic releases assembled", count=len(releases), days_ahead=days_ahead)
    return releases


async def fetch_fred_releases_with_priors(
    api_key: str,
    days_ahead: int = 30,
    max_prior_lookups: int = 10,
) -> list[EconomicRelease]:
    """
    Like fetch_fred_releases, but also fetches prior values for the top high-impact
    releases (capped at max_prior_lookups to avoid hammering the API).
    """
    releases = await fetch_fred_releases(api_key, days_ahead)

    high_with_series = [
        r for r in releases if r.importance == "high" and r.fred_series_ids
    ][:max_prior_lookups]

    async def enrich(rel: EconomicRelease) -> EconomicRelease:
        prior = await fetch_prior_value(api_key, rel.fred_series_ids[0])
        return rel.model_copy(update={"prior": prior})

    enriched_map: dict[tuple[str, str], EconomicRelease] = {}
    enriched = await asyncio.gather(*[enrich(r) for r in high_with_series], return_exceptions=True)
    for original, result in zip(high_with_series, enriched):
        if isinstance(result, EconomicRelease):
            enriched_map[(original.release_id, original.release_date.isoformat())] = result

    final: list[EconomicRelease] = []
    for r in releases:
        key = (r.release_id, r.release_date.isoformat())
        final.append(enriched_map.get(key, r))

    return final


# ── Calendar assembly ─────────────────────────────────────────────────────────

async def get_calendar(api_key: str, days_ahead: int = 30) -> EconomicCalendar:
    """
    Full economic calendar: fetch upcoming FRED releases, sort by date,
    find the next high-impact event, and extract this week's releases.
    """
    releases = await fetch_fred_releases(api_key, days_ahead)

    today = date.today()
    next_high = next(
        (r for r in releases if r.importance == "high" and r.release_date >= today),
        None,
    )

    week_start, week_end = _this_week_window()
    this_week = [r for r in releases if week_start <= r.release_date <= week_end]

    return EconomicCalendar(
        as_of=today,
        releases=releases,
        next_high_impact=next_high,
        releases_this_week=this_week,
    )


def get_releases_in_window(
    calendar: EconomicCalendar,
    start: date,
    end: date,
) -> list[EconomicRelease]:
    """Filter calendar to a specific date window, inclusive on both ends."""
    return [r for r in calendar.releases if start <= r.release_date <= end]


def get_high_impact_releases(calendar: EconomicCalendar) -> list[EconomicRelease]:
    """Return only high-importance releases from the calendar."""
    return [r for r in calendar.releases if r.importance == "high"]


def summarize_calendar(calendar: EconomicCalendar) -> dict:
    """Return a concise summary dict for display or logging."""
    by_importance = {"high": 0, "medium": 0, "low": 0}
    for r in calendar.releases:
        by_importance[r.importance] = by_importance.get(r.importance, 0) + 1

    next_hi = calendar.next_high_impact
    return {
        "as_of": calendar.as_of.isoformat(),
        "total_releases": len(calendar.releases),
        "by_importance": by_importance,
        "this_week_count": len(calendar.releases_this_week),
        "next_high_impact": {
            "name": next_hi.name,
            "date": next_hi.release_date.isoformat(),
            "release_time": next_hi.release_time,
            "series": next_hi.fred_series_ids,
        } if next_hi else None,
    }
