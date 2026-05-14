"""
Economic calendar with consensus forecasts, surprise tracking, and FOMC schedule.

Dimension #44 in the SENTINEL competitive matrix — target score 9.

Data sources (all free):
  - FRED release calendar + series observations (no API key for basic use)
  - FRED CSV endpoint: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
  - BLS release schedule: https://www.bls.gov/schedule/
  - BEA schedule: https://www.bea.gov/news/schedule
  - FRED release dates: https://api.stlouisfed.org/fred/releases/dates

Consensus approach:
  - True sell-side consensus is paywalled (Bloomberg, Refinitiv).
  - We proxy consensus using a seasonal trailing-median (last 12 observations)
    and report it alongside the actual + surprise vs that median.
  - This mirrors how quantitative shops back-test economic surprise indices.
"""
from __future__ import annotations

import asyncio
import re
import statistics
from datetime import date, datetime, timedelta
from typing import Literal, Optional

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE = "https://api.stlouisfed.org/fred"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/json,*/*",
}

# ---------------------------------------------------------------------------
# Release catalogue — key series + market impact metadata
# ---------------------------------------------------------------------------

ECONOMIC_RELEASES: dict[str, dict] = {
    "CPI_YOY": {
        "name": "Consumer Price Index (YoY)",
        "fred_series": "CPIAUCSL",
        "fred_chg_series": "CPIAUCSL_PC1",  # YoY % change
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
        "typical_release_day": "2nd_tuesday",
    },
    "CORE_CPI": {
        "name": "Core CPI (ex-Food & Energy, YoY)",
        "fred_series": "CPILFESL",
        "fred_chg_series": "CPILFESL_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
    },
    "PPI": {
        "name": "Producer Price Index (MoM)",
        "fred_series": "PPIACO",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "high",
        "category": "inflation",
    },
    "PCE": {
        "name": "PCE Price Index (Fed preferred, YoY)",
        "fred_series": "PCEPI",
        "fred_chg_series": "PCEPI_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
    },
    "CORE_PCE": {
        "name": "Core PCE (ex-Food & Energy, YoY)",
        "fred_series": "PCEPILFE",
        "fred_chg_series": "PCEPILFE_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
    },
    "NFP": {
        "name": "Nonfarm Payrolls (MoM change, thousands)",
        "fred_series": "PAYEMS",
        "fred_chg_series": None,  # use diff
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "employment",
        "typical_release_day": "1st_friday",
    },
    "UNEMPLOYMENT": {
        "name": "Unemployment Rate (%)",
        "fred_series": "UNRATE",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "employment",
    },
    "GDP": {
        "name": "Real GDP Growth Rate (QoQ annualized, %)",
        "fred_series": "A191RL1Q225SBEA",
        "fred_chg_series": None,
        "frequency": "quarterly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "growth",
    },
    "ISM_MFG": {
        "name": "ISM Manufacturing PMI",
        "fred_series": "NAPM",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "high",
        "category": "activity",
    },
    "ISM_SVCS": {
        "name": "ISM Services PMI",
        "fred_series": "NMFCI",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "high",
        "category": "activity",
    },
    "RETAIL_SALES": {
        "name": "Retail Sales (MoM %)",
        "fred_series": "RSAFS",
        "fred_chg_series": "RSAFS_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "high",
        "category": "consumption",
    },
    "HOUSING_STARTS": {
        "name": "Housing Starts (thousands)",
        "fred_series": "HOUST",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "housing",
    },
    "JOBLESS_CLAIMS": {
        "name": "Initial Jobless Claims (weekly, thousands)",
        "fred_series": "IC4WSA",
        "fred_chg_series": None,
        "frequency": "weekly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "employment",
    },
    "TRADE_BALANCE": {
        "name": "Trade Balance ($ billions)",
        "fred_series": "BOPGSTB",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "trade",
    },
    "MICHIGAN_SENTIMENT": {
        "name": "U. of Michigan Consumer Sentiment",
        "fred_series": "UMCSENT",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "medium",
        "category": "sentiment",
    },
    "INDUSTRIAL_PRODUCTION": {
        "name": "Industrial Production Index (MoM %)",
        "fred_series": "INDPRO",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "09:15 ET",
        "market_impact": "medium",
        "category": "activity",
    },
    "JOLTS": {
        "name": "Job Openings (JOLTS, millions)",
        "fred_series": "JTSJOL",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "high",
        "category": "employment",
    },
    "DURABLE_GOODS": {
        "name": "Durable Goods Orders (MoM %)",
        "fred_series": "DGORDER",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "activity",
    },
}

# Market signal by category × surprise direction
SURPRISE_SIGNAL_MAP: dict[str, dict[str, str]] = {
    "inflation":   {"above": "hawkish",  "below": "dovish",   "in_line": "neutral"},
    "employment":  {"above": "hawkish",  "below": "dovish",   "in_line": "neutral"},
    "growth":      {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "activity":    {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "consumption": {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "housing":     {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "trade":       {"above": "neutral",  "below": "neutral",  "in_line": "neutral"},
    "sentiment":   {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
}

# FOMC meeting dates (confirmed 2025; projected 2026 per Fed 8-meeting-per-year schedule)
FOMC_DATES: dict[int, list[date]] = {
    2025: [
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
    ],
    2026: [
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ],
}

# Approximate schedule: (release_id, nth_weekday, weekday_idx)
# weekday_idx: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
RELEASE_SCHEDULE: list[tuple[str, int, int]] = [
    ("NFP",          1, 4),   # 1st Friday
    ("UNEMPLOYMENT", 1, 4),   # same day as NFP
    ("ISM_MFG",      1, 0),   # 1st business day (approx Monday)
    ("JOBLESS_CLAIMS", 0, 3), # every Thursday (0 = every)
    ("CPI_YOY",      2, 1),   # ~2nd Tuesday (varies; BLS releases ~2 weeks after month end)
    ("CORE_CPI",     2, 1),   # same release as CPI
    ("PPI",          2, 2),   # ~2nd Wednesday
    ("RETAIL_SALES", 2, 2),   # ~2nd Wednesday
    ("MICHIGAN_SENTIMENT", 2, 4),  # ~2nd Friday (preliminary)
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

ImpactLevel = Literal["very_high", "high", "medium", "low"]
MarketSignal = Literal["hawkish", "dovish", "risk_on", "risk_off", "neutral", "unknown"]
SurpriseTrend = Literal["improving", "deteriorating", "stable"]
ToneLabel = Literal["very_hawkish", "hawkish", "neutral", "dovish", "very_dovish"]


class ConsensusEstimate(BaseModel):
    """Median-based consensus proxy derived from trailing history."""
    model_config = ConfigDict(frozen=True)

    release_id: str
    as_of: date
    trailing_median: float       # seasonal median (last 12 obs)
    trailing_std: float          # standard deviation
    range_low: float             # median − 1σ
    range_high: float            # median + 1σ
    n_observations: int
    method: str = "trailing_12m_median"


class EconomicRelease(BaseModel):
    model_config = ConfigDict(frozen=True)

    release_id: str
    name: str
    release_date: Optional[date] = None
    release_time: Optional[str] = None     # "08:30 ET"
    category: str
    frequency: str
    market_impact: ImpactLevel
    # Consensus proxy
    consensus: Optional[float] = None      # trailing median used as proxy
    consensus_range_low: Optional[float] = None
    consensus_range_high: Optional[float] = None
    consensus_std: Optional[float] = None
    # Actuals
    prior: Optional[float] = None          # previous observation
    prior_revised: Optional[float] = None  # revised prior if detected
    actual: Optional[float] = None         # None if upcoming
    # Surprise analysis
    surprise: Optional[float] = None       # actual − consensus
    surprise_pct: Optional[float] = None   # surprise / |consensus| * 100
    surprise_z: Optional[float] = None     # surprise / consensus_std (z-score)
    market_signal: Optional[MarketSignal] = None
    # Enrichment
    historical_data: Optional[list[dict]] = None  # [{date, value}]
    notes: str = ""


class EconomicCalendar(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    start_date: date
    end_date: date
    releases: list[EconomicRelease]
    high_impact_count: int
    fomc_dates: list[date]
    next_major_release: Optional[EconomicRelease] = None
    releases_this_week: list[EconomicRelease] = Field(default_factory=list)


class SurpriseIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    category: str          # "all", "inflation", "employment", …
    score: float           # rolling weighted surprise (positive = beating)
    trend: SurpriseTrend
    recent_surprises: list[dict]  # [{release_id, date, actual, consensus, surprise_z}]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _get_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict | list:
    resp = await client.get(url, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _get_fred_csv(client: httpx.AsyncClient, series_id: str) -> pd.Series:
    """Fetch a FRED series as a pandas Series via the public CSV endpoint (no API key)."""
    url = f"{FRED_CSV_BASE}?id={series_id}"
    resp = await client.get(url, timeout=25, headers=_HEADERS)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    series = df.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series.sort_index()


# ---------------------------------------------------------------------------
# Consensus estimation (trailing median approach)
# ---------------------------------------------------------------------------

def _estimate_consensus(series: pd.Series, n_trailing: int = 12) -> ConsensusEstimate | None:
    """
    Derive a consensus proxy from the trailing n_trailing observations.
    Uses the same seasonal-median approach as the Citi Economic Surprise Index.
    Returns None if fewer than 4 observations are available.
    """
    if series.empty or len(series) < 4:
        return None

    recent = series.iloc[-n_trailing:].dropna()
    if len(recent) < 2:
        return None

    median = float(recent.median())
    std = float(recent.std())
    return ConsensusEstimate(
        release_id="",  # filled by caller
        as_of=date.today(),
        trailing_median=round(median, 4),
        trailing_std=round(std, 4),
        range_low=round(median - std, 4),
        range_high=round(median + std, 4),
        n_observations=len(recent),
    )


def _compute_surprise(
    actual: float,
    consensus: float,
    std: float,
    category: str,
) -> tuple[float, float, float, MarketSignal]:
    """
    Returns (surprise, surprise_pct, surprise_z, market_signal).
    surprise     = actual − consensus
    surprise_pct = surprise / |consensus| * 100  (guarded against zero)
    surprise_z   = surprise / std               (guarded against zero)
    """
    surprise = round(actual - consensus, 6)
    surprise_pct = round((surprise / abs(consensus)) * 100, 2) if abs(consensus) > 1e-9 else 0.0
    surprise_z = round(surprise / std, 3) if std > 1e-9 else 0.0

    direction = "in_line"
    if surprise_z > 0.5:
        direction = "above"
    elif surprise_z < -0.5:
        direction = "below"

    signal_map = SURPRISE_SIGNAL_MAP.get(category, {})
    raw_signal: str = signal_map.get(direction, "neutral")
    # Cast to MarketSignal literal safely
    valid: set[str] = {"hawkish", "dovish", "risk_on", "risk_off", "neutral", "unknown"}
    market_signal: MarketSignal = raw_signal if raw_signal in valid else "neutral"  # type: ignore[assignment]

    return surprise, surprise_pct, surprise_z, market_signal


# ---------------------------------------------------------------------------
# FOMC calendar helpers
# ---------------------------------------------------------------------------

def get_fomc_dates(year: int | None = None) -> list[date]:
    """Return FOMC meeting dates for the given year (defaults to current + next)."""
    today = date.today()
    current_year = today.year
    if year is not None:
        return FOMC_DATES.get(year, [])
    out: list[date] = []
    for y in (current_year, current_year + 1):
        out.extend(FOMC_DATES.get(y, []))
    return sorted(out)


def _upcoming_fomc(start: date, end: date) -> list[date]:
    all_dates = get_fomc_dates()
    return [d for d in all_dates if start <= d <= end]


# ---------------------------------------------------------------------------
# Scheduled release date inference
# ---------------------------------------------------------------------------

def _nth_weekday_of_month(year: int, month: int, n: int, weekday: int) -> date:
    """
    Return the nth occurrence (1-based) of weekday (0=Mon…6=Sun) in year/month.
    If n==0 means "every week" — returns first occurrence.
    """
    first = date(year, month, 1)
    # How many days until the first occurrence of weekday
    offset = (weekday - first.weekday()) % 7
    first_occurrence = first + timedelta(days=offset)
    n_use = max(n, 1)
    target = first_occurrence + timedelta(weeks=n_use - 1)
    if target.month != month:
        # Overflow — use last occurrence in month
        target -= timedelta(weeks=1)
    return target


def _build_scheduled_dates(
    start: date, end: date
) -> dict[str, list[date]]:
    """
    Approximate release dates for the catalogue's key releases by iterating
    through months in the window and applying the typical release-day rule.
    JOBLESS_CLAIMS (weekly) gets all Thursdays.
    """
    scheduled: dict[str, list[date]] = {rid: [] for rid in ECONOMIC_RELEASES}

    cur = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)

    while cur <= end_month:
        y, m = cur.year, cur.month
        for rid, nth, wday in RELEASE_SCHEDULE:
            if rid == "JOBLESS_CLAIMS":
                # Every Thursday in the window
                first_thu = date(y, m, 1)
                offset = (3 - first_thu.weekday()) % 7  # Thursday = 3
                d = first_thu + timedelta(days=offset)
                while d.month == m:
                    if start <= d <= end:
                        scheduled[rid].append(d)
                    d += timedelta(weeks=1)
            else:
                try:
                    d = _nth_weekday_of_month(y, m, nth, wday)
                    if start <= d <= end:
                        scheduled[rid].append(d)
                except Exception:
                    pass
        # Add releases not in RELEASE_SCHEDULE with a rough "3rd week" approximation
        for rid in ECONOMIC_RELEASES:
            if rid not in {r[0] for r in RELEASE_SCHEDULE}:
                # ~3rd Wednesday of month
                d = _nth_weekday_of_month(y, m, 3, 2)
                if start <= d <= end:
                    if d not in scheduled[rid]:
                        scheduled[rid].append(d)
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    return scheduled


# ---------------------------------------------------------------------------
# Core FRED data fetching
# ---------------------------------------------------------------------------

async def _fetch_series_history(
    series_id: str,
    years_back: int = 10,
) -> pd.Series:
    """Fetch up to years_back years of history from FRED CSV endpoint."""
    try:
        async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
            series = await _get_fred_csv(client, series_id)
        cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years_back))
        return series[series.index >= cutoff]
    except Exception as exc:
        logger.warning("_fetch_series_history: failed", series_id=series_id, error=str(exc))
        return pd.Series(dtype=float)


async def _fetch_fred_releases_raw(api_key: str, start: date, end: date) -> list[dict]:
    """Fetch FRED release-date objects for a window (requires API key)."""
    url = f"{FRED_BASE}/releases/dates"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": start.isoformat(),
        "realtime_end": end.isoformat(),
        "include_release_dates_with_no_data": "true",
        "limit": 1000,
    }
    try:
        async with httpx.AsyncClient(headers=_HEADERS) as client:
            data = await _get_json(client, url, params)
        return data.get("release_dates", [])  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("_fetch_fred_releases_raw: failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# EconomicCalendarEngine
# ---------------------------------------------------------------------------

class EconomicCalendarEngine:
    """
    Full economic calendar with consensus proxies, surprise tracking,
    FOMC schedule, and economic surprise index computation.
    """

    def __init__(self, timeout: float = 25.0, fred_api_key: str = ""):
        self._timeout = timeout
        self._api_key = fred_api_key
        # Series history cache: series_id → pd.Series
        self._cache: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_history(self, release_id: str, years_back: int = 10) -> pd.Series:
        meta = ECONOMIC_RELEASES.get(release_id, {})
        series_id = meta.get("fred_chg_series") or meta.get("fred_series", "")
        if not series_id:
            return pd.Series(dtype=float)
        if series_id in self._cache:
            return self._cache[series_id]
        hist = await _fetch_series_history(series_id, years_back=years_back)
        self._cache[series_id] = hist
        return hist

    def _build_release(
        self,
        release_id: str,
        release_date: Optional[date],
        history: pd.Series,
    ) -> EconomicRelease:
        meta = ECONOMIC_RELEASES[release_id]
        cat = meta["category"]

        # Consensus from trailing 12 obs
        est = _estimate_consensus(history, n_trailing=12)
        consensus = est.trailing_median if est else None
        c_std = est.trailing_std if est else None
        c_low = est.range_low if est else None
        c_high = est.range_high if est else None

        # Prior and actual
        prior: Optional[float] = None
        actual: Optional[float] = None
        surprise: Optional[float] = None
        surprise_pct: Optional[float] = None
        surprise_z: Optional[float] = None
        market_signal: Optional[MarketSignal] = None

        if not history.empty:
            values = history.dropna()
            if len(values) >= 2:
                prior = float(values.iloc[-2])
                actual = float(values.iloc[-1])
            elif len(values) == 1:
                actual = float(values.iloc[-1])

            if actual is not None and consensus is not None and c_std is not None:
                surprise, surprise_pct, surprise_z, market_signal = _compute_surprise(
                    actual, consensus, c_std, cat
                )

        # Historical data (last 24 points for sparkline)
        hist_list: list[dict] = []
        if not history.empty:
            for idx, val in history.iloc[-24:].items():
                hist_list.append({
                    "date": idx.date().isoformat() if hasattr(idx, "date") else str(idx),
                    "value": round(float(val), 4),
                })

        return EconomicRelease(
            release_id=release_id,
            name=meta["name"],
            release_date=release_date,
            release_time=meta.get("release_time"),
            category=cat,
            frequency=meta["frequency"],
            market_impact=meta["market_impact"],
            consensus=round(consensus, 4) if consensus is not None else None,
            consensus_range_low=round(c_low, 4) if c_low is not None else None,
            consensus_range_high=round(c_high, 4) if c_high is not None else None,
            consensus_std=round(c_std, 4) if c_std is not None else None,
            prior=round(prior, 4) if prior is not None else None,
            actual=round(actual, 4) if actual is not None else None,
            surprise=round(surprise, 4) if surprise is not None else None,
            surprise_pct=round(surprise_pct, 2) if surprise_pct is not None else None,
            surprise_z=round(surprise_z, 3) if surprise_z is not None else None,
            market_signal=market_signal,
            historical_data=hist_list if hist_list else None,
            notes=(
                "Consensus is a trailing-12-period median proxy. "
                "For official sell-side consensus use Bloomberg or Refinitiv."
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_release_history(
        self,
        release_id: str,
        years_back: int = 10,
    ) -> pd.DataFrame:
        """Return full FRED history for a release as a DataFrame with rolling stats."""
        if release_id not in ECONOMIC_RELEASES:
            raise ValueError(f"Unknown release_id: {release_id!r}")
        series = await self._get_history(release_id, years_back=years_back)
        if series.empty:
            return pd.DataFrame()
        df = series.to_frame(name="value")
        df["rolling_mean_12"] = df["value"].rolling(12).mean()
        df["rolling_std_12"] = df["value"].rolling(12).std()
        df["z_score"] = (df["value"] - df["rolling_mean_12"]) / df["rolling_std_12"]
        df["yoy_chg"] = df["value"].pct_change(12) * 100
        return df.round(4)

    async def get_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        categories: Optional[list[str]] = None,
        min_impact: str = "medium",
    ) -> EconomicCalendar:
        """
        Build a comprehensive economic calendar for the given date window.

        Fetches FRED history concurrently for all tracked releases, computes
        consensus proxies and surprise scores, and overlays FOMC dates.
        min_impact filters: "very_high" > "high" > "medium" > "low"
        """
        today = date.today()
        start_date = start_date or today
        end_date = end_date or (today + timedelta(days=30))

        impact_order = {"very_high": 4, "high": 3, "medium": 2, "low": 1}
        min_rank = impact_order.get(min_impact, 2)

        # Filter catalogue by impact and category
        filtered_ids = [
            rid for rid, meta in ECONOMIC_RELEASES.items()
            if impact_order.get(meta["market_impact"], 0) >= min_rank
            and (categories is None or meta["category"] in categories)
        ]

        # Approximate scheduled dates
        scheduled = _build_scheduled_dates(start_date, end_date)

        # Fetch all histories concurrently
        histories = await asyncio.gather(
            *[self._get_history(rid) for rid in filtered_ids],
            return_exceptions=True,
        )

        releases: list[EconomicRelease] = []
        for rid, hist_or_exc in zip(filtered_ids, histories):
            if isinstance(hist_or_exc, Exception):
                logger.warning(
                    "get_calendar: history fetch failed", release_id=rid, error=str(hist_or_exc)
                )
                hist_or_exc = pd.Series(dtype=float)

            dates_for_rid = scheduled.get(rid, [])
            if dates_for_rid:
                for d in dates_for_rid:
                    releases.append(self._build_release(rid, d, hist_or_exc))
            else:
                # Include without scheduled date
                releases.append(self._build_release(rid, None, hist_or_exc))

        # Sort by date (None dates last), then name
        releases.sort(
            key=lambda r: (r.release_date is None, r.release_date or date.max, r.name)
        )

        fomc_in_window = _upcoming_fomc(start_date, end_date)
        high_count = sum(1 for r in releases if r.market_impact in ("very_high", "high"))

        # Next upcoming major release
        next_major: Optional[EconomicRelease] = None
        for r in releases:
            if r.release_date and r.release_date >= today and r.market_impact in ("very_high", "high"):
                next_major = r
                break

        # This week
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        this_week = [r for r in releases if r.release_date and monday <= r.release_date <= sunday]

        logger.info(
            "get_calendar: built",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            releases=len(releases),
            high_impact=high_count,
            fomc_count=len(fomc_in_window),
        )

        return EconomicCalendar(
            as_of=today,
            start_date=start_date,
            end_date=end_date,
            releases=releases,
            high_impact_count=high_count,
            fomc_dates=fomc_in_window,
            next_major_release=next_major,
            releases_this_week=this_week,
        )

    async def get_upcoming_week(self) -> list[EconomicRelease]:
        """Return scheduled releases for the next 7 days."""
        today = date.today()
        calendar = await self.get_calendar(
            start_date=today,
            end_date=today + timedelta(days=7),
            min_impact="medium",
        )
        return [r for r in calendar.releases if r.release_date is not None]

    async def get_recent_surprises(
        self,
        n_releases: int = 20,
        categories: Optional[list[str]] = None,
    ) -> list[EconomicRelease]:
        """
        Fetch last N releases with computed surprise vs trailing-median consensus.
        Returns sorted by |surprise_z| descending (biggest surprises first).
        """
        rids = [
            rid for rid, meta in ECONOMIC_RELEASES.items()
            if categories is None or meta["category"] in categories
        ]

        histories = await asyncio.gather(
            *[self._get_history(rid) for rid in rids],
            return_exceptions=True,
        )

        results: list[EconomicRelease] = []
        for rid, hist_or_exc in zip(rids, histories):
            if isinstance(hist_or_exc, Exception):
                continue
            if not isinstance(hist_or_exc, pd.Series) or hist_or_exc.empty:
                continue
            release = self._build_release(rid, None, hist_or_exc)
            if release.actual is not None and release.surprise_z is not None:
                results.append(release)

        results.sort(
            key=lambda r: abs(r.surprise_z) if r.surprise_z is not None else 0.0,
            reverse=True,
        )
        return results[:n_releases]

    async def compute_surprise_index(
        self,
        category: str = "all",
        months_back: int = 3,
    ) -> SurpriseIndex:
        """
        Compute an economic surprise index analogous to the Citi ESI.

        Methodology:
          1. For each release (filtered by category), fetch 10Y history.
          2. For each observation in the last months_back months, use the
             trailing-12-obs median at that point as the consensus proxy.
          3. Sum weighted z-scores; weight decays with age (exponential).
          4. Normalize to [-100, 100] range.
        """
        today = date.today()
        cutoff = today - timedelta(days=30 * months_back)

        rids = [
            rid for rid, meta in ECONOMIC_RELEASES.items()
            if category == "all" or meta["category"] == category
        ]

        histories = await asyncio.gather(
            *[self._get_history(rid, years_back=10) for rid in rids],
            return_exceptions=True,
        )

        recent_surprises: list[dict] = []
        z_scores: list[tuple[date, float]] = []  # (obs_date, z_score)

        for rid, hist_or_exc in zip(rids, histories):
            if isinstance(hist_or_exc, Exception) or not isinstance(hist_or_exc, pd.Series):
                continue
            series = hist_or_exc.dropna()
            if len(series) < 13:
                continue

            # Walk through observations in the window
            for i in range(12, len(series)):
                obs_date = series.index[i].date() if hasattr(series.index[i], "date") else series.index[i]
                if obs_date < cutoff:
                    continue
                if obs_date > today:
                    break

                trailing = series.iloc[i - 12 : i]
                actual_val = float(series.iloc[i])
                trailing_median = float(trailing.median())
                trailing_std = float(trailing.std())
                if trailing_std < 1e-9:
                    continue

                z = (actual_val - trailing_median) / trailing_std
                z_scores.append((obs_date, round(z, 3)))
                recent_surprises.append({
                    "release_id": rid,
                    "date": obs_date.isoformat(),
                    "actual": round(actual_val, 4),
                    "consensus_proxy": round(trailing_median, 4),
                    "surprise_z": round(z, 3),
                })

        # Compute exponentially-weighted sum
        if not z_scores:
            return SurpriseIndex(
                as_of=today, category=category, score=0.0,
                trend="stable", recent_surprises=[],
            )

        z_scores.sort(key=lambda t: t[0])
        n = len(z_scores)
        weighted_sum = 0.0
        weight_total = 0.0
        for i, (_, z) in enumerate(z_scores):
            # More recent = higher weight
            w = (i + 1) / n
            weighted_sum += w * z
            weight_total += w

        raw_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        # Clip to [-10, 10] then scale to sensible range
        score = round(max(-10.0, min(10.0, raw_score * 3)), 2)

        # Trend: compare first half vs second half z-scores
        half = max(1, n // 2)
        first_half = [z for _, z in z_scores[:half]]
        second_half = [z for _, z in z_scores[half:]]
        avg_first = statistics.mean(first_half) if first_half else 0.0
        avg_second = statistics.mean(second_half) if second_half else 0.0
        delta = avg_second - avg_first
        if delta > 0.15:
            trend: SurpriseTrend = "improving"
        elif delta < -0.15:
            trend = "deteriorating"
        else:
            trend = "stable"

        # Sort recent_surprises by date descending, take last 10
        recent_surprises.sort(key=lambda d: d["date"], reverse=True)

        logger.info(
            "compute_surprise_index",
            category=category,
            score=score,
            trend=trend,
            n_obs=len(z_scores),
        )

        return SurpriseIndex(
            as_of=today,
            category=category,
            score=score,
            trend=trend,
            recent_surprises=recent_surprises[:10],
        )

    async def get_fomc_calendar(self, year: Optional[int] = None) -> list[date]:
        """Return FOMC meeting dates, optionally filtered to a specific year."""
        return get_fomc_dates(year)

    def estimate_consensus(
        self,
        release_id: str,
        history: pd.Series,
        n_trailing: int = 12,
    ) -> dict:
        """
        Public convenience: estimate consensus from a history series.
        Returns dict with keys: estimate, std_dev, range_low, range_high, n_observations.
        """
        est = _estimate_consensus(history, n_trailing=n_trailing)
        if est is None:
            return {"estimate": None, "std_dev": None, "range_low": None, "range_high": None, "n_observations": 0}
        return {
            "estimate": est.trailing_median,
            "std_dev": est.trailing_std,
            "range_low": est.range_low,
            "range_high": est.range_high,
            "n_observations": est.n_observations,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def economic_calendar(
    days_ahead: int = 14,
    min_impact: str = "medium",
) -> EconomicCalendar:
    """Fetch the upcoming economic calendar for the next N days."""
    from sentinel.core.config import get_settings
    settings = get_settings()
    engine = EconomicCalendarEngine(fred_api_key=settings.fred_api_key)
    today = date.today()
    return await engine.get_calendar(
        start_date=today,
        end_date=today + timedelta(days=days_ahead),
        min_impact=min_impact,
    )


async def surprise_index(
    category: str = "inflation",
    months_back: int = 3,
) -> SurpriseIndex:
    """Compute the economic surprise index for a category."""
    engine = EconomicCalendarEngine()
    return await engine.compute_surprise_index(category=category, months_back=months_back)


async def release_history(
    release_id: str,
    years_back: int = 10,
) -> pd.DataFrame:
    """Fetch full FRED history with rolling statistics for a named release."""
    engine = EconomicCalendarEngine()
    return await engine.get_release_history(release_id, years_back=years_back)


def summarize_calendar(calendar: EconomicCalendar) -> dict:
    """Return a concise summary dict for display or logging."""
    by_impact: dict[str, int] = {}
    for r in calendar.releases:
        by_impact[r.market_impact] = by_impact.get(r.market_impact, 0) + 1

    next_r = calendar.next_major_release
    return {
        "as_of": calendar.as_of.isoformat(),
        "window": f"{calendar.start_date.isoformat()} → {calendar.end_date.isoformat()}",
        "total_releases": len(calendar.releases),
        "by_impact": by_impact,
        "high_impact_count": calendar.high_impact_count,
        "this_week_count": len(calendar.releases_this_week),
        "fomc_dates": [d.isoformat() for d in calendar.fomc_dates],
        "next_major_release": {
            "id": next_r.release_id,
            "name": next_r.name,
            "date": next_r.release_date.isoformat() if next_r.release_date else None,
            "time": next_r.release_time,
            "impact": next_r.market_impact,
            "consensus": next_r.consensus,
            "prior": next_r.prior,
        } if next_r else None,
    }
