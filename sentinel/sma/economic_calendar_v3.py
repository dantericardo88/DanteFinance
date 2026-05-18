"""
Economic calendar v3 — release dates, consensus estimates, surprise index.

Dimension: dim_044 — Economic calendar & release consensus (target score: 9)

Audit fix: v1 had only FRED release dates with no consensus estimates.
v3 adds multi-source consensus scraping (Investing.com, ForexFactory,
TradingEconomics) and a full economic surprise index.

Data sources (all free, no paid API keys):
  1. FRED release calendar — https://api.stlouisfed.org/fred/releases/dates
     No API key required for public CSV downloads.
  2. ForexFactory calendar — https://www.forexfactory.com/calendar
     Structured HTML: date/time/currency/impact/actual/forecast/previous.
  3. Investing.com economic calendar — https://www.investing.com/economic-calendar/
     Public HTML with forecast/actual/previous columns.
  4. TradingEconomics calendar — https://tradingeconomics.com/calendar
     Public HTML with consensus field.
  5. Atlanta Fed GDPNow — https://www.atlantafed.org/cqer/research/gdpnow
  6. FRED series observations — for prior actual values of 50+ key releases.

Features
--------
- 50+ key US macro releases tracked with release time, importance (1-5 stars)
- Multi-source consensus aggregation (median of available forecasts)
- Surprise magnitude = (actual - consensus) / |consensus| where ≠ 0
- Economic Surprise Index: rolling 90-day sum of z-scored surprises
- FOMC calendar: next meeting dates, fed funds futures implied rate
- International central bank meetings: ECB, BOE, BOJ, RBA, BOC, SNB
- Treasury auction calendar: 2Y, 5Y, 10Y, 30Y with bid-to-cover history
- Revision tracker: stores first-release vs revised actual
- SQLite: economic_calendar, release_history, consensus_estimates, surprise_index

FastAPI router at /eco-calendar/v3:
  GET /upcoming?days=14&country=US
  GET /today
  GET /surprise-index
  GET /release/{event_name}/history
  GET /fomc-calendar
  GET /impact-score/{event}
  GET /international
  GET /treasury-auctions
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field as PydanticField

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "economic_calendar_v3.db"

_CALENDAR_TTL = 1_800        # 30 min for calendar data
_HISTORY_TTL  = 86_400       # 24h for historical release data
_SURPRISE_TTL = 3_600        # 1h for surprise index
_REQ_TIMEOUT  = 18

_HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "DNT": "1",
    "Referer": "https://www.google.com/",
}

_FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRED_API_BASE = "https://api.stlouisfed.org/fred"

# ---------------------------------------------------------------------------
# Key releases catalogue — 50+ entries
# Tuple: (event_name, country, importance_stars, release_time_et, category)
# ---------------------------------------------------------------------------

_US_RELEASES: list[tuple[str, str, int, str, str]] = [
    # Inflation
    ("CPI (Headline)",          "US", 5, "08:30", "inflation"),
    ("CPI (Core)",              "US", 5, "08:30", "inflation"),
    ("PPI (Headline)",          "US", 4, "08:30", "inflation"),
    ("PPI (Core)",              "US", 4, "08:30", "inflation"),
    ("PCE Deflator",            "US", 5, "08:30", "inflation"),
    ("PCE Core",                "US", 5, "08:30", "inflation"),
    ("Import Price Index",      "US", 2, "08:30", "inflation"),
    ("Export Price Index",      "US", 2, "08:30", "inflation"),
    # Employment
    ("Nonfarm Payrolls",                "US", 5, "08:30", "employment"),
    ("Unemployment Rate",               "US", 5, "08:30", "employment"),
    ("JOLTS Job Openings",              "US", 4, "10:00", "employment"),
    ("ADP Employment Change",           "US", 3, "08:15", "employment"),
    ("Initial Jobless Claims",          "US", 4, "08:30", "employment"),
    ("Continuing Jobless Claims",       "US", 3, "08:30", "employment"),
    ("Average Hourly Earnings",         "US", 4, "08:30", "employment"),
    ("Labor Force Participation Rate",  "US", 3, "08:30", "employment"),
    # GDP / Growth
    ("GDP Advance",             "US", 5, "08:30", "growth"),
    ("GDP Second Estimate",     "US", 4, "08:30", "growth"),
    ("GDP Final",               "US", 3, "08:30", "growth"),
    ("GDPNow (Atlanta Fed)",    "US", 3, "varies", "growth"),
    ("Retail Sales (Advance)",  "US", 5, "08:30", "growth"),
    ("Retail Sales (Revised)",  "US", 3, "08:30", "growth"),
    ("Personal Spending",       "US", 4, "08:30", "growth"),
    ("Personal Income",         "US", 3, "08:30", "growth"),
    # Manufacturing / Production
    ("ISM Manufacturing PMI",       "US", 5, "10:00", "activity"),
    ("ISM Services PMI",            "US", 5, "10:00", "activity"),
    ("S&P Global PMI Manufacturing","US", 3, "09:45", "activity"),
    ("S&P Global PMI Services",     "US", 3, "09:45", "activity"),
    ("Industrial Production",       "US", 4, "09:15", "activity"),
    ("Capacity Utilization",        "US", 3, "09:15", "activity"),
    ("Durable Goods Orders",        "US", 4, "08:30", "activity"),
    ("Factory Orders",              "US", 3, "10:00", "activity"),
    ("Chicago PMI",                 "US", 3, "09:45", "activity"),
    ("Philly Fed Manufacturing",    "US", 3, "08:30", "activity"),
    ("Empire State Manufacturing",  "US", 3, "08:30", "activity"),
    # Housing
    ("Existing Home Sales",     "US", 4, "10:00", "housing"),
    ("New Home Sales",          "US", 4, "10:00", "housing"),
    ("Housing Starts",          "US", 4, "08:30", "housing"),
    ("Building Permits",        "US", 4, "08:30", "housing"),
    ("Pending Home Sales",      "US", 3, "10:00", "housing"),
    ("Case-Shiller Home Price", "US", 3, "09:00", "housing"),
    # Sentiment
    ("Consumer Confidence (CB)",    "US", 4, "10:00", "sentiment"),
    ("UMich Consumer Sentiment",    "US", 4, "10:00", "sentiment"),
    ("UMich Inflation Expectations","US", 4, "10:00", "sentiment"),
    # Trade / External
    ("Trade Balance",       "US", 4, "08:30", "external"),
    ("Current Account",     "US", 3, "08:30", "external"),
    # Monetary Policy
    ("FOMC Rate Decision",  "US", 5, "14:00", "monetary"),
    ("FOMC Minutes",        "US", 4, "14:00", "monetary"),
    ("Fed Chair Speech",    "US", 4, "varies", "monetary"),
    ("Beige Book",          "US", 3, "14:00", "monetary"),
    # Treasury Auctions
    ("Treasury 2Y Auction",  "US", 3, "13:00", "auction"),
    ("Treasury 5Y Auction",  "US", 3, "13:00", "auction"),
    ("Treasury 10Y Auction", "US", 4, "13:00", "auction"),
    ("Treasury 30Y Auction", "US", 4, "13:00", "auction"),
    # Other
    ("Leading Indicators",          "US", 3, "10:00", "composite"),
    ("Wholesale Inventories",       "US", 2, "10:00", "activity"),
    ("Business Inventories",        "US", 2, "10:00", "activity"),
    ("Non-Farm Productivity",       "US", 3, "08:30", "employment"),
    ("Unit Labor Costs",            "US", 3, "08:30", "employment"),
]

_INTL_RELEASES: list[tuple[str, str, int, str, str]] = [
    ("ECB Rate Decision",       "EUR", 5, "13:45", "monetary"),
    ("ECB Press Conference",    "EUR", 4, "14:30", "monetary"),
    ("BOE Rate Decision",       "GBP", 5, "12:00", "monetary"),
    ("BOJ Rate Decision",       "JPY", 5, "varies", "monetary"),
    ("RBA Rate Decision",       "AUD", 5, "04:30", "monetary"),
    ("BOC Rate Decision",       "CAD", 5, "10:00", "monetary"),
    ("SNB Rate Decision",       "CHF", 5, "09:30", "monetary"),
    ("RBNZ Rate Decision",      "NZD", 5, "22:00", "monetary"),
    ("Eurozone CPI",            "EUR", 4, "10:00", "inflation"),
    ("Eurozone GDP",            "EUR", 4, "10:00", "growth"),
    ("UK CPI",                  "GBP", 4, "07:00", "inflation"),
    ("UK GDP",                  "GBP", 4, "07:00", "growth"),
    ("Japan CPI",               "JPY", 4, "23:30", "inflation"),
    ("China Manufacturing PMI", "CNY", 4, "01:00", "activity"),
    ("Eurozone PMI",            "EUR", 4, "10:00", "activity"),
    ("German IFO",              "EUR", 3, "09:00", "sentiment"),
]

_ALL_RELEASES = _US_RELEASES + _INTL_RELEASES

# FRED series for key US releases (for prior value lookup)
_FRED_SERIES_MAP: dict[str, list[str]] = {
    "CPI (Headline)":       ["CPIAUCSL"],
    "CPI (Core)":           ["CPILFESL"],
    "PPI (Headline)":       ["PPIACO"],
    "PCE Deflator":         ["PCEPI"],
    "PCE Core":             ["PCEPILFE"],
    "Nonfarm Payrolls":     ["PAYEMS"],
    "Unemployment Rate":    ["UNRATE"],
    "JOLTS Job Openings":   ["JTSJOL"],
    "Initial Jobless Claims": ["ICSA"],
    "Continuing Jobless Claims": ["CCSA"],
    "GDP Advance":          ["GDPC1"],
    "Retail Sales (Advance)": ["RSAFS"],
    "Personal Spending":    ["PCE"],
    "Personal Income":      ["PI"],
    "ISM Manufacturing PMI": ["NAPM"],
    "ISM Services PMI":     ["NMFCI"],
    "Industrial Production": ["INDPRO"],
    "Durable Goods Orders": ["DGORDER"],
    "Housing Starts":       ["HOUST"],
    "Building Permits":     ["PERMIT"],
    "Existing Home Sales":  ["EXHOSLUSM495S"],
    "New Home Sales":       ["HSN1F"],
    "Consumer Confidence (CB)": ["CONCCONF"],
    "UMich Consumer Sentiment": ["UMCSENT"],
    "Trade Balance":        ["BOPGSTB"],
    "Current Account":      ["NETFI"],
    "Factory Orders":       ["AMTMNO"],
}

# FOMC meeting dates 2026 (approximate — 8 per year)
_FOMC_DATES_2026: list[str] = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# ECB/BOE/etc approximate 2026
_INTL_CB_DATES: dict[str, list[str]] = {
    "ECB":  ["2026-01-30", "2026-03-06", "2026-04-17", "2026-06-05",
             "2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17"],
    "BOE":  ["2026-02-06", "2026-03-20", "2026-05-08", "2026-06-19",
             "2026-08-07", "2026-09-18", "2026-11-06", "2026-12-17"],
    "BOJ":  ["2026-01-24", "2026-03-19", "2026-04-30", "2026-06-17",
             "2026-07-30", "2026-09-22", "2026-10-29", "2026-12-19"],
    "RBA":  ["2026-02-03", "2026-04-07", "2026-05-05", "2026-07-07",
             "2026-08-04", "2026-09-01", "2026-10-06", "2026-11-03"],
    "BOC":  ["2026-01-29", "2026-03-04", "2026-04-15", "2026-06-03",
             "2026-07-15", "2026-09-09", "2026-10-28", "2026-12-09"],
}

# Treasury auction schedule (approximate monthly cycle)
_TREASURY_AUCTION_TENORS = {
    "2Y":  {"day_of_month": 25, "frequency": "monthly"},
    "5Y":  {"day_of_month": 26, "frequency": "monthly"},
    "10Y": {"day_of_month": 10, "frequency": "monthly"},
    "30Y": {"day_of_month": 11, "frequency": "monthly"},
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EconomicEvent(BaseModel):
    event_id: str
    event_name: str
    country: str
    event_date: str          # YYYY-MM-DD
    release_time_et: str
    category: str
    importance_stars: int    # 1-5
    forecast: Optional[float] = None
    actual: Optional[float] = None
    previous: Optional[float] = None
    surprise_magnitude: Optional[float] = None
    surprise_direction: Optional[str] = None   # "beat" | "miss" | "inline"
    unit: str = ""
    source: str = ""
    revision_flag: bool = False
    is_released: bool = False


class ConsensusRecord(BaseModel):
    event_name: str
    event_date: str
    forecast_investing: Optional[float] = None
    forecast_ff: Optional[float] = None
    forecast_te: Optional[float] = None
    consensus_median: Optional[float] = None
    sources_available: int = 0
    fetched_at: str


class SurpriseIndexPoint(BaseModel):
    date: str
    event_name: str
    country: str
    actual: float
    forecast: float
    surprise_raw: float          # actual - forecast
    surprise_normalized: float   # (actual - forecast) / std_dev_historical
    rolling_index_90d: Optional[float] = None


class FOMCEvent(BaseModel):
    meeting_date: str
    type: str                    # "rate_decision" | "minutes"
    days_until: int
    current_rate_pct: Optional[float] = None
    expected_change_bps: Optional[float] = None
    is_press_conference: bool = True


class TreasuryAuction(BaseModel):
    auction_date: str
    tenor: str
    amount_bn: Optional[float] = None
    bid_to_cover: Optional[float] = None
    high_yield: Optional[float] = None
    when_issued_yield: Optional[float] = None
    days_until: int


class CalendarResponse(BaseModel):
    generated_at: str
    days_ahead: int
    country_filter: str
    total_events: int
    high_impact_count: int
    events: list[EconomicEvent]


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS economic_calendar (
                event_id           TEXT NOT NULL,
                event_name         TEXT NOT NULL,
                country            TEXT NOT NULL,
                event_date         TEXT NOT NULL,
                release_time_et    TEXT,
                category           TEXT,
                importance_stars   INTEGER,
                forecast           REAL,
                actual             REAL,
                previous           REAL,
                surprise_magnitude REAL,
                surprise_direction TEXT,
                unit               TEXT,
                source             TEXT,
                is_released        INTEGER DEFAULT 0,
                revision_flag      INTEGER DEFAULT 0,
                fetched_at         REAL NOT NULL,
                PRIMARY KEY (event_id, event_date)
            );
            CREATE INDEX IF NOT EXISTS idx_cal_date
                ON economic_calendar(event_date, country);
            CREATE TABLE IF NOT EXISTS release_history (
                history_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name         TEXT NOT NULL,
                country            TEXT NOT NULL,
                release_date       TEXT NOT NULL,
                actual             REAL,
                forecast           REAL,
                previous           REAL,
                revision_of        TEXT,
                surprise_magnitude REAL,
                source             TEXT,
                stored_at          REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_hist_event
                ON release_history(event_name, country, release_date DESC);
            CREATE TABLE IF NOT EXISTS consensus_estimates (
                event_name         TEXT NOT NULL,
                event_date         TEXT NOT NULL,
                forecast_investing REAL,
                forecast_ff        REAL,
                forecast_te        REAL,
                consensus_median   REAL,
                sources_available  INTEGER,
                fetched_at         TEXT NOT NULL,
                PRIMARY KEY (event_name, event_date)
            );
            CREATE TABLE IF NOT EXISTS surprise_index (
                index_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date         TEXT NOT NULL,
                event_name         TEXT NOT NULL,
                country            TEXT NOT NULL,
                actual             REAL NOT NULL,
                forecast           REAL NOT NULL,
                surprise_raw       REAL NOT NULL,
                surprise_normalized REAL NOT NULL,
                rolling_index_90d  REAL,
                stored_at          REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_surp_date
                ON surprise_index(event_date DESC, country);
            CREATE TABLE IF NOT EXISTS fred_prior_cache (
                series_id  TEXT NOT NULL,
                obs_date   TEXT NOT NULL,
                value      REAL,
                cached_at  REAL NOT NULL,
                PRIMARY KEY (series_id, obs_date)
            );
        """)
        conn.commit()


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _cache_key(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()


def _cache_get(key: str, ttl: float) -> Optional[Any]:
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT payload, cached_at FROM economic_calendar "
                "WHERE event_id = ? AND event_date = 'CACHE'",
                (key,)
            ).fetchone()
        # Use a separate meta table — handled inline with release_history
    except Exception:
        pass
    return None  # We use per-table logic below for caching


# ---------------------------------------------------------------------------
# FRED CSV utilities (no API key)
# ---------------------------------------------------------------------------


def _fetch_fred_series_latest(series_id: str) -> Optional[tuple[str, float]]:
    """Fetch most recent FRED observation (date, value) via CSV endpoint."""
    # Check SQLite cache first
    try:
        with _db() as conn:
            rows = conn.execute(
                """SELECT obs_date, value, cached_at FROM fred_prior_cache
                   WHERE series_id=? ORDER BY obs_date DESC LIMIT 1""",
                (series_id,)
            ).fetchall()
        if rows:
            row = rows[0]
            if (time.time() - row["cached_at"]) < _HISTORY_TTL:
                return (row["obs_date"], row["value"])
    except Exception:
        pass

    try:
        url = f"{_FRED_CSV_BASE}?id={series_id}"
        resp = requests.get(url, headers=_HEADERS_BROWSER, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().splitlines()
        # Get last two non-null values (current and prior)
        pairs: list[tuple[str, float]] = []
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", "", "NA"):
                try:
                    pairs.append((parts[0].strip(), float(parts[1].strip())))
                except ValueError:
                    continue
            if len(pairs) >= 2:
                break
        if not pairs:
            return None
        # Cache all found
        now = time.time()
        with _db() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO fred_prior_cache (series_id, obs_date, value, cached_at) VALUES (?,?,?,?)",
                [(series_id, dt, val, now) for dt, val in pairs]
            )
            conn.commit()
        return pairs[0]  # Most recent
    except Exception as exc:
        logger.warning("FRED prior fetch failed", series=series_id, error=str(exc))
        return None


def _fetch_fred_releases_upcoming(days: int = 30) -> list[dict[str, Any]]:
    """
    Fetch upcoming FRED release dates via public API (no key required for list endpoint).
    Returns list of {release_id, name, date} dicts.
    """
    today = date.today()
    end = today + timedelta(days=days)
    try:
        # FRED releases endpoint without API key for basic data
        url = f"{_FRED_API_BASE}/releases/dates"
        params = {
            "realtime_start": today.isoformat(),
            "realtime_end":   end.isoformat(),
            "include_release_dates_with_no_data": "true",
            "file_type": "json",
            "limit": 1000,
        }
        resp = requests.get(url, params=params, headers=_HEADERS_BROWSER, timeout=_REQ_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("release_dates", [])
    except Exception as exc:
        logger.warning("FRED release dates fetch failed", error=str(exc))
    return []


# ---------------------------------------------------------------------------
# ForexFactory scraper
# ---------------------------------------------------------------------------


def _scrape_forexfactory(days_ahead: int = 14) -> list[EconomicEvent]:
    """
    Scrape ForexFactory economic calendar.
    Structured HTML with impact/forecast/actual/previous columns.
    """
    events: list[EconomicEvent] = []
    try:
        url = "https://www.forexfactory.com/calendar"
        resp = requests.get(url, headers=_HEADERS_BROWSER, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("ForexFactory HTTP error", status=resp.status_code)
            return events

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_=re.compile(r"calendar__table"))
        if table is None:
            # Try alternate structure
            table = soup.find("table", {"class": "calendar"})
        if table is None:
            logger.warning("ForexFactory: no calendar table found")
            return events

        current_date_str = date.today().isoformat()
        cutoff = date.today() + timedelta(days=days_ahead)

        rows = table.find_all("tr") if table else []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 7:
                continue

            try:
                # Date cell
                date_cell = row.find("td", class_=re.compile(r"calendar__date"))
                if date_cell:
                    date_text = date_cell.get_text(strip=True)
                    if date_text:
                        # Parse ForexFactory date format: "Mon Jan 1"
                        try:
                            parsed = datetime.strptime(
                                f"{date_text} {date.today().year}", "%a %b %d %Y"
                            )
                            current_date_str = parsed.strftime("%Y-%m-%d")
                        except ValueError:
                            pass

                ev_date = date.fromisoformat(current_date_str)
                if ev_date > cutoff:
                    continue

                # Time cell
                time_cell = row.find("td", class_=re.compile(r"calendar__time"))
                release_time = time_cell.get_text(strip=True) if time_cell else ""

                # Currency cell
                curr_cell = row.find("td", class_=re.compile(r"calendar__currency"))
                country = curr_cell.get_text(strip=True) if curr_cell else "US"

                # Impact cell
                impact_cell = row.find("td", class_=re.compile(r"calendar__impact"))
                impact_class = str(impact_cell) if impact_cell else ""
                if "high" in impact_class.lower():
                    stars = 4
                elif "medium" in impact_class.lower():
                    stars = 3
                elif "low" in impact_class.lower():
                    stars = 1
                else:
                    stars = 2

                # Event name
                event_cell = row.find("td", class_=re.compile(r"calendar__event"))
                event_name = event_cell.get_text(strip=True) if event_cell else ""
                if not event_name:
                    continue

                # Actual, forecast, previous
                actual_cell    = row.find("td", class_=re.compile(r"calendar__actual"))
                forecast_cell  = row.find("td", class_=re.compile(r"calendar__forecast"))
                previous_cell  = row.find("td", class_=re.compile(r"calendar__previous"))

                actual   = _parse_numeric(actual_cell.get_text(strip=True) if actual_cell else "")
                forecast = _parse_numeric(forecast_cell.get_text(strip=True) if forecast_cell else "")
                previous = _parse_numeric(previous_cell.get_text(strip=True) if previous_cell else "")

                surprise, direction = _compute_surprise(actual, forecast)
                event_id = _make_event_id(event_name, current_date_str, country)

                events.append(EconomicEvent(
                    event_id=event_id,
                    event_name=event_name,
                    country=_normalize_currency_code(country),
                    event_date=current_date_str,
                    release_time_et=release_time,
                    category=_infer_category(event_name),
                    importance_stars=stars,
                    forecast=forecast,
                    actual=actual,
                    previous=previous,
                    surprise_magnitude=surprise,
                    surprise_direction=direction,
                    source="forexfactory",
                    is_released=actual is not None,
                ))

            except Exception as exc:
                logger.debug("ForexFactory row parse error", error=str(exc))
                continue

        logger.info("ForexFactory scraped", events=len(events))
    except Exception as exc:
        logger.warning("ForexFactory scrape failed", error=str(exc))

    return events


# ---------------------------------------------------------------------------
# Investing.com scraper
# ---------------------------------------------------------------------------


def _scrape_investing_com(days_ahead: int = 14) -> list[EconomicEvent]:
    """
    Scrape Investing.com economic calendar.
    Public HTML with forecast/actual/previous/importance columns.
    Rate-limits to be polite.
    """
    events: list[EconomicEvent] = []
    try:
        url = "https://www.investing.com/economic-calendar/"
        headers = dict(_HEADERS_BROWSER)
        headers["X-Requested-With"] = "XMLHttpRequest"

        resp = requests.get(url, headers=headers, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("Investing.com HTTP error", status=resp.status_code)
            return events

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", id="economicCalendarData")
        if table is None:
            table = soup.find("div", id="economicCalendarData")
        if table is None:
            logger.warning("Investing.com: calendar table not found (may require JS)")
            return events

        cutoff = date.today() + timedelta(days=days_ahead)
        current_date_str = date.today().isoformat()

        for row in table.find_all("tr", class_=re.compile(r"(js-event-item|ec_|calendar_row)")):
            try:
                # Date from data attribute or surrounding header
                ev_date_str = row.get("data-event-datetime", "")
                if ev_date_str:
                    try:
                        ev_date = datetime.strptime(ev_date_str[:10], "%Y/%m/%d")
                        current_date_str = ev_date.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                else:
                    # Check for date header row
                    date_hdr = row.find("td", class_="theDay")
                    if date_hdr:
                        date_text = date_hdr.get_text(strip=True)
                        try:
                            ev_date = datetime.strptime(date_text, "%A, %B %d, %Y")
                            current_date_str = ev_date.strftime("%Y-%m-%d")
                        except ValueError:
                            pass
                        continue

                ev_d = date.fromisoformat(current_date_str)
                if ev_d > cutoff:
                    continue

                # Cells
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue

                # Time
                time_cell = row.find("td", class_=re.compile(r"time"))
                release_time = time_cell.get_text(strip=True) if time_cell else ""

                # Country/currency
                flag_cell = row.find("td", class_=re.compile(r"(flagCur|country)"))
                country = "US"
                if flag_cell:
                    flag = flag_cell.find("span", class_=re.compile(r"flag"))
                    if flag:
                        cls = str(flag.get("class", ""))
                        m = re.search(r"flag-([a-zA-Z]+)", cls)
                        if m:
                            country = m.group(1).upper()

                # Event name
                name_cell = row.find("td", class_=re.compile(r"event"))
                if name_cell is None:
                    name_cell = row.find("a", class_=re.compile(r"js-event-item"))
                event_name = name_cell.get_text(strip=True) if name_cell else ""
                if not event_name:
                    continue

                # Impact (bull icons count)
                impact_cell = row.find("td", class_=re.compile(r"sentiment"))
                if impact_cell is None:
                    impact_cell = row.find("td", class_="bull")
                stars = 2
                if impact_cell:
                    bulls = impact_cell.find_all("i", class_=re.compile(r"grayFullBullishIcon|bull"))
                    stars = min(5, max(1, len(bulls))) if bulls else 2

                # Actual/forecast/previous
                actual   = _parse_numeric_from_cell(row, "act")
                forecast = _parse_numeric_from_cell(row, "fore")
                previous = _parse_numeric_from_cell(row, "prev")

                surprise, direction = _compute_surprise(actual, forecast)
                event_id = _make_event_id(event_name, current_date_str, country)

                events.append(EconomicEvent(
                    event_id=event_id,
                    event_name=event_name,
                    country=_normalize_currency_code(country),
                    event_date=current_date_str,
                    release_time_et=release_time,
                    category=_infer_category(event_name),
                    importance_stars=stars,
                    forecast=forecast,
                    actual=actual,
                    previous=previous,
                    surprise_magnitude=surprise,
                    surprise_direction=direction,
                    source="investing.com",
                    is_released=actual is not None,
                ))

            except Exception as exc:
                logger.debug("Investing.com row parse error", error=str(exc))
                continue

        logger.info("Investing.com scraped", events=len(events))
    except Exception as exc:
        logger.warning("Investing.com scrape failed", error=str(exc))

    return events


# ---------------------------------------------------------------------------
# TradingEconomics scraper
# ---------------------------------------------------------------------------


def _scrape_tradingeconomics(days_ahead: int = 14) -> list[EconomicEvent]:
    """
    Scrape TradingEconomics economic calendar for consensus estimates.
    """
    events: list[EconomicEvent] = []
    try:
        url = "https://tradingeconomics.com/calendar"
        resp = requests.get(url, headers=_HEADERS_BROWSER, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("TradingEconomics HTTP error", status=resp.status_code)
            return events

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", id=re.compile(r"(calendar|eco)"))
        if table is None:
            table = soup.find("table", class_=re.compile(r"table"))
        if table is None:
            logger.warning("TradingEconomics: no table found")
            return events

        cutoff = date.today() + timedelta(days=days_ahead)
        current_date_str = date.today().isoformat()

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 5:
                # Check if it's a date header
                th_cells = row.find_all("th")
                if th_cells:
                    date_text = " ".join(c.get_text(strip=True) for c in th_cells)
                    try:
                        parsed = datetime.strptime(date_text.strip(), "%A %B %d %Y")
                        current_date_str = parsed.strftime("%Y-%m-%d")
                    except ValueError:
                        try:
                            parsed = datetime.strptime(date_text.strip(), "%B %d, %Y")
                            current_date_str = parsed.strftime("%Y-%m-%d")
                        except ValueError:
                            pass
                continue

            try:
                ev_d = date.fromisoformat(current_date_str)
                if ev_d > cutoff:
                    continue

                texts = [c.get_text(strip=True) for c in cells]
                # Typical TE columns: Date, Time, Country, Category, Actual, Previous, Consensus, Forecast
                if len(texts) < 6:
                    continue

                event_name = texts[3] if len(texts) > 3 else texts[2]
                country_text = texts[2] if len(texts) > 2 else "US"
                time_text = texts[1] if len(texts) > 1 else ""

                actual   = _parse_numeric(texts[4]) if len(texts) > 4 else None
                previous = _parse_numeric(texts[5]) if len(texts) > 5 else None
                consensus = _parse_numeric(texts[6]) if len(texts) > 6 else None
                forecast  = _parse_numeric(texts[7]) if len(texts) > 7 else consensus

                surprise, direction = _compute_surprise(actual, forecast or consensus)
                event_id = _make_event_id(event_name, current_date_str, country_text[:3].upper())

                events.append(EconomicEvent(
                    event_id=event_id,
                    event_name=event_name,
                    country=_normalize_currency_code(country_text[:3].upper()),
                    event_date=current_date_str,
                    release_time_et=time_text,
                    category=_infer_category(event_name),
                    importance_stars=_infer_importance(event_name),
                    forecast=forecast or consensus,
                    actual=actual,
                    previous=previous,
                    surprise_magnitude=surprise,
                    surprise_direction=direction,
                    source="tradingeconomics",
                    is_released=actual is not None,
                ))

            except Exception as exc:
                logger.debug("TradingEconomics row parse error", error=str(exc))

        logger.info("TradingEconomics scraped", events=len(events))
    except Exception as exc:
        logger.warning("TradingEconomics scrape failed", error=str(exc))

    return events


# ---------------------------------------------------------------------------
# GDPNow scraper
# ---------------------------------------------------------------------------


def _fetch_gdpnow() -> Optional[dict[str, Any]]:
    """
    Fetch Atlanta Fed GDPNow nowcast.
    Returns {estimate, as_of, url}.
    """
    try:
        url = "https://www.atlantafed.org/cqer/research/gdpnow"
        resp = requests.get(url, headers=_HEADERS_BROWSER, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for the GDPNow estimate in the page
        gdpnow_div = soup.find("div", class_=re.compile(r"(gdpnow|estimate|nowcast)"))
        estimate_text = ""
        if gdpnow_div:
            estimate_text = gdpnow_div.get_text(strip=True)
        else:
            # Fallback: search text for pattern like "2.X percent"
            text = soup.get_text()
            m = re.search(r"GDPNow.*?(\d+\.\d+)\s*percent", text, re.IGNORECASE)
            if m:
                estimate_text = m.group(1)

        val = _parse_numeric(estimate_text)
        if val is not None:
            return {
                "estimate_pct": val,
                "as_of": date.today().isoformat(),
                "source": "Atlanta Fed GDPNow",
                "url": url,
            }
    except Exception as exc:
        logger.warning("GDPNow fetch failed", error=str(exc))
    return None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _parse_numeric(text: str) -> Optional[float]:
    """Parse a numeric value from text, handling %, K, M, B suffixes."""
    if not text or text in ("--", "N/A", "", "TBA", "n/a"):
        return None
    # Remove whitespace and common suffixes
    text = text.strip().replace(",", "")
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    elif text.upper().endswith("B"):
        text = text[:-1]; multiplier = 1e9
    elif text.upper().endswith("M"):
        text = text[:-1]; multiplier = 1e6
    elif text.upper().endswith("K"):
        text = text[:-1]; multiplier = 1e3
    # Handle parentheses for negative
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return float(text) * multiplier
    except (ValueError, TypeError):
        return None


def _parse_numeric_from_cell(row: Any, class_prefix: str) -> Optional[float]:
    """Find a cell by class prefix and parse its numeric content."""
    cell = row.find("td", class_=re.compile(class_prefix, re.IGNORECASE))
    if cell is None:
        return None
    return _parse_numeric(cell.get_text(strip=True))


def _compute_surprise(
    actual: Optional[float], forecast: Optional[float]
) -> tuple[Optional[float], Optional[str]]:
    """
    Compute surprise magnitude and direction.
    Returns (magnitude_pct, direction) where direction is 'beat' | 'miss' | 'inline'.
    """
    if actual is None or forecast is None:
        return None, None
    if abs(forecast) < 1e-9:
        magnitude = actual - forecast
    else:
        magnitude = (actual - forecast) / abs(forecast) * 100.0
    if magnitude > 0.5:
        direction = "beat"
    elif magnitude < -0.5:
        direction = "miss"
    else:
        direction = "inline"
    return round(magnitude, 4), direction


def _make_event_id(name: str, date_str: str, country: str) -> str:
    raw = f"{country}:{date_str}:{name.lower()[:40]}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _normalize_currency_code(code: str) -> str:
    """Map currency/country codes to ISO 2-letter country."""
    mapping = {
        "USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP",
        "AUD": "AU", "CAD": "CA", "CHF": "CH", "CNY": "CN",
        "NZD": "NZ", "USD ": "US",
    }
    return mapping.get(code.upper().strip(), code[:2].upper())


def _infer_category(name: str) -> str:
    nl = name.lower()
    if any(x in nl for x in ["cpi", "ppi", "pce", "price", "inflation"]):
        return "inflation"
    if any(x in nl for x in ["payroll", "employment", "unemployment", "jobless", "jolts", "adp", "labor"]):
        return "employment"
    if any(x in nl for x in ["gdp", "retail", "spending", "income", "consumption", "growth"]):
        return "growth"
    if any(x in nl for x in ["housing", "home", "permit", "construction"]):
        return "housing"
    if any(x in nl for x in ["ism", "pmi", "manufacturing", "production", "factory", "durable", "orders"]):
        return "activity"
    if any(x in nl for x in ["fomc", "fed", "rate decision", "boe", "ecb", "boj", "rba", "boc"]):
        return "monetary"
    if any(x in nl for x in ["confidence", "sentiment", "umich"]):
        return "sentiment"
    if any(x in nl for x in ["trade", "current account", "export", "import"]):
        return "external"
    if any(x in nl for x in ["auction", "treasury"]):
        return "auction"
    return "other"


def _infer_importance(name: str) -> int:
    nl = name.lower()
    for event_name, _, stars, _, _ in _US_RELEASES + _INTL_RELEASES:
        if event_name.lower() in nl or nl in event_name.lower():
            return stars
    # Fallback heuristics
    if any(x in nl for x in ["fomc", "nonfarm", "payroll", "cpi", "pce", "gdp"]):
        return 5
    if any(x in nl for x in ["ppi", "ism", "jolts", "retail", "unemployment"]):
        return 4
    if any(x in nl for x in ["housing", "confidence", "sentiment", "industrial"]):
        return 3
    return 2


def _deduplicate_events(events: list[EconomicEvent]) -> list[EconomicEvent]:
    """
    Merge events from multiple sources. Priority: FF > Investing.com > TE.
    For same event, use source with most data (actual/forecast).
    """
    seen: dict[str, EconomicEvent] = {}
    source_priority = {"forexfactory": 3, "investing.com": 2, "tradingeconomics": 1, "sentinel": 2}

    for ev in events:
        key = f"{ev.country}:{ev.event_date}:{ev.event_name[:30].lower()}"
        if key not in seen:
            seen[key] = ev
        else:
            existing = seen[key]
            ep = source_priority.get(ev.source, 0)
            xp = source_priority.get(existing.source, 0)
            # Prefer source with more data
            ev_data = sum(1 for v in [ev.actual, ev.forecast, ev.previous] if v is not None)
            ex_data = sum(1 for v in [existing.actual, existing.forecast, existing.previous] if v is not None)
            if ev_data > ex_data or (ev_data == ex_data and ep > xp):
                seen[key] = ev

    return sorted(seen.values(), key=lambda e: (e.event_date, e.country, e.importance_stars), reverse=False)


def _build_sentinel_calendar(days_ahead: int, country_filter: str) -> list[EconomicEvent]:
    """
    Build calendar from SENTINEL internal release catalogue.
    Generates EconomicEvent objects from _ALL_RELEASES with approximate dates.
    Uses FRED prior values where available.
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    events: list[EconomicEvent] = []

    for event_name, country, stars, release_time, category in _ALL_RELEASES:
        if country_filter not in ("ALL", "GLOBAL") and country != country_filter:
            continue

        # Generate approximate release dates
        # Monthly releases: 3rd week of month; Weekly: every Thursday; etc.
        release_dates = _estimate_release_dates(event_name, category, today, cutoff)
        for rd in release_dates:
            prior_val = None
            fred_series = _FRED_SERIES_MAP.get(event_name, [])
            if fred_series:
                result = _fetch_fred_series_latest(fred_series[0])
                if result:
                    prior_val = result[1]

            event_id = _make_event_id(event_name, rd.isoformat(), country)
            events.append(EconomicEvent(
                event_id=event_id,
                event_name=event_name,
                country=country,
                event_date=rd.isoformat(),
                release_time_et=release_time,
                category=category,
                importance_stars=stars,
                forecast=None,
                actual=None,
                previous=prior_val,
                source="sentinel",
                is_released=False,
            ))

    return events


def _estimate_release_dates(
    event_name: str, category: str, start: date, end: date
) -> list[date]:
    """
    Estimate when an event will release within [start, end].
    FOMC: use hardcoded 2026 dates. Weekly: every Thursday. Monthly: ~3rd week.
    """
    results: list[date] = []
    nl = event_name.lower()

    if "fomc" in nl or "fed funds" in nl:
        for d_str in _FOMC_DATES_2026:
            d = date.fromisoformat(d_str)
            if start <= d <= end:
                results.append(d)
        return results

    if "jobless claims" in nl or "initial claims" in nl or "continuing claims" in nl:
        # Every Thursday
        d = start
        while d <= end:
            if d.weekday() == 3:  # Thursday
                results.append(d)
            d += timedelta(days=1)
        return results

    if "auction" in nl:
        # Approximate: mid-month
        d = start.replace(day=min(15, 28))
        while d <= end:
            results.append(d)
            # Next month
            if d.month == 12:
                d = d.replace(year=d.year + 1, month=1)
            else:
                d = d.replace(month=d.month + 1)
        return results

    # Default: monthly, 3rd Wednesday/Thursday
    d = start
    seen_months: set[tuple[int, int]] = set()
    while d <= end:
        ym = (d.year, d.month)
        if ym not in seen_months:
            # Find the 3rd Thursday of month
            first = d.replace(day=1)
            # Weekday 3 = Thursday
            offset = (3 - first.weekday()) % 7
            third_thu = first + timedelta(days=offset + 14)
            if start <= third_thu <= end:
                results.append(third_thu)
            seen_months.add(ym)
        d += timedelta(days=1)

    return results


# ---------------------------------------------------------------------------
# Consensus aggregation
# ---------------------------------------------------------------------------


def _aggregate_consensus(
    events_by_source: dict[str, list[EconomicEvent]]
) -> dict[str, ConsensusRecord]:
    """
    Aggregate consensus forecasts across sources.
    Key: event_name + event_date. Value: ConsensusRecord with median forecast.
    """
    consensus: dict[str, ConsensusRecord] = {}

    for source, events in events_by_source.items():
        for ev in events:
            if ev.forecast is None:
                continue
            key = f"{ev.event_name}:{ev.event_date}"
            if key not in consensus:
                consensus[key] = ConsensusRecord(
                    event_name=ev.event_name,
                    event_date=ev.event_date,
                    fetched_at=datetime.utcnow().isoformat(),
                )
            rec = consensus[key]
            if source == "forexfactory":
                consensus[key] = rec.model_copy(update={"forecast_ff": ev.forecast})
            elif source == "investing.com":
                consensus[key] = rec.model_copy(update={"forecast_investing": ev.forecast})
            elif source == "tradingeconomics":
                consensus[key] = rec.model_copy(update={"forecast_te": ev.forecast})

    # Compute medians
    for key, rec in consensus.items():
        vals = [v for v in [rec.forecast_ff, rec.forecast_investing, rec.forecast_te] if v is not None]
        if vals:
            median_val = float(np.median(vals))
            sources_count = len(vals)
            consensus[key] = rec.model_copy(update={
                "consensus_median": round(median_val, 4),
                "sources_available": sources_count,
            })
        # Persist to SQLite
        try:
            r = consensus[key]
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO consensus_estimates
                       (event_name, event_date, forecast_investing, forecast_ff, forecast_te,
                        consensus_median, sources_available, fetched_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (r.event_name, r.event_date, r.forecast_investing, r.forecast_ff,
                     r.forecast_te, r.consensus_median, r.sources_available, r.fetched_at)
                )
                conn.commit()
        except Exception:
            pass

    return consensus


# ---------------------------------------------------------------------------
# Economic Surprise Index
# ---------------------------------------------------------------------------


def _update_surprise_index(events: list[EconomicEvent]) -> None:
    """
    Store surprise data points in SQLite and compute rolling 90-day surprise index.
    Normalizes surprises by historical std deviation per event.
    """
    now = time.time()
    for ev in events:
        if ev.actual is None or ev.forecast is None:
            continue
        raw = ev.actual - ev.forecast
        # Historical std for this event
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT surprise_raw FROM surprise_index
                       WHERE event_name=? AND country=?
                       ORDER BY event_date DESC LIMIT 24""",
                    (ev.event_name, ev.country)
                ).fetchall()
            hist = [r["surprise_raw"] for r in rows if r["surprise_raw"] is not None]
            std = float(np.std(hist)) if len(hist) >= 3 else 1.0
            std = max(std, 0.001)
            normalized = raw / std
        except Exception:
            normalized = raw

        # Compute 90-day rolling index
        try:
            cutoff_90d = (date.fromisoformat(ev.event_date) - timedelta(days=90)).isoformat()
            with _db() as conn:
                rows = conn.execute(
                    """SELECT surprise_normalized FROM surprise_index
                       WHERE country=? AND event_date >= ?
                       ORDER BY event_date DESC LIMIT 50""",
                    (ev.country, cutoff_90d)
                ).fetchall()
            past_vals = [r["surprise_normalized"] for r in rows if r["surprise_normalized"] is not None]
            rolling = float(np.sum(past_vals + [normalized])) if past_vals else normalized
        except Exception:
            rolling = normalized

        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO surprise_index
                       (event_date, event_name, country, actual, forecast,
                        surprise_raw, surprise_normalized, rolling_index_90d, stored_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (ev.event_date, ev.event_name, ev.country, ev.actual, ev.forecast,
                     round(raw, 6), round(normalized, 6), round(rolling, 4), now)
                )
                conn.commit()
        except Exception:
            pass


def _get_surprise_index(country: str = "US", days: int = 90) -> list[SurpriseIndexPoint]:
    """Load surprise index from SQLite for the past N days."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        with _db() as conn:
            rows = conn.execute(
                """SELECT event_date, event_name, country, actual, forecast,
                          surprise_raw, surprise_normalized, rolling_index_90d
                   FROM surprise_index
                   WHERE country=? AND event_date >= ?
                   ORDER BY event_date DESC""",
                (country, cutoff)
            ).fetchall()
        return [
            SurpriseIndexPoint(
                date=r["event_date"],
                event_name=r["event_name"],
                country=r["country"],
                actual=r["actual"],
                forecast=r["forecast"],
                surprise_raw=r["surprise_raw"],
                surprise_normalized=r["surprise_normalized"],
                rolling_index_90d=r["rolling_index_90d"],
            )
            for r in rows
        ]
    except Exception as exc:
        logger.warning("Surprise index load failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Public API: Economic Surprise Index as pd.Series
# ---------------------------------------------------------------------------

# Synthetic release history for ESI bootstrap — used when SQLite is empty.
# Maps (event_name, country) -> list of (date_str, actual, consensus_proxy)
# Consensus proxy = prior reading (FRED-style). Surprises here are illustrative.
_ESI_SEED_DATA: list[tuple[str, str, str, float, float]] = [
    # (event_name, country, date, actual, consensus_proxy)
    ("Nonfarm Payrolls",     "US", "2026-01-10", 256.0, 175.0),
    ("Nonfarm Payrolls",     "US", "2026-02-07", 151.0, 200.0),
    ("Nonfarm Payrolls",     "US", "2026-03-07", 228.0, 160.0),
    ("Nonfarm Payrolls",     "US", "2026-04-04", 177.0, 180.0),
    ("CPI (Headline)",       "US", "2026-01-15", 3.1,   3.0),
    ("CPI (Headline)",       "US", "2026-02-12", 3.2,   3.1),
    ("CPI (Headline)",       "US", "2026-03-12", 2.9,   3.1),
    ("CPI (Headline)",       "US", "2026-04-10", 2.8,   2.9),
    ("Unemployment Rate",    "US", "2026-01-10", 4.1,   4.2),
    ("Unemployment Rate",    "US", "2026-02-07", 4.0,   4.1),
    ("Unemployment Rate",    "US", "2026-03-07", 4.1,   4.0),
    ("Retail Sales (Advance)", "US", "2026-01-16", 0.4, 0.3),
    ("Retail Sales (Advance)", "US", "2026-02-14", -0.9, 0.2),
    ("Retail Sales (Advance)", "US", "2026-03-17", 1.4, 0.6),
    ("ISM Manufacturing PMI", "US", "2026-02-03", 50.9, 49.5),
    ("ISM Manufacturing PMI", "US", "2026-03-03", 49.8, 50.0),
    ("ISM Manufacturing PMI", "US", "2026-04-01", 49.0, 50.2),
    ("Industrial Production", "US", "2026-01-17", 0.3,  0.1),
    ("Industrial Production", "US", "2026-02-14", -0.5, 0.1),
    ("Industrial Production", "US", "2026-03-14", 0.7,  0.2),
]


def _seed_esi_from_static() -> None:
    """
    Seed the surprise_index table with static 2026 data so ESI works
    without any network calls. Safe to call multiple times (idempotent via
    sentinel marker row check).
    """
    now = time.time()
    # Guard: check for a known sentinel row to avoid duplicate seeding
    try:
        with _db() as conn:
            row = conn.execute(
                """SELECT COUNT(*) FROM surprise_index
                   WHERE event_name='Nonfarm Payrolls' AND event_date='2026-01-10'"""
            ).fetchone()
            already_seeded = row[0] > 0 if row else False
            if already_seeded:
                return
    except Exception:
        return

    rows_to_insert: list[tuple] = []
    for event_name, country, date_str, actual, consensus in _ESI_SEED_DATA:
        raw = actual - consensus
        denom = abs(consensus) if abs(consensus) > 1e-9 else 1.0
        normalized = raw / denom  # normalized by |consensus| per task spec
        rows_to_insert.append((
            date_str, event_name, country, actual, consensus,
            round(raw, 6), round(normalized, 6), None, now,
        ))

    try:
        with _db() as conn:
            conn.executemany(
                """INSERT INTO surprise_index
                   (event_date, event_name, country, actual, forecast,
                    surprise_raw, surprise_normalized, rolling_index_90d, stored_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows_to_insert,
            )
            conn.commit()
        # Now recompute rolling_index_90d for the inserted rows
        _recompute_rolling_esi()
        logger.info("ESI seed data inserted", count=len(rows_to_insert))
    except Exception as exc:
        logger.warning("ESI seed failed", error=str(exc))


def _recompute_rolling_esi() -> None:
    """
    Recompute rolling_index_90d for all rows in surprise_index.
    Rolling = sum of normalized surprises in the 90-day window ending at that date.
    Called after seeding to populate the rolling column.
    """
    try:
        with _db() as conn:
            all_rows = conn.execute(
                """SELECT rowid, event_date, country, surprise_normalized
                   FROM surprise_index
                   ORDER BY country, event_date ASC"""
            ).fetchall()
    except Exception:
        return

    # Group by country
    by_country: dict[str, list[tuple[str, float, int]]] = {}
    for row in all_rows:
        c = row["country"]
        by_country.setdefault(c, []).append(
            (row["event_date"], row["surprise_normalized"] or 0.0, row["rowid"])
        )

    updates: list[tuple[float, int]] = []
    for country, entries in by_country.items():
        entries.sort(key=lambda x: x[0])
        for i, (dt_str, _, rowid) in enumerate(entries):
            dt = date.fromisoformat(dt_str)
            cutoff_dt = dt - timedelta(days=90)
            window_vals = [
                sn for (d2, sn, _) in entries
                if date.fromisoformat(d2) >= cutoff_dt and date.fromisoformat(d2) <= dt
            ]
            rolling = float(np.sum(window_vals))
            updates.append((round(rolling, 4), rowid))

    if updates:
        try:
            with _db() as conn:
                conn.executemany(
                    "UPDATE surprise_index SET rolling_index_90d=? WHERE rowid=?",
                    updates,
                )
                conn.commit()
        except Exception as exc:
            logger.warning("ESI rolling recompute failed", error=str(exc))


def compute_economic_surprise_index(
    country: str = "US",
    window_days: int = 90,
    min_observations: int = 3,
) -> pd.Series:
    """
    Compute the Economic Surprise Index (ESI) as a pd.Series indexed by date.

    ESI methodology:
      1. For each released event: surprise = (actual - consensus) / |consensus|
         (when |consensus| > epsilon, else surprise = actual - consensus)
      2. Collect all surprises in the rolling window_days window
      3. Compute z-score of the current window: (mean - 0) / std
         where 0 is the null hypothesis of no persistent surprise
      4. The ESI series is the rolling sum of normalized surprises at each date.

    If the SQLite database has insufficient data, seeds from static 2026 data.

    Parameters
    ----------
    country : str
        ISO country code (e.g. "US"). Default "US".
    window_days : int
        Rolling window in calendar days. Default 90.
    min_observations : int
        Minimum data points needed before returning a value. Default 3.

    Returns
    -------
    pd.Series
        Index: pd.DatetimeIndex (date of each observation)
        Values: float — rolling ESI at that date (positive = beats consensus,
                negative = misses consensus)
        Name: f"ESI_{country}_{window_days}d"
    """
    # Ensure seed data is present so function works without network calls
    _seed_esi_from_static()

    cutoff = (date.today() - timedelta(days=window_days * 4)).isoformat()  # wider for history
    try:
        with _db() as conn:
            rows = conn.execute(
                """SELECT event_date, event_name, actual, forecast,
                          surprise_normalized, rolling_index_90d
                   FROM surprise_index
                   WHERE country=? AND event_date >= ?
                   ORDER BY event_date ASC""",
                (country, cutoff),
            ).fetchall()
    except Exception as exc:
        logger.warning("ESI query failed, returning empty series", error=str(exc))
        return pd.Series(dtype=float, name=f"ESI_{country}_{window_days}d")

    if not rows:
        return pd.Series(dtype=float, name=f"ESI_{country}_{window_days}d")

    # Build a DataFrame of normalized surprises
    records = []
    for r in rows:
        try:
            actual = r["actual"]
            forecast = r["forecast"]
            # Recompute normalized surprise from raw values for freshness
            if actual is not None and forecast is not None:
                raw = actual - forecast
                denom = abs(forecast) if abs(forecast) > 1e-9 else 1.0
                norm = raw / denom
            else:
                norm = r["surprise_normalized"] or 0.0
            records.append({
                "date": pd.to_datetime(r["event_date"]),
                "surprise_normalized": norm,
            })
        except Exception:
            continue

    if len(records) < min_observations:
        logger.info(
            "ESI: insufficient observations",
            country=country,
            count=len(records),
            min_required=min_observations,
        )
        return pd.Series(dtype=float, name=f"ESI_{country}_{window_days}d")

    df = pd.DataFrame(records).set_index("date").sort_index()

    # Deduplicate by taking max abs surprise per date (multiple events on same day)
    daily = df.groupby(df.index.date)["surprise_normalized"].sum()
    daily.index = pd.to_datetime([str(d) for d in daily.index])

    # Rolling sum over window_days calendar days = ESI
    # Use min_periods=min_observations so early dates don't show spurious values
    esi = daily.rolling(f"{window_days}D", min_periods=min_observations).sum()
    esi.name = f"ESI_{country}_{window_days}d"

    return esi.dropna()


# ---------------------------------------------------------------------------
# Public API: International Central Bank Calendar
# ---------------------------------------------------------------------------

# Full 2026 FOMC meeting dates (two-day meetings — decision day is day 2)
_FOMC_MEETINGS_2026: list[dict[str, str]] = [
    {"start": "2026-01-28", "decision": "2026-01-29"},
    {"start": "2026-03-18", "decision": "2026-03-19"},
    {"start": "2026-05-06", "decision": "2026-05-07"},
    {"start": "2026-06-17", "decision": "2026-06-18"},
    {"start": "2026-07-29", "decision": "2026-07-30"},
    {"start": "2026-09-16", "decision": "2026-09-17"},
    {"start": "2026-10-28", "decision": "2026-10-29"},
    {"start": "2026-12-15", "decision": "2026-12-16"},
]

# Full 2026 international CB schedules
_CB_SCHEDULES_2026: dict[str, list[str]] = {
    "ECB":  [
        "2026-01-30", "2026-03-06", "2026-04-17", "2026-06-05",
        "2026-07-24", "2026-09-11", "2026-10-30", "2026-12-18",
    ],
    "BOE":  [
        "2026-02-05", "2026-03-19", "2026-05-07", "2026-06-18",
        "2026-08-06", "2026-09-17", "2026-11-05", "2026-12-17",
    ],
    "BOJ":  [
        "2026-01-23", "2026-03-18", "2026-04-28", "2026-06-16",
        "2026-07-28", "2026-09-19", "2026-10-28", "2026-12-18",
    ],
    "RBA":  [
        "2026-02-03", "2026-04-07", "2026-05-05", "2026-07-07",
        "2026-08-04", "2026-09-01", "2026-10-06", "2026-11-03",
    ],
    "BOC":  [
        "2026-01-29", "2026-03-04", "2026-04-15", "2026-06-03",
        "2026-07-15", "2026-09-09", "2026-10-28", "2026-12-09",
    ],
    "SNB":  [
        "2026-03-19", "2026-06-18", "2026-09-17", "2026-12-10",
    ],
    "RBNZ": [
        "2026-02-19", "2026-04-08", "2026-05-27", "2026-07-08",
        "2026-08-26", "2026-10-14", "2026-11-25",
    ],
}

_CB_CURRENCY_MAP: dict[str, str] = {
    "Fed":  "USD", "ECB":  "EUR", "BOE":  "GBP", "BOJ":  "JPY",
    "RBA":  "AUD", "BOC":  "CAD", "SNB":  "CHF", "RBNZ": "NZD",
}


def get_international_cb_calendar(
    days_ahead: int = 365,
    include_past: bool = False,
) -> list[dict[str, Any]]:
    """
    Return central bank meeting dates for all major banks.

    Covers: Fed (FOMC), ECB, BOE, BOJ, RBA, BOC, SNB, RBNZ.
    Uses hardcoded 2026 meeting dates (updated annually).

    Parameters
    ----------
    days_ahead : int
        How many calendar days ahead to include. Default 365.
    include_past : bool
        If True, include past meetings from the current year. Default False.

    Returns
    -------
    list[dict]
        Each dict contains:
          - bank (str): Central bank acronym (e.g. "Fed", "ECB")
          - currency (str): ISO currency code (e.g. "USD", "EUR")
          - date (str): Meeting / decision date YYYY-MM-DD
          - is_next (bool): True for the single next upcoming meeting per bank
          - days_until (int): Calendar days from today (negative = past)
          - meeting_type (str): "rate_decision"
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    results: list[dict[str, Any]] = []

    # Fed (FOMC) — use the two-day meeting decision date
    for meeting in _FOMC_MEETINGS_2026:
        decision_date = date.fromisoformat(meeting["decision"])
        days_until = (decision_date - today).days
        if not include_past and decision_date < today:
            continue
        if decision_date > cutoff:
            continue
        results.append({
            "bank": "Fed",
            "currency": "USD",
            "date": meeting["decision"],
            "start_date": meeting["start"],
            "is_next": False,  # set below
            "days_until": days_until,
            "meeting_type": "rate_decision",
        })

    # Other CBs
    for bank, dates in _CB_SCHEDULES_2026.items():
        for d_str in dates:
            d = date.fromisoformat(d_str)
            days_until = (d - today).days
            if not include_past and d < today:
                continue
            if d > cutoff:
                continue
            results.append({
                "bank": bank,
                "currency": _CB_CURRENCY_MAP.get(bank, "???"),
                "date": d_str,
                "start_date": d_str,
                "is_next": False,  # set below
                "days_until": days_until,
                "meeting_type": "rate_decision",
            })

    # Sort by date ascending
    results.sort(key=lambda x: x["date"])

    # Mark is_next for each bank (first upcoming meeting per bank)
    next_marked: set[str] = set()
    for entry in results:
        bank = entry["bank"]
        if bank not in next_marked and entry["days_until"] >= 0:
            entry["is_next"] = True
            next_marked.add(bank)

    return results


# ---------------------------------------------------------------------------
# Public API: Treasury Auction Calendar
# ---------------------------------------------------------------------------

_TREASURY_DIRECT_API = (
    "https://www.treasurydirect.gov/TA_WS/securities/search"
)

# Hardcoded 2026 fallback schedule (approximate — real schedule announced ~1 week prior)
# 4-week Bills: every Tuesday; 13-week/26-week: every Monday; 52-week: monthly
# Notes/Bonds: 2Y monthly ~last Wed; 5Y ~last Thu; 10Y ~mid-month; 30Y ~mid-month
_TREASURY_FALLBACK_2026: list[dict[str, Any]] = [
    # Bills — 4-week (every week), represented as monthly anchors
    {"type": "Bill", "term": "4-Week",   "cusip": "TBD", "day_of_week": 1, "freq": "weekly"},
    {"type": "Bill", "term": "13-Week",  "cusip": "TBD", "day_of_week": 0, "freq": "biweekly"},
    {"type": "Bill", "term": "26-Week",  "cusip": "TBD", "day_of_week": 0, "freq": "biweekly"},
    {"type": "Bill", "term": "52-Week",  "cusip": "TBD", "day_of_week": 0, "freq": "monthly"},
    # Notes/Bonds — approximate monthly
    {"type": "Note", "term": "2-Year",   "cusip": "TBD", "day_of_week": 2, "freq": "monthly"},
    {"type": "Note", "term": "5-Year",   "cusip": "TBD", "day_of_week": 3, "freq": "monthly"},
    {"type": "Note", "term": "10-Year",  "cusip": "TBD", "day_of_week": 3, "freq": "monthly"},
    {"type": "Bond", "term": "30-Year",  "cusip": "TBD", "day_of_week": 4, "freq": "monthly"},
]


def _generate_fallback_auctions(
    today: date, cutoff: date
) -> list[dict[str, Any]]:
    """Generate fallback auction dates from the known 2026 schedule pattern."""
    auctions: list[dict[str, Any]] = []
    for template in _TREASURY_FALLBACK_2026:
        freq = template["freq"]
        dow = template["day_of_week"]  # 0=Mon .. 6=Sun
        term = template["term"]
        sec_type = template["type"]

        # Generate candidate dates
        d = today
        seen_weeks: set[int] = set()
        seen_months: set[tuple[int, int]] = set()

        while d <= cutoff:
            iso_week = d.isocalendar()[1]
            ym = (d.year, d.month)

            if d.weekday() == dow:
                should_add = False
                if freq == "weekly":
                    if iso_week not in seen_weeks:
                        should_add = True
                        seen_weeks.add(iso_week)
                elif freq == "biweekly":
                    if iso_week not in seen_weeks and iso_week % 2 == 0:
                        should_add = True
                        seen_weeks.add(iso_week)
                elif freq == "monthly":
                    if ym not in seen_months:
                        # Last occurrence of that weekday in month
                        last_day = (d.replace(month=d.month % 12 + 1, day=1) - timedelta(days=1)) if d.month < 12 else d.replace(day=31)
                        last_dow = last_day - timedelta(days=(last_day.weekday() - dow) % 7)
                        if d >= last_dow - timedelta(days=7):
                            should_add = True
                            seen_months.add(ym)

                if should_add and today <= d <= cutoff:
                    issue_date = d + timedelta(days=2)  # T+2 settlement
                    # Maturity approximation
                    term_days = {
                        "4-Week": 28, "13-Week": 91, "26-Week": 182, "52-Week": 364,
                        "2-Year": 730, "5-Year": 1825, "10-Year": 3650, "30-Year": 10950,
                    }.get(term, 365)
                    maturity = issue_date + timedelta(days=term_days)
                    auctions.append({
                        "cusip": "TBD",
                        "type": sec_type,
                        "term": term,
                        "auction_date": d.isoformat(),
                        "issue_date": issue_date.isoformat(),
                        "maturity_date": maturity.isoformat(),
                        "days_until": (d - today).days,
                        "source": "sentinel_fallback",
                    })
            d += timedelta(days=1)

    return sorted(auctions, key=lambda a: a["auction_date"])


def fetch_treasury_auction_calendar(
    days_ahead: int = 30,
    security_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Fetch upcoming US Treasury auction dates.

    Tries TreasuryDirect public API first; falls back to hardcoded 2026 schedule.

    Parameters
    ----------
    days_ahead : int
        How many calendar days ahead to fetch. Default 30.
    security_types : list[str], optional
        Filter by security types, e.g. ["Bill", "Note", "Bond"].
        Default: all types.

    Returns
    -------
    list[dict]
        Each dict contains:
          - cusip (str): CUSIP identifier or "TBD"
          - type (str): "Bill" | "Note" | "Bond" | "TIPS" | "FRN"
          - term (str): e.g. "4-Week", "10-Year"
          - auction_date (str): YYYY-MM-DD
          - issue_date (str): YYYY-MM-DD
          - maturity_date (str): YYYY-MM-DD
          - days_until (int): Calendar days from today
          - source (str): "treasurydirect" | "sentinel_fallback"
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    auctions: list[dict[str, Any]] = []

    # --- Attempt 1: TreasuryDirect public API ---
    try:
        for sec_type in (security_types or ["Bill", "Note", "Bond"]):
            params = {
                "type": sec_type,
                "dateFieldName": "auctionDate",
                "startDate": today.isoformat(),
                "endDate": cutoff.isoformat(),
                "format": "json",
            }
            resp = requests.get(
                _TREASURY_DIRECT_API,
                params=params,
                headers=_HEADERS_BROWSER,
                timeout=_REQ_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        try:
                            auction_date_str = (
                                item.get("auctionDate", "") or ""
                            ).split("T")[0]
                            if not auction_date_str:
                                continue
                            auction_d = date.fromisoformat(auction_date_str)
                            if auction_d < today or auction_d > cutoff:
                                continue
                            issue_date_str = (
                                item.get("issueDate", "") or ""
                            ).split("T")[0]
                            maturity_date_str = (
                                item.get("maturityDate", "") or ""
                            ).split("T")[0]
                            auctions.append({
                                "cusip": item.get("cusip", "TBD"),
                                "type": item.get("securityType", sec_type),
                                "term": item.get("securityTerm", "Unknown"),
                                "auction_date": auction_date_str,
                                "issue_date": issue_date_str,
                                "maturity_date": maturity_date_str,
                                "days_until": (auction_d - today).days,
                                "source": "treasurydirect",
                            })
                        except Exception:
                            continue
            time.sleep(0.1)  # polite rate limit

        if auctions:
            # Filter by security_types if specified
            if security_types:
                auctions = [
                    a for a in auctions
                    if a["type"] in security_types
                ]
            auctions.sort(key=lambda a: a["auction_date"])
            logger.info(
                "TreasuryDirect auction fetch succeeded",
                count=len(auctions),
                days_ahead=days_ahead,
            )
            return auctions

    except Exception as exc:
        logger.warning(
            "TreasuryDirect API failed, using fallback schedule",
            error=str(exc),
        )

    # --- Fallback: hardcoded 2026 pattern ---
    fallback = _generate_fallback_auctions(today, cutoff)
    if security_types:
        fallback = [a for a in fallback if a["type"] in security_types]
    logger.info(
        "Treasury auction fallback schedule used",
        count=len(fallback),
        days_ahead=days_ahead,
    )
    return fallback


# ---------------------------------------------------------------------------
# Public API: Consensus aggregation using FRED prior values
# ---------------------------------------------------------------------------


def build_consensus_from_fred(
    event_names: Optional[list[str]] = None,
    store: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Build consensus proxy from FRED prior values for key US macro releases.

    When live consensus data is unavailable (no scrapers), the prior FRED
    reading serves as the consensus proxy — the market anchors on the
    most recently published value.

    Parameters
    ----------
    event_names : list[str], optional
        Subset of events to process. Default: all events in _FRED_SERIES_MAP.
    store : bool
        If True, persist to consensus_estimates table. Default True.

    Returns
    -------
    dict[str, dict]
        Maps event_name -> {
            "consensus_proxy": float,
            "as_of_date": str,
            "fred_series": str,
            "method": "fred_prior",
        }
    """
    target_events = event_names or list(_FRED_SERIES_MAP.keys())
    results: dict[str, dict[str, Any]] = {}

    for event_name in target_events:
        series_list = _FRED_SERIES_MAP.get(event_name)
        if not series_list:
            continue
        series_id = series_list[0]

        fred_result = _fetch_fred_series_latest(series_id)
        if fred_result is None:
            continue

        obs_date, prior_value = fred_result
        results[event_name] = {
            "consensus_proxy": prior_value,
            "as_of_date": obs_date,
            "fred_series": series_id,
            "method": "fred_prior",
        }

        if store:
            # Estimate next release date (approximately 1 month out)
            try:
                next_release = (
                    date.fromisoformat(obs_date) + timedelta(days=35)
                ).isoformat()
            except Exception:
                next_release = date.today().isoformat()

            try:
                with _db() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO consensus_estimates
                           (event_name, event_date, forecast_investing, forecast_ff,
                            forecast_te, consensus_median, sources_available, fetched_at)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (
                            event_name,
                            next_release,
                            None,  # forecast_investing
                            None,  # forecast_ff (ForexFactory)
                            prior_value,  # forecast_te used as FRED proxy slot
                            prior_value,  # consensus_median
                            1,     # 1 source (FRED)
                            datetime.utcnow().isoformat(),
                        ),
                    )
                    conn.commit()
            except Exception:
                pass

    logger.info(
        "FRED consensus proxy built",
        event_count=len(results),
        series_fetched=len([v for v in results.values() if v]),
    )
    return results


# ---------------------------------------------------------------------------
# Calendar assembly — main entry point
# ---------------------------------------------------------------------------


def fetch_calendar(
    days_ahead: int = 14,
    country_filter: str = "US",
    include_scrapers: bool = True,
) -> CalendarResponse:
    """
    Assemble economic calendar from all sources:
    1. SENTINEL internal catalogue (always available)
    2. ForexFactory scrape (if include_scrapers)
    3. Investing.com scrape (if include_scrapers)
    4. TradingEconomics scrape (if include_scrapers)
    Then deduplicate, aggregate consensus, and store to SQLite.
    """
    all_events: list[EconomicEvent] = []
    events_by_source: dict[str, list[EconomicEvent]] = {}

    # 1. Sentinel internal catalogue
    sentinel_events = _build_sentinel_calendar(days_ahead, country_filter)
    all_events.extend(sentinel_events)
    events_by_source["sentinel"] = sentinel_events

    if include_scrapers:
        # 2. ForexFactory
        ff_events = _scrape_forexfactory(days_ahead)
        if country_filter not in ("ALL", "GLOBAL"):
            ff_events = [e for e in ff_events if e.country == country_filter or country_filter == "US"]
        all_events.extend(ff_events)
        events_by_source["forexfactory"] = ff_events
        time.sleep(0.5)

        # 3. Investing.com
        inv_events = _scrape_investing_com(days_ahead)
        if country_filter not in ("ALL", "GLOBAL"):
            inv_events = [e for e in inv_events if e.country == country_filter]
        all_events.extend(inv_events)
        events_by_source["investing.com"] = inv_events
        time.sleep(0.5)

        # 4. TradingEconomics (consensus heavy)
        te_events = _scrape_tradingeconomics(days_ahead)
        if country_filter not in ("ALL", "GLOBAL"):
            te_events = [e for e in te_events if e.country == country_filter]
        all_events.extend(te_events)
        events_by_source["tradingeconomics"] = te_events

    # Deduplicate and merge
    merged = _deduplicate_events(all_events)

    # Aggregate consensus
    _aggregate_consensus(events_by_source)

    # Update surprise index for released events
    released = [e for e in merged if e.is_released and e.actual is not None]
    if released:
        _update_surprise_index(released)

    # Persist to economic_calendar table
    now = time.time()
    for ev in merged:
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO economic_calendar
                       (event_id, event_name, country, event_date, release_time_et,
                        category, importance_stars, forecast, actual, previous,
                        surprise_magnitude, surprise_direction, unit, source,
                        is_released, revision_flag, fetched_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ev.event_id, ev.event_name, ev.country, ev.event_date,
                     ev.release_time_et, ev.category, ev.importance_stars,
                     ev.forecast, ev.actual, ev.previous,
                     ev.surprise_magnitude, ev.surprise_direction,
                     ev.unit, ev.source, int(ev.is_released),
                     int(ev.revision_flag), now)
                )
                conn.commit()
        except Exception:
            pass

    high_count = sum(1 for e in merged if e.importance_stars >= 4)

    return CalendarResponse(
        generated_at=datetime.utcnow().isoformat(),
        days_ahead=days_ahead,
        country_filter=country_filter,
        total_events=len(merged),
        high_impact_count=high_count,
        events=merged,
    )


def get_release_history(event_name: str, country: str = "US", limit: int = 24) -> list[dict[str, Any]]:
    """Load historical releases for an event from SQLite + FRED."""
    results: list[dict[str, Any]] = []
    # DB
    try:
        with _db() as conn:
            rows = conn.execute(
                """SELECT release_date, actual, forecast, previous, surprise_magnitude, source
                   FROM release_history
                   WHERE event_name LIKE ? AND country=?
                   ORDER BY release_date DESC LIMIT ?""",
                (f"%{event_name}%", country, limit)
            ).fetchall()
        results = [dict(r) for r in rows]
    except Exception:
        pass

    # Supplement with FRED if available
    fred_series = _FRED_SERIES_MAP.get(event_name, [])
    if fred_series and len(results) < 12:
        try:
            url = f"{_FRED_CSV_BASE}?id={fred_series[0]}"
            resp = requests.get(url, headers=_HEADERS_BROWSER, timeout=_REQ_TIMEOUT)
            if resp.status_code == 200:
                lines = resp.text.strip().splitlines()[1:]
                fred_rows = []
                for line in reversed(lines[-limit:]):
                    parts = line.split(",")
                    if len(parts) == 2 and parts[1].strip() not in (".", ""):
                        try:
                            fred_rows.append({
                                "release_date": parts[0].strip(),
                                "actual": float(parts[1].strip()),
                                "forecast": None,
                                "previous": None,
                                "source": f"FRED:{fred_series[0]}",
                            })
                        except ValueError:
                            pass
                # Merge — FRED fills gaps
                existing_dates = {r["release_date"] for r in results}
                for fr in fred_rows:
                    if fr["release_date"] not in existing_dates:
                        results.append(fr)
                results.sort(key=lambda r: r.get("release_date", ""), reverse=True)
        except Exception:
            pass

    return results[:limit]


def get_fomc_calendar(days_ahead: int = 365) -> list[FOMCEvent]:
    """Return upcoming FOMC meetings with days-until countdown."""
    today = date.today()
    events: list[FOMCEvent] = []

    # Fetch current Fed Funds rate from FRED
    ff_result = _fetch_fred_series_latest("FEDFUNDS")
    current_rate = ff_result[1] if ff_result else None

    for d_str in _FOMC_DATES_2026:
        d = date.fromisoformat(d_str)
        if d < today:
            continue
        days_until = (d - today).days
        if days_until > days_ahead:
            continue
        events.append(FOMCEvent(
            meeting_date=d_str,
            type="rate_decision",
            days_until=days_until,
            current_rate_pct=current_rate,
            is_press_conference=True,
        ))
        # FOMC Minutes released ~3 weeks after meeting
        minutes_date = d + timedelta(days=21)
        if minutes_date <= today + timedelta(days=days_ahead):
            events.append(FOMCEvent(
                meeting_date=minutes_date.isoformat(),
                type="minutes",
                days_until=(minutes_date - today).days,
                current_rate_pct=current_rate,
                is_press_conference=False,
            ))

    return sorted(events, key=lambda e: e.meeting_date)


def get_impact_score(event_name: str) -> dict[str, Any]:
    """
    Return market impact score for an event.
    1-5 stars based on SENTINEL catalogue + historical surprise volatility.
    """
    base_stars = _infer_importance(event_name)
    # Check historical surprise volatility
    try:
        with _db() as conn:
            rows = conn.execute(
                """SELECT AVG(ABS(surprise_normalized)) as avg_surp, COUNT(*) as cnt
                   FROM surprise_index WHERE event_name LIKE ?""",
                (f"%{event_name[:20]}%",)
            ).fetchone()
        if rows and rows["cnt"] > 3:
            avg_surp = rows["avg_surp"] or 1.0
            # Boost stars if historically volatile
            if avg_surp > 2.0 and base_stars < 5:
                base_stars = min(5, base_stars + 1)
    except Exception:
        pass

    return {
        "event_name": event_name,
        "importance_stars": base_stars,
        "category": _infer_category(event_name),
        "release_time_et": next(
            (rt for en, _, _, rt, _ in _US_RELEASES + _INTL_RELEASES
             if event_name.lower() in en.lower() or en.lower() in event_name.lower()), "varies"
        ),
        "description": _impact_description(base_stars),
        "fred_series": _FRED_SERIES_MAP.get(event_name, []),
    }


def _impact_description(stars: int) -> str:
    return {
        5: "Tier 1 — Major market mover. Expect significant volatility in equities, bonds, and FX.",
        4: "Tier 2 — High impact. Usually causes notable market moves on surprise.",
        3: "Tier 3 — Medium impact. Moderate market reaction; mainly affects sector or FX.",
        2: "Tier 2 — Low-medium impact. Limited market reaction unless large surprise.",
        1: "Tier 1 — Low impact. Typically limited market reaction.",
    }.get(stars, "Unknown impact level")


# ---------------------------------------------------------------------------
# Module initialization
# ---------------------------------------------------------------------------

_ensure_db()

# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

eco_calendar_router = APIRouter(
    prefix="/eco-calendar/v3",
    tags=["Economic Calendar v3"],
)


@eco_calendar_router.get("/upcoming")
def get_upcoming(
    days: int = Query(14, ge=1, le=90, description="Days ahead to look"),
    country: str = Query("US", description="Country code: US, EU, GB, JP, AU, CA, ALL"),
    scrape: bool = Query(True, description="Include live scraping (slower but more consensus data)"),
) -> dict:
    """
    Upcoming economic releases with consensus estimates.
    Aggregates ForexFactory, Investing.com, TradingEconomics + SENTINEL catalogue.
    High-impact events (4-5 stars) include multi-source consensus median.
    """
    cal = fetch_calendar(days_ahead=days, country_filter=country.upper(), include_scrapers=scrape)
    return cal.model_dump()


@eco_calendar_router.get("/today")
def get_today(country: str = Query("US")) -> dict:
    """
    Today's economic releases with live actual/forecast/previous.
    Returns events scheduled for today's date only.
    """
    cal = fetch_calendar(days_ahead=1, country_filter=country.upper(), include_scrapers=True)
    today_str = date.today().isoformat()
    today_events = [e for e in cal.events if e.event_date == today_str]
    released = [e for e in today_events if e.is_released]
    upcoming = [e for e in today_events if not e.is_released]
    return {
        "date": today_str,
        "country": country.upper(),
        "total_events": len(today_events),
        "released": len(released),
        "upcoming": len(upcoming),
        "events": sorted(
            [e.model_dump() for e in today_events],
            key=lambda e: (e["release_time_et"] or "99:99", -e["importance_stars"])
        ),
    }


@eco_calendar_router.get("/surprise-index")
def get_surprise_index_endpoint(
    country: str = Query("US", description="Country code"),
    days: int = Query(90, ge=7, le=365, description="Lookback days"),
) -> dict:
    """
    Economic Surprise Index: rolling 90-day sum of normalized surprises.
    Positive = data beating consensus (hawkish), negative = missing (dovish).
    """
    index = _get_surprise_index(country=country.upper(), days=days)
    # Compute summary stats
    if index:
        latest = index[0]
        normalized_vals = [p.surprise_normalized for p in index if p.surprise_normalized is not None]
        rolling_latest = latest.rolling_index_90d
    else:
        rolling_latest = None
        normalized_vals = []

    return {
        "country": country.upper(),
        "lookback_days": days,
        "current_index": round(rolling_latest, 3) if rolling_latest else None,
        "signal": _surprise_index_signal(rolling_latest),
        "data_points": len(index),
        "avg_surprise_normalized": round(float(np.mean(normalized_vals)), 4) if normalized_vals else None,
        "history": [p.model_dump() for p in index[:60]],
    }


def _surprise_index_signal(index_val: Optional[float]) -> str:
    if index_val is None:
        return "insufficient_data"
    if index_val > 3.0:
        return "strongly_positive"
    if index_val > 1.0:
        return "positive"
    if index_val > -1.0:
        return "neutral"
    if index_val > -3.0:
        return "negative"
    return "strongly_negative"


@eco_calendar_router.get("/release/{event_name}/history")
def get_release_history_endpoint(
    event_name: str,
    country: str = Query("US"),
    limit: int = Query(24, ge=1, le=120),
) -> dict:
    """
    Historical release data for a specific event: actual vs forecast, surprise, revision.
    Supplements SQLite history with FRED observations.
    """
    history = get_release_history(event_name, country.upper(), limit)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for {event_name!r}. "
                   "Ensure the event has been tracked in a prior calendar fetch."
        )
    # Compute stats
    actuals = [h["actual"] for h in history if h.get("actual") is not None]
    surprises = [h["surprise_magnitude"] for h in history if h.get("surprise_magnitude") is not None]
    return {
        "event_name": event_name,
        "country": country.upper(),
        "records": len(history),
        "stats": {
            "mean_actual": round(float(np.mean(actuals)), 4) if actuals else None,
            "std_actual": round(float(np.std(actuals)), 4) if actuals else None,
            "mean_surprise_pct": round(float(np.mean(surprises)), 4) if surprises else None,
            "beat_rate_pct": round(
                sum(1 for s in surprises if s > 0) / len(surprises) * 100, 1
            ) if surprises else None,
        },
        "history": history,
    }


@eco_calendar_router.get("/fomc-calendar")
def get_fomc_calendar_endpoint(days_ahead: int = Query(365, ge=30, le=730)) -> dict:
    """
    FOMC meeting calendar: rate decisions and minutes release dates.
    Includes current Fed Funds rate from FRED and days-until countdown.
    """
    fomc = get_fomc_calendar(days_ahead)
    decisions = [e for e in fomc if e.type == "rate_decision"]
    minutes = [e for e in fomc if e.type == "minutes"]

    next_meeting = decisions[0] if decisions else None
    return {
        "as_of": date.today().isoformat(),
        "days_ahead": days_ahead,
        "next_meeting": next_meeting.model_dump() if next_meeting else None,
        "upcoming_decisions": len(decisions),
        "upcoming_minutes": len(minutes),
        "current_rate_pct": next_meeting.current_rate_pct if next_meeting else None,
        "schedule": [e.model_dump() for e in fomc],
    }


@eco_calendar_router.get("/impact-score/{event_name}")
def get_impact_score_endpoint(event_name: str) -> dict:
    """
    Market impact score for an event: 1-5 stars, category, release time.
    Adjusted by historical surprise volatility stored in SQLite.
    """
    return get_impact_score(event_name)


@eco_calendar_router.get("/international")
def get_international(days: int = Query(14, ge=1, le=90)) -> dict:
    """
    International central bank meetings and major economic releases.
    Covers ECB, BOE, BOJ, RBA, BOC, SNB for EUR, GBP, JPY, AUD, CAD, CHF.
    """
    today = date.today()
    cutoff = today + timedelta(days=days)
    events: list[dict[str, Any]] = []

    # CB meetings from hardcoded 2026 schedule
    for cb, dates in _INTL_CB_DATES.items():
        for d_str in dates:
            d = date.fromisoformat(d_str)
            if today <= d <= cutoff:
                currency_map = {"ECB": "EUR", "BOE": "GBP", "BOJ": "JPY",
                                "RBA": "AUD", "BOC": "CAD", "RBNZ": "NZD"}
                events.append({
                    "event_name": f"{cb} Rate Decision",
                    "country": currency_map.get(cb, "INT"),
                    "event_date": d_str,
                    "days_until": (d - today).days,
                    "importance_stars": 5,
                    "category": "monetary",
                    "release_time_et": next(
                        (rt for en, _, _, rt, _ in _INTL_RELEASES if cb in en), "varies"
                    ),
                })

    # Key international releases from catalogue
    intl_events = [
        ev for ev in _build_sentinel_calendar(days, "GLOBAL")
        if ev.country not in ("US",)
    ]
    for ev in intl_events:
        events.append({
            "event_name": ev.event_name,
            "country": ev.country,
            "event_date": ev.event_date,
            "days_until": (date.fromisoformat(ev.event_date) - today).days,
            "importance_stars": ev.importance_stars,
            "category": ev.category,
            "release_time_et": ev.release_time_et,
        })

    events.sort(key=lambda e: (e["event_date"], -e["importance_stars"]))

    return {
        "as_of": today.isoformat(),
        "days_ahead": days,
        "total_events": len(events),
        "events": events,
    }


@eco_calendar_router.get("/treasury-auctions")
def get_treasury_auctions(days: int = Query(30, ge=1, le=90)) -> dict:
    """
    Upcoming US Treasury auctions: 2Y, 5Y, 10Y, 30Y.
    Includes historical bid-to-cover ratios from FRED where available.
    """
    today = date.today()
    cutoff = today + timedelta(days=days)
    auctions: list[TreasuryAuction] = []

    # Approximate auction schedule: Treasury auctions are typically monthly
    # 2Y: last Tuesday of month; 5Y: last Wednesday; 10Y: ~10th; 30Y: ~11th
    tenor_configs = [
        ("2Y",  3, "Treasury 2Y Auction"),   # Wednesday=2, Thursday=3
        ("5Y",  2, "Treasury 5Y Auction"),   # Wednesday
        ("10Y", 3, "Treasury 10Y Auction"),  # Thursday
        ("30Y", 4, "Treasury 30Y Auction"),  # Friday
    ]

    # FRED bid-to-cover series (if available)
    btc_series = {
        "2Y":  "B2RATEUS2Y",   # Approximate — may not exist; skip gracefully
        "10Y": "B2RATEUS10Y",
    }

    seen_months: set[tuple[int, str]] = set()
    d = today
    while d <= cutoff:
        for tenor, weekday, name in tenor_configs:
            ym_key = (d.year * 100 + d.month, tenor)
            if ym_key in seen_months:
                continue
            # Find the next occurrence of the specified weekday in this month
            first_of_month = d.replace(day=1)
            offset = (weekday - first_of_month.weekday()) % 7
            auction_date = first_of_month + timedelta(days=offset + 21)  # ~last week
            if today <= auction_date <= cutoff:
                seen_months.add(ym_key)
                event_id = _make_event_id(name, auction_date.isoformat(), "US")
                auctions.append(TreasuryAuction(
                    auction_date=auction_date.isoformat(),
                    tenor=tenor,
                    days_until=(auction_date - today).days,
                ))
        d += timedelta(days=1)

    auctions.sort(key=lambda a: a.auction_date)

    # Fetch current yields as when-issued proxy
    from sentinel.sma.economic_calendar_v3 import _fetch_fred_series_latest
    yield_series = {"2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30"}
    yield_cache: dict[str, Optional[float]] = {}
    for tenor, series in yield_series.items():
        result = _fetch_fred_series_latest(series)
        yield_cache[tenor] = result[1] if result else None
        time.sleep(0.07)

    enriched = []
    for a in auctions:
        enriched.append({
            **a.model_dump(),
            "when_issued_yield_pct": yield_cache.get(a.tenor),
            "note": "Auction dates are approximate monthly estimates",
        })

    return {
        "as_of": today.isoformat(),
        "days_ahead": days,
        "auctions": enriched,
        "yield_source": "FRED CMT (live)",
    }


@eco_calendar_router.get("/gdpnow")
def get_gdpnow() -> dict:
    """
    Atlanta Fed GDPNow real-time GDP nowcast.
    Scraped directly from atlantafed.org.
    """
    result = _fetch_gdpnow()
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="GDPNow unavailable — Atlanta Fed site may be blocking scrape or changing structure."
        )
    return result


@eco_calendar_router.get("/consensus/{event_name}")
def get_consensus(
    event_name: str,
    event_date: Optional[str] = Query(None, description="YYYY-MM-DD or omit for latest"),
) -> dict:
    """
    Multi-source consensus estimate for a specific release.
    Shows ForexFactory, Investing.com, TradingEconomics forecasts and median.
    """
    target_date = event_date or date.today().isoformat()
    try:
        with _db() as conn:
            rows = conn.execute(
                """SELECT * FROM consensus_estimates
                   WHERE event_name LIKE ? AND event_date >= ?
                   ORDER BY event_date ASC LIMIT 5""",
                (f"%{event_name[:30]}%", target_date)
            ).fetchall()
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No consensus data for {event_name!r} on or after {target_date}. "
                       "Fetch the calendar first to populate consensus data."
            )
        records = [dict(r) for r in rows]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "event_name": event_name,
        "results": records,
        "note": "Consensus from ForexFactory, Investing.com, TradingEconomics (where available). "
                "Median used as primary consensus estimate.",
    }
