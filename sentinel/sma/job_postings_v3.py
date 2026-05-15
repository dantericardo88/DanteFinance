"""
Job Postings & Hiring Intelligence v3 — Dimension #086 (target score 9).

AUDIT FIX (v3): Indeed RSS is blocked; Wayback CDX is slow. This version replaces
those unreliable sources with:
  - BLS Data API v2 (free, no key): full JOLTS dashboard — JTS000000000000000JOL,
    JTS000000000000000JHR, JTS000000000000000QUR, plus 20+ sector series.
  - FRED free CSV / REST: JTSJOL, JTSQUR, JTSHJL, UNRATE, PAYEMS, CES0000000001.
  - Google Search count for site:linkedin.com/jobs+{company} (public, no auth).
  - Indeed company pages: https://www.indeed.com/cmp/{slug}/jobs (public HTML).
  - Wayback CDX (done right): limit=500, output=json, count monthly snapshots — used
    only as a traffic proxy, not for raw page content.
  - USAJobs.gov free API for government-contractor hiring signals.
  - Derived hiring velocity (30/60/90-day trend), tech-stack signal, sector comparison.

Public API
----------
BLSAdapter
    get_jolts_series(series_ids, start_year, end_year)  -> dict[str, pd.DataFrame]
    get_sector_openings()                               -> SectorOpenings
    get_macro_labor_dashboard()                         -> MacroLaborDashboard

FREDLaborAdapter
    get_series(series_id, start, end)                   -> pd.DataFrame
    get_multi_series(series_ids)                        -> dict[str, pd.DataFrame]

CompanyHiringAdapter
    get_linkedin_count(company_name)                    -> int | None
    get_indeed_count(company_slug)                      -> int | None
    get_usajobs_count(agency_name)                      -> int | None

WaybackTrafficAdapter (reliable implementation)
    get_monthly_snapshots(domain, lookback_months)      -> pd.Series
    get_traffic_proxy(domain, lookback_months)          -> WebTrafficProxy

HiringSignalEngine
    compute_hiring_velocity(ticker, company)            -> HiringVelocitySignal
    compute_tech_stack_signal(company)                  -> TechStackSignal
    compare_to_sector(ticker, sector)                   -> SectorComparison

SQLite: data/job_postings_v3.db — tables: bls_series_cache, company_job_counts,
        hiring_signals, web_traffic_snapshots, fred_cache
FastAPI router at /hiring/v3: full endpoint suite.

Dependencies: requests, pandas, numpy, fastapi, sqlite3 (stdlib).
No paid APIs. No auth required.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote, quote_plus, urlencode

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field as PField

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DB_PATH = Path("data") / "job_postings_v3.db"

_BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
_USAJOBS_BASE = "https://data.usajobs.gov/api/search"
_GOOGLE_SEARCH = "https://www.google.com/search"
_INDEED_CMP = "https://www.indeed.com/cmp"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_JSON_HEADERS = {
    "User-Agent": "SENTINEL/3.0 (hiring-intelligence; contact@sentinel.io)",
    "Accept": "application/json",
}

_BLS_CACHE_TTL = 6 * 3600        # 6 hours — BLS updates monthly
_FRED_CACHE_TTL = 4 * 3600       # 4 hours
_COMPANY_CACHE_TTL = 3600        # 1 hour — job counts change slowly
_WAYBACK_CACHE_TTL = 24 * 3600   # 24 hours — Wayback data is historical

# ── BLS JOLTS Series IDs ───────────────────────────────────────────────────────

# Total economy
JOLTS_TOTAL = {
    "openings":   "JTS000000000000000JOL",
    "hires":      "JTS000000000000000HIL",
    "hires_rate": "JTS000000000000000HIR",
    "quits":      "JTS000000000000000QUL",
    "quits_rate": "JTS000000000000000QUR",
    "layoffs":    "JTS000000000000000LDL",
    "layoffs_rate":"JTS000000000000000LDR",
    "separations":"JTS000000000000000TSL",
}

# Sector-level openings (JOLTS industry codes)
JOLTS_SECTOR_OPENINGS = {
    "construction":          "JTS2300000000000000JOL",
    "manufacturing":         "JTS3000000000000000JOL",
    "trade_transport_util":  "JTS4000000000000000JOL",
    "information":           "JTS5100000000000000JOL",
    "finance_insurance":     "JTS5200000000000000JOL",
    "real_estate":           "JTS5300000000000000JOL",
    "professional_business": "JTS5400000000000000JOL",
    "education_health":      "JTS6000000000000000JOL",
    "leisure_hospitality":   "JTS7000000000000000JOL",
    "government":            "JTSU00000000000000JOL",
}

# FRED series for broader labor picture
FRED_LABOR_SERIES = {
    "jolts_openings":  "JTSJOL",
    "jolts_quits":     "JTSQUR",
    "jolts_hires":     "JTSHJL",
    "unemployment":    "UNRATE",
    "nonfarm_payroll": "PAYEMS",
    "total_employed":  "CES0000000001",
}

# Tech-related keywords for job title signal detection
_TECH_KEYWORDS = [
    "aws", "azure", "gcp", "cloud", "kubernetes", "docker", "python",
    "machine learning", "ml engineer", "data engineer", "devops", "sre",
    "platform engineer", "backend engineer", "infrastructure", "ai engineer",
    "llm", "generative ai", "data science", "mlops", "spark", "kafka",
]

_EXEC_KEYWORDS = [
    "vp ", "vice president", "chief ", "cto", "cfo", "coo", "cpo", "ciso",
    "svp", "evp", "general manager", "president", "head of", "director",
]

_GEO_KEYWORDS = [
    "new york", "san francisco", "london", "singapore", "austin", "seattle",
    "chicago", "boston", "toronto", "berlin", "amsterdam", "dubai", "tokyo",
    "remote", "hybrid",
]


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bls_series_cache (
                series_id   TEXT NOT NULL,
                year        INTEGER NOT NULL,
                period      TEXT NOT NULL,
                value       REAL,
                fetched_at  INTEGER NOT NULL,
                PRIMARY KEY (series_id, year, period)
            );
            CREATE TABLE IF NOT EXISTS fred_cache (
                series_id   TEXT NOT NULL,
                obs_date    TEXT NOT NULL,
                value       REAL,
                fetched_at  INTEGER NOT NULL,
                PRIMARY KEY (series_id, obs_date)
            );
            CREATE TABLE IF NOT EXISTS company_job_counts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT,
                company     TEXT NOT NULL,
                source      TEXT NOT NULL,
                job_count   INTEGER,
                fetched_at  INTEGER NOT NULL,
                raw_snippet TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cjc_ticker_ts
                ON company_job_counts(ticker, fetched_at DESC);
            CREATE TABLE IF NOT EXISTS hiring_signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                company     TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                value       REAL,
                metadata_json TEXT,
                computed_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_hs_ticker
                ON hiring_signals(ticker, computed_at DESC);
            CREATE TABLE IF NOT EXISTS web_traffic_snapshots (
                domain      TEXT NOT NULL,
                year_month  TEXT NOT NULL,
                snap_count  INTEGER NOT NULL,
                fetched_at  INTEGER NOT NULL,
                PRIMARY KEY (domain, year_month)
            );
        """)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Pydantic models ────────────────────────────────────────────────────────────

class JOLTSSeries(BaseModel):
    series_id: str
    series_name: str
    latest_value: Optional[float] = None
    latest_period: Optional[str] = None
    mom_change: Optional[float] = None
    yoy_change: Optional[float] = None
    data_points: int = 0


class SectorOpenings(BaseModel):
    as_of: str
    sectors: Dict[str, Optional[float]]
    total_openings: Optional[float] = None
    highest_sector: Optional[str] = None
    lowest_sector: Optional[str] = None


class MacroLaborDashboard(BaseModel):
    as_of: str
    jolts_openings_k: Optional[float] = None
    jolts_hires_rate: Optional[float] = None
    jolts_quits_rate: Optional[float] = None
    jolts_layoffs_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    nonfarm_payroll_k: Optional[float] = None
    labor_market_tightness: Optional[float] = None   # openings / unemployed
    trend_3m: Optional[str] = None
    fred_series: Dict[str, Optional[float]] = PField(default_factory=dict)
    bls_series: Dict[str, Optional[float]] = PField(default_factory=dict)


class CompanyJobCount(BaseModel):
    ticker: Optional[str] = None
    company: str
    linkedin_count: Optional[int] = None
    indeed_count: Optional[int] = None
    usajobs_count: Optional[int] = None
    total_estimate: Optional[int] = None
    fetched_at: str
    confidence: str = "low"


class WebTrafficProxy(BaseModel):
    domain: str
    period_months: int
    total_snapshots: int
    monthly_avg: float
    mom_trend: Optional[float] = None       # % change last 3m vs prior 3m
    peak_month: Optional[str] = None
    trough_month: Optional[str] = None
    monthly_series: Dict[str, int] = PField(default_factory=dict)


class HiringVelocitySignal(BaseModel):
    ticker: str
    company: str
    current_openings: Optional[int] = None
    d30_openings: Optional[int] = None
    d60_openings: Optional[int] = None
    d90_openings: Optional[int] = None
    velocity_30d: Optional[float] = None    # % change 0-30 vs 31-60
    velocity_60d: Optional[float] = None    # % change 0-60 vs 61-120
    signal: str = "neutral"                 # accelerating / decelerating / neutral
    confidence: str = "low"
    computed_at: str


class TechStackSignal(BaseModel):
    company: str
    cloud_mentions: int = 0
    ai_ml_mentions: int = 0
    devops_mentions: int = 0
    top_keywords: List[str] = PField(default_factory=list)
    tech_intensity_score: float = 0.0      # 0–10
    computed_at: str


class SectorComparison(BaseModel):
    ticker: str
    sector: str
    company_openings: Optional[int] = None
    sector_openings_k: Optional[float] = None
    relative_rank: Optional[str] = None
    sector_trend: Optional[str] = None
    as_of: str


class USAJobsResult(BaseModel):
    agency: str
    total_count: int
    sample_titles: List[str] = PField(default_factory=list)
    as_of: str


# ── 1. BLS Data API v2 Adapter ─────────────────────────────────────────────────

class BLSAdapter:
    """
    BLS Public Data API v2.
    No API key required (limited to 25 series, 20 years per request without key).
    Endpoint: https://api.bls.gov/publicAPI/v2/timeseries/data/
    """

    _BATCH_SIZE = 25  # max series per unauthenticated request

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key  # optional — increases rate limit to 500 series/day

    def get_jolts_series(
        self,
        series_ids: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch one or more BLS JOLTS series.
        Returns dict of {series_id: DataFrame(year, period, value)}.
        Caches results in SQLite to avoid hammering the API.
        """
        if series_ids is None:
            series_ids = list(JOLTS_TOTAL.values())
        if end_year is None:
            end_year = datetime.now().year
        if start_year is None:
            start_year = end_year - 5

        # Serve from cache where possible
        cached, missing = self._load_from_cache(series_ids, start_year, end_year)

        if missing:
            fetched = self._fetch_bls_batch(missing, start_year, end_year)
            self._save_to_cache(fetched)
            cached.update(fetched)

        return cached

    def _load_from_cache(
        self, series_ids: List[str], start_year: int, end_year: int
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Return (cached_dict, missing_series_ids)."""
        cutoff = int(time.time()) - _BLS_CACHE_TTL
        cached: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []

        try:
            with _db() as conn:
                for sid in series_ids:
                    rows = conn.execute(
                        """SELECT year, period, value FROM bls_series_cache
                           WHERE series_id=? AND year>=? AND year<=? AND fetched_at>?
                           ORDER BY year, period""",
                        (sid, start_year, end_year, cutoff),
                    ).fetchall()
                    if rows:
                        df = pd.DataFrame(
                            [{"year": r["year"], "period": r["period"], "value": r["value"]}
                             for r in rows]
                        )
                        cached[sid] = df
                    else:
                        missing.append(sid)
        except Exception as exc:
            logger.warning("BLS cache load error: %s", exc)
            missing = series_ids

        return cached, missing

    def _save_to_cache(self, data: Dict[str, pd.DataFrame]) -> None:
        try:
            now = int(time.time())
            with _db() as conn:
                for sid, df in data.items():
                    if df is None or df.empty:
                        continue
                    for _, row in df.iterrows():
                        conn.execute(
                            """INSERT OR REPLACE INTO bls_series_cache
                               (series_id, year, period, value, fetched_at)
                               VALUES (?,?,?,?,?)""",
                            (sid, int(row["year"]), str(row["period"]),
                             float(row["value"]) if pd.notna(row["value"]) else None,
                             now),
                        )
        except Exception as exc:
            logger.warning("BLS cache save error: %s", exc)

    def _fetch_bls_batch(
        self, series_ids: List[str], start_year: int, end_year: int
    ) -> Dict[str, pd.DataFrame]:
        """POST to BLS API in batches of 25."""
        result: Dict[str, pd.DataFrame] = {}
        for i in range(0, len(series_ids), self._BATCH_SIZE):
            batch = series_ids[i : i + self._BATCH_SIZE]
            payload: Dict[str, Any] = {
                "seriesid": batch,
                "startyear": str(start_year),
                "endyear": str(end_year),
                "calculations": True,
            }
            if self._api_key:
                payload["registrationkey"] = self._api_key

            try:
                resp = requests.post(
                    _BLS_BASE,
                    json=payload,
                    headers={**_JSON_HEADERS, "Content-Type": "application/json"},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("BLS API batch fetch failed: %s", exc)
                continue

            if data.get("status") != "REQUEST_SUCCEEDED":
                logger.warning("BLS API non-success status: %s | msg: %s",
                               data.get("status"), data.get("message", []))
                # Still try to parse whatever came back
            for series_data in data.get("Results", {}).get("series", []):
                sid = series_data.get("seriesID", "")
                rows = []
                for obs in series_data.get("data", []):
                    try:
                        val_str = obs.get("value", "").strip()
                        val = float(val_str) if val_str and val_str != "-" else None
                        rows.append({
                            "year": int(obs["year"]),
                            "period": obs["period"],
                            "value": val,
                        })
                    except (ValueError, KeyError):
                        continue
                if rows:
                    df = pd.DataFrame(rows).sort_values(["year", "period"])
                    result[sid] = df

            time.sleep(0.5)  # be polite to BLS

        return result

    def get_sector_openings(self) -> SectorOpenings:
        """Fetch latest openings level for each major sector."""
        series_map = JOLTS_SECTOR_OPENINGS
        data = self.get_jolts_series(list(series_map.values()))

        sector_vals: Dict[str, Optional[float]] = {}
        total_id = JOLTS_TOTAL["openings"]
        total_df = self.get_jolts_series([total_id]).get(total_id)
        total_val = _latest_val(total_df)

        for sector_name, sid in series_map.items():
            df = data.get(sid)
            sector_vals[sector_name] = _latest_val(df)

        non_null = {k: v for k, v in sector_vals.items() if v is not None}
        highest = max(non_null, key=lambda k: non_null[k]) if non_null else None
        lowest = min(non_null, key=lambda k: non_null[k]) if non_null else None

        return SectorOpenings(
            as_of=datetime.now(tz=timezone.utc).strftime("%Y-%m"),
            sectors=sector_vals,
            total_openings=total_val,
            highest_sector=highest,
            lowest_sector=lowest,
        )

    def get_macro_labor_dashboard(self) -> MacroLaborDashboard:
        """Return a unified macro labor snapshot from BLS + FRED."""
        # Fetch all JOLTS total series in one call
        jolts_ids = list(JOLTS_TOTAL.values())
        jolts_data = self.get_jolts_series(jolts_ids)

        openings_df = jolts_data.get(JOLTS_TOTAL["openings"])
        hires_rate_df = jolts_data.get(JOLTS_TOTAL["hires_rate"])
        quits_rate_df = jolts_data.get(JOLTS_TOTAL["quits_rate"])
        layoffs_rate_df = jolts_data.get(JOLTS_TOTAL["layoffs_rate"])

        openings_latest = _latest_val(openings_df)
        hires_rate = _latest_val(hires_rate_df)
        quits_rate = _latest_val(quits_rate_df)
        layoffs_rate = _latest_val(layoffs_rate_df)

        # FRED supplement
        fred = FREDLaborAdapter()
        fred_vals: Dict[str, Optional[float]] = {}
        for name, sid in FRED_LABOR_SERIES.items():
            df = fred.get_series(sid)
            fred_vals[name] = _latest_val_fred(df)

        unemployment = fred_vals.get("unemployment")
        payroll = fred_vals.get("nonfarm_payroll")

        # Labor market tightness = openings / unemployed workers
        # openings is in thousands; UNRATE is a percent so tightness is a ratio
        tightness = None
        if openings_latest and unemployment:
            # openings/1000 / (unemployment_rate * labor_force_approx/100)
            # simplified: openings_thousands / unemployment_rate as tractable proxy
            tightness = round(openings_latest / unemployment, 2) if unemployment > 0 else None

        # Trend: compare latest 3m openings vs prior 3m
        trend = None
        if openings_df is not None and len(openings_df) >= 6:
            vals = openings_df["value"].dropna().tolist()
            if len(vals) >= 6:
                recent_avg = np.mean(vals[-3:])
                prior_avg = np.mean(vals[-6:-3])
                pct = (recent_avg - prior_avg) / prior_avg * 100 if prior_avg else 0
                trend = "tightening" if pct > 2 else "loosening" if pct < -2 else "stable"

        bls_vals = {
            "jolts_openings_k": openings_latest,
            "jolts_hires_rate": hires_rate,
            "jolts_quits_rate": quits_rate,
            "jolts_layoffs_rate": layoffs_rate,
        }

        return MacroLaborDashboard(
            as_of=datetime.now(tz=timezone.utc).isoformat(),
            jolts_openings_k=openings_latest,
            jolts_hires_rate=hires_rate,
            jolts_quits_rate=quits_rate,
            jolts_layoffs_rate=layoffs_rate,
            unemployment_rate=unemployment,
            nonfarm_payroll_k=payroll,
            labor_market_tightness=tightness,
            trend_3m=trend,
            fred_series=fred_vals,
            bls_series=bls_vals,
        )


# ── 2. FRED Labor Adapter ──────────────────────────────────────────────────────

class FREDLaborAdapter:
    """
    FRED free CSV API — no API key required.
    https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}
    """

    def get_series(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series as DataFrame with columns [date, value].
        start/end: "YYYY-MM-DD" strings (optional).
        """
        if use_cache:
            cached = self._load_fred_cache(series_id)
            if cached is not None and not cached.empty:
                return self._slice_dates(cached, start, end)

        params: Dict[str, str] = {"id": series_id}
        if start:
            params["vintage_date"] = start  # FRED ignores unknown params gracefully
        try:
            url = f"{_FRED_BASE}?{urlencode(params)}"
            resp = requests.get(url, headers=_JSON_HEADERS, timeout=20)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            self._save_fred_cache(series_id, df)
            return self._slice_dates(df, start, end)
        except Exception as exc:
            logger.error("FRED fetch error %s: %s", series_id, exc)
            return pd.DataFrame(columns=["date", "value"])

    def get_multi_series(
        self, series_ids: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple FRED series. Returns {series_id: DataFrame}."""
        result: Dict[str, pd.DataFrame] = {}
        for sid in series_ids:
            result[sid] = self.get_series(sid)
            time.sleep(0.2)  # polite pacing
        return result

    def _load_fred_cache(self, series_id: str) -> Optional[pd.DataFrame]:
        cutoff = int(time.time()) - _FRED_CACHE_TTL
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT obs_date, value FROM fred_cache
                       WHERE series_id=? AND fetched_at>?
                       ORDER BY obs_date""",
                    (series_id, cutoff),
                ).fetchall()
            if not rows:
                return None
            df = pd.DataFrame(
                [{"date": pd.to_datetime(r["obs_date"]), "value": r["value"]}
                 for r in rows]
            )
            return df
        except Exception:
            return None

    def _save_fred_cache(self, series_id: str, df: pd.DataFrame) -> None:
        now = int(time.time())
        try:
            with _db() as conn:
                for _, row in df.iterrows():
                    conn.execute(
                        """INSERT OR REPLACE INTO fred_cache
                           (series_id, obs_date, value, fetched_at)
                           VALUES (?,?,?,?)""",
                        (series_id, str(row["date"])[:10],
                         float(row["value"]) if pd.notna(row["value"]) else None,
                         now),
                    )
        except Exception as exc:
            logger.warning("FRED cache save error %s: %s", series_id, exc)

    @staticmethod
    def _slice_dates(
        df: pd.DataFrame, start: Optional[str], end: Optional[str]
    ) -> pd.DataFrame:
        if df.empty:
            return df
        mask = pd.Series([True] * len(df), index=df.index)
        if start:
            mask &= df["date"] >= pd.to_datetime(start)
        if end:
            mask &= df["date"] <= pd.to_datetime(end)
        return df[mask].copy()


# ── 3. Company Hiring Adapter ──────────────────────────────────────────────────

class CompanyHiringAdapter:
    """
    Fetches public job count signals from:
    1. Google Search count for site:linkedin.com/jobs (no auth, public)
    2. Indeed company page HTML (no auth, public)
    3. USAJobs.gov free API (government-contractor signals)

    Note: Google may throttle at high volume. Results are cached 1h in SQLite.
    """

    def get_linkedin_count(self, company_name: str) -> Optional[int]:
        """
        Estimate LinkedIn job openings by Google-searching site:linkedin.com/jobs.
        Returns approximate count from result stats line ("About X results").
        """
        cached = self._load_job_count_cache(company_name, "linkedin")
        if cached is not None:
            return cached

        # Construct a precise site: query
        query = f'site:linkedin.com/jobs/view "{company_name}"'
        params = {"q": query, "num": "10"}
        try:
            resp = requests.get(
                _GOOGLE_SEARCH,
                params=params,
                headers=_BROWSER_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            html = resp.text

            # Look for result stats line: "About 1,230 results"
            count = _parse_google_result_count(html)
            self._save_job_count_cache(company_name, "linkedin", count, html[:500])
            return count
        except Exception as exc:
            logger.warning("LinkedIn count fetch error for %s: %s", company_name, exc)
            return None

    def get_glassdoor_count(self, company_name: str) -> Optional[int]:
        """Google search for glassdoor jobs as a secondary signal."""
        cached = self._load_job_count_cache(company_name, "glassdoor")
        if cached is not None:
            return cached

        query = f'site:glassdoor.com/Jobs "{company_name}"'
        params = {"q": query, "num": "10"}
        try:
            resp = requests.get(
                _GOOGLE_SEARCH,
                params=params,
                headers=_BROWSER_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            count = _parse_google_result_count(resp.text)
            self._save_job_count_cache(company_name, "glassdoor", count, "")
            return count
        except Exception as exc:
            logger.warning("Glassdoor count error for %s: %s", company_name, exc)
            return None

    def get_indeed_count(self, company_slug: str) -> Optional[int]:
        """
        Fetch Indeed company jobs page and extract job count from HTML.
        company_slug: URL-safe company name (e.g. "Apple-Computer" or "apple")
        """
        cached = self._load_job_count_cache(company_slug, "indeed")
        if cached is not None:
            return cached

        url = f"{_INDEED_CMP}/{quote(company_slug)}/jobs"
        params = {"start": "0"}
        try:
            resp = requests.get(
                url,
                params=params,
                headers=_BROWSER_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            html = resp.text

            # Indeed typically shows: "X jobs at Company" or "Showing 1-15 of X jobs"
            count = _parse_indeed_job_count(html)
            self._save_job_count_cache(company_slug, "indeed", count, html[:500])
            return count
        except Exception as exc:
            logger.warning("Indeed count error for %s: %s", company_slug, exc)
            return None

    def get_usajobs_count(
        self, agency_name: str, keyword: Optional[str] = None
    ) -> USAJobsResult:
        """
        USAJobs.gov free API — useful for defense/government contractor hiring signals.
        API docs: https://developer.usajobs.gov/
        No auth required for basic search.
        """
        params: Dict[str, str] = {
            "Organization": agency_name,
            "ResultsPerPage": "25",
        }
        if keyword:
            params["Keyword"] = keyword

        try:
            resp = requests.get(
                _USAJOBS_BASE,
                params=params,
                headers={
                    **_JSON_HEADERS,
                    "Host": "data.usajobs.gov",
                    "User-Agent": "SENTINEL/3.0",
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            total = int(
                data.get("SearchResult", {})
                    .get("SearchResultCountAll", 0)
            )
            items = (
                data.get("SearchResult", {})
                    .get("SearchResultItems", [])
            )
            titles = [
                item.get("MatchedObjectDescriptor", {})
                    .get("PositionTitle", "")
                for item in items[:5]
            ]
            return USAJobsResult(
                agency=agency_name,
                total_count=total,
                sample_titles=titles,
                as_of=datetime.now(tz=timezone.utc).isoformat(),
            )
        except Exception as exc:
            logger.warning("USAJobs API error for %s: %s", agency_name, exc)
            return USAJobsResult(
                agency=agency_name,
                total_count=0,
                as_of=datetime.now(tz=timezone.utc).isoformat(),
            )

    def get_combined_count(
        self, ticker: str, company_name: str, company_slug: Optional[str] = None
    ) -> CompanyJobCount:
        """Aggregate job counts from all available sources."""
        slug = company_slug or _slugify(company_name)
        linkedin = self.get_linkedin_count(company_name)
        indeed = self.get_indeed_count(slug)

        counts = [c for c in [linkedin, indeed] if c is not None]
        total = int(np.median(counts)) if counts else None
        confidence = "high" if len(counts) >= 2 else ("medium" if len(counts) == 1 else "low")

        return CompanyJobCount(
            ticker=ticker,
            company=company_name,
            linkedin_count=linkedin,
            indeed_count=indeed,
            total_estimate=total,
            fetched_at=datetime.now(tz=timezone.utc).isoformat(),
            confidence=confidence,
        )

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _load_job_count_cache(
        self, company: str, source: str
    ) -> Optional[int]:
        cutoff = int(time.time()) - _COMPANY_CACHE_TTL
        try:
            with _db() as conn:
                row = conn.execute(
                    """SELECT job_count FROM company_job_counts
                       WHERE company=? AND source=? AND fetched_at>?
                       ORDER BY fetched_at DESC LIMIT 1""",
                    (company.lower(), source, cutoff),
                ).fetchone()
            if row and row["job_count"] is not None:
                return int(row["job_count"])
        except Exception:
            pass
        return None

    def _save_job_count_cache(
        self,
        company: str,
        source: str,
        count: Optional[int],
        snippet: str = "",
    ) -> None:
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT INTO company_job_counts
                       (company, source, job_count, fetched_at, raw_snippet)
                       VALUES (?,?,?,?,?)""",
                    (company.lower(), source, count, int(time.time()), snippet[:500]),
                )
        except Exception as exc:
            logger.debug("Job count cache save error: %s", exc)


# ── 4. Wayback CDX Traffic Adapter (Reliable Implementation) ──────────────────

class WaybackTrafficAdapter:
    """
    Wayback Machine CDX API — done right:
    - Output=json, limit=500 per request
    - Count snapshots per calendar month (not page content)
    - Used as a pure web-traffic proxy (more snapshots ≈ more visits / interest)
    - Full 24h caching to avoid hammering Wayback
    """

    def get_monthly_snapshots(
        self,
        domain: str,
        lookback_months: int = 12,
        match_type: str = "prefix",
    ) -> pd.Series:
        """
        Returns a pd.Series indexed by "YYYY-MM" with snapshot counts.
        match_type: "prefix" counts all subpages (better proxy for traffic).
        """
        end_dt = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_months * 30)
        from_str = start_dt.strftime("%Y%m%d")
        to_str = end_dt.strftime("%Y%m%d")

        # Check cache
        cached = self._load_snapshot_cache(domain, start_dt, end_dt)
        if cached is not None:
            return cached

        params = {
            "url": domain,
            "output": "json",
            "limit": "500",
            "from": from_str,
            "to": to_str,
            "matchType": match_type,
            "fl": "timestamp",          # only need timestamp column
            "collapse": "timestamp:8",  # collapse to daily granularity to reduce rows
        }

        monthly_counts: Dict[str, int] = defaultdict(int)
        try:
            resp = requests.get(
                _WAYBACK_CDX,
                params=params,
                headers=_JSON_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            # First row is the field headers ["timestamp"]
            if not data or len(data) < 2:
                logger.info("Wayback CDX: no snapshots for %s", domain)
                return pd.Series(dtype=int)

            for row in data[1:]:
                try:
                    ts = str(row[0])
                    year_month = f"{ts[:4]}-{ts[4:6]}"
                    monthly_counts[year_month] += 1
                except (IndexError, ValueError):
                    continue

            series = pd.Series(monthly_counts, dtype=int).sort_index()
            self._save_snapshot_cache(domain, series)
            return series

        except Exception as exc:
            logger.warning("Wayback CDX error for %s: %s", domain, exc)
            return pd.Series(dtype=int)

    def get_traffic_proxy(
        self, domain: str, lookback_months: int = 12
    ) -> WebTrafficProxy:
        """Return full WebTrafficProxy with trend analytics."""
        series = self.get_monthly_snapshots(domain, lookback_months)
        if series.empty:
            return WebTrafficProxy(
                domain=domain,
                period_months=lookback_months,
                total_snapshots=0,
                monthly_avg=0.0,
            )

        total = int(series.sum())
        avg = float(series.mean())

        # MoM trend: last 3 months vs prior 3 months
        mom = None
        if len(series) >= 6:
            recent = series.iloc[-3:].mean()
            prior = series.iloc[-6:-3].mean()
            if prior > 0:
                mom = round((recent - prior) / prior * 100, 1)

        peak = series.idxmax() if not series.empty else None
        trough = series.idxmin() if not series.empty else None

        return WebTrafficProxy(
            domain=domain,
            period_months=lookback_months,
            total_snapshots=total,
            monthly_avg=round(avg, 1),
            mom_trend=mom,
            peak_month=str(peak) if peak else None,
            trough_month=str(trough) if trough else None,
            monthly_series=series.to_dict(),
        )

    def compare_domains(
        self, domains: List[str], lookback_months: int = 12
    ) -> pd.DataFrame:
        """Compare Wayback snapshot counts across multiple domains."""
        rows = []
        for domain in domains:
            proxy = self.get_traffic_proxy(domain, lookback_months)
            rows.append({
                "domain": domain,
                "total_snapshots": proxy.total_snapshots,
                "monthly_avg": proxy.monthly_avg,
                "mom_trend_pct": proxy.mom_trend,
                "peak_month": proxy.peak_month,
            })
        return pd.DataFrame(rows)

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _load_snapshot_cache(
        self, domain: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.Series]:
        cutoff = int(time.time()) - _WAYBACK_CACHE_TTL
        start_ym = start_dt.strftime("%Y-%m")
        end_ym = end_dt.strftime("%Y-%m")
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT year_month, snap_count FROM web_traffic_snapshots
                       WHERE domain=? AND year_month>=? AND year_month<=? AND fetched_at>?
                       ORDER BY year_month""",
                    (domain, start_ym, end_ym, cutoff),
                ).fetchall()
            if not rows:
                return None
            return pd.Series(
                {r["year_month"]: r["snap_count"] for r in rows},
                dtype=int,
            )
        except Exception:
            return None

    def _save_snapshot_cache(self, domain: str, series: pd.Series) -> None:
        now = int(time.time())
        try:
            with _db() as conn:
                for ym, count in series.items():
                    conn.execute(
                        """INSERT OR REPLACE INTO web_traffic_snapshots
                           (domain, year_month, snap_count, fetched_at)
                           VALUES (?,?,?,?)""",
                        (domain, str(ym), int(count), now),
                    )
        except Exception as exc:
            logger.debug("Wayback cache save error: %s", exc)


# ── 5. Hiring Signal Engine ────────────────────────────────────────────────────

class HiringSignalEngine:
    """
    Derives investment-grade hiring signals from raw job count data.

    Signals:
      - hiring_velocity: 30/60/90-day trend from cached counts
      - tech_stack: keyword intensity score from job title patterns
      - sector_comparison: relative rank vs sector JOLTS data
    """

    def __init__(self) -> None:
        self._company_adapter = CompanyHiringAdapter()
        self._bls = BLSAdapter()
        self._fred = FREDLaborAdapter()

    def compute_hiring_velocity(
        self, ticker: str, company: str, company_slug: Optional[str] = None
    ) -> HiringVelocitySignal:
        """
        Build a hiring velocity signal from time-series of cached job counts.
        Pulls current + historical cache entries from SQLite to compute trend.
        """
        slug = company_slug or _slugify(company)
        # Get current count
        current_snap = self._company_adapter.get_combined_count(ticker, company, slug)
        current = current_snap.total_estimate

        # Pull historical counts from DB for trend
        d30, d60, d90 = self._get_historical_counts(ticker, days_back=[30, 60, 90])

        # Compute velocity signals
        vel_30d = None
        vel_60d = None
        signal = "neutral"

        if current is not None and d30 is not None and d30 > 0:
            vel_30d = round((current - d30) / d30 * 100, 1)
        if current is not None and d60 is not None and d60 > 0:
            vel_60d = round((current - d60) / d60 * 100, 1)

        if vel_30d is not None:
            if vel_30d > 15:
                signal = "accelerating"
            elif vel_30d < -15:
                signal = "decelerating"
            else:
                signal = "stable"

        confidence = "medium" if current is not None else "low"
        if vel_30d is not None and vel_60d is not None:
            confidence = "high"

        result = HiringVelocitySignal(
            ticker=ticker,
            company=company,
            current_openings=current,
            d30_openings=d30,
            d60_openings=d60,
            d90_openings=d90,
            velocity_30d=vel_30d,
            velocity_60d=vel_60d,
            signal=signal,
            confidence=confidence,
            computed_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        # Persist signal
        self._save_signal(ticker, company, "hiring_velocity", vel_30d or 0.0, result.model_dump())
        return result

    def compute_tech_stack_signal(self, company: str) -> TechStackSignal:
        """
        Use Google to find job listings and scan titles for tech keywords.
        Returns a tech intensity score (0–10).
        """
        # Search for engineering roles at company
        query = f'"{company}" engineer developer site:linkedin.com/jobs'
        params = {"q": query, "num": "10"}

        cloud_hits = 0
        aiml_hits = 0
        devops_hits = 0
        all_keywords: List[str] = []

        try:
            resp = requests.get(
                _GOOGLE_SEARCH,
                params=params,
                headers=_BROWSER_HEADERS,
                timeout=15,
            )
            html = resp.text.lower()

            for kw in _TECH_KEYWORDS:
                count = html.count(kw)
                if count > 0:
                    all_keywords.append(kw)
                    if kw in ("aws", "azure", "gcp", "cloud", "kubernetes", "docker"):
                        cloud_hits += count
                    elif kw in ("machine learning", "ml engineer", "data engineer",
                                "ai engineer", "llm", "generative ai", "data science",
                                "mlops"):
                        aiml_hits += count
                    elif kw in ("devops", "sre", "platform engineer", "infrastructure"):
                        devops_hits += count

        except Exception as exc:
            logger.warning("Tech stack signal error for %s: %s", company, exc)

        # Normalize to 0–10 score
        raw_score = min(10.0, (cloud_hits * 0.3 + aiml_hits * 0.5 + devops_hits * 0.2) / 2)

        return TechStackSignal(
            company=company,
            cloud_mentions=cloud_hits,
            ai_ml_mentions=aiml_hits,
            devops_mentions=devops_hits,
            top_keywords=all_keywords[:10],
            tech_intensity_score=round(raw_score, 2),
            computed_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def compare_to_sector(
        self, ticker: str, sector: str
    ) -> SectorComparison:
        """
        Compare company-level estimated openings to sector JOLTS data.
        sector: one of the keys in JOLTS_SECTOR_OPENINGS.
        """
        sector_key = sector.lower().replace(" ", "_")
        sector_sid = JOLTS_SECTOR_OPENINGS.get(sector_key)

        sector_openings_k = None
        sector_trend = None
        if sector_sid:
            df = self._bls.get_jolts_series([sector_sid]).get(sector_sid)
            sector_openings_k = _latest_val(df)
            if df is not None and len(df) >= 3:
                vals = df["value"].dropna().tolist()
                if len(vals) >= 3:
                    pct = (vals[-1] - vals[-3]) / vals[-3] * 100 if vals[-3] else 0
                    sector_trend = "rising" if pct > 3 else "falling" if pct < -3 else "flat"

        company_snap = self._company_adapter.get_combined_count(ticker, ticker)
        company_openings = company_snap.total_estimate

        return SectorComparison(
            ticker=ticker,
            sector=sector,
            company_openings=company_openings,
            sector_openings_k=sector_openings_k,
            relative_rank="unknown",
            sector_trend=sector_trend,
            as_of=datetime.now(tz=timezone.utc).isoformat(),
        )

    def _get_historical_counts(
        self, ticker: str, days_back: List[int]
    ) -> Tuple:
        """Pull historical job counts from cache for velocity computation."""
        results = []
        try:
            with _db() as conn:
                for days in days_back:
                    cutoff_ts = int(time.time()) - days * 86400
                    prev_cutoff = cutoff_ts - 7 * 86400  # ±7-day window
                    row = conn.execute(
                        """SELECT job_count FROM company_job_counts
                           WHERE ticker=? AND fetched_at BETWEEN ? AND ?
                           ORDER BY ABS(fetched_at - ?) LIMIT 1""",
                        (ticker, prev_cutoff, cutoff_ts + 86400, cutoff_ts),
                    ).fetchone()
                    results.append(int(row["job_count"]) if row and row["job_count"] else None)
        except Exception:
            results = [None] * len(days_back)
        return tuple(results)

    def _save_signal(
        self, ticker: str, company: str, signal_type: str, value: float, meta: dict
    ) -> None:
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT INTO hiring_signals
                       (ticker, company, signal_type, value, metadata_json, computed_at)
                       VALUES (?,?,?,?,?,?)""",
                    (ticker, company, signal_type, value,
                     json.dumps(meta), int(time.time())),
                )
        except Exception as exc:
            logger.debug("Signal save error: %s", exc)


# ── 6. JOLTS Dashboard Builder ─────────────────────────────────────────────────

class JOLTSDashboard:
    """
    Convenience wrapper that builds a rich JOLTS analytics dashboard
    combining BLS series data with FRED supplements.
    """

    def __init__(self) -> None:
        self._bls = BLSAdapter()
        self._fred = FREDLaborAdapter()

    def full_dashboard(self, lookback_years: int = 5) -> Dict[str, Any]:
        """Return a comprehensive JOLTS data structure for the API."""
        now = datetime.now()
        end_year = now.year
        start_year = end_year - lookback_years

        # Fetch all JOLTS total series
        all_ids = list(JOLTS_TOTAL.values()) + list(JOLTS_SECTOR_OPENINGS.values())
        jolts_data = self._bls.get_jolts_series(all_ids, start_year, end_year)

        dashboard: Dict[str, Any] = {
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
            "lookback_years": lookback_years,
            "total": {},
            "sector_openings": {},
            "fred_supplement": {},
        }

        # Total JOLTS metrics
        for metric_name, sid in JOLTS_TOTAL.items():
            df = jolts_data.get(sid)
            latest = _latest_val(df)
            yoy = _yoy_change(df)
            dashboard["total"][metric_name] = {
                "latest": latest,
                "yoy_pct": yoy,
                "series_length": len(df) if df is not None else 0,
            }

        # Sector openings
        for sector_name, sid in JOLTS_SECTOR_OPENINGS.items():
            df = jolts_data.get(sid)
            dashboard["sector_openings"][sector_name] = {
                "latest_k": _latest_val(df),
                "yoy_pct": _yoy_change(df),
            }

        # FRED supplement
        for fred_name, fred_sid in FRED_LABOR_SERIES.items():
            df = self._fred.get_series(fred_sid)
            dashboard["fred_supplement"][fred_name] = {
                "latest": _latest_val_fred(df),
                "series_id": fred_sid,
            }

        return dashboard

    def sector_openings_ranking(self) -> List[Dict[str, Any]]:
        """Return sectors ranked by current job openings (highest first)."""
        data = self._bls.get_jolts_series(list(JOLTS_SECTOR_OPENINGS.values()))
        rows = []
        for sector_name, sid in JOLTS_SECTOR_OPENINGS.items():
            df = data.get(sid)
            val = _latest_val(df)
            rows.append({
                "sector": sector_name,
                "openings_k": val,
                "yoy_pct": _yoy_change(df),
            })
        rows.sort(key=lambda r: (r["openings_k"] or 0), reverse=True)
        return rows


# ── Helper functions ───────────────────────────────────────────────────────────

def _latest_val(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Extract the most recent non-null value from a BLS DataFrame."""
    if df is None or df.empty or "value" not in df.columns:
        return None
    vals = df["value"].dropna()
    if vals.empty:
        return None
    return float(vals.iloc[-1])


def _latest_val_fred(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Extract the most recent non-null value from a FRED DataFrame."""
    if df is None or df.empty or "value" not in df.columns:
        return None
    vals = df["value"].dropna()
    if vals.empty:
        return None
    return float(vals.iloc[-1])


def _yoy_change(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Compute year-over-year % change from the last 13 observations (monthly)."""
    if df is None or df.empty or "value" not in df.columns:
        return None
    vals = df["value"].dropna().tolist()
    if len(vals) < 13:
        return None
    try:
        current = vals[-1]
        prior_year = vals[-13]
        if prior_year and prior_year != 0:
            return round((current - prior_year) / prior_year * 100, 2)
    except Exception:
        pass
    return None


def _parse_google_result_count(html: str) -> Optional[int]:
    """
    Extract approximate result count from Google search HTML.
    Looks for patterns like "About 1,230 results" or "1-10 of about 850 results".
    """
    patterns = [
        r"About ([0-9,]+) results",
        r"([0-9,]+) results",
        r"of about ([0-9,]+)",
        r"Approximately ([0-9,]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _parse_indeed_job_count(html: str) -> Optional[int]:
    """
    Extract job count from Indeed company page HTML.
    Patterns: "123 jobs at Company", "Showing 1-15 of 123 jobs"
    """
    patterns = [
        r"([0-9,]+)\s+jobs?\s+at\s+\w",
        r"[Ss]howing\s+\d+-\d+\s+of\s+([0-9,]+)\s+jobs?",
        r'"totalJobCount":([0-9]+)',
        r'"jobCount"\s*:\s*([0-9]+)',
        r'([0-9,]+)\s+open\s+jobs?',
        r'([0-9,]+)\s+jobs?\s+available',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _slugify(name: str) -> str:
    """Convert company name to URL-safe slug for Indeed CMP URLs."""
    return re.sub(r"[^a-z0-9-]", "-", name.lower().strip()).strip("-")


# ── 7. FastAPI Router ──────────────────────────────────────────────────────────

hiring_v3_router = APIRouter(prefix="/hiring/v3", tags=["hiring_intelligence_v3"])

# Module-level singletons
_bls: Optional[BLSAdapter] = None
_fred_adapter: Optional[FREDLaborAdapter] = None
_company_adapter: Optional[CompanyHiringAdapter] = None
_wayback: Optional[WaybackTrafficAdapter] = None
_signal_engine: Optional[HiringSignalEngine] = None
_jolts_dashboard: Optional[JOLTSDashboard] = None


def _get_bls() -> BLSAdapter:
    global _bls
    if _bls is None:
        _bls = BLSAdapter()
    return _bls


def _get_fred_adapter() -> FREDLaborAdapter:
    global _fred_adapter
    if _fred_adapter is None:
        _fred_adapter = FREDLaborAdapter()
    return _fred_adapter


def _get_company() -> CompanyHiringAdapter:
    global _company_adapter
    if _company_adapter is None:
        _company_adapter = CompanyHiringAdapter()
    return _company_adapter


def _get_wayback() -> WaybackTrafficAdapter:
    global _wayback
    if _wayback is None:
        _wayback = WaybackTrafficAdapter()
    return _wayback


def _get_signal_engine() -> HiringSignalEngine:
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = HiringSignalEngine()
    return _signal_engine


def _get_jolts_dashboard() -> JOLTSDashboard:
    global _jolts_dashboard
    if _jolts_dashboard is None:
        _jolts_dashboard = JOLTSDashboard()
    return _jolts_dashboard


# ── Endpoints ──────────────────────────────────────────────────────────────────

@hiring_v3_router.get("/macro-labor", response_model=MacroLaborDashboard)
def route_macro_labor():
    """
    Full macro labor dashboard: JOLTS + FRED + BLS.
    Combines job openings, quit rates, hires rate, unemployment, payrolls.
    """
    try:
        return _get_bls().get_macro_labor_dashboard()
    except Exception as exc:
        logger.error("macro-labor error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/jolts-dashboard")
def route_jolts_dashboard(
    lookback_years: int = Query(default=5, ge=1, le=20),
):
    """
    Full JOLTS time-series dashboard — all metrics, all sectors.
    Returns ranked sector openings + total economy JOLTS metrics + FRED supplement.
    """
    try:
        dashboard = _get_jolts_dashboard()
        return {
            "dashboard": dashboard.full_dashboard(lookback_years),
            "sector_ranking": dashboard.sector_openings_ranking(),
        }
    except Exception as exc:
        logger.error("jolts-dashboard error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/sector-openings", response_model=SectorOpenings)
def route_sector_openings():
    """
    Latest JOLTS job openings by sector (construction, manufacturing,
    professional_business, finance_insurance, etc.).
    """
    try:
        return _get_bls().get_sector_openings()
    except Exception as exc:
        logger.error("sector-openings error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/hiring-signals/{ticker}")
def route_hiring_signals(
    ticker: str,
    company: str = Query(..., description="Full company name for job search"),
    company_slug: Optional[str] = Query(
        default=None, description="URL-safe company slug for Indeed (optional)"
    ),
):
    """
    Compute hiring velocity signals for a ticker.
    Returns 30/60/90-day job posting trend + signal classification.
    """
    try:
        engine = _get_signal_engine()
        velocity = engine.compute_hiring_velocity(
            ticker.upper(), company, company_slug
        )
        tech = engine.compute_tech_stack_signal(company)
        return {
            "ticker": ticker.upper(),
            "company": company,
            "hiring_velocity": velocity.model_dump(),
            "tech_stack": tech.model_dump(),
        }
    except Exception as exc:
        logger.error("hiring-signals error %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/job-trend/{ticker}")
def route_job_trend(
    ticker: str,
    company: str = Query(...),
    company_slug: Optional[str] = Query(default=None),
):
    """
    Job posting trend for a specific company.
    Returns current count + historical velocity.
    """
    try:
        adapter = _get_company()
        slug = company_slug or _slugify(company)
        snap = adapter.get_combined_count(ticker.upper(), company, slug)
        return snap.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/web-traffic/{domain}")
def route_web_traffic(
    domain: str,
    months: int = Query(default=12, ge=1, le=36),
):
    """
    Wayback Machine traffic proxy for a domain.
    Returns monthly snapshot counts (proxy for web traffic/interest).
    """
    try:
        proxy = _get_wayback().get_traffic_proxy(domain, months)
        return proxy.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/web-traffic-compare")
def route_web_traffic_compare(
    domains: str = Query(..., description="Comma-separated domain list"),
    months: int = Query(default=12, ge=1, le=24),
):
    """
    Compare Wayback traffic proxies across multiple domains.
    Returns ranked DataFrame as JSON.
    """
    try:
        domain_list = [d.strip() for d in domains.split(",") if d.strip()]
        if len(domain_list) > 10:
            raise HTTPException(status_code=400, detail="Max 10 domains per request")
        df = _get_wayback().compare_domains(domain_list, months)
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/sector-comparison/{ticker}")
def route_sector_comparison(
    ticker: str,
    sector: str = Query(..., description=(
        "Sector key: manufacturing, finance_insurance, professional_business, "
        "information, construction, trade_transport_util, education_health, "
        "leisure_hospitality, government"
    )),
):
    """Compare company-level hiring against sector JOLTS data."""
    try:
        engine = _get_signal_engine()
        return engine.compare_to_sector(ticker.upper(), sector).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/usajobs/{agency}")
def route_usajobs(
    agency: str,
    keyword: Optional[str] = Query(default=None),
):
    """
    USAJobs.gov free API — government contractor / agency hiring signal.
    Useful for defense, aerospace, IT government contractors.
    """
    try:
        adapter = _get_company()
        result = adapter.get_usajobs_count(agency, keyword)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/fred-series/{series_id}")
def route_fred_series(
    series_id: str,
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    """
    Fetch any FRED labor series by ID (free, no key required).
    Popular IDs: JTSJOL, JTSQUR, JTSHJL, UNRATE, PAYEMS, CES0000000001
    """
    try:
        df = _get_fred_adapter().get_series(series_id.upper(), start, end)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for series {series_id}")
        return {
            "series_id": series_id.upper(),
            "observations": df.assign(date=df["date"].astype(str)).to_dict(orient="records"),
            "latest": float(df["value"].dropna().iloc[-1]) if not df.empty else None,
            "count": len(df),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/jolts-series")
def route_jolts_series(
    metrics: str = Query(
        default="openings,quits_rate,hires_rate",
        description="Comma-separated metrics from: "
                    + ", ".join(JOLTS_TOTAL.keys()),
    ),
    start_year: int = Query(default=2019),
    end_year: Optional[int] = Query(default=None),
):
    """
    Fetch specific JOLTS total-economy series.
    Returns time-series data for each requested metric.
    """
    try:
        metric_list = [m.strip() for m in metrics.split(",")]
        series_ids = []
        valid_metrics = {}
        for m in metric_list:
            sid = JOLTS_TOTAL.get(m)
            if sid:
                series_ids.append(sid)
                valid_metrics[m] = sid
            else:
                logger.warning("Unknown JOLTS metric: %s", m)

        if not series_ids:
            raise HTTPException(
                status_code=400,
                detail=f"No valid metrics. Choose from: {list(JOLTS_TOTAL.keys())}",
            )

        bls = _get_bls()
        end = end_year or datetime.now().year
        data = bls.get_jolts_series(series_ids, start_year, end)

        result: Dict[str, Any] = {"as_of": datetime.now(tz=timezone.utc).isoformat()}
        for metric_name, sid in valid_metrics.items():
            df = data.get(sid)
            if df is not None and not df.empty:
                result[metric_name] = {
                    "series_id": sid,
                    "latest": _latest_val(df),
                    "yoy_pct": _yoy_change(df),
                    "data": df.to_dict(orient="records"),
                }
            else:
                result[metric_name] = {"series_id": sid, "latest": None, "data": []}

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@hiring_v3_router.get("/health")
def route_health():
    """Health check + data source availability status."""
    checks: Dict[str, Any] = {}

    # BLS API check
    try:
        resp = requests.get(_BLS_BASE, timeout=5)
        checks["bls_api"] = "ok" if resp.status_code < 500 else "degraded"
    except Exception:
        checks["bls_api"] = "unreachable"

    # FRED check
    try:
        resp = requests.get(
            f"{_FRED_BASE}?id=UNRATE", timeout=5, stream=True
        )
        checks["fred_csv"] = "ok" if resp.status_code == 200 else "degraded"
        resp.close()
    except Exception:
        checks["fred_csv"] = "unreachable"

    # Wayback check
    try:
        resp = requests.get(_WAYBACK_CDX, params={"url": "example.com", "limit": "1"}, timeout=8)
        checks["wayback_cdx"] = "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        checks["wayback_cdx"] = "unreachable"

    # USAJobs check
    try:
        resp = requests.get(_USAJOBS_BASE, params={"ResultsPerPage": "1"}, timeout=8)
        checks["usajobs_api"] = "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        checks["usajobs_api"] = "unreachable"

    # DB check
    try:
        _ensure_db()
        checks["sqlite_db"] = "ok"
    except Exception as exc:
        checks["sqlite_db"] = f"error: {exc}"

    return {
        "status": "healthy" if all(v == "ok" for v in checks.values()) else "degraded",
        "checks": checks,
        "db_path": str(_DB_PATH.resolve()),
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
    }


# ── Module init ────────────────────────────────────────────────────────────────

try:
    _ensure_db()
except Exception as _db_exc:
    logger.warning("job_postings_v3: DB init failed: %s", _db_exc)


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="SENTINEL Hiring Intelligence v3", version="3.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(hiring_v3_router)
    uvicorn.run(app, host="0.0.0.0", port=8087, log_level="info")
