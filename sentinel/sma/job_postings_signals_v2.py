"""
Job Postings & Web Traffic Alternative Data Signals v2 — Dimension #086 (target score 9).

Comprehensive alternative data signals derived from job postings (Indeed RSS), BLS/JOLTS
FRED series, Wayback Machine CDX API web traffic proxies, and derived hiring intelligence.

Bloomberg equivalent: BDATA <GO> → Alternative Data → Employment.
SENTINEL exclusive on web traffic proxies and hiring composition signals.

Public API
----------
JobPostingsScraper
    fetch_indeed_postings(query, pages)         -> list[JobPosting]
    fetch_company_postings(ticker, company)     -> CompanyHiringSnapshot

BLSFREDAdapter
    get_jolts_fred(series_id, start, end)       -> pd.DataFrame
    get_sector_openings()                       -> dict[str, float]
    get_quits_rate()                            -> dict[str, float]
    get_macro_labor_dashboard()                 -> MacroLaborDashboard

WaybackCDXAdapter
    get_capture_counts(domain, year)            -> pd.Series
    get_traffic_proxy(domain, lookback_months)  -> WebTrafficProxy
    compare_domains(domains, lookback_months)   -> pd.DataFrame

HiringSignalEngine
    compute_hiring_velocity(ticker, company)    -> HiringVelocitySignal
    detect_tech_stack(postings)                 -> TechStackSignal
    detect_geo_expansion(postings)              -> GeoExpansionSignal
    detect_exec_hires(postings)                 -> ExecHireSignal
    compute_role_composition(postings)          -> RoleCompositionSignal
    correlate_with_returns(ticker, prices)      -> HiringReturnCorrelation

SQLite storage: .sentinel/cache/job_postings_v2.db
FastAPI router: job_postings_v2_router, prefix /alt/hiring

Free data sources only:
  - Indeed RSS: https://www.indeed.com/rss?q={company}&sort=date
  - FRED free CSV / API: JTSJOL, JTSQUR, sector JOLTS series
  - Wayback Machine CDX API: http://web.archive.org/cdx/search/cdx
  - BLS Public API v2 (no key required for limited series)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus, urlencode

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "JobPostingsScraper",
    "BLSFREDAdapter",
    "WaybackCDXAdapter",
    "HiringSignalEngine",
    "JobPostingsCache",
    "job_postings_v2_router",
    "JobPosting",
    "CompanyHiringSnapshot",
    "HiringVelocitySignal",
    "TechStackSignal",
    "GeoExpansionSignal",
    "ExecHireSignal",
    "RoleCompositionSignal",
    "MacroLaborDashboard",
    "WebTrafficProxy",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(".sentinel") / "cache"
_CACHE_DB = _CACHE_DIR / "job_postings_v2.db"

_POSTINGS_TTL_HOURS = 6
_WEB_TRAFFIC_TTL_HOURS = 24
_MACRO_TTL_HOURS = 12

_REQUEST_DELAY_MIN = 2.0
_REQUEST_DELAY_MAX = 5.0

INDEED_RSS_URL = "https://www.indeed.com/rss?q={query}&sort=date&limit=50"
WAYBACK_CDX_URL = "http://web.archive.org/cdx/search/cdx"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# FRED series IDs (all free, no API key required for CSV download)
FRED_SERIES: dict[str, str] = {
    "JTSJOL": "Total Job Openings (thousands)",
    "JTSQUR": "Total Quits Rate",
    "JTSHIL": "Total Hires (thousands)",
    "JTSTSL": "Total Separations (thousands)",
    "JTSLOL": "Total Layoffs and Discharges (thousands)",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Total Nonfarm Payrolls (thousands)",
    "MANEMP": "Manufacturing Employment (thousands)",
    "USINFO": "Information Sector Employment (thousands)",
    "USFIRE": "Finance and Insurance Employment (thousands)",
    "USSTHPI": "Professional and Technical Services Employment (thousands)",
}

FRED_SECTOR_OPENINGS: dict[str, str] = {
    "manufacturing": "JTS3000JOL",
    "retail_trade": "JTS4400JOL",
    "information": "JTS5100JOL",
    "finance_insurance": "JTS5200JOL",
    "professional_services": "JTS5400JOL",
    "health_education": "JTS6000JOL",
    "leisure_hospitality": "JTS7000JOL",
    "government": "JTS9000JOL",
}

# Tech stack keywords to detect from job descriptions
TECH_STACK_KEYWORDS: dict[str, list[str]] = {
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "eks", "ecs"],
    "azure": ["azure", "microsoft azure", "azure devops", "aks"],
    "gcp": ["gcp", "google cloud", "bigquery", "gke", "cloud run"],
    "kubernetes": ["kubernetes", "k8s", "helm", "kubectl"],
    "terraform": ["terraform", "infrastructure as code", "iac"],
    "python": ["python", "django", "flask", "fastapi"],
    "java": ["java", "spring boot", "jvm"],
    "golang": ["golang", "go lang"],
    "rust": ["rust", "cargo"],
    "typescript": ["typescript", "react", "nextjs", "next.js"],
    "ml_ai": ["machine learning", "deep learning", "tensorflow", "pytorch", "llm", "nlp"],
    "data_engineering": ["spark", "kafka", "airflow", "dbt", "snowflake", "databricks"],
    "blockchain": ["blockchain", "solidity", "web3", "defi", "smart contract"],
}

# Executive / senior role keywords
EXEC_ROLE_KEYWORDS = [
    "chief executive", "ceo", "chief financial", "cfo", "chief technology", "cto",
    "chief operating", "coo", "chief product", "cpo", "chief marketing", "cmo",
    "chief revenue", "cro", "chief data", "cdo", "chief information", "ciso",
    "vice president", "vp of", "head of", "general manager", "director of",
    "managing director", "president", "chief of staff",
]

# Role category mapping for composition analysis
ROLE_CATEGORIES: dict[str, list[str]] = {
    "engineering": [
        "engineer", "developer", "programmer", "architect", "devops", "sre",
        "data scientist", "ml engineer", "software", "backend", "frontend", "fullstack",
    ],
    "sales": [
        "sales", "account executive", "business development", "account manager",
        "enterprise sales", "regional sales", "sales representative",
    ],
    "marketing": [
        "marketing", "content", "seo", "growth", "brand", "demand generation",
        "product marketing", "social media",
    ],
    "finance": [
        "financial analyst", "accountant", "controller", "finance manager",
        "fp&a", "treasury", "tax", "audit",
    ],
    "operations": [
        "operations", "supply chain", "logistics", "warehouse", "fulfillment",
        "customer success", "customer support", "customer service",
    ],
    "hr": [
        "human resources", "recruiter", "talent acquisition", "people operations",
        "hr business partner", "compensation", "benefits",
    ],
    "product": [
        "product manager", "product owner", "product lead", "ux", "ui designer",
        "user experience", "user research",
    ],
    "legal_compliance": [
        "lawyer", "counsel", "legal", "compliance", "regulatory", "paralegal",
    ],
}

# Major US cities for geographic expansion detection
MAJOR_CITIES = [
    "new york", "san francisco", "los angeles", "chicago", "seattle", "austin",
    "boston", "denver", "miami", "atlanta", "dallas", "houston", "phoenix",
    "portland", "san diego", "washington dc", "nashville", "minneapolis",
    "salt lake city", "raleigh", "charlotte", "detroit", "philadelphia",
    "remote", "hybrid", "nationwide",
]

# Known company domains for web traffic proxies
TICKER_TO_DOMAIN: dict[str, str] = {
    "AAPL": "apple.com",
    "MSFT": "microsoft.com",
    "GOOGL": "google.com",
    "AMZN": "amazon.com",
    "META": "meta.com",
    "NFLX": "netflix.com",
    "TSLA": "tesla.com",
    "NVDA": "nvidia.com",
    "AMD": "amd.com",
    "INTC": "intel.com",
    "CRM": "salesforce.com",
    "ORCL": "oracle.com",
    "IBM": "ibm.com",
    "ADBE": "adobe.com",
    "SHOP": "shopify.com",
    "SQ": "block.xyz",
    "PYPL": "paypal.com",
    "UBER": "uber.com",
    "LYFT": "lyft.com",
    "ABNB": "airbnb.com",
    "SNAP": "snap.com",
    "TWTR": "twitter.com",
    "SPOT": "spotify.com",
    "ZM": "zoom.us",
    "DOCU": "docusign.com",
    "OKTA": "okta.com",
    "SNOW": "snowflake.com",
    "PLTR": "palantir.com",
    "COIN": "coinbase.com",
    "RBLX": "roblox.com",
    "U": "unity.com",
    "NET": "cloudflare.com",
    "DDOG": "datadoghq.com",
    "CRWD": "crowdstrike.com",
    "ZS": "zscaler.com",
    "PANW": "paloaltonetworks.com",
    "NOW": "servicenow.com",
    "WDAY": "workday.com",
    "VEEV": "veeva.com",
    "HUB": "hubspot.com",
}

_HEADERS = {
    "User-Agent": (
        "SENTINEL:FinancialTerminal:2.0 (research; contact: sentinel@example.com)"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

_HTTP_HEADERS = {
    "User-Agent": (
        "SENTINEL:FinancialTerminal:2.0 (research; contact: sentinel@example.com)"
    ),
    "Accept": "application/json, text/csv, */*",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobPosting:
    title: str
    company: str
    location: str
    description: str
    url: str
    posted_date: Optional[date]
    query: str
    raw_categories: list[str] = field(default_factory=list)


@dataclass
class CompanyHiringSnapshot:
    ticker: str
    company: str
    as_of: date
    total_postings_30d: int
    total_postings_90d: int
    postings_yoy_change_pct: Optional[float]
    top_roles: list[str]
    top_locations: list[str]
    role_composition: dict[str, float]
    tech_stack_mentions: dict[str, int]
    exec_hire_count: int
    new_cities: list[str]
    raw_postings: list[JobPosting] = field(default_factory=list)


@dataclass
class HiringVelocitySignal:
    ticker: str
    company: str
    as_of: date
    velocity_90d: float          # rolling 90d posting volume change %
    velocity_30d: float          # rolling 30d posting volume change %
    acceleration: float          # second derivative of hiring rate
    signal: str                  # "accelerating" | "decelerating" | "stable" | "surge" | "collapse"
    z_score: float
    interpretation: str


@dataclass
class TechStackSignal:
    ticker: str
    as_of: date
    cloud_provider: str          # "aws" | "azure" | "gcp" | "multi-cloud" | "unknown"
    cloud_dominance_pct: float   # % of job postings mentioning dominant cloud
    ml_ai_adoption_pct: float
    data_engineering_pct: float
    modernization_score: float   # 0-10, higher = more cloud-native/modern stack
    top_tech_mentions: list[tuple[str, int]]


@dataclass
class GeoExpansionSignal:
    ticker: str
    as_of: date
    current_cities: list[str]
    new_cities_vs_prior: list[str]   # cities appearing for first time vs 90d ago
    expansion_score: float           # 0-10, 10 = aggressive expansion
    primary_markets: list[str]
    remote_pct: float


@dataclass
class ExecHireSignal:
    ticker: str
    as_of: date
    exec_postings: list[str]         # job titles of exec openings
    exec_hire_count: int
    c_suite_count: int
    vp_count: int
    is_strategic_pivot: bool
    pivot_reason: str
    signal_strength: str             # "strong" | "moderate" | "weak" | "none"


@dataclass
class RoleCompositionSignal:
    ticker: str
    as_of: date
    composition: dict[str, float]    # category -> % of postings
    growth_stage: str                # "early" | "scaling" | "mature" | "restructuring"
    eng_to_sales_ratio: float
    eng_to_support_ratio: float
    interpretation: str


@dataclass
class HiringReturnCorrelation:
    ticker: str
    lag_weeks: int
    correlation: float
    p_value: float
    is_significant: bool
    direction: str                   # "positive" | "negative" | "neutral"
    r_squared: float
    sample_size: int


@dataclass
class MacroLaborDashboard:
    as_of: date
    total_job_openings_k: Optional[float]
    total_quits_rate: Optional[float]
    total_hires_k: Optional[float]
    unemployment_rate: Optional[float]
    nonfarm_payrolls_k: Optional[float]
    sector_openings: dict[str, float]
    tight_labor_sectors: list[str]
    slack_labor_sectors: list[str]
    macro_signal: str                # "tight" | "loosening" | "slack" | "neutral"
    yoy_openings_change_pct: Optional[float]
    interpretation: str


@dataclass
class WebTrafficProxy:
    domain: str
    ticker: Optional[str]
    as_of: date
    monthly_captures: pd.Series     # index=date, values=snapshot_count (traffic proxy)
    trend_direction: str            # "growing" | "declining" | "stable"
    yoy_change_pct: Optional[float]
    mom_change_pct: Optional[float]
    z_score_latest: float


# ---------------------------------------------------------------------------
# SQLite Cache
# ---------------------------------------------------------------------------


class JobPostingsCache:
    """
    SQLite-backed cache for job postings, web traffic proxies, and macro data.

    Tables
    ------
    cache_entries      — generic key/value TTL cache (JSON payloads)
    job_postings       — persisted raw postings for velocity computation
    web_traffic        — CDX snapshot counts by domain and month
    """

    def __init__(self, db_path: Path = _CACHE_DB) -> None:
        self._db = db_path
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key   TEXT PRIMARY KEY,
                    payload     TEXT NOT NULL,
                    cached_at   TEXT NOT NULL,
                    ttl_hours   REAL NOT NULL DEFAULT 24
                );
                CREATE INDEX IF NOT EXISTS ix_cache_at
                    ON cache_entries(cached_at);

                CREATE TABLE IF NOT EXISTS job_postings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker          TEXT,
                    company         TEXT NOT NULL,
                    query           TEXT NOT NULL,
                    title           TEXT NOT NULL,
                    location        TEXT,
                    description     TEXT,
                    url             TEXT,
                    posted_date     TEXT,
                    fetched_at      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_jp_ticker
                    ON job_postings(ticker, fetched_at);
                CREATE INDEX IF NOT EXISTS ix_jp_company
                    ON job_postings(company, fetched_at);

                CREATE TABLE IF NOT EXISTS web_traffic_snapshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain      TEXT NOT NULL,
                    year_month  TEXT NOT NULL,
                    capture_count INTEGER NOT NULL,
                    fetched_at  TEXT NOT NULL,
                    UNIQUE(domain, year_month)
                );
                CREATE INDEX IF NOT EXISTS ix_wts_domain
                    ON web_traffic_snapshots(domain, year_month);
                """
            )

    # ── generic cache ─────────────────────────────────────────────────────────

    def get(self, key: str, ttl_hours: float) -> Optional[str]:
        cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
        with self._conn() as con:
            row = con.execute(
                "SELECT payload FROM cache_entries "
                "WHERE cache_key=? AND cached_at>=?",
                (key, cutoff),
            ).fetchone()
        return row["payload"] if row else None

    def set(self, key: str, payload: str, ttl_hours: float = 24.0) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO cache_entries(cache_key, payload, cached_at, ttl_hours)
                VALUES(?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    cached_at=excluded.cached_at,
                    ttl_hours=excluded.ttl_hours
                """,
                (key, payload, now, ttl_hours),
            )

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(list(parts), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_json(self, key: str, ttl_hours: float = 24.0) -> Optional[dict]:
        raw = self.get(key, ttl_hours)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set_json(self, key: str, data: dict, ttl_hours: float = 24.0) -> None:
        self.set(key, json.dumps(data, default=str), ttl_hours)

    # ── postings storage ──────────────────────────────────────────────────────

    def save_postings(
        self,
        postings: list[JobPosting],
        ticker: Optional[str],
        company: str,
        query: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.executemany(
                """
                INSERT INTO job_postings
                    (ticker, company, query, title, location, description, url,
                     posted_date, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        ticker, company, query,
                        p.title, p.location, p.description[:2000],
                        p.url,
                        p.posted_date.isoformat() if p.posted_date else None,
                        now,
                    )
                    for p in postings
                ],
            )

    def load_postings_last_n_days(
        self,
        company: str,
        days: int,
        ticker: Optional[str] = None,
    ) -> list[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT title, company, location, description, url, posted_date,
                       query, fetched_at
                FROM job_postings
                WHERE company=? AND fetched_at>=?
                ORDER BY fetched_at DESC
                """,
                (company, cutoff),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── web traffic storage ───────────────────────────────────────────────────

    def save_web_traffic(self, domain: str, monthly: dict[str, int]) -> None:
        """Save domain -> {year_month -> capture_count} dict."""
        now = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.executemany(
                """
                INSERT INTO web_traffic_snapshots
                    (domain, year_month, capture_count, fetched_at)
                VALUES(?,?,?,?)
                ON CONFLICT(domain, year_month) DO UPDATE SET
                    capture_count=excluded.capture_count,
                    fetched_at=excluded.fetched_at
                """,
                [(domain, ym, cnt, now) for ym, cnt in monthly.items()],
            )

    def load_web_traffic(self, domain: str, months_back: int = 24) -> pd.DataFrame:
        cutoff = (
            datetime.utcnow() - timedelta(days=months_back * 31)
        ).strftime("%Y-%m")
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT year_month, capture_count FROM web_traffic_snapshots
                WHERE domain=? AND year_month>=?
                ORDER BY year_month ASC
                """,
                (domain, cutoff),
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["year_month", "capture_count"])
        return pd.DataFrame([dict(r) for r in rows])

    def evict_expired(self) -> int:
        cutoff = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        with self._conn() as con:
            cur = con.execute(
                "DELETE FROM cache_entries WHERE cached_at<?", (cutoff,)
            )
            return cur.rowcount


# ---------------------------------------------------------------------------
# Rate-limit aware HTTP helper
# ---------------------------------------------------------------------------


def _http_get(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 20.0,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> Optional[requests.Response]:
    """GET with exponential backoff on 429 / connection errors."""
    h = headers or _HTTP_HEADERS
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=h, timeout=timeout)
            if resp.status_code == 429:
                wait = backoff_base ** (attempt + 1) + random.uniform(0, 1)
                logger.warning("Rate limited by %s; sleeping %.1fs", url[:60], wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            wait = backoff_base ** (attempt + 1)
            logger.warning("HTTP error attempt %d/%d: %s — sleeping %.1fs",
                           attempt + 1, retries, exc, wait)
            if attempt < retries - 1:
                time.sleep(wait)
    return None


import random  # noqa: E402 — imported after helper definition for clarity


def _rate_limit_sleep() -> None:
    t = random.uniform(_REQUEST_DELAY_MIN, _REQUEST_DELAY_MAX)
    time.sleep(t)


# ---------------------------------------------------------------------------
# Indeed RSS Job Postings Scraper
# ---------------------------------------------------------------------------


class JobPostingsScraper:
    """
    Scrapes job postings from Indeed RSS feeds (free, no API key).

    Indeed exposes a public RSS endpoint:
        https://www.indeed.com/rss?q={query}&sort=date&limit=50

    Each item contains: title, company, location, description, pubDate, link.
    """

    def __init__(self, cache: Optional[JobPostingsCache] = None) -> None:
        self._cache = cache or JobPostingsCache()
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _build_url(self, query: str, start: int = 0) -> str:
        encoded = quote_plus(query)
        return f"https://www.indeed.com/rss?q={encoded}&sort=date&limit=50&start={start}"

    def _parse_rss(self, xml_text: str, query: str) -> list[JobPosting]:
        postings: list[JobPosting] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("RSS parse error: %s", exc)
            return postings

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        channel = root.find("channel")
        if channel is None:
            return postings

        for item in channel.findall("item"):
            def _text(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""

            title = _text("title")
            link = _text("link")
            description = re.sub(r"<[^>]+>", " ", _text("description"))
            description = re.sub(r"\s+", " ", description).strip()

            # Indeed wraps company/location in the title: "Job Title - Company - Location"
            company = ""
            location = ""
            parts = [p.strip() for p in title.split(" - ")]
            if len(parts) >= 3:
                company = parts[-2]
                location = parts[-1]
            elif len(parts) == 2:
                company = parts[-1]

            pub_date_raw = _text("pubDate")
            posted_date: Optional[date] = None
            for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
                try:
                    posted_date = datetime.strptime(
                        pub_date_raw.strip(), fmt
                    ).date()
                    break
                except ValueError:
                    continue

            postings.append(
                JobPosting(
                    title=title,
                    company=company,
                    location=location,
                    description=description,
                    url=link,
                    posted_date=posted_date,
                    query=query,
                )
            )
        return postings

    def fetch_indeed_postings(
        self,
        query: str,
        pages: int = 3,
        use_cache: bool = True,
    ) -> list[JobPosting]:
        """
        Fetch job postings from Indeed RSS for a query string.

        Parameters
        ----------
        query:
            Search query, e.g. company name or role keyword.
        pages:
            Number of RSS pages to fetch (50 results per page).
        use_cache:
            Use cached results if available (TTL = 6 hours).
        """
        cache_key = self._cache.make_key("indeed_rss", query, pages)
        if use_cache:
            cached = self._cache.get_json(cache_key, ttl_hours=_POSTINGS_TTL_HOURS)
            if cached:
                logger.debug("Cache hit: indeed_rss(%s)", query)
                return [JobPosting(**p) for p in cached.get("postings", [])]

        all_postings: list[JobPosting] = []
        for page in range(pages):
            start = page * 50
            url = self._build_url(query, start)
            resp = _http_get(url, headers=_HEADERS, timeout=15.0)
            if resp is None:
                logger.warning("Indeed RSS fetch failed for query=%r page=%d", query, page)
                break

            new_postings = self._parse_rss(resp.text, query)
            if not new_postings:
                break
            all_postings.extend(new_postings)
            _rate_limit_sleep()

        if use_cache and all_postings:
            self._cache.set_json(
                cache_key,
                {"postings": [p.__dict__ for p in all_postings]},
                ttl_hours=_POSTINGS_TTL_HOURS,
            )

        logger.info("Fetched %d postings for query=%r", len(all_postings), query)
        return all_postings

    def fetch_company_postings(
        self,
        ticker: str,
        company: str,
        pages: int = 5,
        use_cache: bool = True,
    ) -> CompanyHiringSnapshot:
        """
        Fetch all recent job postings for a company and build a hiring snapshot.

        Uses multiple search queries:
          1. Company name exact
          2. Ticker symbol (for well-known companies)
          3. "{company} jobs"

        Parameters
        ----------
        ticker:
            Stock ticker (e.g., "AAPL").
        company:
            Company name (e.g., "Apple").
        pages:
            Pages per query.
        """
        queries = [company, f"{company} jobs", ticker]
        seen_urls: set[str] = set()
        all_postings: list[JobPosting] = []

        for q in queries:
            postings = self.fetch_indeed_postings(q, pages=pages, use_cache=use_cache)
            for p in postings:
                if p.url not in seen_urls:
                    seen_urls.add(p.url)
                    all_postings.append(p)

        # Also check SQLite for historical postings
        historical = self._cache.load_postings_last_n_days(company, days=90, ticker=ticker)

        # Save new postings to SQLite
        if all_postings:
            self._cache.save_postings(all_postings, ticker, company, ",".join(queries))

        # Compute snapshot metrics
        today = date.today()
        cutoff_30d = today - timedelta(days=30)
        cutoff_90d = today - timedelta(days=90)

        postings_30d = [
            p for p in all_postings
            if p.posted_date and p.posted_date >= cutoff_30d
        ]
        postings_90d = [
            p for p in all_postings
            if p.posted_date and p.posted_date >= cutoff_90d
        ]

        # Role composition
        role_comp = _classify_roles(all_postings)

        # Tech stack
        tech_mentions = _count_tech_mentions(all_postings)

        # Exec hires
        exec_count = sum(
            1 for p in all_postings
            if any(kw in p.title.lower() for kw in EXEC_ROLE_KEYWORDS)
        )

        # Top locations
        locations = [p.location for p in all_postings if p.location]
        top_locs = _top_n_strings(locations, n=5)

        # Top roles
        titles = [p.title for p in all_postings if p.title]
        top_roles = _top_n_strings(titles, n=10)

        # New cities vs prior period
        cities_new = _extract_cities(postings_30d)
        cities_prior = _extract_cities(
            [p for p in all_postings if p not in postings_30d]
        )
        new_cities = [c for c in cities_new if c not in cities_prior]

        # YoY change: we use historical SQLite data if available
        yoy_pct: Optional[float] = None
        if historical:
            hist_90d_count = len(historical)
            if hist_90d_count > 0 and len(postings_90d) > 0:
                yoy_pct = ((len(postings_90d) - hist_90d_count) / hist_90d_count) * 100

        return CompanyHiringSnapshot(
            ticker=ticker,
            company=company,
            as_of=today,
            total_postings_30d=len(postings_30d),
            total_postings_90d=len(postings_90d),
            postings_yoy_change_pct=yoy_pct,
            top_roles=top_roles,
            top_locations=top_locs,
            role_composition=role_comp,
            tech_stack_mentions=tech_mentions,
            exec_hire_count=exec_count,
            new_cities=new_cities,
            raw_postings=all_postings,
        )


# ---------------------------------------------------------------------------
# BLS / FRED Adapter
# ---------------------------------------------------------------------------


class BLSFREDAdapter:
    """
    Fetches labor market data from FRED (free CSV downloads) and BLS Public API.

    FRED CSV endpoint does not require an API key.
    BLS Public API v2 allows up to 25 series per request, 50 requests/day without key.
    """

    def __init__(self, cache: Optional[JobPostingsCache] = None) -> None:
        self._cache = cache or JobPostingsCache()

    def get_jolts_fred(
        self,
        series_id: str,
        start: str = "2015-01-01",
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download a FRED time series as CSV and return as DataFrame.

        Parameters
        ----------
        series_id:
            FRED series ID, e.g. "JTSJOL".
        start:
            Start date string "YYYY-MM-DD".
        end:
            End date string "YYYY-MM-DD" (defaults to today).
        """
        if end is None:
            end = date.today().isoformat()

        cache_key = self._cache.make_key("fred_csv", series_id, start, end)
        if use_cache:
            cached_json = self._cache.get(cache_key, ttl_hours=_MACRO_TTL_HOURS)
            if cached_json:
                return pd.read_json(cached_json)

        params = {
            "id": series_id,
            "vintage_date": end,
            "observation_start": start,
            "observation_end": end,
        }
        resp = _http_get(FRED_CSV_URL, params=params, timeout=30.0)
        if resp is None:
            logger.error("FRED CSV fetch failed for series=%s", series_id)
            return pd.DataFrame(columns=["date", "value"])

        try:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "date", series_id.upper(): "value"})
            if "value" not in df.columns:
                # FRED sometimes uses the series ID as column name
                val_col = [c for c in df.columns if c != "date"]
                if val_col:
                    df = df.rename(columns={val_col[0]: "value"})
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            df["series_id"] = series_id

            if use_cache:
                self._cache.set(cache_key, df.to_json(), ttl_hours=_MACRO_TTL_HOURS)

            return df
        except Exception as exc:
            logger.error("FRED CSV parse error for %s: %s", series_id, exc)
            return pd.DataFrame(columns=["date", "value", "series_id"])

    def get_sector_openings(self, use_cache: bool = True) -> dict[str, float]:
        """
        Return latest job openings level for each GICS-mapped sector from FRED.
        """
        cache_key = self._cache.make_key("sector_openings_latest")
        if use_cache:
            cached = self._cache.get_json(cache_key, ttl_hours=_MACRO_TTL_HOURS)
            if cached:
                return cached

        result: dict[str, float] = {}
        for sector, series_id in FRED_SECTOR_OPENINGS.items():
            df = self.get_jolts_fred(series_id, use_cache=use_cache)
            if df.empty:
                continue
            latest = df.sort_values("date").iloc[-1]["value"]
            result[sector] = float(latest)
            _rate_limit_sleep()

        if use_cache and result:
            self._cache.set_json(cache_key, result, ttl_hours=_MACRO_TTL_HOURS)
        return result

    def get_quits_rate(self, use_cache: bool = True) -> dict[str, float]:
        """Return latest total quits rate and historical context."""
        df = self.get_jolts_fred("JTSQUR", use_cache=use_cache)
        if df.empty:
            return {}
        df_sorted = df.sort_values("date")
        latest = float(df_sorted.iloc[-1]["value"])
        avg_5yr = float(df_sorted.tail(60)["value"].mean())
        return {
            "latest": latest,
            "avg_5yr": avg_5yr,
            "vs_avg_pct": round((latest - avg_5yr) / avg_5yr * 100, 2),
        }

    def get_macro_labor_dashboard(self, use_cache: bool = True) -> MacroLaborDashboard:
        """
        Build a comprehensive macro labor dashboard from FRED data.

        Fetches: job openings, quits rate, hires, separations, unemployment,
        nonfarm payrolls, and sector-level openings.
        """
        cache_key = self._cache.make_key("macro_labor_dashboard")
        if use_cache:
            cached = self._cache.get_json(cache_key, ttl_hours=_MACRO_TTL_HOURS)
            if cached:
                # Reconstruct from dict
                return _macro_dashboard_from_dict(cached)

        series_map = {
            "JTSJOL": "total_openings",
            "JTSQUR": "quits_rate",
            "JTSHIL": "hires",
            "UNRATE": "unemployment",
            "PAYEMS": "payrolls",
        }

        values: dict[str, Optional[float]] = {}
        yoy_openings: Optional[float] = None

        for sid, label in series_map.items():
            df = self.get_jolts_fred(sid, use_cache=use_cache)
            if df.empty:
                values[label] = None
                _rate_limit_sleep()
                continue
            df_sorted = df.sort_values("date")
            values[label] = float(df_sorted.iloc[-1]["value"])
            if label == "total_openings" and len(df_sorted) >= 13:
                prev = float(df_sorted.iloc[-13]["value"])
                if prev > 0:
                    yoy_openings = round(
                        (values[label] - prev) / prev * 100, 2
                    )
            _rate_limit_sleep()

        sector_openings = self.get_sector_openings(use_cache=use_cache)

        # Classify sectors as tight/slack based on openings
        median_open = np.median(list(sector_openings.values())) if sector_openings else 1
        tight = [s for s, v in sector_openings.items() if v > median_open * 1.2]
        slack = [s for s, v in sector_openings.items() if v < median_open * 0.8]

        # Overall macro signal
        qr = values.get("quits_rate") or 0
        ur = values.get("unemployment") or 0
        macro_signal = _classify_macro_labor(qr, ur)

        interp = _build_macro_interpretation(values, yoy_openings, macro_signal)

        dashboard = MacroLaborDashboard(
            as_of=date.today(),
            total_job_openings_k=values.get("total_openings"),
            total_quits_rate=values.get("quits_rate"),
            total_hires_k=values.get("hires"),
            unemployment_rate=values.get("unemployment"),
            nonfarm_payrolls_k=values.get("payrolls"),
            sector_openings=sector_openings,
            tight_labor_sectors=tight,
            slack_labor_sectors=slack,
            macro_signal=macro_signal,
            yoy_openings_change_pct=yoy_openings,
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_json(
                cache_key, _macro_dashboard_to_dict(dashboard),
                ttl_hours=_MACRO_TTL_HOURS,
            )

        return dashboard


# ---------------------------------------------------------------------------
# Wayback Machine CDX API Web Traffic Proxy
# ---------------------------------------------------------------------------


class WaybackCDXAdapter:
    """
    Uses the Internet Archive's CDX API to count monthly web snapshots as a
    free proxy for web traffic trends.

    CDX API docs: https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server
    The number of Wayback crawls for a domain correlates with its traffic and
    web visibility. This is a rough proxy (not SimilarWeb), but it's free,
    consistent, and reveals relative trends.

    Endpoint: http://web.archive.org/cdx/search/cdx
    Parameters used:
        url={domain}
        output=json
        fl=timestamp
        collapse=digest (dedupe identical pages)
        from={start}&to={end}
        limit=50000
    """

    def __init__(self, cache: Optional[JobPostingsCache] = None) -> None:
        self._cache = cache or JobPostingsCache()

    def get_capture_counts(
        self,
        domain: str,
        year: int,
        use_cache: bool = True,
    ) -> dict[str, int]:
        """
        Return monthly capture counts for a domain in a given year.

        Returns dict mapping "YYYY-MM" -> count.
        """
        cache_key = self._cache.make_key("cdx_counts", domain, year)
        if use_cache:
            cached = self._cache.get_json(cache_key, ttl_hours=_WEB_TRAFFIC_TTL_HOURS)
            if cached:
                return cached

        start = f"{year}0101000000"
        end = f"{year}1231235959"
        params = {
            "url": f"*.{domain}",
            "output": "json",
            "fl": "timestamp",
            "collapse": "urlkey",
            "from": start,
            "to": end,
            "limit": "100000",
            "filter": "statuscode:200",
        }

        resp = _http_get(WAYBACK_CDX_URL, params=params, timeout=45.0)
        if resp is None:
            logger.warning("CDX fetch failed for domain=%s year=%d", domain, year)
            return {}

        try:
            data = resp.json()
            if not data or len(data) <= 1:
                return {}
            # First row is headers
            timestamps = [row[0] for row in data[1:] if row]
            monthly: dict[str, int] = {}
            for ts in timestamps:
                if len(ts) >= 6:
                    ym = f"{ts[:4]}-{ts[4:6]}"
                    monthly[ym] = monthly.get(ym, 0) + 1
        except Exception as exc:
            logger.error("CDX parse error for %s: %s", domain, exc)
            return {}

        if use_cache and monthly:
            self._cache.set_json(cache_key, monthly, ttl_hours=_WEB_TRAFFIC_TTL_HOURS)
            self._cache.save_web_traffic(domain, monthly)

        return monthly

    def get_traffic_proxy(
        self,
        domain: str,
        lookback_months: int = 24,
        ticker: Optional[str] = None,
        use_cache: bool = True,
    ) -> WebTrafficProxy:
        """
        Build a multi-month traffic proxy series for a domain.

        Fetches CDX capture counts for the past `lookback_months` months
        and returns a WebTrafficProxy with trend analysis.
        """
        cache_key = self._cache.make_key("traffic_proxy", domain, lookback_months)
        today = date.today()

        # Check SQLite for pre-cached traffic data
        existing_df = self._cache.load_web_traffic(domain, months_back=lookback_months)

        # Determine which years we need to fetch
        start_year = (today - timedelta(days=lookback_months * 31)).year
        current_year = today.year
        years_needed = list(range(start_year, current_year + 1))

        # Fetch missing years
        monthly_all: dict[str, int] = {}

        for ym_row in existing_df.itertuples():
            monthly_all[ym_row.year_month] = ym_row.capture_count

        for yr in years_needed:
            yr_key = f"{yr}-01"
            has_data = any(k.startswith(str(yr)) for k in monthly_all)
            if not has_data or yr == current_year:
                counts = self.get_capture_counts(domain, yr, use_cache=use_cache)
                monthly_all.update(counts)
                _rate_limit_sleep()

        if not monthly_all:
            return WebTrafficProxy(
                domain=domain,
                ticker=ticker,
                as_of=today,
                monthly_captures=pd.Series(dtype=float),
                trend_direction="unknown",
                yoy_change_pct=None,
                mom_change_pct=None,
                z_score_latest=0.0,
            )

        # Build series
        series = pd.Series(monthly_all).sort_index()
        series.index = pd.to_datetime(series.index)

        # Trim to lookback window
        cutoff_dt = today - timedelta(days=lookback_months * 31)
        series = series[series.index >= pd.Timestamp(cutoff_dt)]

        # Trend analysis
        if len(series) < 2:
            trend = "stable"
            yoy_pct = None
            mom_pct = None
            z_score = 0.0
        else:
            # Month-over-month change
            mom_pct = None
            if len(series) >= 2:
                if series.iloc[-2] > 0:
                    mom_pct = round(
                        (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] * 100, 2
                    )

            # Year-over-year change
            yoy_pct = None
            if len(series) >= 13:
                if series.iloc[-13] > 0:
                    yoy_pct = round(
                        (series.iloc[-1] - series.iloc[-13]) / series.iloc[-13] * 100, 2
                    )

            # Z-score of latest vs trailing 12m
            window = series.tail(12)
            if len(window) >= 3:
                mean = window.mean()
                std = window.std()
                z_score = float((series.iloc[-1] - mean) / std) if std > 0 else 0.0
            else:
                z_score = 0.0

            # Trend direction via linear regression
            if len(series) >= 4:
                x = np.arange(len(series))
                slope, _, r, _, _ = stats.linregress(x, series.values)
                if r ** 2 > 0.3:
                    trend = "growing" if slope > 0 else "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

        return WebTrafficProxy(
            domain=domain,
            ticker=ticker,
            as_of=today,
            monthly_captures=series,
            trend_direction=trend,
            yoy_change_pct=yoy_pct,
            mom_change_pct=mom_pct,
            z_score_latest=round(z_score, 3),
        )

    def compare_domains(
        self,
        domains: list[str],
        lookback_months: int = 12,
    ) -> pd.DataFrame:
        """
        Compare web traffic proxy across multiple domains.

        Returns DataFrame with domain as index and:
        latest_captures, yoy_change_pct, trend_direction, z_score.
        """
        records: list[dict] = []
        for domain in domains:
            proxy = self.get_traffic_proxy(domain, lookback_months=lookback_months)
            records.append(
                {
                    "domain": domain,
                    "latest_captures": (
                        int(proxy.monthly_captures.iloc[-1])
                        if not proxy.monthly_captures.empty
                        else None
                    ),
                    "yoy_change_pct": proxy.yoy_change_pct,
                    "mom_change_pct": proxy.mom_change_pct,
                    "trend_direction": proxy.trend_direction,
                    "z_score": proxy.z_score_latest,
                }
            )
            _rate_limit_sleep()

        return pd.DataFrame(records).set_index("domain")


# ---------------------------------------------------------------------------
# Hiring Signal Engine
# ---------------------------------------------------------------------------


class HiringSignalEngine:
    """
    Derives actionable investment signals from job postings data.

    Signals produced:
      - HiringVelocitySignal: Is the company accelerating or decelerating hiring?
      - TechStackSignal: What cloud/tech infrastructure is the company building?
      - GeoExpansionSignal: Which new markets is the company entering?
      - ExecHireSignal: Are C-suite/VP hires signaling a strategic pivot?
      - RoleCompositionSignal: What growth stage does the hiring mix suggest?
      - HiringReturnCorrelation: Does lagged hiring velocity predict stock returns?
    """

    def __init__(
        self,
        scraper: Optional[JobPostingsScraper] = None,
        cache: Optional[JobPostingsCache] = None,
    ) -> None:
        self._scraper = scraper or JobPostingsScraper()
        self._cache = cache or JobPostingsCache()

    def compute_hiring_velocity(
        self,
        ticker: str,
        company: str,
        pages: int = 4,
    ) -> HiringVelocitySignal:
        """
        Compute hiring velocity signal: rolling 90-day job posting volume change.

        Methodology:
          - Fetch current postings
          - Load historical postings from SQLite
          - Compute 30d and 90d volumes
          - Calculate z-score vs 12-month baseline
          - Classify signal: surge / accelerating / stable / decelerating / collapse
        """
        snapshot = self._scraper.fetch_company_postings(ticker, company, pages=pages)

        # Historical data from SQLite
        hist_30d = self._cache.load_postings_last_n_days(company, days=30)
        hist_60d = self._cache.load_postings_last_n_days(company, days=60)
        hist_90d = self._cache.load_postings_last_n_days(company, days=90)
        hist_180d = self._cache.load_postings_last_n_days(company, days=180)

        n_30d = max(snapshot.total_postings_30d, len(hist_30d))
        n_90d = max(snapshot.total_postings_90d, len(hist_90d))
        n_prior_90d = max(len(hist_180d) - n_90d, 1)

        velocity_90d = ((n_90d - n_prior_90d) / n_prior_90d * 100) if n_prior_90d > 0 else 0.0

        # 30d vs 60d velocity
        n_60d_count = len(hist_60d) if hist_60d else n_30d
        velocity_30d = ((n_30d - n_60d_count / 2) / (n_60d_count / 2) * 100) if n_60d_count > 0 else 0.0

        # Acceleration: is velocity_30d accelerating vs velocity_90d?
        acceleration = velocity_30d - velocity_90d / 3.0

        # Z-score: treat historical monthly counts as baseline
        baseline_monthly = [
            max(len(hist_30d), 1),
            max(n_90d // 3, 1),
            max(n_prior_90d // 3, 1),
        ]
        mu = np.mean(baseline_monthly)
        sigma = np.std(baseline_monthly)
        z = (n_30d - mu) / sigma if sigma > 0 else 0.0

        # Classify
        if z > 2.5 or velocity_90d > 100:
            signal = "surge"
        elif velocity_90d > 20 or (velocity_30d > 30 and acceleration > 5):
            signal = "accelerating"
        elif velocity_90d < -40 or z < -2.0:
            signal = "collapse"
        elif velocity_90d < -10 or (velocity_30d < -20 and acceleration < -5):
            signal = "decelerating"
        else:
            signal = "stable"

        interp = _interpret_velocity(signal, velocity_90d, velocity_30d, ticker)

        return HiringVelocitySignal(
            ticker=ticker,
            company=company,
            as_of=date.today(),
            velocity_90d=round(velocity_90d, 2),
            velocity_30d=round(velocity_30d, 2),
            acceleration=round(acceleration, 2),
            signal=signal,
            z_score=round(z, 3),
            interpretation=interp,
        )

    def detect_tech_stack(
        self, postings: list[JobPosting]
    ) -> dict[str, Any]:
        """
        Analyze tech stack from job descriptions to infer cloud adoption,
        AI/ML maturity, and infrastructure modernization.
        """
        if not postings:
            return {}

        n_total = len(postings)
        tech_counts = _count_tech_mentions(postings)

        # Cloud provider analysis
        aws_count = tech_counts.get("aws", 0)
        azure_count = tech_counts.get("azure", 0)
        gcp_count = tech_counts.get("gcp", 0)

        cloud_total = aws_count + azure_count + gcp_count
        if cloud_total == 0:
            cloud_provider = "unknown"
            cloud_dominance = 0.0
        else:
            max_count = max(aws_count, azure_count, gcp_count)
            cloud_dominance = max_count / cloud_total
            if cloud_dominance < 0.5:
                cloud_provider = "multi-cloud"
            elif aws_count >= azure_count and aws_count >= gcp_count:
                cloud_provider = "aws"
            elif azure_count >= aws_count and azure_count >= gcp_count:
                cloud_provider = "azure"
            else:
                cloud_provider = "gcp"

        ml_ai_pct = (tech_counts.get("ml_ai", 0) / n_total * 100) if n_total > 0 else 0.0
        data_eng_pct = (
            tech_counts.get("data_engineering", 0) / n_total * 100
        ) if n_total > 0 else 0.0

        # Modernization score: weighted sum of cloud-native, containerization, ML signals
        modern_signals = {
            "kubernetes": 2.0,
            "terraform": 1.5,
            "ml_ai": 2.0,
            "data_engineering": 1.5,
            "golang": 1.0,
            "rust": 1.0,
            "typescript": 0.5,
        }
        mod_score = sum(
            (tech_counts.get(tech, 0) / n_total * 10) * weight
            for tech, weight in modern_signals.items()
        )
        mod_score = min(10.0, round(mod_score, 2))

        top_tech = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:8]

        return {
            "cloud_provider": cloud_provider,
            "cloud_dominance_pct": round(cloud_dominance * 100, 1),
            "ml_ai_adoption_pct": round(ml_ai_pct, 1),
            "data_engineering_pct": round(data_eng_pct, 1),
            "modernization_score": mod_score,
            "top_tech_mentions": top_tech,
            "raw_counts": tech_counts,
        }

    def detect_geo_expansion(
        self, postings: list[JobPosting], ticker: str = ""
    ) -> GeoExpansionSignal:
        """
        Detect geographic expansion patterns from job posting locations.

        New cities appearing in recent postings vs prior period signal
        market expansion or new office openings.
        """
        today = date.today()
        cutoff_recent = today - timedelta(days=30)
        cutoff_prior = today - timedelta(days=90)

        recent = [
            p for p in postings
            if p.posted_date and p.posted_date >= cutoff_recent
        ]
        prior = [
            p for p in postings
            if p.posted_date and p.posted_date < cutoff_recent
            and p.posted_date >= cutoff_prior
        ]

        cities_recent = _extract_cities(recent)
        cities_prior = _extract_cities(prior)

        new_cities = [c for c in cities_recent if c not in cities_prior and c]

        all_cities = _extract_cities(postings)
        remote_count = sum(
            1 for p in postings
            if any(kw in (p.location or "").lower() for kw in ["remote", "hybrid", "anywhere"])
        )
        remote_pct = (remote_count / len(postings) * 100) if postings else 0.0

        # Expansion score: new cities weighted by number
        n_new = len(new_cities)
        expansion_score = min(10.0, n_new * 1.5 + (len(all_cities) / 10))

        # Primary markets = top-3 cities by posting frequency
        loc_counts: dict[str, int] = {}
        for p in postings:
            locs = _extract_cities([p])
            for loc in locs:
                loc_counts[loc] = loc_counts.get(loc, 0) + 1
        primary = sorted(loc_counts, key=loc_counts.get, reverse=True)[:3]

        return GeoExpansionSignal(
            ticker=ticker,
            as_of=today,
            current_cities=all_cities[:20],
            new_cities_vs_prior=new_cities,
            expansion_score=round(expansion_score, 2),
            primary_markets=primary,
            remote_pct=round(remote_pct, 1),
        )

    def detect_exec_hires(
        self, postings: list[JobPosting], ticker: str = ""
    ) -> ExecHireSignal:
        """
        Detect executive / C-suite hiring signals.

        Executive openings indicate:
          - CEO/CFO opening → leadership transition risk
          - CTO/VP Eng opening → technology pivot
          - CMO/CRO opening → go-to-market shift
          - Multiple C-suite openings → major strategic reset
        """
        today = date.today()
        exec_postings: list[str] = []
        c_suite_count = 0
        vp_count = 0

        for p in postings:
            title_lower = p.title.lower()
            is_csuite = any(
                kw in title_lower
                for kw in [
                    "chief ", "ceo", "cfo", "cto", "coo", "cpo", "cmo", "cdo",
                    "ciso", "cro", "chief of staff",
                ]
            )
            is_vp = any(
                kw in title_lower
                for kw in ["vice president", "vp of", "head of", "svp", "evp"]
            )

            if is_csuite:
                exec_postings.append(p.title)
                c_suite_count += 1
            elif is_vp:
                exec_postings.append(p.title)
                vp_count += 1

        exec_count = c_suite_count + vp_count
        is_pivot = c_suite_count >= 2 or (c_suite_count == 1 and vp_count >= 3)

        pivot_reason = _classify_exec_pivot(exec_postings)

        if c_suite_count >= 3:
            strength = "strong"
        elif c_suite_count >= 1 or vp_count >= 3:
            strength = "moderate"
        elif vp_count >= 1:
            strength = "weak"
        else:
            strength = "none"

        return ExecHireSignal(
            ticker=ticker,
            as_of=today,
            exec_postings=exec_postings[:20],
            exec_hire_count=exec_count,
            c_suite_count=c_suite_count,
            vp_count=vp_count,
            is_strategic_pivot=is_pivot,
            pivot_reason=pivot_reason,
            signal_strength=strength,
        )

    def compute_role_composition(
        self, postings: list[JobPosting], ticker: str = ""
    ) -> RoleCompositionSignal:
        """
        Analyze the mix of engineering, sales, support, and other roles.

        Growth stage inference:
          - Early stage: >50% engineering, <15% sales/marketing
          - Scaling: 35-50% engineering, 20-30% sales
          - Mature: <30% engineering, balanced support/ops
          - Restructuring: >20% ops/HR + layoff-adjacent keywords
        """
        composition = _classify_roles(postings)
        today = date.today()

        eng_pct = composition.get("engineering", 0.0)
        sales_pct = composition.get("sales", 0.0)
        ops_pct = composition.get("operations", 0.0)
        hr_pct = composition.get("hr", 0.0)

        eng_to_sales = (eng_pct / sales_pct) if sales_pct > 0 else float("inf")
        eng_to_ops = (eng_pct / ops_pct) if ops_pct > 0 else float("inf")

        # Growth stage classification
        if eng_pct > 50 and sales_pct < 15:
            stage = "early"
        elif eng_pct > 35 and sales_pct >= 20:
            stage = "scaling"
        elif eng_pct < 30 and (ops_pct + hr_pct) > 25:
            stage = "restructuring"
        else:
            stage = "mature"

        interp = _interpret_role_composition(stage, composition, ticker)

        return RoleCompositionSignal(
            ticker=ticker,
            as_of=today,
            composition=composition,
            growth_stage=stage,
            eng_to_sales_ratio=round(eng_to_sales, 2) if eng_to_sales != float("inf") else 99.0,
            eng_to_support_ratio=round(eng_to_ops, 2) if eng_to_ops != float("inf") else 99.0,
            interpretation=interp,
        )

    def correlate_with_returns(
        self,
        ticker: str,
        prices: pd.Series,
        company: str = "",
        lag_weeks: int = 4,
    ) -> HiringReturnCorrelation:
        """
        Compute Granger-like lagged correlation between hiring velocity and stock returns.

        Methodology:
          - Load historical posting counts from SQLite (by week)
          - Compute weekly return from price series
          - Align and lag hiring series by `lag_weeks`
          - Run Pearson correlation + OLS regression for significance
        """
        hist = self._cache.load_postings_last_n_days(company or ticker, days=365)
        if len(hist) < 8:
            return HiringReturnCorrelation(
                ticker=ticker,
                lag_weeks=lag_weeks,
                correlation=0.0,
                p_value=1.0,
                is_significant=False,
                direction="neutral",
                r_squared=0.0,
                sample_size=len(hist),
            )

        # Build weekly hiring counts
        df_hist = pd.DataFrame(hist)
        df_hist["fetched_at"] = pd.to_datetime(df_hist["fetched_at"])
        df_hist = df_hist.set_index("fetched_at").resample("W").size()
        df_hist.name = "posting_count"

        # Compute weekly returns
        if prices.empty or len(prices) < 2:
            weekly_returns = pd.Series(dtype=float)
        else:
            prices.index = pd.to_datetime(prices.index)
            weekly_returns = prices.resample("W").last().pct_change().dropna()
            weekly_returns.name = "weekly_return"

        # Align and lag
        combined = pd.concat(
            [df_hist, weekly_returns], axis=1
        ).dropna()

        if len(combined) < lag_weeks + 4:
            return HiringReturnCorrelation(
                ticker=ticker,
                lag_weeks=lag_weeks,
                correlation=0.0,
                p_value=1.0,
                is_significant=False,
                direction="neutral",
                r_squared=0.0,
                sample_size=len(combined),
            )

        x = combined["posting_count"].values[:-lag_weeks]
        y = combined["weekly_return"].values[lag_weeks:]

        if len(x) < 4:
            return HiringReturnCorrelation(
                ticker=ticker, lag_weeks=lag_weeks, correlation=0.0,
                p_value=1.0, is_significant=False, direction="neutral",
                r_squared=0.0, sample_size=len(x),
            )

        corr, p_val = stats.pearsonr(x, y)
        slope, intercept, r_val, p_val2, se = stats.linregress(x, y)

        return HiringReturnCorrelation(
            ticker=ticker,
            lag_weeks=lag_weeks,
            correlation=round(float(corr), 4),
            p_value=round(float(p_val), 4),
            is_significant=float(p_val) < 0.05,
            direction="positive" if corr > 0.1 else ("negative" if corr < -0.1 else "neutral"),
            r_squared=round(float(r_val ** 2), 4),
            sample_size=len(x),
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _classify_roles(postings: list[JobPosting]) -> dict[str, float]:
    """Return percentage composition of roles by category."""
    counts: dict[str, int] = {cat: 0 for cat in ROLE_CATEGORIES}
    counts["other"] = 0

    for p in postings:
        title_lower = p.title.lower()
        matched = False
        for cat, keywords in ROLE_CATEGORIES.items():
            if any(kw in title_lower for kw in keywords):
                counts[cat] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1

    total = max(sum(counts.values()), 1)
    return {cat: round(cnt / total * 100, 1) for cat, cnt in counts.items()}


def _count_tech_mentions(postings: list[JobPosting]) -> dict[str, int]:
    """Count tech stack keyword mentions across all job descriptions."""
    counts: dict[str, int] = {tech: 0 for tech in TECH_STACK_KEYWORDS}
    for p in postings:
        text = (p.description + " " + p.title).lower()
        for tech, keywords in TECH_STACK_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                counts[tech] += 1
    return counts


def _extract_cities(postings: list[JobPosting]) -> list[str]:
    """Extract known US cities from job posting locations."""
    cities: set[str] = set()
    for p in postings:
        loc = (p.location or "").lower()
        for city in MAJOR_CITIES:
            if city in loc:
                cities.add(city)
    return sorted(cities)


def _top_n_strings(items: list[str], n: int = 5) -> list[str]:
    """Return top-N most frequent strings."""
    counts: dict[str, int] = {}
    for item in items:
        key = item.strip().lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]]


def _interpret_velocity(
    signal: str, velocity_90d: float, velocity_30d: float, ticker: str
) -> str:
    if signal == "surge":
        return (
            f"{ticker} showing hiring surge: +{velocity_90d:.0f}% 90-day posting volume. "
            "Historically associated with product launches or post-fundraise expansion."
        )
    elif signal == "accelerating":
        return (
            f"{ticker} hiring accelerating: {velocity_30d:.0f}% 30d vs {velocity_90d:.0f}% 90d. "
            "Leading indicator of revenue growth 1-2 quarters forward."
        )
    elif signal == "collapse":
        return (
            f"{ticker} hiring collapse: {velocity_90d:.0f}% 90-day decline. "
            "Potential restructuring, cash preservation, or demand slowdown."
        )
    elif signal == "decelerating":
        return (
            f"{ticker} hiring decelerating: {velocity_30d:.0f}% 30d vs {velocity_90d:.0f}% 90d. "
            "May indicate normalizing growth or cost discipline."
        )
    return f"{ticker} hiring stable: {velocity_90d:.0f}% 90-day change within normal range."


def _interpret_role_composition(
    stage: str, composition: dict[str, float], ticker: str
) -> str:
    eng = composition.get("engineering", 0)
    sales = composition.get("sales", 0)
    ops = composition.get("operations", 0)
    if stage == "early":
        return (
            f"{ticker} in early-stage growth mode: {eng:.0f}% engineering, {sales:.0f}% sales. "
            "Product-building phase; pre-commercialization signals."
        )
    elif stage == "scaling":
        return (
            f"{ticker} scaling GTM: {eng:.0f}% engineering, {sales:.0f}% sales. "
            "Classic Series C+ expansion pattern — monetizing product-market fit."
        )
    elif stage == "restructuring":
        return (
            f"{ticker} showing restructuring signals: {ops:.0f}% operations/support. "
            "Possible headcount rebalancing or cost optimization cycle."
        )
    return (
        f"{ticker} mature hiring profile: {eng:.0f}% engineering, {sales:.0f}% sales. "
        "Balanced mix consistent with established market position."
    )


def _classify_exec_pivot(exec_postings: list[str]) -> str:
    """Infer strategic pivot type from executive role titles."""
    titles = " ".join(exec_postings).lower()
    if any(k in titles for k in ["chief technology", "cto", "vp engineering", "engineering"]):
        return "technology / infrastructure pivot"
    elif any(k in titles for k in ["chief revenue", "cro", "chief marketing", "cmo", "sales"]):
        return "go-to-market / revenue model shift"
    elif any(k in titles for k in ["chief financial", "cfo", "finance"]):
        return "financial restructuring or pre-IPO preparation"
    elif any(k in titles for k in ["chief executive", "ceo", "president"]):
        return "leadership transition — high uncertainty signal"
    elif any(k in titles for k in ["chief data", "cdo", "data", "ai"]):
        return "AI / data strategy transformation"
    elif exec_postings:
        return "operational leadership expansion"
    return "no significant executive hiring signal"


def _classify_macro_labor(quits_rate: float, unemployment: float) -> str:
    if quits_rate > 2.5 and unemployment < 4.5:
        return "tight"
    elif quits_rate < 2.0 and unemployment > 5.0:
        return "slack"
    elif quits_rate > 2.5 or unemployment < 4.0:
        return "tight"
    elif quits_rate < 2.0 or unemployment > 5.0:
        return "loosening"
    return "neutral"


def _build_macro_interpretation(
    values: dict, yoy_openings: Optional[float], signal: str
) -> str:
    ur = values.get("unemployment", 0) or 0
    qr = values.get("quits_rate", 0) or 0
    opens = values.get("total_openings", 0) or 0
    lines = [
        f"Labor market signal: {signal.upper()}.",
        f"Job openings: {opens:,.0f}k " + (f"(YoY: {yoy_openings:+.1f}%)" if yoy_openings else ""),
        f"Unemployment: {ur:.1f}% | Quits rate: {qr:.1f}%",
    ]
    if signal == "tight":
        lines.append("Tight labor market → wage pressure risk, Fed hawkish bias, margin headwinds.")
    elif signal == "slack":
        lines.append("Slack labor market → disinflation tailwind, consumer stress risk.")
    elif signal == "loosening":
        lines.append("Labor loosening → cooling demand signal, potential Fed pivot catalyst.")
    return " ".join(lines)


def _macro_dashboard_to_dict(d: MacroLaborDashboard) -> dict:
    return {
        "as_of": d.as_of.isoformat(),
        "total_job_openings_k": d.total_job_openings_k,
        "total_quits_rate": d.total_quits_rate,
        "total_hires_k": d.total_hires_k,
        "unemployment_rate": d.unemployment_rate,
        "nonfarm_payrolls_k": d.nonfarm_payrolls_k,
        "sector_openings": d.sector_openings,
        "tight_labor_sectors": d.tight_labor_sectors,
        "slack_labor_sectors": d.slack_labor_sectors,
        "macro_signal": d.macro_signal,
        "yoy_openings_change_pct": d.yoy_openings_change_pct,
        "interpretation": d.interpretation,
    }


def _macro_dashboard_from_dict(d: dict) -> MacroLaborDashboard:
    return MacroLaborDashboard(
        as_of=date.fromisoformat(d["as_of"]),
        total_job_openings_k=d.get("total_job_openings_k"),
        total_quits_rate=d.get("total_quits_rate"),
        total_hires_k=d.get("total_hires_k"),
        unemployment_rate=d.get("unemployment_rate"),
        nonfarm_payrolls_k=d.get("nonfarm_payrolls_k"),
        sector_openings=d.get("sector_openings", {}),
        tight_labor_sectors=d.get("tight_labor_sectors", []),
        slack_labor_sectors=d.get("slack_labor_sectors", []),
        macro_signal=d.get("macro_signal", "neutral"),
        yoy_openings_change_pct=d.get("yoy_openings_change_pct"),
        interpretation=d.get("interpretation", ""),
    )


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

job_postings_v2_router = APIRouter(
    prefix="/alt/hiring",
    tags=["Alternative Data — Hiring Signals"],
)

# Module-level singletons (initialized lazily)
_cache: Optional[JobPostingsCache] = None
_scraper: Optional[JobPostingsScraper] = None
_bls: Optional[BLSFREDAdapter] = None
_cdx: Optional[WaybackCDXAdapter] = None
_engine: Optional[HiringSignalEngine] = None


def _get_components() -> tuple[
    JobPostingsCache,
    JobPostingsScraper,
    BLSFREDAdapter,
    WaybackCDXAdapter,
    HiringSignalEngine,
]:
    global _cache, _scraper, _bls, _cdx, _engine
    if _cache is None:
        _cache = JobPostingsCache()
    if _scraper is None:
        _scraper = JobPostingsScraper(cache=_cache)
    if _bls is None:
        _bls = BLSFREDAdapter(cache=_cache)
    if _cdx is None:
        _cdx = WaybackCDXAdapter(cache=_cache)
    if _engine is None:
        _engine = HiringSignalEngine(scraper=_scraper, cache=_cache)
    return _cache, _scraper, _bls, _cdx, _engine


def _company_from_ticker(ticker: str) -> str:
    """Best-effort company name from ticker."""
    TICKER_COMPANY: dict[str, str] = {
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
        "AMZN": "Amazon", "META": "Meta", "NFLX": "Netflix",
        "TSLA": "Tesla", "NVDA": "Nvidia", "AMD": "AMD",
        "INTC": "Intel", "CRM": "Salesforce", "ORCL": "Oracle",
        "IBM": "IBM", "ADBE": "Adobe", "SHOP": "Shopify",
        "SQ": "Block", "PYPL": "PayPal", "UBER": "Uber",
        "LYFT": "Lyft", "ABNB": "Airbnb", "SNAP": "Snap",
        "SPOT": "Spotify", "ZM": "Zoom", "DOCU": "DocuSign",
        "OKTA": "Okta", "SNOW": "Snowflake", "PLTR": "Palantir",
        "COIN": "Coinbase", "NET": "Cloudflare", "DDOG": "Datadog",
        "CRWD": "CrowdStrike", "NOW": "ServiceNow", "WDAY": "Workday",
        "PANW": "Palo Alto Networks", "ZS": "Zscaler", "VEEV": "Veeva",
    }
    return TICKER_COMPANY.get(ticker.upper(), ticker)


@job_postings_v2_router.get(
    "/hiring-signals/{ticker}",
    summary="Full hiring intelligence for a ticker",
    response_model=dict,
)
def get_hiring_signals(
    ticker: str,
    company: Optional[str] = Query(None, description="Company name override"),
    pages: int = Query(3, ge=1, le=8, description="Indeed RSS pages to fetch"),
) -> dict:
    """
    Return comprehensive hiring signals for a ticker:
    velocity, tech stack, geo expansion, exec hires, role composition.
    """
    _, scraper, _, _, engine = _get_components()
    ticker = ticker.upper()
    cname = company or _company_from_ticker(ticker)

    try:
        snapshot = scraper.fetch_company_postings(ticker, cname, pages=pages)
        velocity = engine.compute_hiring_velocity(ticker, cname, pages=pages)
        tech = engine.detect_tech_stack(snapshot.raw_postings)
        geo = engine.detect_geo_expansion(snapshot.raw_postings, ticker=ticker)
        exec_sig = engine.detect_exec_hires(snapshot.raw_postings, ticker=ticker)
        role_comp = engine.compute_role_composition(snapshot.raw_postings, ticker=ticker)

        return {
            "ticker": ticker,
            "company": cname,
            "as_of": date.today().isoformat(),
            "hiring_snapshot": {
                "total_postings_30d": snapshot.total_postings_30d,
                "total_postings_90d": snapshot.total_postings_90d,
                "postings_yoy_change_pct": snapshot.postings_yoy_change_pct,
                "top_roles": snapshot.top_roles,
                "top_locations": snapshot.top_locations,
                "exec_hire_count": snapshot.exec_hire_count,
                "new_cities": snapshot.new_cities,
            },
            "hiring_velocity": {
                "velocity_90d_pct": velocity.velocity_90d,
                "velocity_30d_pct": velocity.velocity_30d,
                "acceleration": velocity.acceleration,
                "signal": velocity.signal,
                "z_score": velocity.z_score,
                "interpretation": velocity.interpretation,
            },
            "tech_stack": tech,
            "geo_expansion": {
                "current_cities": geo.current_cities,
                "new_cities_vs_prior": geo.new_cities_vs_prior,
                "expansion_score": geo.expansion_score,
                "primary_markets": geo.primary_markets,
                "remote_pct": geo.remote_pct,
            },
            "exec_hires": {
                "exec_postings": exec_sig.exec_postings,
                "exec_hire_count": exec_sig.exec_hire_count,
                "c_suite_count": exec_sig.c_suite_count,
                "vp_count": exec_sig.vp_count,
                "is_strategic_pivot": exec_sig.is_strategic_pivot,
                "pivot_reason": exec_sig.pivot_reason,
                "signal_strength": exec_sig.signal_strength,
            },
            "role_composition": {
                "composition_pct": role_comp.composition,
                "growth_stage": role_comp.growth_stage,
                "eng_to_sales_ratio": role_comp.eng_to_sales_ratio,
                "eng_to_support_ratio": role_comp.eng_to_support_ratio,
                "interpretation": role_comp.interpretation,
            },
        }
    except Exception as exc:
        logger.error("hiring-signals error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/job-volume/{ticker}",
    summary="Job posting volume trend for a ticker",
    response_model=dict,
)
def get_job_volume(
    ticker: str,
    company: Optional[str] = Query(None),
    pages: int = Query(3, ge=1, le=6),
) -> dict:
    """Return job posting volume and velocity for a ticker."""
    _, scraper, _, _, engine = _get_components()
    ticker = ticker.upper()
    cname = company or _company_from_ticker(ticker)

    try:
        snapshot = scraper.fetch_company_postings(ticker, cname, pages=pages)
        velocity = engine.compute_hiring_velocity(ticker, cname, pages=pages)
        return {
            "ticker": ticker,
            "company": cname,
            "as_of": date.today().isoformat(),
            "postings_30d": snapshot.total_postings_30d,
            "postings_90d": snapshot.total_postings_90d,
            "yoy_change_pct": snapshot.postings_yoy_change_pct,
            "velocity": {
                "90d_change_pct": velocity.velocity_90d,
                "30d_change_pct": velocity.velocity_30d,
                "signal": velocity.signal,
                "z_score": velocity.z_score,
                "interpretation": velocity.interpretation,
            },
        }
    except Exception as exc:
        logger.error("job-volume error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/tech-stack/{ticker}",
    summary="Tech stack analysis from job descriptions",
    response_model=dict,
)
def get_tech_stack(
    ticker: str,
    company: Optional[str] = Query(None),
    pages: int = Query(3, ge=1, le=6),
) -> dict:
    """Return cloud adoption and tech stack signals for a ticker."""
    _, scraper, _, _, engine = _get_components()
    ticker = ticker.upper()
    cname = company or _company_from_ticker(ticker)

    try:
        snapshot = scraper.fetch_company_postings(ticker, cname, pages=pages)
        tech = engine.detect_tech_stack(snapshot.raw_postings)
        return {
            "ticker": ticker,
            "company": cname,
            "as_of": date.today().isoformat(),
            "total_postings_analyzed": len(snapshot.raw_postings),
            **tech,
        }
    except Exception as exc:
        logger.error("tech-stack error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/macro-labor",
    summary="Macro labor market dashboard (FRED / BLS JOLTS)",
    response_model=dict,
)
def get_macro_labor(use_cache: bool = Query(True)) -> dict:
    """
    Return comprehensive macro labor market dashboard:
    JOLTS openings, quits rate, hires, unemployment, nonfarm payrolls,
    sector-level openings, and macro signal classification.
    """
    _, _, bls, _, _ = _get_components()
    try:
        dashboard = bls.get_macro_labor_dashboard(use_cache=use_cache)
        return _macro_dashboard_to_dict(dashboard)
    except Exception as exc:
        logger.error("macro-labor error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/web-traffic/{ticker}",
    summary="Wayback Machine CDX web traffic proxy",
    response_model=dict,
)
def get_web_traffic(
    ticker: str,
    lookback_months: int = Query(24, ge=3, le=60),
) -> dict:
    """
    Return Wayback Machine CDX web traffic proxy for a ticker's domain.
    Counts monthly snapshot crawls as a proxy for web visibility / traffic.
    """
    _, _, _, cdx, _ = _get_components()
    ticker = ticker.upper()
    domain = TICKER_TO_DOMAIN.get(ticker, f"{ticker.lower()}.com")

    try:
        proxy = cdx.get_traffic_proxy(domain, lookback_months=lookback_months, ticker=ticker)
        return {
            "ticker": ticker,
            "domain": domain,
            "as_of": proxy.as_of.isoformat(),
            "trend_direction": proxy.trend_direction,
            "yoy_change_pct": proxy.yoy_change_pct,
            "mom_change_pct": proxy.mom_change_pct,
            "z_score_latest": proxy.z_score_latest,
            "monthly_captures": {
                k.isoformat(): int(v)
                for k, v in proxy.monthly_captures.items()
            } if not proxy.monthly_captures.empty else {},
        }
    except Exception as exc:
        logger.error("web-traffic error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/geo-expansion/{ticker}",
    summary="Geographic expansion signals from job postings",
    response_model=dict,
)
def get_geo_expansion(
    ticker: str,
    company: Optional[str] = Query(None),
    pages: int = Query(3, ge=1, le=6),
) -> dict:
    """Return geographic expansion signals derived from job posting locations."""
    _, scraper, _, _, engine = _get_components()
    ticker = ticker.upper()
    cname = company or _company_from_ticker(ticker)

    try:
        snapshot = scraper.fetch_company_postings(ticker, cname, pages=pages)
        geo = engine.detect_geo_expansion(snapshot.raw_postings, ticker=ticker)
        return {
            "ticker": ticker,
            "company": cname,
            "as_of": geo.as_of.isoformat(),
            "current_cities": geo.current_cities,
            "new_cities_vs_prior_30d": geo.new_cities_vs_prior,
            "expansion_score": geo.expansion_score,
            "primary_markets": geo.primary_markets,
            "remote_pct": geo.remote_pct,
        }
    except Exception as exc:
        logger.error("geo-expansion error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/exec-hires/{ticker}",
    summary="Executive / C-suite hiring signals",
    response_model=dict,
)
def get_exec_hires(
    ticker: str,
    company: Optional[str] = Query(None),
    pages: int = Query(4, ge=1, le=8),
) -> dict:
    """Detect executive and C-suite hiring patterns that may signal strategic pivots."""
    _, scraper, _, _, engine = _get_components()
    ticker = ticker.upper()
    cname = company or _company_from_ticker(ticker)

    try:
        snapshot = scraper.fetch_company_postings(ticker, cname, pages=pages)
        sig = engine.detect_exec_hires(snapshot.raw_postings, ticker=ticker)
        return {
            "ticker": ticker,
            "company": cname,
            "as_of": sig.as_of.isoformat(),
            "exec_postings": sig.exec_postings,
            "exec_hire_count": sig.exec_hire_count,
            "c_suite_count": sig.c_suite_count,
            "vp_count": sig.vp_count,
            "is_strategic_pivot": sig.is_strategic_pivot,
            "pivot_reason": sig.pivot_reason,
            "signal_strength": sig.signal_strength,
        }
    except Exception as exc:
        logger.error("exec-hires error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/role-composition/{ticker}",
    summary="Role composition and growth stage inference",
    response_model=dict,
)
def get_role_composition(
    ticker: str,
    company: Optional[str] = Query(None),
    pages: int = Query(3, ge=1, le=6),
) -> dict:
    """
    Return role composition (engineering/sales/support ratios) and
    inferred growth stage classification.
    """
    _, scraper, _, _, engine = _get_components()
    ticker = ticker.upper()
    cname = company or _company_from_ticker(ticker)

    try:
        snapshot = scraper.fetch_company_postings(ticker, cname, pages=pages)
        role_sig = engine.compute_role_composition(snapshot.raw_postings, ticker=ticker)
        return {
            "ticker": ticker,
            "company": cname,
            "as_of": role_sig.as_of.isoformat(),
            "composition_pct": role_sig.composition,
            "growth_stage": role_sig.growth_stage,
            "eng_to_sales_ratio": role_sig.eng_to_sales_ratio,
            "eng_to_support_ratio": role_sig.eng_to_support_ratio,
            "interpretation": role_sig.interpretation,
        }
    except Exception as exc:
        logger.error("role-composition error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/sector-openings",
    summary="JOLTS sector-level job openings from FRED",
    response_model=dict,
)
def get_sector_openings(use_cache: bool = Query(True)) -> dict:
    """Return latest JOLTS job openings by sector (FRED free CSV)."""
    _, _, bls, _, _ = _get_components()
    try:
        openings = bls.get_sector_openings(use_cache=use_cache)
        quits = bls.get_quits_rate(use_cache=use_cache)
        return {
            "as_of": date.today().isoformat(),
            "sector_openings_k": openings,
            "quits_rate": quits,
        }
    except Exception as exc:
        logger.error("sector-openings error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/domain-traffic-compare",
    summary="Compare web traffic proxy across multiple tickers",
    response_model=dict,
)
def compare_web_traffic(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL"),
    lookback_months: int = Query(12, ge=3, le=36),
) -> dict:
    """Compare Wayback CDX web traffic proxies across a set of tickers."""
    _, _, _, cdx, _ = _get_components()
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")

    domains = [TICKER_TO_DOMAIN.get(t, f"{t.lower()}.com") for t in ticker_list]
    try:
        df = cdx.compare_domains(domains, lookback_months=lookback_months)
        return {
            "as_of": date.today().isoformat(),
            "tickers": ticker_list,
            "domains": domains,
            "comparison": df.reset_index().to_dict(orient="records"),
        }
    except Exception as exc:
        logger.error("domain-traffic-compare error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@job_postings_v2_router.get(
    "/fred-series/{series_id}",
    summary="Raw FRED time series data",
    response_model=dict,
)
def get_fred_series(
    series_id: str,
    start: str = Query("2020-01-01"),
    end: Optional[str] = Query(None),
    use_cache: bool = Query(True),
) -> dict:
    """Download any FRED series as JSON. No API key required."""
    _, _, bls, _, _ = _get_components()
    try:
        df = bls.get_jolts_fred(series_id.upper(), start=start, end=end, use_cache=use_cache)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for series {series_id}")
        return {
            "series_id": series_id.upper(),
            "description": FRED_SERIES.get(series_id.upper(), ""),
            "start": start,
            "end": end or date.today().isoformat(),
            "data": df[["date", "value"]].assign(
                date=df["date"].astype(str)
            ).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("fred-series error for %s: %s", series_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Module-level initialization helper
# ---------------------------------------------------------------------------


def build_job_postings_engine(
    db_path: Optional[Path] = None,
) -> tuple[JobPostingsCache, JobPostingsScraper, BLSFREDAdapter, WaybackCDXAdapter, HiringSignalEngine]:
    """
    Build and return all components for the job postings signals system.

    Parameters
    ----------
    db_path:
        Override default SQLite database path.

    Returns
    -------
    (cache, scraper, bls_adapter, cdx_adapter, signal_engine)
    """
    cache = JobPostingsCache(db_path or _CACHE_DB)
    scraper = JobPostingsScraper(cache=cache)
    bls = BLSFREDAdapter(cache=cache)
    cdx = WaybackCDXAdapter(cache=cache)
    engine = HiringSignalEngine(scraper=scraper, cache=cache)
    return cache, scraper, bls, cdx, engine


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    company_arg = sys.argv[2] if len(sys.argv) > 2 else _company_from_ticker(ticker_arg)

    logger.info("=== SENTINEL Job Postings Signals v2 ===")
    logger.info("Ticker: %s | Company: %s", ticker_arg, company_arg)

    cache, scraper, bls, cdx, engine = build_job_postings_engine()

    # 1. Macro labor dashboard
    logger.info("\n[1/5] Fetching macro labor dashboard...")
    dash = bls.get_macro_labor_dashboard()
    print(f"  Job Openings: {dash.total_job_openings_k:,.0f}k" if dash.total_job_openings_k else "  Job Openings: N/A")
    print(f"  Unemployment: {dash.unemployment_rate}% | Quits: {dash.total_quits_rate}%")
    print(f"  Signal: {dash.macro_signal}")
    print(f"  Interpretation: {dash.interpretation}")

    # 2. Company hiring snapshot
    logger.info("\n[2/5] Fetching company postings for %s...", ticker_arg)
    snapshot = scraper.fetch_company_postings(ticker_arg, company_arg, pages=2)
    print(f"  30d postings: {snapshot.total_postings_30d}")
    print(f"  90d postings: {snapshot.total_postings_90d}")
    print(f"  Top roles: {snapshot.top_roles[:3]}")
    print(f"  Top locations: {snapshot.top_locations[:3]}")

    # 3. Hiring velocity
    logger.info("\n[3/5] Computing hiring velocity...")
    velocity = engine.compute_hiring_velocity(ticker_arg, company_arg, pages=2)
    print(f"  Velocity 90d: {velocity.velocity_90d:.1f}%")
    print(f"  Signal: {velocity.signal} (z={velocity.z_score:.2f})")
    print(f"  {velocity.interpretation}")

    # 4. Tech stack
    logger.info("\n[4/5] Analyzing tech stack...")
    tech = engine.detect_tech_stack(snapshot.raw_postings)
    print(f"  Cloud: {tech.get('cloud_provider')} ({tech.get('cloud_dominance_pct')}%)")
    print(f"  Modernization score: {tech.get('modernization_score')}/10")

    # 5. Web traffic proxy
    logger.info("\n[5/5] Fetching web traffic proxy for %s...", ticker_arg)
    domain = TICKER_TO_DOMAIN.get(ticker_arg.upper(), f"{ticker_arg.lower()}.com")
    proxy = cdx.get_traffic_proxy(domain, lookback_months=12, ticker=ticker_arg)
    print(f"  Domain: {domain}")
    print(f"  Trend: {proxy.trend_direction}")
    print(f"  YoY change: {proxy.yoy_change_pct}%")
    print(f"  Z-score: {proxy.z_score_latest:.2f}")

    logger.info("\n=== Done ===")
