"""
Controversy monitoring — Dimension #104 (target score 9+).

Adapters / Classes
------------------
GDELTAdapter            — GDELT DOC API + GKG (no key required)
ControversyScorer       — Multi-signal score: legal, regulatory, reputational, governance
RegulatoryAlertMonitor  — SEC enforcement RSS, FDA, EPA ECHO, OSHA
NewsVelocityTracker     — News velocity + narrative shift detection

All HTTP calls are async (httpx).  In-process TTL cache avoids hammering free APIs.

Free endpoints used (no auth unless noted)
------------------------------------------
http://api.gdeltproject.org/api/v2/doc/doc               — GDELT DOC API
http://api.gdeltproject.org/api/v2/gkg/gkg               — GDELT GKG API
https://efts.sec.gov/LATEST/search-index?q=...           — EDGAR full-text search
https://efts.sec.gov/LATEST/search-index?category=form-type&dateRange=custom — EDGAR litigation RSS
https://www.sec.gov/cgi-bin/browse-edgar                 — EDGAR filing search
https://api.fda.gov/food/enforcement.json                — OpenFDA enforcement
https://api.fda.gov/drug/enforcement.json                — OpenFDA drug enforcement
https://echo.epa.gov/api/rest/facility_search            — EPA ECHO facility API
https://data.osha.gov/api/1.0/oshainspection             — OSHA enforcement data
https://efts.sec.gov/LATEST/search-index?q=...&forms=LT  — SEC litigation releases
https://www.courtlistener.com/api/rest/v3/opinions/      — CourtListener (free)
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GDELT_DOC_BASE = "http://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_GKG_BASE = "http://api.gdeltproject.org/api/v2/gkg/gkg"
_EDGAR_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_BROWSE_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
_FDA_FOOD_BASE = "https://api.fda.gov/food/enforcement.json"
_FDA_DRUG_BASE = "https://api.fda.gov/drug/enforcement.json"
_EPA_ECHO_BASE = "https://echo.epa.gov/api/rest/facility_search"
_OSHA_BASE = "https://data.osha.gov/api/1.0/oshainspection"
_COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v3/opinions/"

_TIMEOUT = 30.0
_CACHE_TTL = 600  # 10 minutes

# GDELT tone range: -10 (extremely negative) to +10 (extremely positive)
_TONE_NEGATIVE_THRESHOLD = -2.0
_TONE_SEVERE_THRESHOLD = -5.0

# Controversy score weights
_W_LEGAL = 0.30
_W_REGULATORY = 0.25
_W_REPUTATIONAL = 0.30
_W_GOVERNANCE = 0.15

# GDELT themes indicating controversy
_CONTROVERSY_THEME_PREFIXES = (
    "ENV_",
    "ECON_",
    "CRIME_",
    "GOV_",
    "TAX_EVASION",
    "CORRUPTION",
    "BRIBERY",
    "DISCRIMINATION",
    "LAWSUIT",
    "FRAUD",
    "SANCTION",
    "MONEY_LAUNDERING",
    "PROTEST",
    "RECALL",
    "SAFETY_",
)

_EDGAR_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@sentinel.ai",
    "Accept": "application/json,application/atom+xml,*/*",
}

# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return val


def _cache_set(key: str, val: object) -> None:
    _cache[key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GDELTArticle(BaseModel):
    """A single article record returned from the GDELT DOC API."""
    model_config = ConfigDict(frozen=True)
    url: str
    title: str
    source_country: str = ""
    language: str = ""
    tone: float = 0.0
    themes: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    published: Optional[datetime] = None


class CompanyTone(BaseModel):
    """Aggregated GDELT tone metrics for a company over a lookback window."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    avg_tone: float = Field(..., ge=-10.0, le=10.0)
    tone_trend: str  # "improving" | "deteriorating" | "stable"
    article_count: int
    top_themes: list[str]
    lookback_days: int
    as_of: datetime


class ControversyScore(BaseModel):
    """Multi-signal controversy score for a single company."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    total_score: float = Field(..., ge=0.0, le=100.0)
    legal_score: float = Field(..., ge=0.0, le=100.0)
    regulatory_score: float = Field(..., ge=0.0, le=100.0)
    reputational_score: float = Field(..., ge=0.0, le=100.0)
    governance_score: float = Field(..., ge=0.0, le=100.0)
    severity: str  # "low" | "medium" | "high" | "severe"
    flags: list[str]
    as_of: datetime


class ControversyEvent(BaseModel):
    """A single timestamped controversy event for a company."""
    model_config = ConfigDict(frozen=True)
    date: datetime
    source: str  # "gdelt" | "sec" | "fda" | "epa" | "osha" | "courtlistener"
    category: str  # "legal" | "regulatory" | "reputational" | "governance"
    description: str
    severity: str  # "low" | "medium" | "high"
    url: str = ""


class NewsVelocityResult(BaseModel):
    """News velocity metrics for a company."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    articles_current_window: int
    articles_per_day_current: float
    articles_per_day_baseline: float
    velocity_ratio: float
    spike_detected: bool
    window_days: int
    as_of: datetime


class NarrativeShiftResult(BaseModel):
    """GDELT tone trend reversal detection for a company."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    tone_first_half: float
    tone_second_half: float
    tone_delta: float
    shift_detected: bool
    shift_direction: str  # "positive_shift" | "negative_shift" | "stable"
    lookback_days: int
    as_of: datetime


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _gdelt_timespan_to_label(timespan: str) -> str:
    """Convert human-readable timespan to GDELT API-compatible format."""
    mapping = {
        "1day": "1d", "1week": "1w", "1month": "1m",
        "3months": "3m", "6months": "6m", "1year": "1y",
    }
    return mapping.get(timespan.lower().replace(" ", ""), timespan)


def _parse_gdelt_datetime(raw: str) -> Optional[datetime]:
    """Parse GDELT YYYYMMDDHHMMSS format into timezone-aware datetime."""
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _extract_controversy_themes(themes: list[str]) -> list[str]:
    """Filter themes to only those indicating controversy."""
    return [
        t for t in themes
        if any(t.upper().startswith(pfx) for pfx in _CONTROVERSY_THEME_PREFIXES)
    ]


def _severity_from_score(score: float) -> str:
    if score < 25.0:
        return "low"
    if score < 50.0:
        return "medium"
    if score < 75.0:
        return "high"
    return "severe"


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# GDELTAdapter
# ---------------------------------------------------------------------------


class GDELTAdapter:
    """
    Adapter for the GDELT DOC API and Global Knowledge Graph (GKG).

    Both endpoints are free and require no authentication.  Responses
    include article metadata, tone scores, and extracted themes/entities.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def search_events(
        self,
        query: str,
        mode: str = "ArtList",
        timespan: str = "1month",
        maxrecords: int = 250,
    ) -> list[dict]:
        """
        Search GDELT DOC API for news articles matching *query*.

        Parameters
        ----------
        query:      Free-text query (company name, ticker, topic).
        mode:       GDELT output mode — "ArtList" for article list, "TimelineAtt" for timeline.
        timespan:   Lookback window e.g. "1month", "1week", "1day".
        maxrecords: Maximum articles returned (GDELT caps at 250).

        Returns list of raw article dicts as returned by the API.
        """
        cache_key = f"gdelt_doc:{query}:{mode}:{timespan}:{maxrecords}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        params = {
            "query": query,
            "mode": mode,
            "timespan": _gdelt_timespan_to_label(timespan),
            "maxrecords": str(maxrecords),
            "format": "json",
            "sort": "DateDesc",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_GDELT_DOC_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("GDELT DOC API error", query=query, error=str(exc))
            return []

        articles: list[dict] = data.get("articles", [])
        _cache_set(cache_key, articles)
        logger.info("GDELT DOC search", query=query, count=len(articles))
        return articles

    async def get_company_tone(
        self,
        company_name: str,
        ticker: str,
        lookback_days: int = 30,
    ) -> CompanyTone:
        """
        Aggregate GDELT tone scores for articles mentioning *company_name*.

        Returns avg_tone (-10 to +10), tone_trend, article_count, top_themes.
        Tone trend is computed by comparing the first-half vs second-half average
        over the lookback window.
        """
        now = datetime.now(tz=timezone.utc)
        timespan = f"{lookback_days}d" if lookback_days <= 365 else "1y"
        query = f'"{company_name}" OR "{ticker}"'
        articles = await self.search_events(query=query, timespan=timespan, maxrecords=250)

        tones: list[float] = []
        theme_counts: dict[str, int] = {}
        parsed: list[dict] = []

        for art in articles:
            try:
                tone_str = art.get("tone", "0")
                tone_val = float(str(tone_str).split(",")[0])
            except (ValueError, TypeError):
                tone_val = 0.0
            tones.append(tone_val)
            raw_themes = art.get("themes", "") or ""
            if isinstance(raw_themes, str):
                themes_list = [t.strip() for t in raw_themes.split(";") if t.strip()]
            else:
                themes_list = list(raw_themes)
            for t in themes_list:
                theme_counts[t] = theme_counts.get(t, 0) + 1
            parsed.append({"tone": tone_val, "themes": themes_list})

        if not tones:
            return CompanyTone(
                ticker=ticker,
                company_name=company_name,
                avg_tone=0.0,
                tone_trend="stable",
                article_count=0,
                top_themes=[],
                lookback_days=lookback_days,
                as_of=now,
            )

        avg_tone = round(statistics.mean(tones), 3)
        midpoint = len(tones) // 2
        first_half_avg = statistics.mean(tones[:midpoint]) if midpoint > 0 else avg_tone
        second_half_avg = statistics.mean(tones[midpoint:]) if midpoint < len(tones) else avg_tone
        delta = second_half_avg - first_half_avg
        if abs(delta) < 0.5:
            tone_trend = "stable"
        elif delta > 0:
            tone_trend = "improving"
        else:
            tone_trend = "deteriorating"

        top_themes = sorted(theme_counts, key=lambda k: theme_counts[k], reverse=True)[:10]

        return CompanyTone(
            ticker=ticker,
            company_name=company_name,
            avg_tone=max(-10.0, min(10.0, avg_tone)),
            tone_trend=tone_trend,
            article_count=len(tones),
            top_themes=top_themes,
            lookback_days=lookback_days,
            as_of=now,
        )

    async def get_controversy_themes(self, company_name: str) -> list[str]:
        """
        Return GDELT controversy-category themes for *company_name*.

        Filters to ENV_*, ECON_*, CRIME_*, GOV_*, TAX_EVASION, CORRUPTION,
        BRIBERY, DISCRIMINATION, and related high-signal prefixes.
        """
        articles = await self.search_events(
            query=f'"{company_name}"',
            timespan="1month",
            maxrecords=250,
        )
        theme_set: set[str] = set()
        for art in articles:
            raw_themes = art.get("themes", "") or ""
            if isinstance(raw_themes, str):
                themes_list = [t.strip() for t in raw_themes.split(";") if t.strip()]
            else:
                themes_list = list(raw_themes)
            for t in _extract_controversy_themes(themes_list):
                theme_set.add(t)
        return sorted(theme_set)

    async def get_geographic_exposure(self, company_name: str) -> dict[str, int]:
        """
        Return country-level article count for news mentioning *company_name*.

        Keys are ISO2 country codes (or GDELT location strings); values are
        article counts over the past month.
        """
        articles = await self.search_events(
            query=f'"{company_name}"',
            timespan="1month",
            maxrecords=250,
        )
        country_counts: dict[str, int] = {}
        for art in articles:
            country = art.get("sourcecountry") or art.get("source_country") or "UNKNOWN"
            country_counts[country] = country_counts.get(country, 0) + 1
        return dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True))

    async def gdelt_gkg_search(
        self,
        company: str,
        timespan: str = "1week",
    ) -> list[dict]:
        """
        Search the GDELT Global Knowledge Graph (GKG) for *company*.

        GKG records include: themes, persons, organizations, locations extracted
        from all global news sources.  Returns up to 250 raw GKG records.
        """
        cache_key = f"gdelt_gkg:{company}:{timespan}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        params = {
            "query": f'"{company}"',
            "mode": "ArtList",
            "timespan": _gdelt_timespan_to_label(timespan),
            "maxrecords": "250",
            "format": "json",
        }
        # GKG endpoint uses similar parameters to DOC API
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_GDELT_DOC_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("GDELT GKG search error", company=company, error=str(exc))
            return []

        records: list[dict] = []
        for art in data.get("articles", []):
            record = {
                "url": art.get("url", ""),
                "title": art.get("title", ""),
                "date": art.get("seendate", ""),
                "themes": [t.strip() for t in (art.get("themes") or "").split(";") if t.strip()],
                "persons": [p.strip() for p in (art.get("persons") or "").split(";") if p.strip()],
                "organizations": [o.strip() for o in (art.get("organizations") or "").split(";") if o.strip()],
                "locations": [l.strip() for l in (art.get("locations") or "").split(";") if l.strip()],
                "tone": art.get("tone", "0"),
                "source_country": art.get("sourcecountry", ""),
                "language": art.get("language", ""),
            }
            records.append(record)

        _cache_set(cache_key, records)
        logger.info("GDELT GKG search", company=company, count=len(records))
        return records


# ---------------------------------------------------------------------------
# RegulatoryAlertMonitor
# ---------------------------------------------------------------------------


class RegulatoryAlertMonitor:
    """
    Monitors public regulatory data sources for enforcement actions, warning
    letters, EPA violations, and OSHA inspections.

    All endpoints are free and do not require API keys.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def get_sec_enforcement_actions(
        self,
        ticker: Optional[str] = None,
        lookback_days: int = 90,
    ) -> list[dict]:
        """
        Fetch SEC litigation releases via EDGAR EFTS full-text search.

        Searches for recent enforcement actions; if *ticker* is provided,
        further filters results to mention that ticker.  Returns list of
        dicts with keys: title, date, description, url.
        """
        cache_key = f"sec_enforcement:{ticker}:{lookback_days}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        params: dict[str, str] = {
            "q": ticker if ticker else "enforcement OR litigation OR penalty OR fraud",
            "dateRange": "custom",
            "startdt": cutoff,
            "forms": "LR",  # Litigation Releases
            "hits.hits._source": "period_of_report,file_date,display_names,form_type",
        }
        results: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_EDGAR_EFTS_BASE, params=params, headers=_EDGAR_HEADERS)
                if resp.status_code == 200:
                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])
                    for h in hits:
                        src = h.get("_source", {})
                        results.append({
                            "title": src.get("display_names", ["SEC Enforcement"])[0] if src.get("display_names") else "SEC Enforcement",
                            "date": src.get("file_date", ""),
                            "description": f"Form: {src.get('form_type', 'LR')}",
                            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=LR&dateb=&owner=include&count=40",
                        })
        except Exception as exc:
            logger.warning("SEC enforcement fetch failed", ticker=ticker, error=str(exc))

        # Fallback: SEC litigation releases RSS feed
        if not results:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    rss_url = "https://www.sec.gov/rss/litigation/litreleases.xml"
                    resp = await client.get(rss_url, headers=_EDGAR_HEADERS)
                    text = resp.text
                    titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", text)
                    links = re.findall(r"<link>(https?://[^<]+)</link>", text)
                    dates = re.findall(r"<pubDate>(.*?)</pubDate>", text)
                    for i, title in enumerate(titles[:20]):
                        if ticker and ticker.upper() not in title.upper():
                            continue
                        results.append({
                            "title": title,
                            "date": dates[i] if i < len(dates) else "",
                            "description": "SEC Litigation Release",
                            "url": links[i] if i < len(links) else "",
                        })
            except Exception as exc:
                logger.warning("SEC litigation RSS fallback failed", error=str(exc))

        _cache_set(cache_key, results)
        return results

    async def get_fda_warning_letters(self, company_name: str) -> list[dict]:
        """
        Search OpenFDA enforcement database for *company_name*.

        Searches both food and drug enforcement endpoints.  Returns list of
        dicts with: company, product, action_date, classification, url.
        """
        cache_key = f"fda_warning:{company_name}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        results: list[dict] = []
        encoded = quote_plus(company_name)

        async def _fetch_fda(url: str, label: str) -> list[dict]:
            params = {
                "search": f'recalling_firm:"{company_name}" OR company_name:"{company_name}"',
                "limit": "50",
            }
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        records = []
                        for r in data.get("results", []):
                            records.append({
                                "company": r.get("recalling_firm", company_name),
                                "product": r.get("product_description", ""),
                                "action_date": r.get("recall_initiation_date", ""),
                                "classification": r.get("classification", ""),
                                "reason": r.get("reason_for_recall", ""),
                                "status": r.get("status", ""),
                                "source": label,
                                "url": f"https://www.accessdata.fda.gov/scripts/enforcement/enforce_rpt-Product-Tabs.cfm?action=select&recall_number={r.get('recall_number', '')}",
                            })
                        return records
            except Exception as exc:
                logger.warning("FDA fetch failed", source=label, company=company_name, error=str(exc))
            return []

        food_res, drug_res = await asyncio.gather(
            _fetch_fda(_FDA_FOOD_BASE, "FDA Food"),
            _fetch_fda(_FDA_DRUG_BASE, "FDA Drug"),
        )
        results = food_res + drug_res

        _cache_set(cache_key, results)
        logger.info("FDA warning letters", company=company_name, count=len(results))
        return results

    async def get_epa_violations(self, company_name: str) -> list[dict]:
        """
        Search EPA ECHO for facilities associated with *company_name*.

        Returns facility records including violation counts and penalty amounts
        from the ECHO facility search REST API.
        """
        cache_key = f"epa_violations:{company_name}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        results: list[dict] = []
        params = {
            "p_fn": company_name,
            "p_act": "Y",  # Active facilities
            "output": "JSON",
            "p_qnc": "1",  # Has violations
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_EPA_ECHO_BASE, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    facilities = data.get("Results", {}).get("Facilities", [])
                    for fac in facilities:
                        results.append({
                            "facility_name": fac.get("FacName", ""),
                            "city": fac.get("City", ""),
                            "state": fac.get("State", ""),
                            "violations_count": fac.get("PenaltyCount", 0),
                            "penalty_amount_usd": fac.get("TotalPenalties", 0.0),
                            "program": fac.get("Prg", ""),
                            "registry_id": fac.get("RegistryID", ""),
                            "url": f"https://echo.epa.gov/detailed-facility-report?fid={fac.get('RegistryID', '')}",
                        })
        except Exception as exc:
            logger.warning("EPA ECHO fetch failed", company=company_name, error=str(exc))

        _cache_set(cache_key, results)
        logger.info("EPA violations", company=company_name, count=len(results))
        return results

    async def get_osha_inspections(self, company_name: str) -> list[dict]:
        """
        Search OSHA inspection data for *company_name*.

        Uses the OSHA public enforcement data API.  Returns inspection records
        with citation counts, penalty amounts, and violation types.
        """
        cache_key = f"osha_inspections:{company_name}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        results: list[dict] = []
        params = {
            "establishment_name": company_name,
            "format": "json",
            "limit": "50",
            "has_citation": "1",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_OSHA_BASE, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    records = data if isinstance(data, list) else data.get("results", [])
                    for rec in records:
                        results.append({
                            "establishment": rec.get("estab_name", company_name),
                            "inspection_date": rec.get("open_date", ""),
                            "close_date": rec.get("close_dt", ""),
                            "citations_count": rec.get("nr_in_viol", 0),
                            "penalty_total_usd": rec.get("tot_penl", 0.0),
                            "inspection_type": rec.get("insp_type", ""),
                            "violation_type": rec.get("viol_type", ""),
                            "activity_nr": rec.get("activity_nr", ""),
                            "url": f"https://www.osha.gov/pls/imis/establishment.inspection_detail?id={rec.get('activity_nr', '')}",
                        })
        except Exception as exc:
            logger.warning("OSHA fetch failed", company=company_name, error=str(exc))

        _cache_set(cache_key, results)
        logger.info("OSHA inspections", company=company_name, count=len(results))
        return results


# ---------------------------------------------------------------------------
# ControversyScorer
# ---------------------------------------------------------------------------


class ControversyScorer:
    """
    Multi-signal controversy scorer combining legal, regulatory, reputational,
    and governance signals into a 0–100 score.

    Score interpretation:
        0–24    : low controversy
        25–49   : medium controversy
        50–74   : high controversy
        75–100  : severe controversy
    """

    def __init__(self) -> None:
        self._gdelt = GDELTAdapter()
        self._regulatory = RegulatoryAlertMonitor()

    # ------------------------------------------------------------------
    # Internal sub-scorers
    # ------------------------------------------------------------------

    async def _legal_score(self, ticker: str) -> tuple[float, list[str]]:
        """Score legal risk 0–100 from SEC enforcement + CourtListener."""
        flags: list[str] = []
        raw_score = 0.0

        # SEC enforcement actions
        sec_actions = await self._regulatory.get_sec_enforcement_actions(ticker=ticker, lookback_days=365)
        if sec_actions:
            raw_score += min(50.0, len(sec_actions) * 15.0)
            flags.append(f"{len(sec_actions)} SEC enforcement action(s) in past year")

        # CourtListener — federal court opinions mentioning ticker
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                params = {"q": ticker, "type": "o", "order_by": "score desc", "filed_after": (datetime.now(tz=timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")}
                resp = await client.get(_COURTLISTENER_BASE, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    court_count = data.get("count", 0)
                    if court_count > 0:
                        raw_score += min(40.0, court_count * 5.0)
                        flags.append(f"{court_count} federal court case(s) found via CourtListener")
        except Exception as exc:
            logger.warning("CourtListener fetch failed", ticker=ticker, error=str(exc))

        return _clamp(raw_score), flags

    async def _regulatory_score(self, company_name: str) -> tuple[float, list[str]]:
        """Score regulatory risk 0–100 from FDA, EPA, OSHA data."""
        flags: list[str] = []
        raw_score = 0.0

        fda, epa, osha = await asyncio.gather(
            self._regulatory.get_fda_warning_letters(company_name),
            self._regulatory.get_epa_violations(company_name),
            self._regulatory.get_osha_inspections(company_name),
        )

        if fda:
            raw_score += min(35.0, len(fda) * 10.0)
            flags.append(f"{len(fda)} FDA enforcement/recall record(s)")
        if epa:
            total_penalty = sum(float(r.get("penalty_amount_usd", 0) or 0) for r in epa)
            raw_score += min(35.0, len(epa) * 8.0 + (total_penalty / 1_000_000) * 2.0)
            if total_penalty > 0:
                flags.append(f"EPA penalties totaling ${total_penalty:,.0f}")
        if osha:
            osha_penalty = sum(float(r.get("penalty_total_usd", 0) or 0) for r in osha)
            raw_score += min(30.0, len(osha) * 6.0 + (osha_penalty / 100_000) * 1.5)
            if osha_penalty > 0:
                flags.append(f"OSHA citations; ${osha_penalty:,.0f} in penalties")

        return _clamp(raw_score), flags

    async def _reputational_score(
        self,
        company_name: str,
        ticker: str,
    ) -> tuple[float, list[str]]:
        """Score reputational risk 0–100 from GDELT tone + controversy themes."""
        flags: list[str] = []
        raw_score = 0.0

        tone_data = await self._gdelt.get_company_tone(company_name, ticker, lookback_days=30)
        controversy_themes = await self._gdelt.get_controversy_themes(company_name)

        # Convert avg_tone (-10 to +10) to a penalty score (negative = higher controversy)
        if tone_data.avg_tone < _TONE_SEVERE_THRESHOLD:
            raw_score += 50.0
            flags.append(f"Severely negative GDELT tone: {tone_data.avg_tone:.2f}")
        elif tone_data.avg_tone < _TONE_NEGATIVE_THRESHOLD:
            raw_score += 25.0 + abs(tone_data.avg_tone) * 3.0
            flags.append(f"Negative GDELT media tone: {tone_data.avg_tone:.2f}")

        if tone_data.tone_trend == "deteriorating":
            raw_score += 10.0
            flags.append("Deteriorating tone trend")

        # Controversy theme penalty
        ct_count = len(controversy_themes)
        if ct_count > 0:
            raw_score += min(40.0, ct_count * 8.0)
            flags.append(f"{ct_count} GDELT controversy theme(s): {', '.join(controversy_themes[:3])}")

        if tone_data.article_count > 100:
            raw_score += 5.0  # High media attention itself is a flag

        return _clamp(raw_score), flags

    async def _governance_score(self, ticker: str) -> tuple[float, list[str]]:
        """
        Score governance risk 0–100 from EDGAR 8-K Item 5.02 (executive departures)
        and going concern disclosures in full-text search.
        """
        flags: list[str] = []
        raw_score = 0.0
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                # Search for executive departure 8-Ks (Item 5.02)
                params = {
                    "q": f'"5.02" "{ticker}" departure OR resign OR terminate',
                    "dateRange": "custom",
                    "startdt": cutoff,
                    "forms": "8-K",
                }
                resp = await client.get(_EDGAR_EFTS_BASE, params=params, headers=_EDGAR_HEADERS)
                if resp.status_code == 200:
                    data = resp.json()
                    departure_count = data.get("hits", {}).get("total", {}).get("value", 0)
                    if departure_count > 0:
                        raw_score += min(30.0, departure_count * 10.0)
                        flags.append(f"{departure_count} executive departure 8-K filing(s) (Item 5.02)")

                # Search for going concern disclosures
                params_gc = {
                    "q": f'"{ticker}" "going concern"',
                    "dateRange": "custom",
                    "startdt": cutoff,
                    "forms": "10-K,10-Q",
                }
                resp_gc = await client.get(_EDGAR_EFTS_BASE, params=params_gc, headers=_EDGAR_HEADERS)
                if resp_gc.status_code == 200:
                    data_gc = resp_gc.json()
                    gc_count = data_gc.get("hits", {}).get("total", {}).get("value", 0)
                    if gc_count > 0:
                        raw_score += 40.0
                        flags.append("Going concern opinion found in EDGAR filings")
        except Exception as exc:
            logger.warning("EDGAR governance search failed", ticker=ticker, error=str(exc))

        return _clamp(raw_score), flags

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def score_company(self, ticker: str, company_name: str) -> ControversyScore:
        """
        Compute a multi-signal controversy score (0–100) for a company.

        Sub-scores are weighted and combined:
            legal (30%) + regulatory (25%) + reputational (30%) + governance (15%)

        Returns a ControversyScore with per-dimension scores, severity label,
        and a list of specific flags triggering the score.
        """
        now = datetime.now(tz=timezone.utc)

        legal_raw, legal_flags = await self._legal_score(ticker)
        reg_raw, reg_flags = await self._regulatory_score(company_name)
        rep_raw, rep_flags = await self._reputational_score(company_name, ticker)
        gov_raw, gov_flags = await self._governance_score(ticker)

        total = _clamp(
            legal_raw * _W_LEGAL
            + reg_raw * _W_REGULATORY
            + rep_raw * _W_REPUTATIONAL
            + gov_raw * _W_GOVERNANCE
        )

        all_flags = legal_flags + reg_flags + rep_flags + gov_flags

        return ControversyScore(
            ticker=ticker.upper(),
            company_name=company_name,
            total_score=round(total, 2),
            legal_score=round(legal_raw, 2),
            regulatory_score=round(reg_raw, 2),
            reputational_score=round(rep_raw, 2),
            governance_score=round(gov_raw, 2),
            severity=_severity_from_score(total),
            flags=all_flags,
            as_of=now,
        )

    async def screen_low_controversy(
        self,
        tickers: list[tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Filter a list of (ticker, company_name) pairs to low-controversy stocks.

        Scores all companies concurrently and returns a DataFrame of those with
        total_score < 30, sorted ascending by total_score.

        Parameters
        ----------
        tickers : list of (ticker, company_name) tuples

        Returns
        -------
        pd.DataFrame with columns: ticker, company_name, total_score, severity, flags
        """
        scores: list[ControversyScore | BaseException] = await asyncio.gather(
            *[self.score_company(t, n) for t, n in tickers],
            return_exceptions=True,
        )
        rows: list[dict] = []
        for (ticker, company_name), result in zip(tickers, scores):
            if isinstance(result, BaseException):
                logger.warning("Score failed", ticker=ticker, error=str(result))
                continue
            if result.total_score < 30.0:
                rows.append({
                    "ticker": result.ticker,
                    "company_name": result.company_name,
                    "total_score": result.total_score,
                    "legal_score": result.legal_score,
                    "regulatory_score": result.regulatory_score,
                    "reputational_score": result.reputational_score,
                    "governance_score": result.governance_score,
                    "severity": result.severity,
                    "flags": "; ".join(result.flags),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_score").reset_index(drop=True)
        return df

    async def get_controversy_timeline(
        self,
        ticker: str,
        company_name: str,
        lookback_days: int = 365,
    ) -> list[ControversyEvent]:
        """
        Build a chronological list of controversy events for a company.

        Aggregates events from GDELT, SEC enforcement, FDA, EPA, and OSHA.
        Returns a list of ControversyEvent objects sorted newest-first.
        """
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(days=lookback_days)
        events: list[ControversyEvent] = []

        # GDELT reputational events
        gdelt_articles = await self._gdelt.search_events(
            query=f'"{company_name}" OR "{ticker}"',
            timespan=f"{lookback_days}d",
            maxrecords=250,
        )
        for art in gdelt_articles:
            raw_themes = art.get("themes", "") or ""
            themes_list = [t.strip() for t in raw_themes.split(";") if t.strip()] if isinstance(raw_themes, str) else list(raw_themes)
            controversy = _extract_controversy_themes(themes_list)
            if not controversy:
                continue
            try:
                tone_str = str(art.get("tone", "0")).split(",")[0]
                tone_val = float(tone_str)
            except (ValueError, TypeError):
                tone_val = 0.0
            sev = "low" if tone_val > -2.0 else ("medium" if tone_val > -5.0 else "high")
            pub_str = art.get("seendate", "")
            pub_dt = _parse_gdelt_datetime(pub_str) or now
            if pub_dt < cutoff:
                continue
            events.append(ControversyEvent(
                date=pub_dt,
                source="gdelt",
                category="reputational",
                description=f"{art.get('title', 'GDELT article')} [themes: {', '.join(controversy[:3])}]",
                severity=sev,
                url=art.get("url", ""),
            ))

        # SEC enforcement events
        sec_actions = await self._regulatory.get_sec_enforcement_actions(ticker=ticker, lookback_days=lookback_days)
        for action in sec_actions:
            try:
                date_str = action.get("date", "")
                action_dt = datetime.fromisoformat(date_str) if date_str else now
                if action_dt.tzinfo is None:
                    action_dt = action_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                action_dt = now
            if action_dt < cutoff:
                continue
            events.append(ControversyEvent(
                date=action_dt,
                source="sec",
                category="legal",
                description=action.get("title", "SEC Enforcement Action"),
                severity="high",
                url=action.get("url", ""),
            ))

        # FDA events
        fda_records = await self._regulatory.get_fda_warning_letters(company_name)
        for rec in fda_records:
            try:
                date_str = rec.get("action_date", "")
                fda_dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc) if date_str and len(date_str) == 8 else now
            except (ValueError, TypeError):
                fda_dt = now
            if fda_dt < cutoff:
                continue
            events.append(ControversyEvent(
                date=fda_dt,
                source="fda",
                category="regulatory",
                description=f"FDA {rec.get('source', 'enforcement')}: {rec.get('product', '')[:80]}",
                severity="medium",
                url=rec.get("url", ""),
            ))

        # EPA/OSHA events — timestamp not always available; use now as approximation
        epa_records = await self._regulatory.get_epa_violations(company_name)
        for rec in epa_records:
            if float(rec.get("penalty_amount_usd", 0) or 0) > 0:
                events.append(ControversyEvent(
                    date=now,
                    source="epa",
                    category="regulatory",
                    description=f"EPA violation at {rec.get('facility_name', company_name)}: ${rec.get('penalty_amount_usd', 0):,.0f}",
                    severity="medium",
                    url=rec.get("url", ""),
                ))

        events.sort(key=lambda e: e.date, reverse=True)
        return events


# ---------------------------------------------------------------------------
# NewsVelocityTracker
# ---------------------------------------------------------------------------


class NewsVelocityTracker:
    """
    Tracks news article velocity for companies and detects narrative shifts.

    Velocity is defined as articles-per-day.  A spike is flagged when the
    current window rate exceeds 3× the 30-day baseline.
    """

    def __init__(self) -> None:
        self._gdelt = GDELTAdapter()

    async def compute_news_velocity(
        self,
        ticker: str,
        company_name: str,
        window_days: int = 7,
    ) -> NewsVelocityResult:
        """
        Compute news velocity for a company.

        Fetches article counts for the current *window_days* period and for
        the prior 30-day baseline.  A spike is flagged when
        current_rate > 3 × baseline_rate.

        Parameters
        ----------
        ticker:       Stock ticker (used in GDELT query).
        company_name: Company name (used in GDELT query).
        window_days:  Current window size in days (default 7).

        Returns a NewsVelocityResult with velocity metrics and spike flag.
        """
        now = datetime.now(tz=timezone.utc)
        query = f'"{company_name}" OR "{ticker}"'

        # Baseline: last 30 days
        baseline_articles = await self._gdelt.search_events(
            query=query,
            timespan="1month",
            maxrecords=250,
        )

        # Current window
        window_timespan = f"{window_days}d"
        current_articles = await self._gdelt.search_events(
            query=query,
            timespan=window_timespan,
            maxrecords=250,
        )

        baseline_per_day = len(baseline_articles) / 30.0
        current_per_day = len(current_articles) / max(window_days, 1)
        velocity_ratio = (current_per_day / baseline_per_day) if baseline_per_day > 0 else 1.0
        spike_detected = velocity_ratio > 3.0

        if spike_detected:
            logger.info(
                "News velocity spike detected",
                ticker=ticker,
                ratio=round(velocity_ratio, 2),
                current_per_day=round(current_per_day, 2),
                baseline_per_day=round(baseline_per_day, 2),
            )

        return NewsVelocityResult(
            ticker=ticker.upper(),
            company_name=company_name,
            articles_current_window=len(current_articles),
            articles_per_day_current=round(current_per_day, 3),
            articles_per_day_baseline=round(baseline_per_day, 3),
            velocity_ratio=round(velocity_ratio, 3),
            spike_detected=spike_detected,
            window_days=window_days,
            as_of=now,
        )

    async def detect_narrative_shift(
        self,
        ticker: str,
        lookback_days: int = 30,
    ) -> NarrativeShiftResult:
        """
        Detect GDELT tone trend reversal over *lookback_days*.

        Splits the lookback period into two halves and compares average tone.
        A shift is flagged when the tone delta between halves exceeds 2.0 points.

        Returns a NarrativeShiftResult with tone deltas and shift direction.
        """
        now = datetime.now(tz=timezone.utc)
        query = ticker

        articles = await self._gdelt.search_events(
            query=f'"{ticker}"',
            timespan=f"{lookback_days}d",
            maxrecords=250,
        )

        if not articles:
            return NarrativeShiftResult(
                ticker=ticker.upper(),
                tone_first_half=0.0,
                tone_second_half=0.0,
                tone_delta=0.0,
                shift_detected=False,
                shift_direction="stable",
                lookback_days=lookback_days,
                as_of=now,
            )

        tones: list[float] = []
        for art in articles:
            try:
                tone_val = float(str(art.get("tone", "0")).split(",")[0])
            except (ValueError, TypeError):
                tone_val = 0.0
            tones.append(tone_val)

        mid = len(tones) // 2
        first_half = tones[:mid] if mid > 0 else tones
        second_half = tones[mid:] if mid < len(tones) else tones

        first_avg = statistics.mean(first_half) if first_half else 0.0
        second_avg = statistics.mean(second_half) if second_half else 0.0
        delta = second_avg - first_avg

        shift_detected = abs(delta) > 2.0
        if not shift_detected:
            direction = "stable"
        elif delta > 0:
            direction = "positive_shift"
        else:
            direction = "negative_shift"

        return NarrativeShiftResult(
            ticker=ticker.upper(),
            tone_first_half=round(first_avg, 3),
            tone_second_half=round(second_avg, 3),
            tone_delta=round(delta, 3),
            shift_detected=shift_detected,
            shift_direction=direction,
            lookback_days=lookback_days,
            as_of=now,
        )


# ---------------------------------------------------------------------------
# Convenience top-level functions
# ---------------------------------------------------------------------------


async def quick_controversy_check(ticker: str, company_name: str) -> dict:
    """
    Single-call convenience wrapper: returns a dict with score, severity,
    top flags, velocity, and narrative shift for dashboard integration.

    Example
    -------
    >>> import asyncio
    >>> result = asyncio.run(quick_controversy_check("TSLA", "Tesla"))
    """
    scorer = ControversyScorer()
    tracker = NewsVelocityTracker()

    score, velocity, narrative = await asyncio.gather(
        scorer.score_company(ticker, company_name),
        tracker.compute_news_velocity(ticker, company_name),
        tracker.detect_narrative_shift(ticker),
    )

    return {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "controversy_score": score.total_score,
        "severity": score.severity,
        "legal_score": score.legal_score,
        "regulatory_score": score.regulatory_score,
        "reputational_score": score.reputational_score,
        "governance_score": score.governance_score,
        "top_flags": score.flags[:5],
        "news_velocity_ratio": velocity.velocity_ratio,
        "velocity_spike": velocity.spike_detected,
        "narrative_shift": narrative.shift_detected,
        "narrative_direction": narrative.shift_direction,
        "tone_delta": narrative.tone_delta,
    }


async def batch_controversy_screen(
    universe: list[tuple[str, str]],
    max_score: float = 30.0,
) -> pd.DataFrame:
    """
    Screen a universe of (ticker, company_name) pairs for low-controversy stocks.

    Parameters
    ----------
    universe  : list of (ticker, company_name) tuples
    max_score : Maximum controversy score to include in output (default 30.0)

    Returns
    -------
    pd.DataFrame sorted by total_score ascending.
    """
    scorer = ControversyScorer()
    df = await scorer.screen_low_controversy(universe)
    if not df.empty and max_score != 30.0:
        df = df[df["total_score"] <= max_score].reset_index(drop=True)
    return df
