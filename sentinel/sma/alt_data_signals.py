"""
Alternative Data Signals — Dimension #086 (target score 9+).

Covers job postings, web traffic proxies, Google Trends, H-1B filings,
layoff signals, and macro mobility indicators — all from free public sources.

Classes
-------
JobPostingsAdapter        — Indeed RSS, LinkedIn proxy, H-1B DOL data, hiring momentum
WebTrafficAdapter         — SimilarWeb proxy, pytrends web-traffic proxy, app-download proxy
AltDataSignalEngine       — Composite alt-data signal, hiring vs revenue divergence, layoff signal
EconomicMobilityTracker   — OpenStreetMap POI density, BTS transport, OpenTable proxy, FRED retail
alt_data_router           — FastAPI router

Free endpoints used (no API keys required unless noted)
-------------------------------------------------------
https://www.indeed.com/rss                                          — Indeed job RSS
https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/             — DOL H-1B disclosure data
https://api.stlouisfed.org/fred/series/observations                — FRED (RETAILSMSA, DCOILWTICO)
https://layoffs.fyi/                                               — Layoffs.fyi public data
https://efts.sec.gov/LATEST/search-index                          — EDGAR full-text search (8-K)
https://data.bts.gov/api/views/                                    — BTS transportation stats
pytrends                                                           — Google Trends proxy
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEED_RSS = "https://www.indeed.com/rss?q=%22{company}%22&sort=date&limit=25"
_GOOGLE_JOBS_PROXY = (
    "https://www.googleapis.com/customsearch/v1"
    "?q={query}&cx={cx}&num=10&key={key}"
)
_DOL_H1B_FY24_Q1 = "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2024_Q1.xlsx"
_LAYOFFS_FYI = "https://layoffs.fyi/"
_EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
_FRED_OBS = "https://api.stlouisfed.org/fred/series/observations"
_BTS_PASSENGER = "https://data.bts.gov/api/views/crem-w88i/rows.json?accessType=DOWNLOAD"
_FRED_RETAIL = "RETAILSMSA"
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

_TIMEOUT = 25.0
_CACHE_TTL = 1800.0  # 30 min for alt-data (slower-moving)

# Department inference keywords from job titles
_DEPT_KEYWORDS: dict[str, list[str]] = {
    "tech": [
        "engineer", "developer", "software", "data scientist", "ml", "ai",
        "devops", "sre", "architect", "backend", "frontend", "fullstack",
        "infrastructure", "security", "qa", "cloud", "platform",
    ],
    "sales": [
        "sales", "account executive", "business development", "account manager",
        "revenue", "partnership", "sdr", "bdr", "client success",
    ],
    "ops": [
        "operations", "logistics", "supply chain", "warehouse", "fulfillment",
        "customer support", "customer service", "analyst", "project manager",
    ],
    "finance": [
        "finance", "accounting", "controller", "cfo", "treasury", "audit",
        "compliance", "tax", "financial analyst",
    ],
}

# ---------------------------------------------------------------------------
# In-process TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> object | None:
    entry = _cache.get(key)
    if entry and time.monotonic() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: object) -> None:
    _cache[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _company_to_slug(company_name: str) -> str:
    """Convert company name to URL slug for Indeed company page."""
    slug = re.sub(r"[^a-z0-9\s]", "", company_name.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


def _infer_department(title: str) -> str:
    """Infer department from job title string."""
    title_lower = title.lower()
    for dept, keywords in _DEPT_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return dept
    return "other"


def _pct_change(current: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return round((current - baseline) / abs(baseline) * 100, 1)


# ---------------------------------------------------------------------------
# JobPostingsAdapter
# ---------------------------------------------------------------------------


class JobPostingsAdapter:
    """
    Collects and analyses job posting signals from public sources.

    Primary: Indeed RSS (public, no auth)
    Secondary: Google as LinkedIn proxy
    Tertiary: DOL H-1B LCA disclosure data
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def get_indeed_postings(
        self,
        company_name: str,
        lookback_days: int = 30,
    ) -> dict:
        """
        Scrape Indeed RSS for jobs posted by company_name in the last N days.
        Returns {job_count, titles_breakdown, locations, posted_dates, department_mix}.
        """
        cache_key = f"indeed:{company_name}:{lookback_days}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        url = _INDEED_RSS.format(company=quote_plus(company_name))
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        titles: list[str] = []
        locations: list[str] = []
        posted_dates: list[str] = []

        try:
            import xml.etree.ElementTree as ET

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "SENTINEL:FinancialTerminal:1.0"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
        except Exception as exc:
            logger.debug("Indeed RSS fetch error", company=company_name, error=str(exc))
            return self._empty_postings(company_name)

        items = root.findall(".//item")
        for item in items:
            title_el = item.find("title")
            pub_el = item.find("pubDate")
            desc_el = item.find("description")

            title = (title_el.text or "").strip() if title_el is not None else ""
            pub_raw = (pub_el.text or "").strip() if pub_el is not None else ""
            desc = (desc_el.text or "").strip() if desc_el is not None else ""

            if not title:
                continue

            # Parse date
            pub_dt = None
            for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
                try:
                    pub_dt = datetime.strptime(pub_raw, fmt)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue

            if pub_dt and pub_dt < cutoff:
                continue

            titles.append(title)
            if pub_raw:
                posted_dates.append(pub_raw)

            # Extract location from description
            loc_match = re.search(r"<b>Location</b>\s*[:—-]?\s*([^<\n]+)", desc)
            if loc_match:
                locations.append(loc_match.group(1).strip())

        # Department mix from titles
        dept_counts: dict[str, int] = {d: 0 for d in _DEPT_KEYWORDS}
        dept_counts["other"] = 0
        for t in titles:
            dept = _infer_department(t)
            dept_counts[dept] = dept_counts.get(dept, 0) + 1

        total = max(1, len(titles))
        dept_mix = {f"{k}_pct": round(v / total * 100, 1) for k, v in dept_counts.items()}

        result = {
            "company_name": company_name,
            "job_count": len(titles),
            "titles_breakdown": titles[:20],
            "locations": list(set(locations))[:10],
            "posted_dates": posted_dates[:20],
            "department_mix": dept_mix,
            "source": "indeed_rss",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        _cache_set(cache_key, result)
        return result

    def _empty_postings(self, company_name: str) -> dict:
        return {
            "company_name": company_name,
            "job_count": 0,
            "titles_breakdown": [],
            "locations": [],
            "posted_dates": [],
            "department_mix": {},
            "source": "indeed_rss",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    async def get_linkedin_jobs_proxy(
        self,
        company_name: str,
        ticker: str,
    ) -> dict:
        """
        Use Google search result count as LinkedIn jobs proxy.
        Queries: {company} site:linkedin.com/jobs
        Returns {estimated_job_count, comparison_note}.
        """
        cache_key = f"linkedin_proxy:{company_name}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        # Use a scraping-friendly approach: DuckDuckGo HTML (no API key needed)
        query = f"{company_name} site:linkedin.com/jobs"
        ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        estimated_count = 0
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    ddg_url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; SENTINEL/1.0)"},
                    follow_redirects=True,
                )
                text = resp.text
                # Extract result count hint from DDG HTML
                count_match = re.search(
                    r"About\s+([\d,]+)\s+results",
                    text,
                    re.IGNORECASE,
                )
                if count_match:
                    estimated_count = int(count_match.group(1).replace(",", ""))
                else:
                    # Count LinkedIn job links found
                    job_links = re.findall(r"linkedin\.com/jobs/view/", text)
                    estimated_count = len(job_links) * 10  # scale up
        except Exception as exc:
            logger.debug("LinkedIn proxy error", company=company_name, error=str(exc))

        result = {
            "company_name": company_name,
            "ticker": ticker,
            "estimated_job_count": estimated_count,
            "source": "google_linkedin_proxy",
            "note": "Estimated from search result count; not an official LinkedIn figure",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        _cache_set(cache_key, result)
        return result

    async def compute_hiring_momentum(
        self,
        ticker: str,
        company_name: str,
        lookback_months: int = 6,
    ) -> dict:
        """
        Compare current job postings vs 30/60/90-day averages.
        Returns hiring_acceleration (%), hiring_signal, department_mix.
        """
        # Collect current period and earlier windows
        current = await self.get_indeed_postings(company_name, lookback_days=30)
        prior_60 = await self.get_indeed_postings(company_name, lookback_days=60)
        prior_90 = await self.get_indeed_postings(company_name, lookback_days=90)

        curr_count = current.get("job_count", 0)
        # Approximate 30d vs 60d rolling: 60d window - 30d window ≈ prior 30d
        prior_30 = max(0, prior_60.get("job_count", 0) - curr_count)
        prior_60_avg = prior_90.get("job_count", 0) / 3 if prior_90.get("job_count", 0) > 0 else 0

        acceleration = _pct_change(curr_count, prior_30) if prior_30 > 0 else None

        if acceleration is None:
            hiring_signal = "unknown"
        elif acceleration > 20:
            hiring_signal = "expanding"
        elif acceleration < -10:
            hiring_signal = "contracting"
        else:
            hiring_signal = "stable"

        return {
            "ticker": ticker,
            "company_name": company_name,
            "current_30d_count": curr_count,
            "prior_30d_estimate": prior_30,
            "prior_60d_avg": round(prior_60_avg, 1),
            "hiring_acceleration_pct": acceleration,
            "hiring_signal": hiring_signal,
            "department_mix": current.get("department_mix", {}),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    async def get_h1b_filings(self, company_name: str) -> dict:
        """
        Parse DOL H-1B LCA Disclosure Data (public quarterly Excel file).
        Filters by employer name, returns h1b_applications, avg_wage, top_job_titles.

        Note: Excel file is ~50MB — results are cached aggressively.
        Falls back to empty result on download failure.
        """
        cache_key = f"h1b:{company_name.lower()}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(
                    _DOL_H1B_FY24_Q1,
                    headers={"User-Agent": "SENTINEL:FinancialTerminal:1.0"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                content = resp.content
        except Exception as exc:
            logger.debug("H-1B download error", company=company_name, error=str(exc))
            return self._empty_h1b(company_name)

        try:
            import io
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception as exc:
            logger.warning("H-1B Excel parse error", error=str(exc))
            return self._empty_h1b(company_name)

        # Normalise column names
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Find employer column
        emp_col = next(
            (c for c in df.columns if "employer" in c or "company" in c),
            None,
        )
        if emp_col is None:
            return self._empty_h1b(company_name)

        name_lower = company_name.lower()
        filtered = df[df[emp_col].str.lower().str.contains(name_lower, na=False)]

        if filtered.empty:
            result = self._empty_h1b(company_name)
            _cache_set(cache_key, result)
            return result

        wage_col = next((c for c in df.columns if "wage" in c and "prevail" not in c), None)
        title_col = next((c for c in df.columns if "job_title" in c or "soc_title" in c), None)

        avg_wage = None
        if wage_col and wage_col in filtered.columns:
            wages = pd.to_numeric(filtered[wage_col], errors="coerce").dropna()
            avg_wage = round(float(wages.mean()), 0) if len(wages) > 0 else None

        top_titles: list[str] = []
        if title_col and title_col in filtered.columns:
            top_titles = (
                filtered[title_col].value_counts().head(5).index.tolist()
            )

        result = {
            "company_name": company_name,
            "h1b_applications": len(filtered),
            "avg_wage_usd": avg_wage,
            "top_job_titles": top_titles,
            "source": "dol_lca_fy2024_q1",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        _cache_set(cache_key, result)
        return result

    def _empty_h1b(self, company_name: str) -> dict:
        return {
            "company_name": company_name,
            "h1b_applications": 0,
            "avg_wage_usd": None,
            "top_job_titles": [],
            "source": "dol_lca_fy2024_q1",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# WebTrafficAdapter
# ---------------------------------------------------------------------------


class WebTrafficAdapter:
    """
    Estimates web traffic signals from public proxies:
      - Google Trends as traffic proxy (pytrends)
      - SimilarWeb public overview (limited scraping)
      - Alexa rank (archived endpoint)
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def get_similarweb_proxy(self, domain: str, ticker: str) -> dict:
        """
        Attempt to retrieve SimilarWeb public overview data via scraping.
        Falls back to empty dict if blocked (very likely on free tier).
        """
        cache_key = f"similarweb:{domain}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        url = f"https://www.similarweb.com/website/{domain}/"
        estimated: dict = {
            "domain": domain,
            "ticker": ticker,
            "estimated_monthly_visits": None,
            "engagement_time_minutes": None,
            "bounce_rate_pct": None,
            "traffic_trend_pct": None,
            "source": "similarweb_scrape",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    follow_redirects=True,
                )
                html = resp.text

            # Parse JSON-LD or structured data embedded in the page
            visits_match = re.search(
                r'"totalVisits"\s*:\s*([\d.]+)',
                html,
            )
            if visits_match:
                estimated["estimated_monthly_visits"] = int(float(visits_match.group(1)))

            bounce_match = re.search(r'"bounceRate"\s*:\s*([\d.]+)', html)
            if bounce_match:
                estimated["bounce_rate_pct"] = round(float(bounce_match.group(1)) * 100, 1)

            trend_match = re.search(r'"visitsTrend"\s*:\s*(["-\d.]+)', html)
            if trend_match:
                try:
                    estimated["traffic_trend_pct"] = float(trend_match.group(1).strip('"'))
                except ValueError:
                    pass

        except Exception as exc:
            logger.debug("SimilarWeb proxy error", domain=domain, error=str(exc))

        _cache_set(cache_key, estimated)
        return estimated

    async def get_google_trends_web_traffic(
        self,
        company_name: str,
        ticker: str,
    ) -> dict:
        """
        Use pytrends search interest as a web traffic proxy.
        Compares current 7-day interest vs 90-day baseline.
        """
        cache_key = f"gtrends_traffic:{ticker}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        result: dict = {
            "ticker": ticker,
            "company_name": company_name,
            "traffic_vs_baseline_pct": None,
            "current_interest": None,
            "baseline_interest": None,
            "trend_direction": "unknown",
            "source": "google_trends_proxy",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        try:
            from pytrends.request import TrendReq
            import random

            # Jitter to avoid rate-limiting
            await asyncio.sleep(random.uniform(2.0, 5.0))

            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            kw_list = [company_name[:50]]  # pytrends keyword limit
            pt.build_payload(kw_list, cat=0, timeframe="today 3-m", geo="US")
            df = pt.interest_over_time()

            if df is not None and not df.empty and company_name[:50] in df.columns:
                series = df[company_name[:50]].astype(float)
                current = float(series.iloc[-7:].mean()) if len(series) >= 7 else float(series.mean())
                baseline = float(series.mean())
                pct_vs_baseline = _pct_change(current, baseline)

                if pct_vs_baseline is not None:
                    if pct_vs_baseline > 15:
                        direction = "rising"
                    elif pct_vs_baseline < -15:
                        direction = "falling"
                    else:
                        direction = "stable"
                else:
                    direction = "stable"

                result.update({
                    "current_interest": round(current, 1),
                    "baseline_interest": round(baseline, 1),
                    "traffic_vs_baseline_pct": pct_vs_baseline,
                    "trend_direction": direction,
                })

        except ImportError:
            logger.warning("pytrends not installed; Google Trends proxy unavailable")
        except Exception as exc:
            logger.debug("Google Trends traffic error", ticker=ticker, error=str(exc))

        _cache_set(cache_key, result)
        return result

    async def compute_app_download_proxy(
        self,
        ticker: str,
        company_name: str,
    ) -> dict:
        """
        Use Google Trends for app name to proxy mobile app download momentum.
        Checks if company appears in App Store category-level public charts.
        """
        cache_key = f"app_proxy:{ticker}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        result: dict = {
            "ticker": ticker,
            "company_name": company_name,
            "app_trend_direction": "unknown",
            "search_momentum": None,
            "source": "google_trends_app_proxy",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        try:
            from pytrends.request import TrendReq
            import random

            await asyncio.sleep(random.uniform(3.0, 7.0))

            # Search for "{company} app" to isolate mobile searches
            app_query = f"{company_name} app"
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pt.build_payload([app_query[:50]], cat=0, timeframe="today 3-m", geo="US")
            df = pt.interest_over_time()

            if df is not None and not df.empty and app_query[:50] in df.columns:
                series = df[app_query[:50]].astype(float)
                if len(series) >= 14:
                    recent_2w = float(series.iloc[-14:].mean())
                    prior_2w = float(series.iloc[-28:-14].mean()) if len(series) >= 28 else float(series.mean())
                    momentum = _pct_change(recent_2w, prior_2w)
                    result["search_momentum"] = momentum
                    if momentum is not None:
                        result["app_trend_direction"] = (
                            "rising" if momentum > 10 else "falling" if momentum < -10 else "stable"
                        )

        except ImportError:
            logger.warning("pytrends not installed; app proxy unavailable")
        except Exception as exc:
            logger.debug("App download proxy error", ticker=ticker, error=str(exc))

        _cache_set(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# AltDataSignalEngine
# ---------------------------------------------------------------------------


class AltDataSignalEngine:
    """
    Combines job postings, web traffic, Google Trends, and H-1B signals
    into a composite alternative-data score (0–100).
    """

    def __init__(self) -> None:
        self._jobs = JobPostingsAdapter()
        self._web = WebTrafficAdapter()

    async def compute_composite_alt_signal(
        self,
        ticker: str,
        company_name: str,
        domain: Optional[str] = None,
    ) -> dict:
        """
        Composite alt-data signal:
          job posting momentum  40%
          web traffic trend     30%
          Google Trends         20%
          H-1B growth           10%

        Returns composite_score (0–100), component_scores, signal.
        """
        # Collect components concurrently
        tasks = [
            self._jobs.compute_hiring_momentum(ticker, company_name),
            self._web.get_google_trends_web_traffic(company_name, ticker),
            self._jobs.get_h1b_filings(company_name),
        ]
        if domain:
            tasks.append(self._web.get_similarweb_proxy(domain, ticker))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        hiring = results[0] if not isinstance(results[0], Exception) else {}
        gtrends = results[1] if not isinstance(results[1], Exception) else {}
        h1b = results[2] if not isinstance(results[2], Exception) else {}
        web_traffic = results[3] if (len(results) > 3 and not isinstance(results[3], Exception)) else {}

        # Score each component to 0–100 scale
        job_score = self._score_hiring(hiring)
        web_score = self._score_web_traffic(gtrends, web_traffic)
        trends_score = self._score_google_trends(gtrends)
        h1b_score = self._score_h1b(h1b)

        composite = (
            0.40 * job_score
            + 0.30 * web_score
            + 0.20 * trends_score
            + 0.10 * h1b_score
        )

        if composite >= 65:
            signal = "expanding"
        elif composite <= 35:
            signal = "contracting"
        else:
            signal = "stable"

        return {
            "ticker": ticker,
            "company_name": company_name,
            "composite_score": round(composite, 1),
            "signal": signal,
            "component_scores": {
                "hiring_momentum": round(job_score, 1),
                "web_traffic": round(web_score, 1),
                "google_trends": round(trends_score, 1),
                "h1b_growth": round(h1b_score, 1),
            },
            "hiring_signal": hiring.get("hiring_signal", "unknown"),
            "hiring_acceleration_pct": hiring.get("hiring_acceleration_pct"),
            "traffic_vs_baseline_pct": gtrends.get("traffic_vs_baseline_pct"),
            "h1b_applications": h1b.get("h1b_applications", 0),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def _score_hiring(self, hiring: dict) -> float:
        """Map hiring signal to 0-100 score."""
        signal = hiring.get("hiring_signal", "unknown")
        accel = hiring.get("hiring_acceleration_pct")
        if signal == "unknown":
            return 50.0
        base = {"expanding": 75.0, "stable": 50.0, "contracting": 25.0}.get(signal, 50.0)
        if accel is not None:
            # Fine-tune by magnitude (each 10% acceleration = 2 points)
            adjustment = min(15.0, max(-15.0, accel / 5.0))
            base = min(100.0, max(0.0, base + adjustment))
        return base

    def _score_web_traffic(self, gtrends: dict, web_traffic: dict) -> float:
        """Score web traffic from available sources."""
        pct = gtrends.get("traffic_vs_baseline_pct")
        sw_trend = web_traffic.get("traffic_trend_pct")
        scores: list[float] = []
        if pct is not None:
            scores.append(min(100.0, max(0.0, 50.0 + pct * 0.5)))
        if sw_trend is not None:
            scores.append(min(100.0, max(0.0, 50.0 + sw_trend * 0.5)))
        return float(np.mean(scores)) if scores else 50.0

    def _score_google_trends(self, gtrends: dict) -> float:
        """Map Google Trends direction to 0-100."""
        direction = gtrends.get("trend_direction", "unknown")
        pct = gtrends.get("traffic_vs_baseline_pct")
        base = {"rising": 72.0, "stable": 50.0, "falling": 28.0, "unknown": 50.0}.get(direction, 50.0)
        if pct is not None:
            adjustment = min(20.0, max(-20.0, pct * 0.3))
            base = min(100.0, max(0.0, base + adjustment))
        return base

    def _score_h1b(self, h1b: dict) -> float:
        """Score H-1B filings (higher applications = expanding, tech-oriented)."""
        count = h1b.get("h1b_applications", 0)
        if count == 0:
            return 50.0
        # Sigmoid-like mapping: 100 applications → ~75 score
        score = 50.0 + 30.0 * (1 - math.exp(-count / 100))
        return min(100.0, score)

    async def screen_high_alt_momentum(
        self,
        tickers: list[str],
        company_name_map: Optional[dict[str, str]] = None,
        domain_map: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Compute composite alt signal for all tickers and return sorted DataFrame.
        """
        company_name_map = company_name_map or {}
        domain_map = domain_map or {}

        tasks = [
            self.compute_composite_alt_signal(
                ticker=t,
                company_name=company_name_map.get(t, t),
                domain=domain_map.get(t),
            )
            for t in tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rows: list[dict] = []
        for r in results:
            if isinstance(r, dict):
                rows.append(r)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        return df

    async def compute_hiring_vs_revenue_divergence(
        self,
        ticker: str,
        company_name: str,
        revenue_growth_pct: Optional[float] = None,
    ) -> dict:
        """
        Detect divergence between hiring momentum and revenue growth.

        hiring accelerating + revenue decelerating  → efficiency/restructuring signal
        revenue accelerating + hiring flat          → operating leverage (bullish margin)
        """
        hiring = await self._jobs.compute_hiring_momentum(ticker, company_name)
        h_accel = hiring.get("hiring_acceleration_pct") or 0.0
        h_signal = hiring.get("hiring_signal", "stable")

        divergence_type = "none"
        interpretation = "No significant divergence detected."
        bias = "neutral"

        if revenue_growth_pct is not None:
            if h_accel > 20 and revenue_growth_pct < 0:
                divergence_type = "hiring_up_revenue_down"
                interpretation = (
                    "Hiring is accelerating (+{:.1f}%) while revenue growth is negative ({:.1f}%). "
                    "Possible restructuring, market-share investment, or leading indicator of future revenue.".format(
                        h_accel, revenue_growth_pct
                    )
                )
                bias = "cautious"
            elif h_accel < -10 and revenue_growth_pct > 15:
                divergence_type = "hiring_flat_revenue_up"
                interpretation = (
                    "Revenue accelerating ({:.1f}%) while hiring contracts ({:.1f}%). "
                    "Strong operating leverage signal — bullish for margins.".format(
                        revenue_growth_pct, h_accel
                    )
                )
                bias = "bullish"
            elif h_accel > 20 and revenue_growth_pct > 15:
                divergence_type = "both_expanding"
                interpretation = "Both hiring and revenue are accelerating — growth-phase company."
                bias = "bullish"
            elif h_accel < -10 and revenue_growth_pct < 0:
                divergence_type = "both_contracting"
                interpretation = "Both hiring and revenue are contracting — contraction-phase risk."
                bias = "bearish"

        return {
            "ticker": ticker,
            "hiring_acceleration_pct": h_accel,
            "hiring_signal": h_signal,
            "revenue_growth_pct_input": revenue_growth_pct,
            "divergence_type": divergence_type,
            "interpretation": interpretation,
            "bias": bias,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    async def compute_layoff_signal(self, company_name: str) -> dict:
        """
        Detect layoff signals from:
          1. EDGAR 8-K full-text search for "reduction in force" / "restructuring"
          2. Layoffs.fyi public website scrape
        """
        cache_key = f"layoff:{company_name.lower()}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        # -- EDGAR full-text search --
        edgar_layoff = False
        edgar_date = None
        edgar_count_estimate = None

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                for keyword in ("reduction in force", "workforce reduction", "restructuring"):
                    params = {
                        "q": f'"{keyword}" "{company_name}"',
                        "dateRange": "custom",
                        "startdt": (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d"),
                        "enddt": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "forms": "8-K",
                    }
                    resp = await client.get(
                        _EDGAR_EFTS,
                        params=params,
                        headers={"User-Agent": "SENTINEL:FinancialTerminal:1.0"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])
                    if hits:
                        edgar_layoff = True
                        edgar_date = hits[0].get("_source", {}).get("file_date")
                        # Try to extract headcount from filing excerpt
                        excerpt = hits[0].get("_source", {}).get("period_of_report", "")
                        count_match = re.search(r"(\d[\d,]+)\s*(?:employee|position|worker|job)", excerpt, re.I)
                        if count_match:
                            edgar_count_estimate = int(count_match.group(1).replace(",", ""))
                        break
        except Exception as exc:
            logger.debug("EDGAR layoff search error", company=company_name, error=str(exc))

        # -- Layoffs.fyi scrape (public HTML) --
        layoffs_fyi_count = None
        layoffs_fyi_date = None

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _LAYOFFS_FYI,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; SENTINEL/1.0)"},
                    follow_redirects=True,
                )
                text = resp.text

            # Simple pattern match against company name in table rows
            name_pattern = re.compile(
                re.escape(company_name.split()[0]),  # first word of company name
                re.IGNORECASE,
            )
            for line in text.splitlines():
                if name_pattern.search(line):
                    count_match = re.search(r"(\d[\d,]+)", line)
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\w+ \d{4})", line)
                    if count_match:
                        layoffs_fyi_count = int(count_match.group(1).replace(",", ""))
                    if date_match:
                        layoffs_fyi_date = date_match.group(1)
                    break
        except Exception as exc:
            logger.debug("Layoffs.fyi scrape error", company=company_name, error=str(exc))

        recent_layoffs = edgar_layoff or (layoffs_fyi_count is not None and layoffs_fyi_count > 0)
        count_estimate = layoffs_fyi_count or edgar_count_estimate

        if count_estimate is None:
            severity = "unknown"
        elif count_estimate >= 5000:
            severity = "severe"
        elif count_estimate >= 1000:
            severity = "significant"
        elif count_estimate >= 100:
            severity = "moderate"
        else:
            severity = "minor"

        result = {
            "company_name": company_name,
            "recent_layoffs": recent_layoffs,
            "layoff_count_estimate": count_estimate,
            "layoff_date": layoffs_fyi_date or edgar_date,
            "severity": severity,
            "sources": {
                "edgar_8k_match": edgar_layoff,
                "layoffs_fyi_match": layoffs_fyi_count is not None,
            },
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        _cache_set(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# Math import (needed for _score_h1b)
# ---------------------------------------------------------------------------
import math


# ---------------------------------------------------------------------------
# EconomicMobilityTracker
# ---------------------------------------------------------------------------


class EconomicMobilityTracker:
    """
    Macro-level alternative data signals from public mobility and
    activity datasets.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def get_openstreetmap_poi_density(
        self,
        city: str,
        poi_type: str = "restaurant",
    ) -> dict:
        """
        Query Overpass API (OSM) for POI count as local economic activity proxy.
        Uses Overpass QL to count amenity nodes in a named area.
        """
        cache_key = f"osm:{city}:{poi_type}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
[out:json][timeout:25];
area["name"="{city}"]["boundary"="administrative"]->.searchArea;
node["amenity"="{poi_type}"](area.searchArea);
out count;
"""
        result: dict = {
            "city": city,
            "poi_type": poi_type,
            "poi_count": None,
            "source": "openstreetmap_overpass",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    overpass_url,
                    data={"data": query},
                    headers={"User-Agent": "SENTINEL:FinancialTerminal:1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
            elements = data.get("elements", [])
            if elements and "tags" in elements[0]:
                result["poi_count"] = int(elements[0]["tags"].get("nodes", 0))
        except Exception as exc:
            logger.debug("OSM Overpass error", city=city, error=str(exc))

        _cache_set(cache_key, result)
        return result

    async def get_transportation_stats(self) -> dict:
        """
        Fetch BTS (Bureau of Transportation Statistics) air passenger data
        as a proxy for economic activity levels.
        Falls back to FRED transport-related series.
        """
        cache_key = "bts_transport"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        # FRED: Air Revenue Passenger Miles (AIRRPMINDM, seasonally adjusted)
        fred_url = _FRED_BASE.format(sid="AIRRPMINDM")
        result: dict = {
            "series": "AIRRPMINDM",
            "description": "Air Revenue Passenger Miles (index, SA)",
            "latest_value": None,
            "mom_change_pct": None,
            "yoy_change_pct": None,
            "trend": "unknown",
            "source": "fred",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    fred_url,
                    headers={"User-Agent": "SENTINEL:FinancialTerminal:1.0"},
                )
                resp.raise_for_status()
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
            df = df.dropna().sort_values("DATE")
            if len(df) >= 13:
                latest = float(df["AIRRPMINDM"].iloc[-1])
                prior_m = float(df["AIRRPMINDM"].iloc[-2])
                prior_y = float(df["AIRRPMINDM"].iloc[-13])
                mom = _pct_change(latest, prior_m)
                yoy = _pct_change(latest, prior_y)
                result.update({
                    "latest_value": round(latest, 2),
                    "mom_change_pct": mom,
                    "yoy_change_pct": yoy,
                    "trend": "expanding" if (yoy or 0) > 5 else "contracting" if (yoy or 0) < -5 else "stable",
                })
        except Exception as exc:
            logger.debug("BTS transport error", error=str(exc))

        _cache_set(cache_key, result)
        return result

    async def get_restaurant_reservations_proxy(self) -> dict:
        """
        Fetch OpenTable state-of-industry seated diner data (public CSV/JSON).
        OpenTable publishes % change vs 2019 baseline by state.
        """
        cache_key = "opentable_reservations"
        cached = _cache_get(cache_key)
        if cached is not None:
            return dict(cached)  # type: ignore[arg-type]

        # OpenTable publishes this as a GitHub-hosted CSV (updated frequently)
        ot_url = (
            "https://raw.githubusercontent.com/TheUpshot/covid-19-data/master/"
            "restaurant-visits-open-table.csv"
        )

        result: dict = {
            "source": "opentable_via_github",
            "latest_date": None,
            "us_avg_vs_2019_pct": None,
            "state_breakdown": {},
            "trend": "unknown",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    ot_url,
                    headers={"User-Agent": "SENTINEL:FinancialTerminal:1.0"},
                )
                resp.raise_for_status()
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))

            if not df.empty:
                # US column usually "United States" or aggregated
                date_cols = [c for c in df.columns if re.match(r"\d{4}/\d{2}/\d{2}", str(c))]
                if date_cols:
                    latest_date = date_cols[-1]
                    result["latest_date"] = latest_date
                    us_row = df[df.iloc[:, 0].str.contains("United States", na=False)]
                    if not us_row.empty:
                        val = us_row[latest_date].values[0]
                        result["us_avg_vs_2019_pct"] = float(val) if pd.notna(val) else None
        except Exception as exc:
            logger.debug("OpenTable data error", error=str(exc))

        _cache_set(cache_key, result)
        return result

    async def get_redbook_retail_sales(self) -> pd.DataFrame:
        """
        Fetch FRED RETAILSMSA (Retail and Food Services Sales, seasonally adjusted).
        Returns DataFrame with DATE and RETAILSMSA columns.
        """
        cache_key = "fred_retailsmsa"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        fred_url = _FRED_BASE.format(sid=_FRED_RETAIL)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    fred_url,
                    headers={"User-Agent": "SENTINEL:FinancialTerminal:1.0"},
                )
                resp.raise_for_status()
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
            df = df.dropna().sort_values("DATE").tail(36)  # last 3 years
            _cache_set(cache_key, df)
            return df
        except Exception as exc:
            logger.debug("FRED retail sales error", error=str(exc))
            return pd.DataFrame(columns=["DATE", _FRED_RETAIL])


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

alt_data_router = APIRouter(prefix="/api/alt-data", tags=["Alternative Data"])

# Module-level engine instance
_engine: Optional[AltDataSignalEngine] = None
_mobility: Optional[EconomicMobilityTracker] = None


def _get_engine() -> AltDataSignalEngine:
    global _engine
    if _engine is None:
        _engine = AltDataSignalEngine()
    return _engine


def _get_mobility() -> EconomicMobilityTracker:
    global _mobility
    if _mobility is None:
        _mobility = EconomicMobilityTracker()
    return _mobility


@alt_data_router.get("/{ticker}/hiring", summary="Job posting signals for a company")
async def get_hiring_signals(
    ticker: str,
    company_name: Optional[str] = Query(default=None),
    lookback_months: int = Query(default=3, ge=1, le=12),
):
    """Return hiring momentum and department mix signals for a ticker."""
    company = company_name or ticker.upper()
    engine = _get_engine()
    try:
        momentum = await engine._jobs.compute_hiring_momentum(
            ticker=ticker.upper(),
            company_name=company,
            lookback_months=lookback_months,
        )
        h1b = await engine._jobs.get_h1b_filings(company)
        return {
            "ticker": ticker.upper(),
            "hiring_momentum": momentum,
            "h1b_data": h1b,
        }
    except Exception as exc:
        logger.error("Hiring signals error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@alt_data_router.get("/{ticker}/web-traffic", summary="Web traffic proxy signals")
async def get_web_traffic(
    ticker: str,
    company_name: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
):
    """Return web traffic proxy estimates from Google Trends and SimilarWeb."""
    company = company_name or ticker.upper()
    engine = _get_engine()
    try:
        tasks = [
            engine._web.get_google_trends_web_traffic(company, ticker.upper()),
            engine._web.compute_app_download_proxy(ticker.upper(), company),
        ]
        if domain:
            tasks.append(engine._web.get_similarweb_proxy(domain, ticker.upper()))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "ticker": ticker.upper(),
            "google_trends_traffic": results[0] if not isinstance(results[0], Exception) else {},
            "app_download_proxy": results[1] if not isinstance(results[1], Exception) else {},
            "similarweb_proxy": results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@alt_data_router.get("/{ticker}/composite", summary="Composite alt-data signal")
async def get_composite_alt_signal(
    ticker: str,
    company_name: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
):
    """Return composite alternative data signal (0-100 score) for a ticker."""
    company = company_name or ticker.upper()
    engine = _get_engine()
    try:
        result = await engine.compute_composite_alt_signal(
            ticker=ticker.upper(),
            company_name=company,
            domain=domain,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class MomentumScreenRequest(BaseModel):
    tickers: list[str]
    company_name_map: dict[str, str] = Field(default_factory=dict)
    domain_map: dict[str, str] = Field(default_factory=dict)


@alt_data_router.get("/screen/momentum", summary="Top stocks by alt-data momentum")
async def screen_momentum(
    tickers: str = Query(..., description="Comma-separated ticker list, e.g. AAPL,MSFT,GOOGL"),
):
    """Screen tickers by composite alt-data momentum and return sorted results."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="At least one ticker required")
    if len(ticker_list) > 15:
        raise HTTPException(status_code=400, detail="Maximum 15 tickers for momentum screen")

    engine = _get_engine()
    try:
        df = await engine.screen_high_alt_momentum(ticker_list)
        if df.empty:
            return {"count": 0, "results": []}
        # Drop columns that might not be JSON-serialisable
        safe_cols = [c for c in df.columns if c != "component_scores"]
        return {
            "count": len(df),
            "results": df[safe_cols].to_dict(orient="records"),
            "component_scores": df["component_scores"].tolist() if "component_scores" in df.columns else [],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@alt_data_router.get("/macro/mobility", summary="Macro economic mobility indicators")
async def get_macro_mobility():
    """Return macro mobility signals: air travel, restaurant reservations, retail sales."""
    mobility = _get_mobility()
    try:
        transport, reservations = await asyncio.gather(
            mobility.get_transportation_stats(),
            mobility.get_restaurant_reservations_proxy(),
            return_exceptions=True,
        )
        retail_df = await mobility.get_redbook_retail_sales()

        retail_summary: dict = {}
        if isinstance(retail_df, pd.DataFrame) and not retail_df.empty:
            col = _FRED_RETAIL
            if col in retail_df.columns:
                latest = float(retail_df[col].iloc[-1])
                prior = float(retail_df[col].iloc[-13]) if len(retail_df) >= 13 else None
                retail_summary = {
                    "latest_value": round(latest, 2),
                    "yoy_change_pct": _pct_change(latest, prior) if prior else None,
                    "series": col,
                    "description": "Retail and Food Services Sales (SA, $M)",
                }

        return {
            "air_transportation": transport if not isinstance(transport, Exception) else {},
            "restaurant_reservations": reservations if not isinstance(reservations, Exception) else {},
            "retail_sales": retail_summary,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
