"""
sentinel/sma/controversy_monitor_v3.py
dim_104: Controversy Monitoring (GDELT + EPA + OSHA + SEC EDGAR)

Production-grade ESG controversy monitoring using free data sources only.
Score target: 6 → 9

Sources:
  - GDELT API v2 (no key required)
  - SEC EDGAR EFTS (full-text search)
  - EPA ECHO API (free)
  - OSHA enforcement data (DOL open data)
  - DOJ / FTC press release scraping
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import quote_plus

import requests
from requests.adapters import HTTPAdapter

try:
    import pandas as pd
    from pandas import Series as PdSeries, DataFrame as PdDataFrame
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from urllib3.util.retry import Retry
    HAS_URLLIB3_RETRY = True
except ImportError:
    HAS_URLLIB3_RETRY = False

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_GEO_API = "https://api.gdeltproject.org/api/v2/geo/geo"
EPA_ECHO_FACILITY_SEARCH = "https://echo.epa.gov/api/compliance/rest/facilities/search"
EPA_ECHO_VIOLATIONS = "https://echo.epa.gov/api/compliance/rest/facilities/{fac_id}/violations"
OSHA_DATA_PORTAL = "https://data.dol.gov/api/3/action/datastore_search"
EDGAR_EFTS_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_FULLTEXT = "https://efts.sec.gov/LATEST/search-index?q={query}&forms=8-K&dateRange=custom&startdt={start}&enddt={end}&entity={entity}"
DOJ_PRESSREL_API = "https://www.justice.gov/api/v1/press_releases.json"
FTC_PRESSREL_RSS = "https://www.ftc.gov/news-events/news/press-releases/rss"

REQUEST_TIMEOUT = 20
USER_AGENT = "SentinelFinance/3.0 (contact@sentinelfinance.io)"
GDELT_RATE_LIMIT_SECS = 0.5  # be polite
MAX_ARTICLES_PER_QUERY = 250

HALF_LIFE_DAYS = 90.0  # exponential decay half-life for recency weighting
RECIDIVISM_PENALTY = 1.3  # multiplier per repeat violation in same category


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class ContCategory(str, Enum):
    FRAUD = "FRAUD"
    REGULATORY = "REGULATORY"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    LABOR = "LABOR"
    DATA_PRIVACY = "DATA_PRIVACY"
    PRODUCT_SAFETY = "PRODUCT_SAFETY"
    GOVERNANCE = "GOVERNANCE"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    SOCIAL = "SOCIAL"
    FINANCIAL = "FINANCIAL"
    UNKNOWN = "UNKNOWN"


CATEGORY_SEVERITY: dict[ContCategory, int] = {
    ContCategory.FRAUD: 4,
    ContCategory.REGULATORY: 3,
    ContCategory.ENVIRONMENTAL: 3,
    ContCategory.LABOR: 2,
    ContCategory.DATA_PRIVACY: 3,
    ContCategory.PRODUCT_SAFETY: 3,
    ContCategory.GOVERNANCE: 2,
    ContCategory.SUPPLY_CHAIN: 2,
    ContCategory.SOCIAL: 1,
    ContCategory.FINANCIAL: 3,
    ContCategory.UNKNOWN: 1,
}


@dataclass
class ControversyArticle:
    url: str
    title: str
    seendate: datetime
    domain: str
    language: str = "English"
    sourcecountry: str = ""
    socialimage: str = ""
    tone: float = 0.0
    category: ContCategory = ContCategory.UNKNOWN
    severity: int = 1


@dataclass
class ControversyCategory:
    category: ContCategory
    severity: int
    keywords_matched: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ControversyScore:
    ticker: str
    total_score: float          # 0-100, higher = more controversial
    category_breakdown: dict[str, float] = field(default_factory=dict)
    article_count: int = 0
    weighted_article_count: float = 0.0
    avg_tone: float = 0.0
    risk_rating: str = "NEGLIGIBLE"  # NEGLIGIBLE / LOW / MODERATE / HIGH / SEVERE
    trend: str = "STABLE"            # IMPROVING / STABLE / DETERIORATING
    top_categories: list[str] = field(default_factory=list)
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EPAViolation:
    facility_name: str
    facility_id: str
    violation_type: str
    violation_date: Optional[datetime]
    penalty_amount: float = 0.0
    program: str = ""
    status: str = ""


@dataclass
class OSHAViolation:
    establishment: str
    inspection_id: str
    violation_type: str
    penalty: float = 0.0
    inspection_date: Optional[datetime] = None
    isic_code: str = ""
    gravity: str = ""


@dataclass
class ControversyEvent:
    ticker: str
    event_date: datetime
    headline: str
    category: ContCategory
    severity: int
    source: str                  # GDELT / SEC / EPA / OSHA
    resolved: bool = False
    resolution_date: Optional[datetime] = None
    url: str = ""


@dataclass
class ControversySpike:
    spike_date: datetime
    article_count: int
    z_score: float
    dominant_category: ContCategory
    sample_headlines: list[str] = field(default_factory=list)


@dataclass
class SECAction:
    ticker: str
    filing_type: str
    filing_date: datetime
    description: str
    action_type: str             # INVESTIGATION / CONSENT_DECREE / PENALTY / OTHER
    url: str = ""
    amount: float = 0.0


@dataclass
class ControversyDashboard:
    ticker: str
    score: ControversyScore
    recent_articles: list[ControversyArticle]
    timeline: list[ControversyEvent]
    spikes: list[ControversySpike]
    epa_violations: list[EPAViolation]
    osha_violations: list[OSHAViolation]
    sec_actions: list[SECAction]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ControversyAlert:
    ticker: str
    alert_type: str
    category: ContCategory
    severity: int
    description: str
    score: float
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# HTTP Session Helper
# ---------------------------------------------------------------------------

def _build_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    if HAS_URLLIB3_RETRY:
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    return session


_SESSION = _build_session()


def _get(url: str, params: Optional[dict] = None, timeout: int = REQUEST_TIMEOUT) -> Optional[dict]:
    try:
        resp = _SESSION.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            return resp.json()
        return {"_text": resp.text}
    except requests.exceptions.RequestException as exc:
        log.warning("HTTP GET failed: %s — %s", url, exc)
        return None


def _parse_gdelt_date(ds: str) -> Optional[datetime]:
    """Parse GDELT seendate like '20240115T120000Z'."""
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(ds, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Keyword → Category Mapping (200+ keywords)
# ---------------------------------------------------------------------------

KEYWORD_MAP: dict[ContCategory, list[str]] = {
    ContCategory.FRAUD: [
        "fraud", "accounting fraud", "securities fraud", "insider trading",
        "market manipulation", "ponzi", "embezzlement", "falsification",
        "fictitious", "misrepresentation", "deceptive", "misleading investors",
        "books cooked", "earnings manipulation", "channel stuffing", "round-tripping",
        "bribery", "kickback", "corruption", "illicit payment", "slush fund",
        "wire fraud", "mail fraud", "conspiracy to defraud", "criminal charges",
        "indicted", "criminal complaint", "arrested", "convicted", "plea deal",
        "whistleblower", "restatement", "audit failure", "material weakness",
    ],
    ContCategory.REGULATORY: [
        "sec investigation", "sec enforcement", "sec charges", "sec subpoena",
        "doj investigation", "department of justice", "consent decree", "cease and desist",
        "civil penalty", "regulatory fine", "enforcement action", "non-prosecution agreement",
        "deferred prosecution", "monitor appointed", "regulatory settlement",
        "antitrust", "monopoly investigation", "price fixing", "cartel",
        "competition authority", "ftc investigation", "ftc action",
        "occ order", "federal reserve order", "cfpb action", "finra sanction",
        "cftc charges", "bank examination", "supervisory action", "regulatory order",
    ],
    ContCategory.ENVIRONMENTAL: [
        "epa violation", "epa fine", "environmental violation", "pollution",
        "toxic spill", "oil spill", "chemical release", "hazardous waste",
        "superfund", "clean water act", "clean air act", "greenhouse gas",
        "climate lawsuit", "carbon emissions", "methane leak", "contamination",
        "environmental damage", "ecosystem damage", "wildlife harm", "deforestation",
        "illegal dumping", "environmental cleanup", "remediation order",
        "air quality violation", "water quality", "wastewater discharge",
        "environmental settlement", "environmental penalty", "greenwashing lawsuit",
    ],
    ContCategory.LABOR: [
        "nlrb complaint", "labor violation", "worker strike", "union dispute",
        "unsafe working conditions", "osha citation", "workplace injury",
        "wage theft", "overtime violation", "minimum wage violation",
        "discrimination lawsuit", "sexual harassment", "hostile work environment",
        "wrongful termination", "retaliation", "whistleblower retaliation",
        "child labor", "forced overtime", "labor trafficking", "worker exploitation",
        "employee lawsuit", "class action employees", "labor relations",
        "collective bargaining dispute", "lockout", "picket line",
    ],
    ContCategory.DATA_PRIVACY: [
        "data breach", "data hack", "cyberattack", "ransomware", "data leak",
        "gdpr fine", "gdpr violation", "privacy violation", "ftc privacy",
        "data theft", "personal data exposed", "social security numbers leaked",
        "credit card data stolen", "identity theft", "biometric data",
        "surveillance", "unauthorized access", "information security failure",
        "ico fine", "ccpa violation", "hipaa violation", "data protection",
        "cyber incident", "security incident", "intrusion detected",
    ],
    ContCategory.PRODUCT_SAFETY: [
        "product recall", "fda warning letter", "safety defect", "safety recall",
        "personal injury lawsuit", "class action product", "death defect",
        "dangerous product", "consumer safety", "nhtsa recall",
        "cpsc recall", "pharmaceutical recall", "drug recall", "medical device recall",
        "safety investigation", "injury claims", "product liability",
        "toxic product", "contaminated product", "mislabeled", "adulterated",
    ],
    ContCategory.GOVERNANCE: [
        "board dispute", "shareholder lawsuit", "derivative lawsuit",
        "executive misconduct", "ceo fired", "cfo resignation scandal",
        "board member ousted", "proxy fight", "activist investor",
        "executive pay scandal", "golden parachute controversy",
        "conflict of interest", "related party transaction",
        "audit committee failure", "internal controls failure",
        "poison pill", "entrenchment", "corporate governance failure",
    ],
    ContCategory.SUPPLY_CHAIN: [
        "forced labor", "slave labor", "child labor supply chain",
        "conflict minerals", "cobalt mining", "supplier violation",
        "human trafficking", "uighur forced labor", "xinjiang",
        "supply chain audit failure", "supplier scandal", "sweatshop",
        "modern slavery", "uyghur forced labor act", "uflpa",
        "conflict-free", "responsible sourcing failure", "supply chain risk",
    ],
    ContCategory.SOCIAL: [
        "boycott", "protest", "brand crisis", "reputational damage",
        "negative press", "social media backlash", "public outrage",
        "controversy", "scandal", "criticism", "consumer anger",
        "misinformation", "disinformation campaign", "hate speech platform",
        "civil rights", "racial bias", "gender bias", "ageism",
        "ableism", "community opposition", "nimby", "social license",
    ],
    ContCategory.FINANCIAL: [
        "accounting restatement", "going concern", "debt default",
        "bond default", "credit downgrade", "rating downgrade",
        "bankruptcy filing", "chapter 11", "chapter 7", "liquidation",
        "covenant breach", "debt restructuring", "financial distress",
        "liquidity crisis", "solvency concerns", "auditor resignation",
        "accounting irregularities", "financial misstatement",
        "earnings manipulation", "revenue recognition problem",
    ],
}


def _build_keyword_index() -> dict[str, ContCategory]:
    """Build a flat keyword → category lookup for O(1) classification."""
    index: dict[str, ContCategory] = {}
    for cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            index[kw.lower()] = cat
    return index


_KEYWORD_INDEX = _build_keyword_index()


# ---------------------------------------------------------------------------
# GDELT Controversy Fetcher
# ---------------------------------------------------------------------------

class GDELTControversyFetcher:
    """Fetches controversy data from GDELT API v2 (no API key required)."""

    def __init__(self, rate_limit_secs: float = GDELT_RATE_LIMIT_SECS):
        self._rate_limit = rate_limit_secs
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    def _build_query(self, company_name: str, topic: Optional[str] = None) -> str:
        q = f'"{company_name}" sourcelang:english'
        if topic:
            q += f" {topic}"
        return q

    def _parse_articles(self, data: dict, cutoff: datetime) -> list[ControversyArticle]:
        articles: list[ControversyArticle] = []
        items = data.get("articles", [])
        if not items:
            return articles
        for item in items:
            raw_date = item.get("seendate", "")
            dt = _parse_gdelt_date(raw_date)
            if dt is None or dt < cutoff:
                continue
            try:
                tone = float(item.get("tone", 0.0))
            except (TypeError, ValueError):
                tone = 0.0
            art = ControversyArticle(
                url=item.get("url", ""),
                title=item.get("title", ""),
                seendate=dt,
                domain=item.get("domain", ""),
                language=item.get("language", "English"),
                sourcecountry=item.get("sourcecountry", ""),
                socialimage=item.get("socialimage", ""),
                tone=tone,
            )
            articles.append(art)
        return articles

    def fetch_articles(self, company_name: str, days: int = 30) -> list[ControversyArticle]:
        """Fetch recent news articles mentioning the company from GDELT."""
        self._throttle()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        query = self._build_query(company_name)
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": str(MAX_ARTICLES_PER_QUERY),
            "format": "json",
            "sort": "DateDesc",
        }
        data = _get(GDELT_DOC_API, params=params)
        if not data:
            log.warning("GDELT: no data returned for '%s'", company_name)
            return []
        articles = self._parse_articles(data, cutoff)
        log.info("GDELT: fetched %d articles for '%s' (last %d days)", len(articles), company_name, days)
        return articles

    def fetch_timeline(self, company_name: str, days: int = 90) -> "PdSeries | dict":
        """Fetch GDELT article volume timeline (TimelineVol mode)."""
        self._throttle()
        query = self._build_query(company_name)
        params = {
            "query": query,
            "mode": "TimelineVol",
            "format": "json",
            "smoothing": "3",
        }
        data = _get(GDELT_DOC_API, params=params)
        if not data:
            return {} if not HAS_PANDAS else pd.Series(dtype=float)

        timeline_data: dict[datetime, float] = {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        for series in data.get("timeline", []):
            for point in series.get("data", []):
                raw_date = point.get("date", "")
                dt = _parse_gdelt_date(raw_date)
                if dt is None or dt < cutoff:
                    continue
                val = float(point.get("value", 0.0))
                timeline_data[dt] = val

        if not timeline_data:
            return {} if not HAS_PANDAS else pd.Series(dtype=float)

        if HAS_PANDAS:
            s = pd.Series(timeline_data).sort_index()
            max_val = s.max()
            if max_val > 0:
                s = (s / max_val) * 100.0
            return s
        else:
            max_val = max(timeline_data.values()) or 1.0
            return {dt: (v / max_val) * 100.0 for dt, v in sorted(timeline_data.items())}

    def fetch_by_topic(self, company_name: str, topic: str) -> list[ControversyArticle]:
        """Fetch articles combining company name with a specific controversy topic keyword."""
        self._throttle()
        topic_map = {
            "lawsuit": "lawsuit OR litigation OR sued",
            "fraud": "fraud OR fraudulent OR embezzlement",
            "scandal": "scandal OR controversy OR misconduct",
            "environmental": "environmental OR pollution OR EPA OR spill",
            "labor": "strike OR union OR OSHA OR \"workplace injury\"",
            "data breach": "\"data breach\" OR cyberattack OR hack OR ransomware",
            "fine": "fine OR penalty OR settlement OR sanction",
            "regulatory": "\"SEC investigation\" OR \"DOJ investigation\" OR \"regulatory action\"",
        }
        topic_query = topic_map.get(topic.lower(), topic)
        query = f'"{company_name}" sourcelang:english ({topic_query})'
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": str(MAX_ARTICLES_PER_QUERY),
            "format": "json",
            "sort": "DateDesc",
        }
        data = _get(GDELT_DOC_API, params=params)
        if not data:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        return self._parse_articles(data, cutoff)

    def fetch_gdelt_tone(self, company_name: str, days: int = 30) -> float:
        """Compute average GDELT tone for company articles (negative = bad sentiment)."""
        articles = self.fetch_articles(company_name, days=days)
        if not articles:
            return 0.0
        return sum(a.tone for a in articles) / len(articles)

    def fetch_geo_mentions(self, company_name: str) -> dict[str, int]:
        """Fetch geographic distribution of mentions."""
        self._throttle()
        params = {
            "query": f'"{company_name}"',
            "format": "json",
        }
        data = _get(GDELT_GEO_API, params=params)
        if not data:
            return {}
        result: dict[str, int] = {}
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            country = props.get("countrycode", "")
            count = int(props.get("count", 0))
            if country:
                result[country] = result.get(country, 0) + count
        return result


# ---------------------------------------------------------------------------
# Controversy Classifier
# ---------------------------------------------------------------------------

class ControversyClassifier:
    """Classify news articles into controversy categories using keyword matching."""

    def __init__(self):
        self._index = _KEYWORD_INDEX
        self._category_keywords = KEYWORD_MAP

    def _extract_text(self, article: ControversyArticle) -> str:
        return (article.title + " " + article.url).lower()

    def classify(self, article: ControversyArticle) -> ControversyCategory:
        text = self._extract_text(article)
        category_hits: dict[ContCategory, list[str]] = {}

        for kw, cat in self._index.items():
            if kw in text:
                category_hits.setdefault(cat, []).append(kw)

        if not category_hits:
            return ControversyCategory(
                category=ContCategory.UNKNOWN,
                severity=1,
                keywords_matched=[],
                confidence=0.0,
            )

        # Pick the category with highest severity * hit count
        def _priority(item: tuple[ContCategory, list[str]]) -> float:
            cat, hits = item
            return CATEGORY_SEVERITY[cat] * len(hits)

        best_cat, best_hits = max(category_hits.items(), key=_priority)
        confidence = min(1.0, len(best_hits) / 3.0)

        return ControversyCategory(
            category=best_cat,
            severity=CATEGORY_SEVERITY[best_cat],
            keywords_matched=best_hits[:10],
            confidence=confidence,
        )

    def classify_bulk(self, articles: list[ControversyArticle]) -> list[ControversyArticle]:
        """Classify articles in-place and return them."""
        for art in articles:
            cat_result = self.classify(art)
            art.category = cat_result.category
            art.severity = cat_result.severity
        return articles

    def compute_severity(self, articles: list[ControversyArticle]) -> float:
        """Compute an aggregate severity score 0-100 from a list of classified articles."""
        if not articles:
            return 0.0

        now = datetime.now(timezone.utc)
        total_weight = 0.0
        weighted_severity = 0.0
        decay_lambda = math.log(2) / HALF_LIFE_DAYS

        for art in articles:
            age_days = max(0.0, (now - art.seendate).total_seconds() / 86400.0)
            decay = math.exp(-decay_lambda * age_days)
            weight = decay * art.severity
            weighted_severity += weight
            total_weight += decay

        if total_weight == 0:
            return 0.0

        avg_weighted = weighted_severity / total_weight
        raw_score = avg_weighted * len(articles) / 4.0  # normalise by max severity=4
        return min(100.0, raw_score * 10.0)

    def get_category_distribution(self, articles: list[ControversyArticle]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for art in articles:
            key = art.category.value
            dist[key] = dist.get(key, 0) + 1
        return dist


# ---------------------------------------------------------------------------
# EPA ECHO Integration
# ---------------------------------------------------------------------------

class EPAECHOAdapter:
    """Fetch EPA ECHO facility violations for a company."""

    def search_facilities(self, company_name: str) -> list[dict]:
        params = {
            "p_name": company_name,
            "output": "JSON",
            "p_active": "Y",
        }
        data = _get(EPA_ECHO_FACILITY_SEARCH, params=params)
        if not data:
            return []
        results = data.get("Results", {}) or {}
        facilities = results.get("Facilities", [])
        if isinstance(facilities, list):
            return facilities
        return []

    def fetch_violations(self, facility_id: str) -> list[dict]:
        url = EPA_ECHO_VIOLATIONS.format(fac_id=facility_id)
        data = _get(url)
        if not data:
            return []
        return data.get("Results", {}).get("Violations", []) or []

    def get_company_violations(self, company_name: str) -> list[EPAViolation]:
        """Aggregate violations across all matching EPA facilities."""
        facilities = self.search_facilities(company_name)
        violations: list[EPAViolation] = []
        for fac in facilities[:5]:  # limit to top 5 facilities to avoid rate limiting
            fac_id = fac.get("RegistryID", "")
            fac_name = fac.get("FacilityName", "")
            if not fac_id:
                continue
            raw_viols = self.fetch_violations(fac_id)
            for v in raw_viols[:20]:
                vdate_raw = v.get("ViolationDate", "")
                vdate = None
                for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
                    try:
                        vdate = datetime.strptime(vdate_raw, fmt).replace(tzinfo=timezone.utc)
                        break
                    except (ValueError, TypeError):
                        continue
                try:
                    penalty = float(v.get("penaltyamount", 0) or 0)
                except (ValueError, TypeError):
                    penalty = 0.0
                violations.append(EPAViolation(
                    facility_name=fac_name,
                    facility_id=fac_id,
                    violation_type=v.get("violationtype", ""),
                    violation_date=vdate,
                    penalty_amount=penalty,
                    program=v.get("programid", ""),
                    status=v.get("violationstatus", ""),
                ))
            time.sleep(0.3)  # be polite
        return violations


# ---------------------------------------------------------------------------
# OSHA Integration
# ---------------------------------------------------------------------------

class OSHAAdapter:
    """Fetch OSHA enforcement data from DOL open data portal."""

    OSHA_RESOURCE_ID = "ffd45ddd-4f33-489d-a58e-fa4e14c5c89b"  # OSHA inspections

    def fetch_violations(self, company_name: str, limit: int = 50) -> list[OSHAViolation]:
        params = {
            "resource_id": self.OSHA_RESOURCE_ID,
            "q": company_name,
            "limit": str(limit),
        }
        data = _get(OSHA_DATA_PORTAL, params=params)
        if not data:
            return []
        records = data.get("result", {}).get("records", [])
        violations: list[OSHAViolation] = []
        for rec in records:
            insp_date_raw = rec.get("open_date", "")
            insp_date = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                try:
                    insp_date = datetime.strptime(insp_date_raw[:10], fmt).replace(tzinfo=timezone.utc)
                    break
                except (ValueError, TypeError):
                    continue
            try:
                penalty = float(rec.get("total_current_penalty", 0) or 0)
            except (ValueError, TypeError):
                penalty = 0.0
            violations.append(OSHAViolation(
                establishment=rec.get("establishment_name", ""),
                inspection_id=str(rec.get("activity_nr", "")),
                violation_type=rec.get("violation_type", ""),
                penalty=penalty,
                inspection_date=insp_date,
                isic_code=str(rec.get("sic_code", "")),
                gravity=str(rec.get("gravity", "")),
            ))
        return violations


# ---------------------------------------------------------------------------
# SEC EDGAR Regulatory Monitor
# ---------------------------------------------------------------------------

class RegulatoryMonitor:
    """Monitor regulatory actions via SEC EDGAR EFTS and government press releases."""

    SEC_ACTION_KEYWORDS = [
        "SEC investigation", "consent decree", "civil penalty", "enforcement action",
        "cease and desist", "deferred prosecution", "non-prosecution agreement",
        "permanent injunction", "disgorgement", "securities fraud", "insider trading",
        "material weakness", "going concern", "restatement",
    ]

    DOJ_ACTION_KEYWORDS = [
        "indicted", "criminal charges", "plea agreement", "conviction", "guilty plea",
        "deferred prosecution", "corporate monitor", "antitrust", "wire fraud",
    ]

    def _build_efts_url(self, ticker: str, query: str, days: int = 365) -> str:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        encoded_q = quote_plus(f'"{query}"')
        return (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q={encoded_q}&forms=8-K&dateRange=custom"
            f"&startdt={start_str}&enddt={end_str}&entity={quote_plus(ticker)}"
        )

    def _classify_sec_action(self, text: str) -> str:
        text_lower = text.lower()
        if any(k in text_lower for k in ["investigation", "subpoena", "inquiry"]):
            return "INVESTIGATION"
        if any(k in text_lower for k in ["consent decree", "cease and desist", "injunction"]):
            return "CONSENT_DECREE"
        if any(k in text_lower for k in ["penalty", "fine", "disgorgement", "settlement"]):
            return "PENALTY"
        return "OTHER"

    def fetch_sec_enforcement_actions(self, ticker: str, days: int = 365) -> list[SECAction]:
        """Search EDGAR EFTS for 8-K filings with enforcement-related language."""
        actions: list[SECAction] = []
        for keyword in self.SEC_ACTION_KEYWORDS[:6]:  # avoid rate-limiting, top keywords only
            url = self._build_efts_url(ticker, keyword, days)
            data = _get(url)
            if not data:
                continue
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits[:5]:
                src = hit.get("_source", {})
                raw_date = src.get("file_date", "")
                filing_date = None
                for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        filing_date = datetime.strptime(raw_date[:10], fmt).replace(tzinfo=timezone.utc)
                        break
                    except (ValueError, TypeError):
                        continue
                if filing_date is None:
                    continue
                description = src.get("period_of_report", "") or src.get("entity_name", "")
                accession = src.get("file_num", "") or hit.get("_id", "")
                actions.append(SECAction(
                    ticker=ticker,
                    filing_type=src.get("form_type", "8-K"),
                    filing_date=filing_date,
                    description=f"{keyword}: {description}",
                    action_type=self._classify_sec_action(keyword),
                    url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K",
                ))
            time.sleep(0.3)
        # deduplicate by (date, action_type)
        seen: set[str] = set()
        unique: list[SECAction] = []
        for a in actions:
            key = f"{a.filing_date.date()}_{a.action_type}" if a.filing_date else a.action_type
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    def fetch_doj_actions(self, company: str) -> list[dict]:
        """Fetch DOJ press releases mentioning the company."""
        params = {
            "search": company,
            "sort_by": "date",
            "sort_order": "DESC",
            "items_per_page": "20",
            "page": "0",
        }
        data = _get(DOJ_PRESSREL_API, params=params)
        if not data:
            return []
        results = []
        for item in data.get("results", [])[:10]:
            results.append({
                "title": item.get("title", ""),
                "date": item.get("date", ""),
                "url": item.get("url", ""),
                "body_summary": (item.get("body", "") or "")[:500],
            })
        return results

    def fetch_ftc_actions(self, company: str) -> list[dict]:
        """Fetch FTC press releases via RSS feed."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            return []
        resp = None
        try:
            resp = _SESSION.get(FTC_PRESSREL_RSS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("FTC RSS fetch failed: %s", exc)
            return []
        results = []
        company_lower = company.lower()
        try:
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:50]:
                title = (item.findtext("title") or "").lower()
                desc = (item.findtext("description") or "").lower()
                if company_lower in title or company_lower in desc:
                    results.append({
                        "title": item.findtext("title", ""),
                        "link": item.findtext("link", ""),
                        "pubDate": item.findtext("pubDate", ""),
                        "description": item.findtext("description", "")[:300],
                    })
        except ET.ParseError as exc:
            log.warning("FTC RSS parse error: %s", exc)
        return results[:10]

    def compute_regulatory_risk_score(self, ticker: str) -> float:
        """Compute a regulatory risk score 0-100 from SEC + DOJ actions."""
        sec_actions = self.fetch_sec_enforcement_actions(ticker)
        score = 0.0
        severity_map = {
            "INVESTIGATION": 15.0,
            "CONSENT_DECREE": 25.0,
            "PENALTY": 20.0,
            "OTHER": 5.0,
        }
        now = datetime.now(timezone.utc)
        for action in sec_actions:
            base = severity_map.get(action.action_type, 5.0)
            if action.filing_date:
                age_days = max(0.0, (now - action.filing_date).total_seconds() / 86400.0)
                decay = math.exp(-math.log(2) / HALF_LIFE_DAYS * age_days)
                score += base * decay
            else:
                score += base * 0.5
        return min(100.0, score)


# ---------------------------------------------------------------------------
# ESG Controversy Scorer
# ---------------------------------------------------------------------------

class ESGControversyScorer:
    """Compute comprehensive ESG controversy scores."""

    def __init__(self):
        self._gdelt = GDELTControversyFetcher()
        self._classifier = ControversyClassifier()
        self._epa = EPAECHOAdapter()

    def _compute_recency_weight(self, article: ControversyArticle) -> float:
        now = datetime.now(timezone.utc)
        age_days = max(0.0, (now - article.seendate).total_seconds() / 86400.0)
        return math.exp(-math.log(2) / HALF_LIFE_DAYS * age_days)

    def _compute_recidivism_penalty(self, articles: list[ControversyArticle]) -> float:
        """Extra penalty for companies with repeat violations in the same category."""
        cat_counts: dict[ContCategory, int] = {}
        for art in articles:
            if art.category not in (ContCategory.UNKNOWN, ContCategory.SOCIAL):
                cat_counts[art.category] = cat_counts.get(art.category, 0) + 1
        penalty = 1.0
        for cat, count in cat_counts.items():
            if count >= 3:
                penalty *= RECIDIVISM_PENALTY ** (count // 3)
        return min(penalty, 3.0)  # cap at 3x

    def compute_controversy_score(self, ticker: str, company_name: str, days: int = 365) -> ControversyScore:
        """Full controversy score computation pipeline."""
        articles = self._gdelt.fetch_articles(company_name, days=days)
        articles = self._classifier.classify_bulk(articles)

        if not articles:
            return ControversyScore(
                ticker=ticker,
                total_score=0.0,
                article_count=0,
                risk_rating="NEGLIGIBLE",
                trend="STABLE",
            )

        now = datetime.now(timezone.utc)
        weighted_sum = 0.0
        total_decay = 0.0
        tone_sum = 0.0
        category_scores: dict[str, float] = {}

        for art in articles:
            w = self._compute_recency_weight(art)
            severity_w = w * art.severity
            weighted_sum += severity_w
            total_decay += w
            tone_sum += art.tone
            cat_key = art.category.value
            category_scores[cat_key] = category_scores.get(cat_key, 0.0) + severity_w

        avg_tone = tone_sum / len(articles)
        recidivism_mult = self._compute_recidivism_penalty(articles)

        raw_score = (weighted_sum / max(total_decay, 1e-9)) * (len(articles) ** 0.5) * recidivism_mult
        total_score = min(100.0, raw_score * 5.0)

        # Normalize category scores 0-100
        max_cat_score = max(category_scores.values()) if category_scores else 1.0
        norm_categories = {k: min(100.0, v / max(max_cat_score, 1e-9) * 100.0) for k, v in category_scores.items()}

        risk_rating = self.compute_esg_risk_rating_from_score(total_score)
        trend = self.compute_controversy_trend(ticker, company_name)
        top_cats = sorted(category_scores.keys(), key=lambda k: category_scores[k], reverse=True)[:3]

        return ControversyScore(
            ticker=ticker,
            total_score=total_score,
            category_breakdown=norm_categories,
            article_count=len(articles),
            weighted_article_count=weighted_sum,
            avg_tone=avg_tone,
            risk_rating=risk_rating,
            trend=trend,
            top_categories=top_cats,
        )

    def compute_esg_risk_rating_from_score(self, score: float) -> str:
        if score >= 80:
            return "SEVERE"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MODERATE"
        elif score >= 20:
            return "LOW"
        else:
            return "NEGLIGIBLE"

    def compute_esg_risk_rating(self, ticker: str, company_name: str) -> str:
        score = self.compute_controversy_score(ticker, company_name)
        return score.risk_rating

    def compute_controversy_trend(self, ticker: str, company_name: str) -> str:
        """Compare recent 90 days vs 90-180 days ago to determine trend."""
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=90)
        older_cutoff = now - timedelta(days=180)

        all_articles = self._gdelt.fetch_articles(company_name, days=180)
        all_articles = self._classifier.classify_bulk(all_articles)

        recent = [a for a in all_articles if a.seendate >= recent_cutoff]
        older = [a for a in all_articles if older_cutoff <= a.seendate < recent_cutoff]

        recent_severity = self._classifier.compute_severity(recent)
        older_severity = self._classifier.compute_severity(older)

        if older_severity < 1.0:
            return "STABLE"
        change_pct = (recent_severity - older_severity) / older_severity * 100.0
        if change_pct > 20.0:
            return "DETERIORATING"
        elif change_pct < -20.0:
            return "IMPROVING"
        return "STABLE"

    def fetch_epa_violations(self, ticker: str, company_name: str) -> list[EPAViolation]:
        return self._epa.get_company_violations(company_name)

    def fetch_osha_violations(self, company_name: str) -> list[OSHAViolation]:
        osha = OSHAAdapter()
        return osha.fetch_violations(company_name)


# ---------------------------------------------------------------------------
# Controversy Timeline
# ---------------------------------------------------------------------------

class ControversyTimeline:
    """Build and analyze controversy event timelines."""

    def __init__(self):
        self._gdelt = GDELTControversyFetcher()
        self._classifier = ControversyClassifier()
        self._epa = EPAECHOAdapter()
        self._osha = OSHAAdapter()
        self._regulatory = RegulatoryMonitor()

    def _articles_to_events(self, ticker: str, articles: list[ControversyArticle]) -> list[ControversyEvent]:
        events: list[ControversyEvent] = []
        for art in articles:
            if art.category in (ContCategory.UNKNOWN, ContCategory.SOCIAL):
                if art.severity < 2:
                    continue
            events.append(ControversyEvent(
                ticker=ticker,
                event_date=art.seendate,
                headline=art.title[:200],
                category=art.category,
                severity=art.severity,
                source="GDELT",
                url=art.url,
            ))
        return events

    def _epa_to_events(self, ticker: str, violations: list[EPAViolation]) -> list[ControversyEvent]:
        events: list[ControversyEvent] = []
        for v in violations:
            if v.violation_date is None:
                continue
            events.append(ControversyEvent(
                ticker=ticker,
                event_date=v.violation_date,
                headline=f"EPA Violation: {v.violation_type} at {v.facility_name}",
                category=ContCategory.ENVIRONMENTAL,
                severity=3,
                source="EPA",
            ))
        return events

    def _osha_to_events(self, ticker: str, violations: list[OSHAViolation]) -> list[ControversyEvent]:
        events: list[ControversyEvent] = []
        for v in violations:
            if v.inspection_date is None:
                continue
            events.append(ControversyEvent(
                ticker=ticker,
                event_date=v.inspection_date,
                headline=f"OSHA Violation: {v.violation_type} at {v.establishment} (${v.penalty:,.0f})",
                category=ContCategory.LABOR,
                severity=2,
                source="OSHA",
            ))
        return events

    def _sec_to_events(self, ticker: str, actions: list[SECAction]) -> list[ControversyEvent]:
        events: list[ControversyEvent] = []
        for a in actions:
            if a.filing_date is None:
                continue
            events.append(ControversyEvent(
                ticker=ticker,
                event_date=a.filing_date,
                headline=f"SEC {a.action_type}: {a.description[:150]}",
                category=ContCategory.REGULATORY,
                severity=3,
                source="SEC",
                url=a.url,
            ))
        return events

    def build_timeline(self, ticker: str, company_name: str, years: int = 5) -> list[ControversyEvent]:
        """Build a full controversy timeline from all available sources."""
        days = years * 365
        articles = self._gdelt.fetch_articles(company_name, days=days)
        articles = self._classifier.classify_bulk(articles)

        events = self._articles_to_events(ticker, articles)

        try:
            epa_viols = self._epa.get_company_violations(company_name)
            events.extend(self._epa_to_events(ticker, epa_viols))
        except Exception as exc:
            log.warning("EPA timeline fetch failed: %s", exc)

        try:
            osha_viols = self._osha.fetch_violations(company_name)
            events.extend(self._osha_to_events(ticker, osha_viols))
        except Exception as exc:
            log.warning("OSHA timeline fetch failed: %s", exc)

        try:
            sec_actions = self._regulatory.fetch_sec_enforcement_actions(ticker, days=days)
            events.extend(self._sec_to_events(ticker, sec_actions))
        except Exception as exc:
            log.warning("SEC timeline fetch failed: %s", exc)

        events.sort(key=lambda e: e.event_date, reverse=True)
        return events

    def detect_controversy_spikes(self, company_name: str, days: int = 365) -> list[ControversySpike]:
        """Detect days with statistically significant article volume spikes."""
        articles = self._gdelt.fetch_articles(company_name, days=days)
        if not articles:
            return []

        # Bin by day
        day_counts: dict[str, list[ControversyArticle]] = {}
        for art in articles:
            day_key = art.seendate.strftime("%Y-%m-%d")
            day_counts.setdefault(day_key, []).append(art)

        counts = [len(v) for v in day_counts.values()]
        if len(counts) < 7:
            return []

        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        std_count = math.sqrt(variance) if variance > 0 else 0.0
        threshold = mean_count + 2.0 * std_count

        spikes: list[ControversySpike] = []
        for day_key, day_arts in day_counts.items():
            count = len(day_arts)
            if count >= threshold and std_count > 0:
                z = (count - mean_count) / std_count
                classified = self._classifier.classify_bulk(day_arts)
                cat_dist: dict[ContCategory, int] = {}
                for art in classified:
                    cat_dist[art.category] = cat_dist.get(art.category, 0) + 1
                dominant_cat = max(cat_dist, key=lambda k: cat_dist[k])
                headlines = [a.title[:100] for a in classified[:3]]
                spikes.append(ControversySpike(
                    spike_date=datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    article_count=count,
                    z_score=round(z, 2),
                    dominant_category=dominant_cat,
                    sample_headlines=headlines,
                ))
        spikes.sort(key=lambda s: s.z_score, reverse=True)
        return spikes

    def compute_resolution_rate(self, events: list[ControversyEvent]) -> float:
        """Estimate what fraction of events have been resolved."""
        if not events:
            return 0.0
        resolved = sum(1 for e in events if e.resolved)
        return resolved / len(events)

    def compute_recidivism(self, ticker: str, company_name: str, category: str) -> int:
        """Count how many times a company has had violations in the same category."""
        articles = self._gdelt.fetch_articles(company_name, days=365 * 5)
        articles = self._classifier.classify_bulk(articles)
        target_cat = ContCategory(category) if category in ContCategory.__members__.values() else ContCategory.UNKNOWN
        return sum(1 for a in articles if a.category == target_cat)


# ---------------------------------------------------------------------------
# Controversy Screener
# ---------------------------------------------------------------------------

class ControversyScreener:
    """Screen a universe of tickers for controversy risk."""

    PRESETS: dict[str, dict] = {
        "esg_risk": {"threshold": 60.0},
        "regulatory_watch": {"category": "REGULATORY"},
        "environmental": {"category": "ENVIRONMENTAL"},
        "governance": {"category": "GOVERNANCE"},
        "clean": {"max_threshold": 20.0},
    }

    def __init__(self, company_name_map: Optional[dict[str, str]] = None):
        """
        Args:
            company_name_map: ticker → company name. If not provided, ticker is used as company name.
        """
        self._scorer = ESGControversyScorer()
        self._name_map = company_name_map or {}

    def _get_name(self, ticker: str) -> str:
        return self._name_map.get(ticker, ticker)

    def _score_universe(self, universe: list[str], days: int = 180) -> list[ControversyScore]:
        scores: list[ControversyScore] = []
        for ticker in universe:
            company = self._get_name(ticker)
            try:
                score = self._scorer.compute_controversy_score(ticker, company, days=days)
                scores.append(score)
            except Exception as exc:
                log.warning("Scoring failed for %s: %s", ticker, exc)
        return scores

    def screen_high_risk(self, universe: list[str], threshold: float = 60.0) -> "PdDataFrame | list[dict]":
        scores = self._score_universe(universe)
        filtered = [s for s in scores if s.total_score >= threshold]
        filtered.sort(key=lambda s: s.total_score, reverse=True)
        rows = [
            {
                "ticker": s.ticker,
                "score": round(s.total_score, 1),
                "risk_rating": s.risk_rating,
                "trend": s.trend,
                "top_categories": ", ".join(s.top_categories),
                "article_count": s.article_count,
            }
            for s in filtered
        ]
        if HAS_PANDAS:
            return pd.DataFrame(rows)
        return rows

    def screen_improving(self, universe: list[str]) -> "PdDataFrame | list[dict]":
        scores = self._score_universe(universe)
        improving = [s for s in scores if s.trend == "IMPROVING"]
        improving.sort(key=lambda s: s.total_score, reverse=True)
        rows = [
            {
                "ticker": s.ticker,
                "score": round(s.total_score, 1),
                "trend": s.trend,
                "risk_rating": s.risk_rating,
            }
            for s in improving
        ]
        if HAS_PANDAS:
            return pd.DataFrame(rows)
        return rows

    def screen_by_category(self, universe: list[str], category: str) -> "PdDataFrame | list[dict]":
        scores = self._score_universe(universe)
        filtered = [s for s in scores if category in (s.category_breakdown or {})]
        filtered.sort(key=lambda s: s.category_breakdown.get(category, 0.0), reverse=True)
        rows = [
            {
                "ticker": s.ticker,
                "category": category,
                "category_score": round(s.category_breakdown.get(category, 0.0), 1),
                "total_score": round(s.total_score, 1),
                "risk_rating": s.risk_rating,
            }
            for s in filtered
        ]
        if HAS_PANDAS:
            return pd.DataFrame(rows)
        return rows

    def get_sector_controversy_rankings(self, sector: str) -> "PdDataFrame | list[dict]":
        """Placeholder — in production, filter universe by GICS sector."""
        log.info("Sector ranking for '%s' requires a sector → ticker universe mapping.", sector)
        return pd.DataFrame() if HAS_PANDAS else []

    def screen_by_preset(self, universe: list[str], preset: str) -> "PdDataFrame | list[dict]":
        cfg = self.PRESETS.get(preset)
        if cfg is None:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(self.PRESETS)}")
        if "category" in cfg:
            return self.screen_by_category(universe, cfg["category"])
        threshold = cfg.get("threshold", 60.0)
        if "max_threshold" in cfg:
            scores = self._score_universe(universe)
            clean = [s for s in scores if s.total_score <= cfg["max_threshold"]]
            rows = [{"ticker": s.ticker, "score": round(s.total_score, 1)} for s in clean]
            return pd.DataFrame(rows) if HAS_PANDAS else rows
        return self.screen_high_risk(universe, threshold=threshold)


# ---------------------------------------------------------------------------
# Controversy Monitor Engine (Orchestrator)
# ---------------------------------------------------------------------------

class ControversyMonitorEngine:
    """Top-level orchestrator for controversy monitoring."""

    def __init__(self, company_name_map: Optional[dict[str, str]] = None):
        self._name_map = company_name_map or {}
        self._gdelt = GDELTControversyFetcher()
        self._classifier = ControversyClassifier()
        self._scorer = ESGControversyScorer()
        self._timeline = ControversyTimeline()
        self._regulatory = RegulatoryMonitor()
        self._screener = ControversyScreener(company_name_map)
        self._epa = EPAECHOAdapter()
        self._osha = OSHAAdapter()

    def _get_name(self, ticker: str) -> str:
        return self._name_map.get(ticker, ticker)

    def get_controversy_dashboard(self, ticker: str) -> ControversyDashboard:
        """Full controversy dashboard for a single ticker."""
        company = self._get_name(ticker)
        log.info("Building controversy dashboard for %s (%s)…", ticker, company)

        articles = self._gdelt.fetch_articles(company, days=90)
        articles = self._classifier.classify_bulk(articles)

        score = self._scorer.compute_controversy_score(ticker, company, days=365)

        timeline = self._timeline.build_timeline(ticker, company, years=3)
        spikes = self._timeline.detect_controversy_spikes(company, days=365)

        epa_viols: list[EPAViolation] = []
        osha_viols: list[OSHAViolation] = []
        sec_actions: list[SECAction] = []

        try:
            epa_viols = self._epa.get_company_violations(company)
        except Exception as exc:
            log.warning("EPA fetch failed: %s", exc)

        try:
            osha_viols = self._osha.fetch_violations(company)
        except Exception as exc:
            log.warning("OSHA fetch failed: %s", exc)

        try:
            sec_actions = self._regulatory.fetch_sec_enforcement_actions(ticker)
        except Exception as exc:
            log.warning("SEC fetch failed: %s", exc)

        return ControversyDashboard(
            ticker=ticker,
            score=score,
            recent_articles=articles[:20],
            timeline=timeline[:50],
            spikes=spikes[:10],
            epa_violations=epa_viols,
            osha_violations=osha_viols,
            sec_actions=sec_actions,
        )

    def run_daily_scan(self, universe: list[str]) -> list[ControversyAlert]:
        """Scan universe for new controversy signals."""
        alerts: list[ControversyAlert] = []
        for ticker in universe:
            company = self._get_name(ticker)
            try:
                articles = self._gdelt.fetch_articles(company, days=2)
                articles = self._classifier.classify_bulk(articles)
                for art in articles:
                    if art.severity >= 3:
                        alerts.append(ControversyAlert(
                            ticker=ticker,
                            alert_type="HIGH_SEVERITY_ARTICLE",
                            category=art.category,
                            severity=art.severity,
                            description=art.title[:200],
                            score=float(art.severity * 25),
                        ))
                    elif art.severity == 4:
                        alerts.append(ControversyAlert(
                            ticker=ticker,
                            alert_type="CRITICAL_CONTROVERSY",
                            category=art.category,
                            severity=art.severity,
                            description=art.title[:200],
                            score=100.0,
                        ))
            except Exception as exc:
                log.warning("Daily scan failed for %s: %s", ticker, exc)
        alerts.sort(key=lambda a: a.severity, reverse=True)
        return alerts

    def get_market_controversy_pulse(self, sp500_sample: Optional[list[str]] = None) -> dict:
        """Aggregate controversy level for a representative sample of S&P 500."""
        universe = sp500_sample or [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "XOM",
            "JPM", "JNJ", "PFE", "BAC", "WMT", "CVX", "MCD",
        ]
        name_map = {
            "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Google",
            "META": "Meta Platforms", "TSLA": "Tesla", "NVDA": "NVIDIA", "XOM": "Exxon Mobil",
            "JPM": "JPMorgan Chase", "JNJ": "Johnson Johnson", "PFE": "Pfizer",
            "BAC": "Bank of America", "WMT": "Walmart", "CVX": "Chevron", "MCD": "McDonald's",
        }
        scores: list[float] = []
        risk_counts: dict[str, int] = {"NEGLIGIBLE": 0, "LOW": 0, "MODERATE": 0, "HIGH": 0, "SEVERE": 0}
        for ticker in universe:
            company = name_map.get(ticker, ticker)
            try:
                s = self._scorer.compute_controversy_score(ticker, company, days=30)
                scores.append(s.total_score)
                risk_counts[s.risk_rating] = risk_counts.get(s.risk_rating, 0) + 1
            except Exception as exc:
                log.warning("Market pulse: failed for %s: %s", ticker, exc)

        if not scores:
            return {"error": "No scores computed"}

        avg_score = sum(scores) / len(scores)
        pulse_label = "ELEVATED" if avg_score > 40 else "NORMAL" if avg_score > 20 else "CALM"
        return {
            "avg_controversy_score": round(avg_score, 1),
            "market_pulse": pulse_label,
            "risk_distribution": risk_counts,
            "tickers_scanned": len(scores),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def export_report(self, ticker: str, path: str) -> None:
        """Export a full controversy report to JSON."""
        dashboard = self.get_controversy_dashboard(ticker)

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, "__dict__"):
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            return obj

        report = _serialize(dashboard)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        log.info("Controversy report for %s written to %s", ticker, path)

    def get_controversy_summary_table(self, tickers: list[str]) -> "PdDataFrame | list[dict]":
        """Quick summary table for a list of tickers."""
        rows = []
        for ticker in tickers:
            company = self._get_name(ticker)
            try:
                score = self._scorer.compute_controversy_score(ticker, company, days=365)
                rows.append({
                    "ticker": ticker,
                    "company": company,
                    "score": round(score.total_score, 1),
                    "risk_rating": score.risk_rating,
                    "trend": score.trend,
                    "article_count": score.article_count,
                    "avg_tone": round(score.avg_tone, 2),
                    "top_categories": " | ".join(score.top_categories[:2]),
                })
            except Exception as exc:
                log.warning("Summary table: failed for %s: %s", ticker, exc)
        if HAS_PANDAS:
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("score", ascending=False).reset_index(drop=True)
            return df
        return sorted(rows, key=lambda r: r["score"], reverse=True)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    COMPANY_MAP = {
        "AAPL": "Apple",
        "META": "Meta Platforms",
        "XOM": "Exxon Mobil",
    }

    engine = ControversyMonitorEngine(company_name_map=COMPANY_MAP)
    tickers = list(COMPANY_MAP.keys())

    print("\n" + "=" * 70)
    print("  SENTINEL — Controversy Monitor v3  |  dim_104")
    print("=" * 70)

    print("\n[1] Controversy Summary Table")
    table = engine.get_controversy_summary_table(tickers)
    if HAS_PANDAS and hasattr(table, "to_string"):
        print(table.to_string(index=False))
    else:
        for row in table:
            print(row)

    print("\n[2] AAPL — Controversy Dashboard")
    dashboard = engine.get_controversy_dashboard("AAPL")
    score = dashboard.score
    print(f"  Score       : {score.total_score:.1f}/100")
    print(f"  Risk Rating : {score.risk_rating}")
    print(f"  Trend       : {score.trend}")
    print(f"  Articles    : {score.article_count}")
    print(f"  Avg Tone    : {score.avg_tone:.2f}")
    print(f"  Top Cats    : {', '.join(score.top_categories)}")

    print("\n[3] XOM — EPA Violations (sample)")
    epa_viols = dashboard.epa_violations[:3]
    for v in epa_viols:
        print(f"  {v.facility_name}: {v.violation_type} (${v.penalty_amount:,.0f})")
    if not epa_viols:
        print("  No EPA violations returned (may be rate-limited or no match)")

    print("\n[4] Controversy Spikes (AAPL)")
    for spike in dashboard.spikes[:3]:
        print(f"  {spike.spike_date.date()} | z={spike.z_score:.2f} | {spike.dominant_category.value} | n={spike.article_count}")

    print("\n[5] Daily Scan Alerts")
    alerts = engine.run_daily_scan(tickers)
    for alert in alerts[:5]:
        print(f"  [{alert.ticker}] {alert.alert_type} | {alert.category.value} | sev={alert.severity}")
    if not alerts:
        print("  No alerts triggered in last 48h")

    print("\nDone.")


# ---------------------------------------------------------------------------
# dim_104 wave-9 additions: severity index, reputation cascade, market impact
# ---------------------------------------------------------------------------


def compute_controversy_severity_index(
    regulatory_score: float,
    legal_score: float,
    esg_violation_score: float,
    media_score: float,
) -> float:
    """
    Compute a weighted controversy severity index (0–100).

    Weights:
        regulatory      : 40%
        legal           : 30%
        ESG violation   : 20%
        media           : 10%

    Parameters
    ----------
    regulatory_score    : Regulatory controversy score 0–100
    legal_score         : Legal controversy score 0–100
    esg_violation_score : ESG violation score 0–100
    media_score         : Media controversy score 0–100

    Returns
    -------
    float: weighted severity index 0–100
    """
    index = (
        regulatory_score    * 0.40 +
        legal_score         * 0.30 +
        esg_violation_score * 0.20 +
        media_score         * 0.10
    )
    return round(max(0.0, min(100.0, index)), 4)


def detect_reputation_cascade(
    controversy_events: list,
    window_days: int = 90,
) -> dict:
    """
    Detect whether a company is in a reputation cascade.

    A reputation cascade is flagged when 3 or more controversy events occur
    within a rolling 90-day window.

    Parameters
    ----------
    controversy_events : list of datetime objects representing controversy event dates
    window_days        : rolling window in days (default 90)

    Returns
    -------
    dict with keys:
        cascade_risk  : bool — True if 3+ events within window_days
        event_count   : number of events in the detection window
        window_days   : the window used
    """
    from datetime import datetime, timedelta, timezone

    if len(controversy_events) < 3:
        return {
            "cascade_risk": False,
            "event_count": len(controversy_events),
            "window_days": window_days,
        }

    # Sort events and check any consecutive 3+ within window_days
    events_sorted = sorted(controversy_events)
    cascade_risk = False
    for i in range(len(events_sorted) - 2):
        window_start = events_sorted[i]
        window_end = events_sorted[i + 2]
        # Check if 3 events (i, i+1, i+2) all fall within window_days
        try:
            delta = (window_end - window_start).days
        except Exception:
            # Handle naive vs aware datetimes
            delta = abs((window_end - window_start).total_seconds()) / 86400
        if delta <= window_days:
            cascade_risk = True
            break

    return {
        "cascade_risk": cascade_risk,
        "event_count": len(controversy_events),
        "window_days": window_days,
    }


def compute_controversy_market_impact(
    tier1_count: int,
    tier2_count: int,
    days: int = 5,
) -> dict:
    """
    Estimate market price impact from controversies in the first N days.

    Impact model:
        -2% per Tier-1 controversy (major: regulatory, legal, ESG)
        -0.5% per Tier-2 controversy (minor: media, social)

    Parameters
    ----------
    tier1_count : Number of Tier-1 (major) controversies
    tier2_count : Number of Tier-2 (minor) controversies
    days        : Time horizon for impact estimate (default 5 trading days)

    Returns
    -------
    dict with keys:
        estimated_impact_pct  : total estimated price impact (negative)
        tier1_impact_pct      : contribution from Tier-1 controversies
        tier2_impact_pct      : contribution from Tier-2 controversies
        days                  : time horizon used
    """
    tier1_impact = tier1_count * -2.0
    tier2_impact = tier2_count * -0.5
    total_impact = tier1_impact + tier2_impact
    return {
        "estimated_impact_pct": round(total_impact, 2),
        "tier1_impact_pct": round(tier1_impact, 2),
        "tier2_impact_pct": round(tier2_impact, 2),
        "tier1_count": tier1_count,
        "tier2_count": tier2_count,
        "days": days,
    }
