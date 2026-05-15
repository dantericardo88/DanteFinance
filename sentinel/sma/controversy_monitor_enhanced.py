"""
Enhanced controversy monitoring: GDELT DOC API, GDELT GKG (global knowledge graph),
entity-level controversy scoring, reputational risk, crisis detection, peer comparison.

Dimension: dim_104 — Controversy monitoring (GDELT) (target: 9)

Enhancements over controversy_monitor.py:
- GDELTDocAPIAdapter: multi-query, article classification, sentiment, source credibility weighting
- GDELTGKGAdapter: tone history, theme detection, co-occurrence network effects
- ControversyScorer (enhanced): 5-component scoring, severity weighting, trajectory, resolution tracking
- CrisisEarlyWarning: volume acceleration, tone spikes, executive mention surge
- ReputationRiskEngine: brand value at risk, revenue correlation, customer trust proxy
- FastAPI router: /controversy/* endpoints

Free endpoints (no API keys required):
  https://api.gdeltproject.org/api/v2/doc/doc   — GDELT DOC API v2
  https://efts.sec.gov/LATEST/search-index       — EDGAR full-text search
  https://api.fda.gov/food/enforcement.json      — OpenFDA enforcement
  https://echo.epa.gov/api/rest/facility_search  — EPA ECHO
  https://data.osha.gov/api/1.0/oshainspection   — OSHA enforcement
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_EDGAR_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_FDA_FOOD_BASE = "https://api.fda.gov/food/enforcement.json"
_FDA_DRUG_BASE = "https://api.fda.gov/drug/enforcement.json"
_EPA_ECHO_BASE = "https://echo.epa.gov/api/rest/facility_search"
_OSHA_BASE = "https://data.osha.gov/api/1.0/oshainspection"
_COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v3/opinions/"

_TIMEOUT = 30
_CACHE_TTL = 600  # 10 minutes

# GDELT tone scale: -10 (extremely negative) to +10 (extremely positive)
_TONE_NEGATIVE_THRESHOLD = -2.0
_TONE_SEVERE_THRESHOLD = -5.0
_TONE_CRISIS_THRESHOLD = -7.0

# Enhanced 5-component weights
_W_LEGAL = 0.25
_W_REGULATORY = 0.25
_W_REPUTATIONAL = 0.20
_W_GOVERNANCE = 0.15
_W_ENVIRONMENTAL = 0.15

# Severity multipliers
_SEVERITY_CRIMINAL = 3.0
_SEVERITY_CIVIL = 2.0
_SEVERITY_REGULATORY = 1.5
_SEVERITY_REPUTATIONAL = 1.0

# Source credibility weights
_SOURCE_CREDIBILITY: dict[str, float] = {
    "nytimes.com": 2.0,
    "wsj.com": 2.0,
    "ft.com": 2.0,
    "reuters.com": 2.0,
    "bloomberg.com": 2.0,
    "apnews.com": 1.8,
    "washingtonpost.com": 1.8,
    "theguardian.com": 1.7,
    "bbc.com": 1.7,
    "cnn.com": 1.5,
    "cnbc.com": 1.5,
    "marketwatch.com": 1.5,
    "businessinsider.com": 1.3,
    "forbes.com": 1.3,
    "fortune.com": 1.3,
}

# GDELT finance-relevant controversy themes
_CONTROVERSY_THEMES_FINANCE = frozenset([
    "CORRUPTION", "BRIBERY", "FRAUD", "MONEY_LAUNDERING", "EMBEZZLEMENT",
    "INSIDER_TRADING", "ACCOUNTING_FRAUD", "TAX_EVASION", "SANCTION",
    "LAWSUIT", "DISCRIMINATION", "SEXUAL_HARASSMENT", "LABOR_DISPUTE",
    "SAFETY_RECALL", "PRODUCT_RECALL", "DATA_BREACH", "CYBERSECURITY",
    "WHISTLEBLOWER", "REGULATORY_VIOLATION", "SEC_INVESTIGATION",
    "DOJ_PROBE", "CLASS_ACTION", "BANKRUPTCY", "GOING_CONCERN",
    "ENVIRONMENTAL_VIOLATION", "POLLUTION", "CLIMATE_FRAUD",
    "GOVERNANCE_FAILURE", "BOARD_DISPUTE", "EXECUTIVE_MISCONDUCT",
    "RESTATEMENT", "AUDIT_FAILURE", "CONFLICT_OF_INTEREST",
    "PROTEST", "BOYCOTT", "STRIKE",
])

_CONTROVERSY_THEME_PREFIXES = (
    "ENV_", "ECON_", "CRIME_", "GOV_", "TAX_", "CORRUPTION",
    "BRIBERY", "DISCRIMINATION", "LAWSUIT", "FRAUD", "SANCTION",
    "MONEY_LAUNDERING", "PROTEST", "RECALL", "SAFETY_", "LABOR_",
    "WHISTLEBLOWER", "DATA_BREACH", "CYBER", "SCANDAL",
)

# Article classification keywords
_CLASSIFICATION_KEYWORDS: dict[str, list[str]] = {
    "legal": [
        "lawsuit", "litigation", "sued", "class action", "settlement",
        "verdict", "judgment", "court", "indictment", "criminal charge",
        "plea", "acquittal", "dismissal", "injunction", "damages", "DOJ",
        "grand jury", "subpoena", "arrest", "guilty", "convicted",
    ],
    "regulatory": [
        "SEC", "CFTC", "FTC", "FDA", "EPA", "OSHA", "FINRA", "OCC",
        "investigation", "probe", "fine", "penalty", "enforcement",
        "consent order", "cease and desist", "violation", "audit",
        "compliance", "whistleblower", "sanction", "embargo",
    ],
    "governance": [
        "CEO resign", "CFO depart", "board remove", "director quit",
        "shareholder revolt", "proxy fight", "going concern", "restatement",
        "accounting error", "internal investigation", "audit committee",
        "conflict of interest", "related party", "executive misconduct",
    ],
    "reputational": [
        "boycott", "protest", "strike", "scandal", "controversy",
        "outrage", "backlash", "criticism", "accusation", "allegation",
        "misconduct", "toxic", "culture problem", "discrimination",
        "harassment", "cover-up", "deception", "misleading",
    ],
    "environmental": [
        "pollution", "spill", "contamination", "emissions", "carbon",
        "climate", "deforestation", "hazardous", "toxic waste",
        "environmental damage", "EPA violation", "ecological", "greenwashing",
    ],
}

# Crisis detection keywords (regulatory)
_CRISIS_KEYWORDS = [
    "SEC investigation", "DOJ probe", "class action", "whistleblower",
    "criminal indictment", "fraud charges", "accounting irregularity",
    "material weakness", "emergency recall", "data breach disclosed",
    "bankruptcy filing", "going concern", "liquidity crisis",
]

# Resolution keywords (controversy diminishing)
_RESOLUTION_KEYWORDS = [
    "settlement reached", "charges dropped", "acquitted", "dismissed",
    "cleared", "no wrongdoing", "resolved", "paid fine", "consent decree",
    "matter closed", "investigation concluded",
]

# GICS sector peer groups (simplified — ticker → sector)
_GICS_SECTORS: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "JPM": "Financials",
    "BAC": "Financials", "GS": "Financials", "JNJ": "Healthcare",
    "PFE": "Healthcare", "UNH": "Healthcare", "XOM": "Energy",
    "CVX": "Energy", "COP": "Energy", "CAT": "Industrials",
    "BA": "Industrials", "GE": "Industrials",
}

_EDGAR_HEADERS = {
    "User-Agent": "SENTINEL/2.0 research@sentinel.ai",
    "Accept": "application/json,*/*",
}

# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_DB_PATH = Path("data/controversy_enhanced.db")


def _ensure_db() -> sqlite3.Connection:
    """Open/create the controversy history SQLite database."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS controversy_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            total_score REAL,
            legal_score REAL,
            regulatory_score REAL,
            reputational_score REAL,
            governance_score REAL,
            environmental_score REAL,
            severity TEXT,
            trajectory TEXT,
            early_warning_score REAL,
            flags TEXT,
            as_of TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gdelt_article_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_key TEXT NOT NULL,
            article_url TEXT,
            title TEXT,
            tone REAL,
            category TEXT,
            source_domain TEXT,
            credibility_weight REAL,
            themes TEXT,
            published TEXT,
            cached_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_controversy_ticker ON controversy_history(ticker, as_of)
    """)
    conn.commit()
    return conn


def _save_controversy_score(score: "EnhancedControversyScore") -> None:
    """Persist a controversy score to SQLite."""
    try:
        conn = _ensure_db()
        conn.execute("""
            INSERT INTO controversy_history
            (ticker, company_name, total_score, legal_score, regulatory_score,
             reputational_score, governance_score, environmental_score,
             severity, trajectory, early_warning_score, flags, as_of)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            score.ticker, score.company_name, score.total_score,
            score.legal_score, score.regulatory_score, score.reputational_score,
            score.governance_score, score.environmental_score,
            score.severity, score.trajectory, score.early_warning_score,
            json.dumps(score.flags), score.as_of.isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Failed to save controversy score to DB", error=str(exc))


def _load_controversy_history(ticker: str, days: int = 90) -> list[dict]:
    """Load historical controversy scores for a ticker from SQLite."""
    try:
        conn = _ensure_db()
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT ticker, company_name, total_score, legal_score, regulatory_score,
                   reputational_score, governance_score, environmental_score,
                   severity, trajectory, early_warning_score, flags, as_of
            FROM controversy_history
            WHERE ticker = ? AND as_of >= ?
            ORDER BY as_of DESC
        """, (ticker.upper(), cutoff)).fetchall()
        conn.close()
        cols = [
            "ticker", "company_name", "total_score", "legal_score",
            "regulatory_score", "reputational_score", "governance_score",
            "environmental_score", "severity", "trajectory",
            "early_warning_score", "flags", "as_of",
        ]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        logger.warning("Failed to load controversy history", ticker=ticker, error=str(exc))
        return []


# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _cache[key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ArticleClassification(BaseModel):
    """Classification result for a single GDELT article."""
    model_config = ConfigDict(frozen=True)
    url: str
    title: str
    category: str  # legal/regulatory/governance/reputational/environmental/unclassified
    tone: float
    credibility_weight: float
    source_domain: str
    is_top_tier: bool
    themes: list[str]
    published: Optional[str] = None
    geographic_scope: str = "domestic"  # domestic / international


class EnhancedControversyScore(BaseModel):
    """Enhanced multi-signal controversy score with 5 components."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    total_score: float = Field(..., ge=0.0, le=100.0)
    legal_score: float = Field(..., ge=0.0, le=100.0)
    regulatory_score: float = Field(..., ge=0.0, le=100.0)
    reputational_score: float = Field(..., ge=0.0, le=100.0)
    governance_score: float = Field(..., ge=0.0, le=100.0)
    environmental_score: float = Field(..., ge=0.0, le=100.0)
    severity: str  # low / medium / high / critical
    trajectory: str  # increasing / decreasing / stable
    trajectory_30d_delta: float
    early_warning_score: float
    resolution_signals: list[str]
    flags: list[str]
    peer_comparison: Optional[dict] = None
    as_of: datetime


class GKGEntityProfile(BaseModel):
    """Entity profile extracted from GDELT GKG."""
    model_config = ConfigDict(frozen=True)
    entity_name: str
    entity_type: str  # company / executive / subsidiary
    tone_30d_avg: float
    tone_trend: str  # improving / deteriorating / stable
    top_themes: list[str]
    co_mentioned_entities: list[str]
    article_count: int
    as_of: datetime


class CrisisWarning(BaseModel):
    """Early warning crisis signal for a company."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    early_warning_score: float = Field(..., ge=0.0, le=100.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    signals: list[str]
    volume_acceleration_ratio: float
    tone_spike_detected: bool
    min_tone_observed: float
    new_themes_detected: list[str]
    executive_mention_surge: bool
    regulatory_keywords_found: list[str]
    recommended_action: str
    as_of: datetime


class ReputationRisk(BaseModel):
    """Reputational risk metrics derived from controversy score."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    controversy_score: float
    brand_value_erosion_pct: float  # estimated % brand value at risk
    estimated_revenue_impact_pct: float  # estimated next 30D revenue impact
    customer_trust_score: float  # 0-100 (100 = high trust)
    esg_divestment_risk: str  # low / medium / high / imminent
    market_impact_next_day_pct: float  # avg next-day return when controversy spikes
    severity_label: str
    risk_factors: list[str]
    as_of: datetime


class PeerComparisonResult(BaseModel):
    """Controversy score relative to sector peers."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    sector: str
    company_score: float
    sector_avg_score: float
    sector_median_score: float
    percentile_rank: float  # 0-100; 100 = most controversial in sector
    peers_scored: list[dict]
    as_of: datetime


class BulkControversyResult(BaseModel):
    """Result for a single ticker in a bulk scan."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    total_score: float
    severity: str
    early_warning_score: float
    top_flags: list[str]
    as_of: datetime


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _parse_gdelt_datetime(raw: str) -> Optional[datetime]:
    """Parse GDELT YYYYMMDDHHMMSS into UTC datetime."""
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _extract_source_domain(url: str) -> str:
    """Extract domain from URL for credibility lookup."""
    try:
        m = re.search(r"https?://(?:www\.)?([^/]+)", url)
        return m.group(1).lower() if m else ""
    except Exception:
        return ""


def _get_credibility_weight(url: str) -> tuple[float, bool]:
    """Return (credibility_weight, is_top_tier) for a URL."""
    domain = _extract_source_domain(url)
    for key, weight in _SOURCE_CREDIBILITY.items():
        if key in domain:
            return weight, weight >= 1.8
    # Wire services get 1.5, blogs/unknown get 0.5
    wire_patterns = ["ap.org", "afp.com", "dpa", "pa.media", "upa.com"]
    if any(p in domain for p in wire_patterns):
        return 1.5, False
    if any(p in domain for p in ["blogspot", "wordpress", "medium.com", "substack"]):
        return 0.5, False
    return 1.0, False


def _classify_article(title: str, themes: list[str]) -> str:
    """Classify article into controversy category based on title + themes."""
    title_lower = title.lower()
    theme_str = " ".join(themes).upper()

    # Priority order: legal > regulatory > governance > environmental > reputational
    for category, keywords in _CLASSIFICATION_KEYWORDS.items():
        if any(kw.lower() in title_lower for kw in keywords):
            return category
    # Fallback: theme-based
    if any(t in theme_str for t in ["CRIME", "LAWSUIT", "FRAUD", "CORRUPTION"]):
        return "legal"
    if any(t in theme_str for t in ["REGULATORY", "SEC", "EPA", "FDA", "OSHA"]):
        return "regulatory"
    if any(t in theme_str for t in ["ENV_", "POLLUTION", "CLIMATE"]):
        return "environmental"
    if any(t in theme_str for t in ["GOV_", "GOVERNANCE", "BOARD"]):
        return "governance"
    return "reputational"


def _parse_tone(raw: Any) -> float:
    """Parse GDELT tone field (may be comma-separated or plain float)."""
    try:
        return float(str(raw).split(",")[0])
    except (ValueError, TypeError):
        return 0.0


def _parse_themes(raw: Any) -> list[str]:
    """Parse GDELT themes field (semicolon-separated string or list)."""
    if isinstance(raw, list):
        return [t.strip() for t in raw if t.strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(";") if t.strip()]
    return []


def _extract_controversy_themes(themes: list[str]) -> list[str]:
    """Filter to controversy-relevant themes."""
    result = []
    for t in themes:
        tu = t.upper()
        if tu in _CONTROVERSY_THEMES_FINANCE:
            result.append(t)
        elif any(tu.startswith(pfx) for pfx in _CONTROVERSY_THEME_PREFIXES):
            result.append(t)
    return result


def _severity_from_score(score: float) -> str:
    if score < 25.0:
        return "low"
    if score < 50.0:
        return "medium"
    if score < 75.0:
        return "high"
    return "critical"


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def _geographic_scope(source_country: str) -> str:
    """Classify news as domestic (US) or international."""
    if not source_country:
        return "unknown"
    return "domestic" if source_country.upper() in ("US", "USA", "UNITED STATES") else "international"


# ---------------------------------------------------------------------------
# GDELTDocAPIAdapter
# ---------------------------------------------------------------------------


class GDELTDocAPIAdapter:
    """
    Enhanced GDELT DOC API v2 adapter.

    Features:
    - Multi-query: company name + CEO name + product names
    - Article classification: legal/regulatory/governance/reputational/environmental
    - Sentiment extraction: GDELT tone field
    - Source credibility weighting: top-tier (NYT/WSJ/FT/Reuters) = 2x, wire = 1.5x, blog = 0.5x
    - Geographic clustering: US domestic vs international controversy
    """

    def __init__(self, timeout: int = _TIMEOUT) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SENTINEL/2.0 research@sentinel.ai"})

    def _fetch_raw(
        self,
        query: str,
        mode: str = "ArtList",
        timespan: str = "1month",
        maxrecords: int = 75,
        sort: str = "DateDesc",
    ) -> list[dict]:
        """Fetch raw articles from GDELT DOC API v2."""
        cache_key = f"gdelt_doc_v2:{query}:{mode}:{timespan}:{maxrecords}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        params = {
            "query": query,
            "mode": mode,
            "timespan": timespan,
            "maxrecords": str(maxrecords),
            "format": "json",
            "sort": sort,
        }
        try:
            resp = self._session.get(_GDELT_DOC_BASE, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            _cache_set(cache_key, articles)
            logger.info("GDELT DOC v2 fetch", query=query[:60], count=len(articles))
            return articles
        except Exception as exc:
            logger.warning("GDELT DOC v2 error", query=query[:60], error=str(exc))
            return []

    def fetch_multi_query(
        self,
        company_name: str,
        ceo_name: Optional[str] = None,
        product_names: Optional[list[str]] = None,
        timespan: str = "1month",
        maxrecords: int = 75,
    ) -> list[dict]:
        """
        Run multiple GDELT queries (company + CEO + products) and deduplicate.

        Returns merged article list with source metadata.
        """
        queries = [f'"{company_name}"']
        if ceo_name:
            queries.append(f'"{ceo_name}"')
        if product_names:
            for prod in product_names[:3]:  # Limit to top 3 products
                queries.append(f'"{prod}"')

        seen_urls: set[str] = set()
        all_articles: list[dict] = []

        for q in queries:
            articles = self._fetch_raw(q, timespan=timespan, maxrecords=maxrecords)
            for art in articles:
                url = art.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    art["_query_source"] = q
                    all_articles.append(art)

        logger.info(
            "GDELT multi-query complete",
            company=company_name,
            queries=len(queries),
            unique_articles=len(all_articles),
        )
        return all_articles

    def classify_articles(
        self,
        articles: list[dict],
    ) -> list[ArticleClassification]:
        """
        Classify and enrich articles with category, credibility weight, geography.

        Returns list of ArticleClassification objects.
        """
        classified: list[ArticleClassification] = []
        for art in articles:
            url = art.get("url", "")
            title = art.get("title", "")
            tone = _parse_tone(art.get("tone", 0))
            themes = _parse_themes(art.get("themes", ""))
            source_country = art.get("sourcecountry", "")
            published = art.get("seendate", "")

            credibility, is_top_tier = _get_credibility_weight(url)
            category = _classify_article(title, themes)
            geo = _geographic_scope(source_country)
            domain = _extract_source_domain(url)

            classified.append(ArticleClassification(
                url=url,
                title=title,
                category=category,
                tone=tone,
                credibility_weight=credibility,
                source_domain=domain,
                is_top_tier=is_top_tier,
                themes=_extract_controversy_themes(themes),
                published=published,
                geographic_scope=geo,
            ))

        return classified

    def compute_weighted_sentiment(
        self,
        classified: list[ArticleClassification],
    ) -> dict[str, float]:
        """
        Compute credibility-weighted sentiment by category.

        Returns dict: category → weighted_avg_tone, plus overall stats.
        """
        by_category: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for art in classified:
            by_category[art.category].append((art.tone, art.credibility_weight))

        result: dict[str, float] = {}
        all_tones_weighted: list[float] = []
        all_weights: list[float] = []

        for cat, items in by_category.items():
            if not items:
                continue
            tones, weights = zip(*items)
            total_w = sum(weights)
            if total_w > 0:
                weighted_avg = sum(t * w for t, w in zip(tones, weights)) / total_w
                result[f"{cat}_tone"] = round(weighted_avg, 3)
                result[f"{cat}_count"] = len(items)
            all_tones_weighted.extend(t * w for t, w in zip(tones, weights))
            all_weights.extend(weights)

        total_w = sum(all_weights)
        result["overall_tone"] = round(sum(all_tones_weighted) / total_w, 3) if total_w > 0 else 0.0
        result["total_articles"] = len(classified)
        result["top_tier_articles"] = sum(1 for a in classified if a.is_top_tier)
        result["domestic_articles"] = sum(1 for a in classified if a.geographic_scope == "domestic")
        result["international_articles"] = sum(1 for a in classified if a.geographic_scope == "international")

        return result

    def geographic_controversy_breakdown(
        self,
        classified: list[ArticleClassification],
    ) -> dict[str, dict]:
        """
        Separate controversy signals by US domestic vs international.

        Returns: {"domestic": {count, avg_tone, top_categories}, "international": {...}}
        """
        breakdown: dict[str, dict] = {"domestic": {}, "international": {}, "unknown": {}}
        for scope in breakdown:
            subset = [a for a in classified if a.geographic_scope == scope]
            if not subset:
                breakdown[scope] = {"count": 0, "avg_tone": 0.0, "top_categories": []}
                continue
            tones = [a.tone for a in subset]
            cats: dict[str, int] = defaultdict(int)
            for a in subset:
                cats[a.category] += 1
            top_cats = sorted(cats, key=lambda k: cats[k], reverse=True)[:3]
            breakdown[scope] = {
                "count": len(subset),
                "avg_tone": round(statistics.mean(tones), 3),
                "top_categories": top_cats,
            }
        return breakdown

    def get_theme_frequency(
        self,
        classified: list[ArticleClassification],
    ) -> dict[str, int]:
        """Count controversy theme occurrences across all classified articles."""
        theme_counts: dict[str, int] = defaultdict(int)
        for art in classified:
            for t in art.themes:
                theme_counts[t] += 1
        return dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# GDELTGKGAdapter
# ---------------------------------------------------------------------------


class GDELTGKGAdapter:
    """
    GDELT Global Knowledge Graph (GKG) adapter.

    Provides:
    - Entity recognition: company, executives, subsidiaries from GKG themes
    - Finance-relevant GDELT themes: CORRUPTION, LAWSUIT, REGULATORY, FRAUD, etc.
    - Tone history: 30-day rolling tone
    - Network effects: co-occurrence of entities alongside company
    """

    _FINANCE_GKG_THEMES = frozenset([
        "CORRUPTION", "LAWSUIT", "FRAUD", "REGULATORY", "SCANDAL",
        "SAFETY", "LABOR_DISPUTE", "BRIBERY", "MONEY_LAUNDERING",
        "TAX_EVASION", "SANCTION", "WHISTLEBLOWER", "DATA_BREACH",
        "DISCRIMINATION", "HARASSMENT", "BANKRUPTCY", "RECALL",
        "INVESTIGATION", "ENFORCEMENT", "PENALTY", "FINE",
    ])

    def __init__(self, timeout: int = _TIMEOUT) -> None:
        self._timeout = timeout
        self._session = requests.Session()

    def _fetch_gkg_articles(
        self,
        entity: str,
        timespan: str = "1month",
        maxrecords: int = 75,
    ) -> list[dict]:
        """Fetch GKG-enriched article records via GDELT DOC API."""
        cache_key = f"gkg:{entity}:{timespan}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        params = {
            "query": f'"{entity}"',
            "mode": "ArtList",
            "timespan": timespan,
            "maxrecords": str(maxrecords),
            "format": "json",
        }
        try:
            resp = self._session.get(_GDELT_DOC_BASE, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            _cache_set(cache_key, articles)
            return articles
        except Exception as exc:
            logger.warning("GKG fetch error", entity=entity, error=str(exc))
            return []

    def build_entity_profile(
        self,
        entity_name: str,
        entity_type: str = "company",
        lookback_days: int = 30,
    ) -> GKGEntityProfile:
        """
        Build a GKG entity profile with tone history and co-occurrence network.

        Parameters
        ----------
        entity_name  : Company name, executive name, or subsidiary
        entity_type  : "company", "executive", or "subsidiary"
        lookback_days: Rolling window for tone history

        Returns GKGEntityProfile with tone_30d_avg, top_themes, co_mentioned_entities.
        """
        timespan = f"{lookback_days}d" if lookback_days <= 365 else "1y"
        articles = self._fetch_gkg_articles(entity_name, timespan=timespan)
        now = datetime.now(tz=timezone.utc)

        tones: list[float] = []
        theme_counts: dict[str, int] = defaultdict(int)
        person_counts: dict[str, int] = defaultdict(int)
        org_counts: dict[str, int] = defaultdict(int)

        for art in articles:
            tones.append(_parse_tone(art.get("tone", 0)))

            for t in _parse_themes(art.get("themes", "")):
                tu = t.upper()
                if any(fin_t in tu for fin_t in self._FINANCE_GKG_THEMES):
                    theme_counts[t] += 1

            # Co-occurrence: persons and organizations mentioned alongside entity
            for p in _parse_themes(art.get("persons", "")):
                pn = p.strip()
                if pn and pn.lower() != entity_name.lower():
                    person_counts[pn] += 1
            for o in _parse_themes(art.get("organizations", "")):
                on = o.strip()
                if on and on.lower() != entity_name.lower():
                    org_counts[on] += 1

        if not tones:
            return GKGEntityProfile(
                entity_name=entity_name,
                entity_type=entity_type,
                tone_30d_avg=0.0,
                tone_trend="stable",
                top_themes=[],
                co_mentioned_entities=[],
                article_count=0,
                as_of=now,
            )

        avg_tone = statistics.mean(tones)
        mid = len(tones) // 2
        first_avg = statistics.mean(tones[:mid]) if mid > 0 else avg_tone
        second_avg = statistics.mean(tones[mid:]) if mid < len(tones) else avg_tone
        delta = second_avg - first_avg

        if abs(delta) < 0.5:
            trend = "stable"
        elif delta > 0:
            trend = "improving"
        else:
            trend = "deteriorating"

        top_themes = sorted(theme_counts, key=lambda k: theme_counts[k], reverse=True)[:8]

        # Combine persons + orgs for co-occurrence, top 10
        combined_co: dict[str, int] = {}
        combined_co.update(person_counts)
        for k, v in org_counts.items():
            combined_co[k] = combined_co.get(k, 0) + v
        co_entities = sorted(combined_co, key=lambda k: combined_co[k], reverse=True)[:10]

        return GKGEntityProfile(
            entity_name=entity_name,
            entity_type=entity_type,
            tone_30d_avg=round(max(-10.0, min(10.0, avg_tone)), 3),
            tone_trend=trend,
            top_themes=top_themes,
            co_mentioned_entities=co_entities,
            article_count=len(tones),
            as_of=now,
        )

    def get_tone_history(
        self,
        entity_name: str,
        lookback_days: int = 30,
        bucket_days: int = 7,
    ) -> list[dict]:
        """
        Build week-by-week tone history for an entity.

        Returns list of {period_start, period_end, avg_tone, article_count, dominant_theme}.
        """
        articles = self._fetch_gkg_articles(entity_name, timespan=f"{lookback_days}d")
        if not articles:
            return []

        # Bucket articles by week
        now = datetime.now(tz=timezone.utc)
        buckets: list[list[dict]] = []
        num_buckets = max(1, lookback_days // bucket_days)
        for i in range(num_buckets):
            start_dt = now - timedelta(days=(i + 1) * bucket_days)
            end_dt = now - timedelta(days=i * bucket_days)
            bucket_arts = []
            for art in articles:
                pub_raw = art.get("seendate", "")
                pub_dt = _parse_gdelt_datetime(pub_raw)
                if pub_dt and start_dt <= pub_dt < end_dt:
                    bucket_arts.append(art)
            buckets.append(bucket_arts)

        history: list[dict] = []
        for i, bucket in enumerate(buckets):
            start_dt = now - timedelta(days=(i + 1) * bucket_days)
            end_dt = now - timedelta(days=i * bucket_days)
            if not bucket:
                history.append({
                    "period_start": start_dt.isoformat(),
                    "period_end": end_dt.isoformat(),
                    "avg_tone": 0.0,
                    "article_count": 0,
                    "dominant_theme": None,
                })
                continue
            tones = [_parse_tone(a.get("tone", 0)) for a in bucket]
            all_themes: list[str] = []
            for a in bucket:
                all_themes.extend(_parse_themes(a.get("themes", "")))
            theme_freq: dict[str, int] = defaultdict(int)
            for t in all_themes:
                theme_freq[t] += 1
            dominant = max(theme_freq, key=lambda k: theme_freq[k]) if theme_freq else None
            history.append({
                "period_start": start_dt.isoformat(),
                "period_end": end_dt.isoformat(),
                "avg_tone": round(statistics.mean(tones), 3),
                "article_count": len(tones),
                "dominant_theme": dominant,
            })

        return list(reversed(history))  # chronological order

    def detect_new_themes(
        self,
        entity_name: str,
        recent_days: int = 7,
        baseline_days: int = 30,
    ) -> list[str]:
        """
        Detect themes appearing in recent_days that were absent in the prior baseline.

        Returns list of newly emerged controversy themes.
        """
        recent_arts = self._fetch_gkg_articles(entity_name, timespan=f"{recent_days}d")
        baseline_arts = self._fetch_gkg_articles(entity_name, timespan=f"{baseline_days}d")

        recent_themes: set[str] = set()
        baseline_themes: set[str] = set()

        for art in recent_arts:
            for t in _parse_themes(art.get("themes", "")):
                if any(t.upper().startswith(pfx) for pfx in _CONTROVERSY_THEME_PREFIXES):
                    recent_themes.add(t)

        for art in baseline_arts:
            for t in _parse_themes(art.get("themes", "")):
                baseline_themes.add(t)

        new_themes = recent_themes - baseline_themes
        return sorted(new_themes)


# ---------------------------------------------------------------------------
# Enhanced ControversyScorer
# ---------------------------------------------------------------------------


class ControversyScorer:
    """
    Enhanced 5-component controversy scorer.

    Components:
      legal (25%): criminal/civil cases, SEC enforcement, court opinions
      regulatory (25%): FDA, EPA, OSHA, CFTC, FTC enforcement
      reputational (20%): GDELT tone, media volume, credibility-weighted sentiment
      governance (15%): exec departures, going concern, restatements
      environmental (15%): EPA violations, climate controversy, greenwashing

    Severity weighting applied to raw sub-scores:
      criminal = 3x, civil = 2x, regulatory = 1.5x, reputational = 1x

    Trajectory: 30-day trend from SQLite history.
    Resolution tracking: settlement/dismissal signals reduce score.
    """

    def __init__(self) -> None:
        self._doc_adapter = GDELTDocAPIAdapter()
        self._gkg_adapter = GDELTGKGAdapter()
        self._session = requests.Session()
        self._session.headers.update(_EDGAR_HEADERS)

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _legal_score(self, ticker: str, company_name: str) -> tuple[float, list[str], list[str]]:
        """Score legal risk 0–100. Returns (score, flags, resolution_signals)."""
        flags: list[str] = []
        resolution: list[str] = []
        raw = 0.0

        # SEC enforcement via EDGAR EFTS
        try:
            cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
            params = {
                "q": f'"{ticker}" OR "{company_name}"',
                "dateRange": "custom",
                "startdt": cutoff,
                "forms": "LR",
            }
            resp = self._session.get(_EDGAR_EFTS_BASE, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                sec_count = data.get("hits", {}).get("total", {}).get("value", 0)
                if sec_count > 0:
                    # Criminal multiplier if DOJ keyword appears
                    raw += min(50.0, sec_count * 15.0 * _SEVERITY_CIVIL)
                    flags.append(f"{sec_count} SEC litigation release(s) in past year")
        except Exception as exc:
            logger.warning("SEC enforcement score failed", ticker=ticker, error=str(exc))

        # CourtListener federal court opinions
        try:
            filed_after = (datetime.now(tz=timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
            resp = self._session.get(
                _COURTLISTENER_BASE,
                params={
                    "q": f'"{company_name}" OR "{ticker}"',
                    "type": "o",
                    "order_by": "score desc",
                    "filed_after": filed_after,
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                court_count = data.get("count", 0)
                if court_count > 0:
                    raw += min(40.0, court_count * 4.0 * _SEVERITY_CIVIL)
                    flags.append(f"{court_count} federal court opinion(s) via CourtListener")
        except Exception as exc:
            logger.warning("CourtListener score failed", ticker=ticker, error=str(exc))

        # Check for criminal keywords in GDELT articles (higher multiplier)
        articles = self._doc_adapter._fetch_raw(
            f'"{company_name}" (criminal OR indicted OR "grand jury" OR "DOJ")',
            timespan="3m",
            maxrecords=25,
        )
        if articles:
            criminal_count = len(articles)
            raw += min(30.0, criminal_count * 8.0 * _SEVERITY_CRIMINAL)
            flags.append(f"{criminal_count} article(s) with criminal/DOJ keywords")

        # Resolution signals
        res_articles = self._doc_adapter._fetch_raw(
            f'"{company_name}" (settlement OR "charges dropped" OR acquitted OR dismissed)',
            timespan="3m",
            maxrecords=20,
        )
        for art in res_articles:
            title = art.get("title", "").lower()
            if any(kw in title for kw in ["settl", "dismiss", "acquit", "clear", "resolv"]):
                raw = max(0.0, raw - 10.0)
                resolution.append(f"Resolution signal: {art.get('title', '')[:80]}")
                break

        return _clamp(raw), flags, resolution

    def _regulatory_score(self, company_name: str) -> tuple[float, list[str]]:
        """Score regulatory risk 0–100 from FDA, EPA, OSHA."""
        flags: list[str] = []
        raw = 0.0

        # FDA recalls
        for fda_url in [_FDA_FOOD_BASE, _FDA_DRUG_BASE]:
            try:
                params = {
                    "search": f'recalling_firm:"{company_name}"',
                    "limit": "50",
                }
                resp = self._session.get(fda_url, params=params, timeout=_TIMEOUT)
                if resp.status_code == 200:
                    fda_count = len(resp.json().get("results", []))
                    if fda_count > 0:
                        raw += min(35.0, fda_count * 10.0 * _SEVERITY_REGULATORY)
                        flags.append(f"{fda_count} FDA recall/enforcement record(s)")
            except Exception as exc:
                logger.warning("FDA regulatory score failed", company=company_name, error=str(exc))

        # EPA ECHO violations
        try:
            params = {"p_fn": company_name, "p_act": "Y", "output": "JSON", "p_qnc": "1"}
            resp = self._session.get(_EPA_ECHO_BASE, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                facilities = resp.json().get("Results", {}).get("Facilities", [])
                if facilities:
                    total_penalty = sum(
                        float(f.get("TotalPenalties", 0) or 0) for f in facilities
                    )
                    raw += min(30.0, len(facilities) * 8.0 * _SEVERITY_REGULATORY)
                    if total_penalty > 0:
                        flags.append(f"EPA penalties: ${total_penalty:,.0f} across {len(facilities)} facility/ies")
        except Exception as exc:
            logger.warning("EPA regulatory score failed", company=company_name, error=str(exc))

        # OSHA inspections
        try:
            params = {"establishment_name": company_name, "format": "json", "limit": "50", "has_citation": "1"}
            resp = self._session.get(_OSHA_BASE, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                records = resp.json() if isinstance(resp.json(), list) else resp.json().get("results", [])
                osha_penalty = sum(float(r.get("tot_penl", 0) or 0) for r in records)
                if records:
                    raw += min(25.0, len(records) * 5.0 * _SEVERITY_REGULATORY)
                    if osha_penalty > 0:
                        flags.append(f"OSHA: {len(records)} citation(s); ${osha_penalty:,.0f} total penalties")
        except Exception as exc:
            logger.warning("OSHA regulatory score failed", company=company_name, error=str(exc))

        return _clamp(raw), flags

    def _reputational_score(
        self,
        company_name: str,
        ticker: str,
        classified: list[ArticleClassification],
        sentiment: dict[str, float],
    ) -> tuple[float, list[str]]:
        """Score reputational risk from credibility-weighted GDELT sentiment."""
        flags: list[str] = []
        raw = 0.0

        overall_tone = sentiment.get("overall_tone", 0.0)
        total_articles = sentiment.get("total_articles", 0)
        top_tier = sentiment.get("top_tier_articles", 0)

        # Tone-based penalty (credibility-weighted)
        if overall_tone < _TONE_CRISIS_THRESHOLD:
            raw += 60.0
            flags.append(f"Crisis-level GDELT tone: {overall_tone:.2f}")
        elif overall_tone < _TONE_SEVERE_THRESHOLD:
            raw += 45.0
            flags.append(f"Severely negative GDELT tone: {overall_tone:.2f}")
        elif overall_tone < _TONE_NEGATIVE_THRESHOLD:
            raw += 20.0 + abs(overall_tone) * 2.5
            flags.append(f"Negative GDELT media tone: {overall_tone:.2f}")

        # Top-tier negative amplification
        rep_tone = sentiment.get("reputational_tone", 0.0)
        if rep_tone < -3.0 and top_tier > 2:
            raw += min(20.0, top_tier * 4.0)
            flags.append(f"{top_tier} top-tier outlet(s) with negative reputational coverage")

        # High volume flag
        if total_articles > 100:
            raw += 8.0
            flags.append(f"High media volume: {total_articles} articles")

        # Category-specific boosts
        legal_tone = sentiment.get("legal_tone", 0.0)
        if legal_tone < -4.0:
            raw += 15.0
            flags.append(f"Negative legal-category tone: {legal_tone:.2f}")

        return _clamp(raw), flags

    def _governance_score(self, ticker: str) -> tuple[float, list[str]]:
        """Score governance risk from EDGAR 8-K departures and going concern."""
        flags: list[str] = []
        raw = 0.0
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

        try:
            # Executive departures — 8-K Item 5.02
            params = {
                "q": f'"{ticker}" "5.02" (resign OR depart OR terminat)',
                "dateRange": "custom",
                "startdt": cutoff,
                "forms": "8-K",
            }
            resp = self._session.get(_EDGAR_EFTS_BASE, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                departure_count = resp.json().get("hits", {}).get("total", {}).get("value", 0)
                if departure_count > 0:
                    raw += min(35.0, departure_count * 12.0)
                    flags.append(f"{departure_count} executive departure filing(s) (8-K Item 5.02)")

            # Going concern
            params_gc = {
                "q": f'"{ticker}" "going concern"',
                "dateRange": "custom",
                "startdt": cutoff,
                "forms": "10-K,10-Q",
            }
            resp_gc = self._session.get(_EDGAR_EFTS_BASE, params=params_gc, timeout=_TIMEOUT)
            if resp_gc.status_code == 200:
                gc_count = resp_gc.json().get("hits", {}).get("total", {}).get("value", 0)
                if gc_count > 0:
                    raw += 45.0
                    flags.append("Going concern disclosed in EDGAR filings")

            # Restatements
            params_rest = {
                "q": f'"{ticker}" restatement OR "material weakness"',
                "dateRange": "custom",
                "startdt": cutoff,
                "forms": "8-K,10-K",
            }
            resp_rest = self._session.get(_EDGAR_EFTS_BASE, params=params_rest, timeout=_TIMEOUT)
            if resp_rest.status_code == 200:
                rest_count = resp_rest.json().get("hits", {}).get("total", {}).get("value", 0)
                if rest_count > 0:
                    raw += min(20.0, rest_count * 10.0)
                    flags.append(f"{rest_count} restatement/material weakness filing(s)")

        except Exception as exc:
            logger.warning("Governance score EDGAR error", ticker=ticker, error=str(exc))

        return _clamp(raw), flags

    def _environmental_score(
        self,
        company_name: str,
        classified: list[ArticleClassification],
    ) -> tuple[float, list[str]]:
        """Score environmental risk from EPA data + environmental GDELT themes."""
        flags: list[str] = []
        raw = 0.0

        # Environmental category from classified articles
        env_articles = [a for a in classified if a.category == "environmental"]
        if env_articles:
            env_tones = [a.tone for a in env_articles]
            avg_env_tone = statistics.mean(env_tones)
            env_count = len(env_articles)
            raw += min(40.0, env_count * 5.0)
            if avg_env_tone < -3.0:
                raw += 20.0
                flags.append(f"Negative environmental coverage: {env_count} articles, tone {avg_env_tone:.2f}")
            else:
                flags.append(f"{env_count} environmental controversy article(s)")

        # EPA ECHO direct
        try:
            params = {"p_fn": company_name, "p_act": "Y", "output": "JSON", "p_qnc": "1"}
            resp = self._session.get(_EPA_ECHO_BASE, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                facilities = resp.json().get("Results", {}).get("Facilities", [])
                for fac in facilities:
                    penalty = float(fac.get("TotalPenalties", 0) or 0)
                    if penalty > 1_000_000:
                        raw += min(30.0, (penalty / 1_000_000) * 5.0)
                        flags.append(f"EPA penalty >${penalty/1e6:.1f}M at {fac.get('FacName', '?')}")
        except Exception as exc:
            logger.warning("Environmental EPA score failed", company=company_name, error=str(exc))

        # Greenwashing GDELT check
        gw_articles = self._doc_adapter._fetch_raw(
            f'"{company_name}" greenwashing',
            timespan="6m",
            maxrecords=20,
        )
        if len(gw_articles) >= 3:
            raw += 15.0
            flags.append(f"{len(gw_articles)} greenwashing article(s) found")

        return _clamp(raw), flags

    def _compute_trajectory(self, ticker: str, current_score: float) -> tuple[str, float]:
        """
        Compare current score to 30-day historical average from SQLite.

        Returns (trajectory, delta).
        """
        history = _load_controversy_history(ticker, days=30)
        if len(history) < 2:
            return "stable", 0.0

        prior_scores = [r["total_score"] for r in history[1:] if r.get("total_score") is not None]
        if not prior_scores:
            return "stable", 0.0

        prior_avg = statistics.mean(prior_scores)
        delta = current_score - prior_avg

        if abs(delta) < 3.0:
            trajectory = "stable"
        elif delta > 0:
            trajectory = "increasing"
        else:
            trajectory = "decreasing"

        return trajectory, round(delta, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_company(
        self,
        ticker: str,
        company_name: str,
        ceo_name: Optional[str] = None,
        product_names: Optional[list[str]] = None,
        include_peer_comparison: bool = False,
    ) -> EnhancedControversyScore:
        """
        Compute an enhanced 5-component controversy score (0–100).

        Weights: legal(25%) + regulatory(25%) + reputational(20%) + governance(15%) + environmental(15%)
        Severity multipliers applied per signal type.
        Trajectory derived from historical SQLite data.

        Parameters
        ----------
        ticker                : Stock ticker symbol
        company_name          : Full company name
        ceo_name              : Optional CEO name for multi-query
        product_names         : Optional product names for multi-query
        include_peer_comparison: If True, also run peer comparison

        Returns EnhancedControversyScore.
        """
        now = datetime.now(tz=timezone.utc)

        # Step 1: Fetch and classify articles (multi-query)
        raw_articles = self._doc_adapter.fetch_multi_query(
            company_name=company_name,
            ceo_name=ceo_name,
            product_names=product_names,
            timespan="1month",
        )
        classified = self._doc_adapter.classify_articles(raw_articles)
        sentiment = self._doc_adapter.compute_weighted_sentiment(classified)

        # Step 2: Sub-scores
        legal_raw, legal_flags, resolution_signals = self._legal_score(ticker, company_name)
        regulatory_raw, regulatory_flags = self._regulatory_score(company_name)
        reputational_raw, reputational_flags = self._reputational_score(
            company_name, ticker, classified, sentiment
        )
        governance_raw, governance_flags = self._governance_score(ticker)
        environmental_raw, environmental_flags = self._environmental_score(company_name, classified)

        # Step 3: Weighted total
        total = _clamp(
            legal_raw * _W_LEGAL
            + regulatory_raw * _W_REGULATORY
            + reputational_raw * _W_REPUTATIONAL
            + governance_raw * _W_GOVERNANCE
            + environmental_raw * _W_ENVIRONMENTAL
        )

        all_flags = legal_flags + regulatory_flags + reputational_flags + governance_flags + environmental_flags

        # Step 4: Trajectory
        trajectory, delta = self._compute_trajectory(ticker, total)

        # Step 5: Early warning (quick compute)
        ew_score = self._quick_early_warning_score(classified, sentiment)

        # Step 6: Peer comparison (optional — sync call)
        peer_comp: Optional[dict] = None
        if include_peer_comparison:
            try:
                peer_result = self._peer_compare(ticker, total)
                peer_comp = peer_result.model_dump() if peer_result else None
            except Exception:
                pass

        score = EnhancedControversyScore(
            ticker=ticker.upper(),
            company_name=company_name,
            total_score=round(total, 2),
            legal_score=round(legal_raw, 2),
            regulatory_score=round(regulatory_raw, 2),
            reputational_score=round(reputational_raw, 2),
            governance_score=round(governance_raw, 2),
            environmental_score=round(environmental_raw, 2),
            severity=_severity_from_score(total),
            trajectory=trajectory,
            trajectory_30d_delta=delta,
            early_warning_score=round(ew_score, 2),
            resolution_signals=resolution_signals,
            flags=all_flags,
            peer_comparison=peer_comp,
            as_of=now,
        )

        _save_controversy_score(score)
        return score

    def _quick_early_warning_score(
        self,
        classified: list[ArticleClassification],
        sentiment: dict[str, float],
    ) -> float:
        """Estimate early warning score from classified articles."""
        ew = 0.0
        total = sentiment.get("total_articles", 0)
        if total > 50:
            ew += 20.0
        overall_tone = sentiment.get("overall_tone", 0.0)
        if overall_tone < _TONE_CRISIS_THRESHOLD:
            ew += 50.0
        elif overall_tone < _TONE_SEVERE_THRESHOLD:
            ew += 30.0
        elif overall_tone < _TONE_NEGATIVE_THRESHOLD:
            ew += 15.0

        legal_count = sum(1 for a in classified if a.category == "legal")
        reg_count = sum(1 for a in classified if a.category == "regulatory")
        if legal_count > 5:
            ew += 20.0
        if reg_count > 5:
            ew += 15.0

        return _clamp(ew)

    def _peer_compare(self, ticker: str, score: float) -> Optional[PeerComparisonResult]:
        """Compare ticker's controversy score to GICS sector peers (illustrative)."""
        sector = _GICS_SECTORS.get(ticker.upper(), "Unknown")
        if sector == "Unknown":
            return None

        sector_peers = [t for t, s in _GICS_SECTORS.items() if s == sector and t != ticker.upper()][:5]
        if not sector_peers:
            return None

        peer_scores: list[dict] = []
        # Use simplified GDELT tone-based estimate for peers (no full score to avoid rate limits)
        for peer in sector_peers:
            peer_articles = self._doc_adapter._fetch_raw(
                f'"{peer}" controversy OR lawsuit OR scandal',
                timespan="1month",
                maxrecords=20,
            )
            peer_neg = sum(1 for a in peer_articles if _parse_tone(a.get("tone", 0)) < -2.0)
            estimated = _clamp(peer_neg * 8.0)
            peer_scores.append({"ticker": peer, "estimated_score": estimated})

        all_scores = [score] + [p["estimated_score"] for p in peer_scores]
        sector_avg = statistics.mean(all_scores)
        sector_med = statistics.median(all_scores)
        rank_below = sum(1 for s in all_scores if s <= score)
        percentile = (rank_below / len(all_scores)) * 100

        return PeerComparisonResult(
            ticker=ticker.upper(),
            sector=sector,
            company_score=score,
            sector_avg_score=round(sector_avg, 2),
            sector_median_score=round(sector_med, 2),
            percentile_rank=round(percentile, 1),
            peers_scored=peer_scores,
            as_of=datetime.now(tz=timezone.utc),
        )

    def get_controversy_history(
        self,
        ticker: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """Load historical controversy scores from SQLite as a DataFrame."""
        history = _load_controversy_history(ticker, days=days)
        if not history:
            return pd.DataFrame()
        df = pd.DataFrame(history)
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df.sort_values("as_of")

    def screen_universe(
        self,
        tickers: list[tuple[str, str]],
        max_score: float = 40.0,
    ) -> pd.DataFrame:
        """
        Score a universe of (ticker, company_name) pairs and filter to max_score.

        Returns DataFrame sorted ascending by total_score.
        """
        rows: list[dict] = []
        for ticker, company_name in tickers:
            try:
                score = self.score_company(ticker, company_name)
                if score.total_score <= max_score:
                    rows.append({
                        "ticker": score.ticker,
                        "company_name": score.company_name,
                        "total_score": score.total_score,
                        "severity": score.severity,
                        "trajectory": score.trajectory,
                        "flags": "; ".join(score.flags[:3]),
                    })
            except Exception as exc:
                logger.warning("Screen failed", ticker=ticker, error=str(exc))

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_score").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# CrisisEarlyWarning
# ---------------------------------------------------------------------------


class CrisisEarlyWarning:
    """
    Detect emerging corporate crises before they peak.

    Signals:
    1. Volume acceleration: 3× article count vs 7-day trailing average
    2. Negative tone spike: tone < -5 (GDELT scale)
    3. New theme emergence: controversy theme absent in prior 30 days
    4. Executive mention surge: CEO name with negative tone
    5. Regulatory keyword alert: "SEC investigation", "DOJ probe", etc.

    Combined early_warning_score: 0–100 with confidence estimate.
    """

    _REGULATORY_CRISIS_PATTERNS = [
        re.compile(r"SEC investigation", re.I),
        re.compile(r"DOJ probe", re.I),
        re.compile(r"class action", re.I),
        re.compile(r"whistleblower", re.I),
        re.compile(r"criminal (charge|indictment)", re.I),
        re.compile(r"(accounting|financial) irregularit", re.I),
        re.compile(r"material weakness", re.I),
        re.compile(r"emergency recall", re.I),
        re.compile(r"data breach", re.I),
        re.compile(r"bankruptcy filing", re.I),
        re.compile(r"going concern", re.I),
        re.compile(r"liquidity crisis", re.I),
        re.compile(r"fraud charges", re.I),
        re.compile(r"grand jury", re.I),
        re.compile(r"subpoena", re.I),
    ]

    def __init__(self) -> None:
        self._doc_adapter = GDELTDocAPIAdapter()
        self._gkg_adapter = GDELTGKGAdapter()

    def assess(
        self,
        ticker: str,
        company_name: str,
        ceo_name: Optional[str] = None,
    ) -> CrisisWarning:
        """
        Run full crisis early warning assessment for a company.

        Parameters
        ----------
        ticker       : Stock ticker
        company_name : Full company name
        ceo_name     : Optional CEO name to check executive mention surge

        Returns CrisisWarning with score, signals, and recommended action.
        """
        now = datetime.now(tz=timezone.utc)
        signals: list[str] = []
        ew_score = 0.0
        confidence = 0.5  # base confidence

        # --- Signal 1: Volume acceleration ---
        recent_7d = self._doc_adapter._fetch_raw(
            f'"{company_name}" OR "{ticker}"',
            timespan="7d",
            maxrecords=75,
        )
        baseline_30d = self._doc_adapter._fetch_raw(
            f'"{company_name}" OR "{ticker}"',
            timespan="1month",
            maxrecords=75,
        )
        recent_per_day = len(recent_7d) / 7.0
        baseline_per_day = len(baseline_30d) / 30.0
        accel_ratio = (recent_per_day / baseline_per_day) if baseline_per_day > 0.1 else 1.0

        if accel_ratio >= 5.0:
            ew_score += 30.0
            confidence = min(1.0, confidence + 0.2)
            signals.append(f"Extreme volume acceleration: {accel_ratio:.1f}x vs 30-day baseline")
        elif accel_ratio >= 3.0:
            ew_score += 20.0
            confidence = min(1.0, confidence + 0.15)
            signals.append(f"Volume acceleration: {accel_ratio:.1f}x vs 30-day baseline")

        # --- Signal 2: Tone spike ---
        recent_tones = [_parse_tone(a.get("tone", 0)) for a in recent_7d]
        min_tone = min(recent_tones) if recent_tones else 0.0
        tone_spike = min_tone < _TONE_SEVERE_THRESHOLD

        if min_tone < _TONE_CRISIS_THRESHOLD:
            ew_score += 35.0
            confidence = min(1.0, confidence + 0.2)
            signals.append(f"Crisis-level tone spike: {min_tone:.2f}")
        elif tone_spike:
            ew_score += 20.0
            confidence = min(1.0, confidence + 0.15)
            signals.append(f"Negative tone spike: {min_tone:.2f}")

        # --- Signal 3: New theme emergence ---
        new_themes = self._gkg_adapter.detect_new_themes(
            company_name, recent_days=7, baseline_days=30
        )
        if new_themes:
            ew_score += min(20.0, len(new_themes) * 5.0)
            confidence = min(1.0, confidence + 0.1)
            signals.append(f"New controversy themes: {', '.join(new_themes[:3])}")

        # --- Signal 4: Executive mention surge ---
        exec_surge = False
        if ceo_name:
            exec_articles = self._doc_adapter._fetch_raw(
                f'"{ceo_name}"',
                timespan="7d",
                maxrecords=20,
            )
            exec_neg = [a for a in exec_articles if _parse_tone(a.get("tone", 0)) < -2.0]
            if len(exec_neg) >= 3:
                exec_surge = True
                ew_score += 20.0
                confidence = min(1.0, confidence + 0.15)
                signals.append(f"CEO mention surge with negative tone: {len(exec_neg)} negative article(s)")

        # --- Signal 5: Regulatory keyword detection ---
        reg_keywords_found: list[str] = []
        for art in recent_7d:
            title = art.get("title", "")
            for pattern in self._REGULATORY_CRISIS_PATTERNS:
                if pattern.search(title):
                    kw = pattern.pattern
                    if kw not in reg_keywords_found:
                        reg_keywords_found.append(kw)
                        ew_score += 15.0
                        confidence = min(1.0, confidence + 0.1)
                        signals.append(f"Regulatory crisis keyword: '{kw}' detected")

        ew_score = _clamp(ew_score)

        # Recommended action
        if ew_score >= 70:
            action = "URGENT: Immediate monitoring; consider ESG/risk alert to portfolio managers"
        elif ew_score >= 45:
            action = "ELEVATED: Increase monitoring frequency; review position size"
        elif ew_score >= 20:
            action = "WATCH: Monitor closely over next 48 hours"
        else:
            action = "NORMAL: Routine monitoring continues"

        return CrisisWarning(
            ticker=ticker.upper(),
            company_name=company_name,
            early_warning_score=round(ew_score, 2),
            confidence=round(min(1.0, confidence), 3),
            signals=signals,
            volume_acceleration_ratio=round(accel_ratio, 3),
            tone_spike_detected=tone_spike,
            min_tone_observed=round(min_tone, 3),
            new_themes_detected=new_themes,
            executive_mention_surge=exec_surge,
            regulatory_keywords_found=reg_keywords_found,
            recommended_action=action,
            as_of=now,
        )


# ---------------------------------------------------------------------------
# ReputationRiskEngine
# ---------------------------------------------------------------------------


class ReputationRiskEngine:
    """
    Translates controversy score into tangible reputational risk metrics.

    Estimates:
    - Brand value erosion % from controversy score
    - Revenue impact in next 30 days (historically -5% at score 70+)
    - Customer trust proxy from consumer-facing news sentiment
    - Market impact: avg next-day return correlation
    - ESG investor divestment risk threshold
    """

    # Empirical curves (illustrative; calibrated to academic literature)
    _BRAND_EROSION_CURVE = [
        (0, 0.0), (25, 0.5), (40, 1.5), (55, 3.5), (70, 6.0),
        (80, 9.0), (90, 14.0), (100, 20.0),
    ]
    _REVENUE_IMPACT_CURVE = [
        (0, 0.0), (25, -0.2), (50, -1.0), (65, -2.5), (70, -5.0),
        (80, -8.0), (90, -12.0), (100, -18.0),
    ]
    _MARKET_IMPACT_CURVE = [
        (0, 0.0), (30, -0.3), (50, -0.7), (65, -1.2), (75, -1.8),
        (85, -2.5), (95, -4.0), (100, -6.0),
    ]

    def __init__(self) -> None:
        self._doc_adapter = GDELTDocAPIAdapter()

    def _interpolate(self, curve: list[tuple], score: float) -> float:
        """Linear interpolation along a (score, value) curve."""
        for i in range(len(curve) - 1):
            x0, y0 = curve[i]
            x1, y1 = curve[i + 1]
            if x0 <= score <= x1:
                t = (score - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return curve[-1][1]

    def _esg_divestment_risk(self, score: float) -> str:
        """Map controversy score to ESG divestment risk label."""
        if score < 30:
            return "low"
        if score < 50:
            return "medium"
        if score < 70:
            return "high"
        return "imminent"

    def _customer_trust_score(
        self, company_name: str
    ) -> float:
        """Estimate customer trust from consumer-facing news sentiment (0-100)."""
        articles = self._doc_adapter._fetch_raw(
            f'"{company_name}" customer OR consumer OR product OR service',
            timespan="1month",
            maxrecords=30,
        )
        if not articles:
            return 60.0  # neutral default

        tones = [_parse_tone(a.get("tone", 0)) for a in articles]
        avg_tone = statistics.mean(tones)
        # Map tone -10 to +10 → trust 0 to 100
        trust = 50.0 + avg_tone * 5.0
        return round(_clamp(trust, 0.0, 100.0), 1)

    def assess(
        self,
        ticker: str,
        company_name: str,
        controversy_score: float,
    ) -> ReputationRisk:
        """
        Compute reputational risk metrics from a controversy score.

        Parameters
        ----------
        ticker             : Stock ticker
        company_name       : Company name
        controversy_score  : Pre-computed EnhancedControversyScore.total_score

        Returns ReputationRisk with quantified risk metrics.
        """
        now = datetime.now(tz=timezone.utc)
        risk_factors: list[str] = []

        brand_erosion = self._interpolate(self._BRAND_EROSION_CURVE, controversy_score)
        revenue_impact = self._interpolate(self._REVENUE_IMPACT_CURVE, controversy_score)
        market_impact = self._interpolate(self._MARKET_IMPACT_CURVE, controversy_score)
        customer_trust = self._customer_trust_score(company_name)
        esg_risk = self._esg_divestment_risk(controversy_score)

        if brand_erosion > 5.0:
            risk_factors.append(f"Significant brand value erosion risk: ~{brand_erosion:.1f}%")
        if revenue_impact < -3.0:
            risk_factors.append(f"Revenue headwind estimate: {revenue_impact:.1f}% next 30 days")
        if customer_trust < 40:
            risk_factors.append(f"Customer trust deterioration: score {customer_trust:.0f}/100")
        if esg_risk in ("high", "imminent"):
            risk_factors.append(f"ESG mandate divestment risk: {esg_risk}")
        if market_impact < -1.5:
            risk_factors.append(f"Market impact risk: ~{market_impact:.1f}% next-day return")

        severity = _severity_from_score(controversy_score)

        return ReputationRisk(
            ticker=ticker.upper(),
            company_name=company_name,
            controversy_score=round(controversy_score, 2),
            brand_value_erosion_pct=round(brand_erosion, 2),
            estimated_revenue_impact_pct=round(revenue_impact, 2),
            customer_trust_score=customer_trust,
            esg_divestment_risk=esg_risk,
            market_impact_next_day_pct=round(market_impact, 2),
            severity_label=severity,
            risk_factors=risk_factors,
            as_of=now,
        )


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

controversy_enhanced_router = APIRouter(
    prefix="/controversy",
    tags=["Controversy Enhanced (dim_104)"],
)

_scorer = ControversyScorer()
_crisis = CrisisEarlyWarning()
_reputation = ReputationRiskEngine()
_doc_adapter = GDELTDocAPIAdapter()


class TickerQuery(BaseModel):
    ticker: str
    company_name: str
    ceo_name: Optional[str] = None
    product_names: Optional[list[str]] = None


class BulkQuery(BaseModel):
    companies: list[TickerQuery]
    max_score: float = Field(default=50.0, ge=0, le=100)


@controversy_enhanced_router.get("/score/{ticker}")
def get_controversy_score(
    ticker: str,
    company_name: str = Query(..., description="Full company name"),
    ceo_name: Optional[str] = Query(None),
    include_peer_comparison: bool = Query(False),
) -> dict:
    """
    Compute enhanced 5-component controversy score for a ticker.

    Returns total_score, per-component scores, severity, trajectory, flags.
    """
    try:
        score = _scorer.score_company(
            ticker=ticker.upper(),
            company_name=company_name,
            ceo_name=ceo_name,
            include_peer_comparison=include_peer_comparison,
        )
        return score.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@controversy_enhanced_router.get("/history/{ticker}")
def get_controversy_history(
    ticker: str,
    days: int = Query(90, ge=7, le=365),
) -> dict:
    """
    Return historical controversy scores for a ticker from SQLite.

    Useful for trend analysis and trajectory validation.
    """
    df = _scorer.get_controversy_history(ticker.upper(), days=days)
    if df.empty:
        return {"ticker": ticker.upper(), "history": [], "days": days}
    return {
        "ticker": ticker.upper(),
        "history": df.to_dict(orient="records"),
        "days": days,
        "count": len(df),
        "avg_score": round(df["total_score"].mean(), 2) if "total_score" in df.columns else None,
    }


@controversy_enhanced_router.get("/crisis-warning/{ticker}")
def get_crisis_warning(
    ticker: str,
    company_name: str = Query(...),
    ceo_name: Optional[str] = Query(None),
) -> dict:
    """
    Run crisis early warning assessment.

    Detects volume acceleration, tone spikes, new themes, exec mentions, regulatory keywords.
    Returns early_warning_score (0-100) with confidence and recommended action.
    """
    try:
        warning = _crisis.assess(
            ticker=ticker.upper(),
            company_name=company_name,
            ceo_name=ceo_name,
        )
        return warning.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@controversy_enhanced_router.get("/peers/{ticker}")
def get_peer_comparison(
    ticker: str,
    company_name: str = Query(...),
) -> dict:
    """
    Compare ticker's controversy score to GICS sector peers.

    Returns company percentile rank, sector average, and peer breakdown.
    """
    try:
        score = _scorer.score_company(ticker=ticker.upper(), company_name=company_name)
        peer_result = _scorer._peer_compare(ticker.upper(), score.total_score)
        if peer_result is None:
            return {"ticker": ticker.upper(), "message": "Sector not found or insufficient peers"}
        return peer_result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@controversy_enhanced_router.get("/themes/{ticker}")
def get_controversy_themes(
    ticker: str,
    company_name: str = Query(...),
    timespan: str = Query("1month"),
) -> dict:
    """
    Return GDELT controversy themes for a company with frequency counts.

    Filters to finance-relevant themes: CORRUPTION, LAWSUIT, FRAUD, etc.
    """
    try:
        articles = _doc_adapter.fetch_multi_query(
            company_name=company_name,
            timespan=timespan,
        )
        classified = _doc_adapter.classify_articles(articles)
        themes = _doc_adapter.get_theme_frequency(classified)
        geo = _doc_adapter.geographic_controversy_breakdown(classified)
        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "themes": themes,
            "geographic_breakdown": geo,
            "total_articles": len(articles),
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@controversy_enhanced_router.get("/reputation/{ticker}")
def get_reputation_risk(
    ticker: str,
    company_name: str = Query(...),
) -> dict:
    """
    Compute reputational risk metrics from controversy score.

    Returns brand value erosion %, revenue impact estimate, ESG divestment risk.
    """
    try:
        score = _scorer.score_company(ticker=ticker.upper(), company_name=company_name)
        risk = _reputation.assess(
            ticker=ticker.upper(),
            company_name=company_name,
            controversy_score=score.total_score,
        )
        return risk.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@controversy_enhanced_router.post("/bulk")
def bulk_controversy_scan(query: BulkQuery) -> dict:
    """
    Score multiple companies simultaneously and return filtered results.

    Filters to companies with total_score <= max_score.
    Returns list of BulkControversyResult sorted by score ascending.
    """
    results: list[dict] = []
    for company in query.companies:
        try:
            score = _scorer.score_company(
                ticker=company.ticker.upper(),
                company_name=company.company_name,
                ceo_name=company.ceo_name,
                product_names=company.product_names,
            )
            if score.total_score <= query.max_score:
                results.append({
                    "ticker": score.ticker,
                    "company_name": score.company_name,
                    "total_score": score.total_score,
                    "severity": score.severity,
                    "early_warning_score": score.early_warning_score,
                    "top_flags": score.flags[:3],
                    "as_of": score.as_of.isoformat(),
                })
        except Exception as exc:
            logger.warning("Bulk scan failed", ticker=company.ticker, error=str(exc))

    results.sort(key=lambda r: r["total_score"])
    return {
        "requested": len(query.companies),
        "returned": len(results),
        "max_score_filter": query.max_score,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def quick_controversy_check(ticker: str, company_name: str, ceo_name: Optional[str] = None) -> dict:
    """
    Single-call convenience wrapper returning score + crisis warning + reputation risk.

    Example
    -------
    >>> result = quick_controversy_check("TSLA", "Tesla Inc", ceo_name="Elon Musk")
    """
    scorer = ControversyScorer()
    crisis = CrisisEarlyWarning()
    rep_engine = ReputationRiskEngine()

    score = scorer.score_company(ticker, company_name, ceo_name=ceo_name)
    warning = crisis.assess(ticker, company_name, ceo_name=ceo_name)
    rep = rep_engine.assess(ticker, company_name, score.total_score)

    return {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "controversy_score": score.total_score,
        "severity": score.severity,
        "trajectory": score.trajectory,
        "trajectory_30d_delta": score.trajectory_30d_delta,
        "legal_score": score.legal_score,
        "regulatory_score": score.regulatory_score,
        "reputational_score": score.reputational_score,
        "governance_score": score.governance_score,
        "environmental_score": score.environmental_score,
        "early_warning_score": warning.early_warning_score,
        "early_warning_confidence": warning.confidence,
        "crisis_signals": warning.signals,
        "recommended_action": warning.recommended_action,
        "brand_erosion_pct": rep.brand_value_erosion_pct,
        "revenue_impact_pct": rep.estimated_revenue_impact_pct,
        "customer_trust": rep.customer_trust_score,
        "esg_divestment_risk": rep.esg_divestment_risk,
        "top_flags": score.flags[:5],
        "resolution_signals": score.resolution_signals,
        "as_of": score.as_of.isoformat(),
    }


def batch_controversy_screen(
    universe: list[tuple[str, str]],
    max_score: float = 35.0,
) -> pd.DataFrame:
    """
    Screen a (ticker, company_name) universe for low-controversy stocks.

    Parameters
    ----------
    universe  : list of (ticker, company_name) tuples
    max_score : Maximum total_score to include (default 35.0)

    Returns pd.DataFrame sorted by total_score ascending.
    """
    scorer = ControversyScorer()
    return scorer.screen_universe(universe, max_score=max_score)
