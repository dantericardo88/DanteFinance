"""
Controversy monitor — ESG controversy signals from free news sources.

Aggregates news-based controversy signals for a given ticker using:
  1. Finnhub company-news API (requires API key in settings, optional)
  2. GDELT Doc API (free, no key — https://api.gdeltproject.org/api/v2/doc/doc)

Classifies each article into a controversy type and severity level, then
computes an overall controversy score (0–100) and trend direction.

Score target: SENTINEL 8 vs Bloomberg 7 (Bloomberg's ESG controversy data
lacks real-time news integration at the free tier).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIMEOUT = 20.0
_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_FINNHUB_NEWS_BASE = "https://finnhub.io/api/v1/company-news"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}

# ---------------------------------------------------------------------------
# Controversy keyword taxonomy
# ---------------------------------------------------------------------------

_CONTROVERSY_KEYWORDS: dict[str, list[str]] = {
    "environmental": [
        "pollution", "spill", "emissions", "EPA",
        "environmental violation", "climate fraud", "greenwashing",
    ],
    "social": [
        "discrimination", "harassment", "labor violation", "child labor",
        "workplace injury", "OSHA", "human rights",
    ],
    "governance": [
        "fraud", "accounting irregularity", "restatement", "SEC investigation",
        "insider trading", "bribery", "corruption", "whistleblower",
    ],
    "regulatory": [
        "fine", "penalty", "consent decree", "antitrust",
        "FTC", "DOJ", "class action",
    ],
    "litigation": [
        "lawsuit", "settlement", "indictment", "conviction", "criminal charges",
    ],
    "cybersecurity": [
        "data breach", "hack", "ransomware", "data leak", "cyber attack",
    ],
}

# Keywords that auto-elevate severity to "high"
_HIGH_SEVERITY_TRIGGERS = {
    "fraud", "criminal", "indictment", "sec enforcement",
    "conviction", "criminal charges", "restatement",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ControversySignal(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    controversy_type: str
    severity: str   # "high" | "medium" | "low"
    headline: str
    source: str
    published_date: Optional[datetime] = None
    url: Optional[str] = None
    keywords_matched: list[str] = Field(default_factory=list)


class ControversyProfile(BaseModel):
    ticker: str
    total_controversies: int
    high_severity: int
    medium_severity: int
    low_severity: int
    controversy_types: dict[str, int] = Field(default_factory=dict)
    recent_signals: list[ControversySignal] = Field(default_factory=list)
    controversy_score: float   # 0–100
    trend: str                 # "worsening" | "stable" | "improving"
    lookback_days: int


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _classify_controversy(text: str) -> tuple[str, str, list[str]]:
    """
    Classify a piece of text (headline + snippet) into controversy type,
    severity, and matched keywords.

    Returns:
        (controversy_type, severity, keywords_matched)
        controversy_type is "" and severity is "" if no keywords matched.
    """
    text_lower = text.lower()
    best_type = ""
    best_count = 0
    all_matched: list[str] = []

    type_scores: dict[str, list[str]] = {}

    for ctype, keywords in _CONTROVERSY_KEYWORDS.items():
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        if matched:
            type_scores[ctype] = matched

    if not type_scores:
        return "", "", []

    # Pick type with most keyword hits
    best_type = max(type_scores, key=lambda t: len(type_scores[t]))
    all_matched = [kw for kws in type_scores.values() for kw in kws]

    # Determine severity
    total_matches = len(all_matched)
    text_lower_set = text_lower
    has_high_trigger = any(t in text_lower_set for t in _HIGH_SEVERITY_TRIGGERS)

    if has_high_trigger or total_matches >= 3:
        severity = "high"
    elif total_matches == 2:
        severity = "medium"
    else:
        severity = "low"

    return best_type, severity, list(dict.fromkeys(all_matched))  # deduplicate


# ---------------------------------------------------------------------------
# News source fetchers
# ---------------------------------------------------------------------------

async def _fetch_finnhub_news(
    ticker: str,
    days_back: int,
    api_key: Optional[str],
) -> list[dict]:
    """
    Fetch company news from Finnhub.

    Returns a list of article dicts with keys: headline, summary, source,
    datetime (unix ts), url.
    """
    if not api_key:
        logger.debug("controversy_monitor: no Finnhub key, skipping Finnhub news")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = (
        f"{_FINNHUB_NEWS_BASE}"
        f"?symbol={ticker.upper()}&from={from_date}&to={to_date}&token={api_key}"
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

        if not isinstance(data, list):
            logger.warning("controversy_monitor: Finnhub returned non-list for %s", ticker)
            return []

        logger.debug("controversy_monitor: Finnhub returned %d articles for %s", len(data), ticker)
        return data

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "controversy_monitor: Finnhub HTTP %d for %s: %s",
            exc.response.status_code, ticker, exc,
        )
        return []
    except Exception as exc:
        logger.error("controversy_monitor: Finnhub fetch error for %s: %s", ticker, exc)
        return []


async def _fetch_gdelt_news(company_name: str, days_back: int) -> list[dict]:
    """
    Fetch recent news articles mentioning `company_name` from the GDELT Doc API.

    GDELT is a free, no-key API covering global news. We request artlist mode
    which returns a JSON array of article metadata.

    Returns a list of article dicts with keys: title, seendate, domain, url, language.
    """
    # GDELT date filter: last N days using SMOOTHTONE mode isn't needed; timespan works
    # Use timespan parameter: e.g. "90d" for 90 days
    timespan = f"{days_back}d"
    query_encoded = quote(f'"{company_name}"')

    url = (
        f"{_GDELT_BASE}"
        f"?query={query_encoded}"
        f"&mode=artlist"
        f"&format=json"
        f"&maxrecords=50"
        f"&timespan={timespan}"
        f"&sort=DateDesc"
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code == 200:
                # GDELT sometimes returns empty body for no results
                if not resp.content:
                    return []
                data = resp.json()
                articles = data.get("articles", [])
                if not isinstance(articles, list):
                    return []
                logger.debug(
                    "controversy_monitor: GDELT returned %d articles for '%s'",
                    len(articles), company_name,
                )
                return articles
            else:
                logger.warning(
                    "controversy_monitor: GDELT HTTP %d for '%s'",
                    resp.status_code, company_name,
                )
                return []
    except Exception as exc:
        logger.error("controversy_monitor: GDELT fetch error for '%s': %s", company_name, exc)
        return []


# ---------------------------------------------------------------------------
# Article normalisation
# ---------------------------------------------------------------------------

def _normalise_finnhub_article(article: dict, ticker: str, company_name: Optional[str]) -> Optional[ControversySignal]:
    """Convert a Finnhub article dict to a ControversySignal, or None if not controversial."""
    headline = article.get("headline", "")
    summary = article.get("summary", "")
    combined_text = f"{headline} {summary}"

    ctype, severity, keywords = _classify_controversy(combined_text)
    if not ctype:
        return None

    ts = article.get("datetime")
    pub_date: Optional[datetime] = None
    if ts:
        try:
            pub_date = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (ValueError, OSError):
            pass

    return ControversySignal(
        ticker=ticker,
        company_name=company_name,
        controversy_type=ctype,
        severity=severity,
        headline=headline,
        source=article.get("source", "Finnhub"),
        published_date=pub_date,
        url=article.get("url"),
        keywords_matched=keywords,
    )


def _normalise_gdelt_article(article: dict, ticker: str, company_name: Optional[str]) -> Optional[ControversySignal]:
    """Convert a GDELT article dict to a ControversySignal, or None if not controversial."""
    title = article.get("title", "")
    # GDELT doesn't always have a body — use title only
    ctype, severity, keywords = _classify_controversy(title)
    if not ctype:
        return None

    seen_date_str = article.get("seendate", "")
    pub_date: Optional[datetime] = None
    if seen_date_str:
        # GDELT format: "20240115T120000Z" or "20240115"
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d"):
            try:
                pub_date = datetime.strptime(seen_date_str[:15], fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue

    return ControversySignal(
        ticker=ticker,
        company_name=company_name,
        controversy_type=ctype,
        severity=severity,
        headline=title,
        source=article.get("domain", "GDELT"),
        published_date=pub_date,
        url=article.get("url"),
        keywords_matched=keywords,
    )


# ---------------------------------------------------------------------------
# Score and trend computation
# ---------------------------------------------------------------------------

def _compute_controversy_score(signals: list[ControversySignal]) -> float:
    """
    Compute a 0–100 controversy score.

    Weighting:
      - high severity: 10 points each (capped contribution)
      - medium severity: 4 points each
      - low severity: 1 point each
    Score is then log-scaled to 0–100 and capped.
    """
    raw = sum(
        10 if s.severity == "high" else (4 if s.severity == "medium" else 1)
        for s in signals
    )
    if raw == 0:
        return 0.0
    # log-scale: raw=100 → ~100, raw=10 → ~46, raw=1 → ~0
    score = min(100.0, math.log1p(raw) / math.log1p(100) * 100)
    return round(score, 2)


def _compute_trend(
    signals: list[ControversySignal],
    lookback_days: int,
) -> str:
    """
    Compare controversy density in the first half vs second half of the window.
    Returns "worsening", "stable", or "improving".
    """
    if not signals or lookback_days < 2:
        return "stable"

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days // 2)

    recent_signals = [
        s for s in signals
        if s.published_date and s.published_date >= cutoff
    ]
    older_signals = [
        s for s in signals
        if s.published_date and s.published_date < cutoff
    ]

    recent_count = len(recent_signals)
    older_count = len(older_signals)

    if recent_count > older_count * 1.5 + 1:
        return "worsening"
    elif older_count > recent_count * 1.5 + 1:
        return "improving"
    return "stable"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def get_controversy_profile(
    ticker: str,
    days_back: int = 90,
    company_name: Optional[str] = None,
    finnhub_api_key: Optional[str] = None,
) -> ControversyProfile:
    """
    Fetch and classify controversy signals for `ticker` from all available
    news sources, then return a ControversyProfile.

    Args:
        ticker: Company ticker symbol (e.g. "XOM").
        days_back: Lookback window in calendar days.
        company_name: Human-readable company name for GDELT queries. If None,
                      ticker is used as a fallback.
        finnhub_api_key: Finnhub API key. If None, Finnhub is skipped.

    Returns:
        ControversyProfile with computed score, trend, and signal breakdown.
    """
    if company_name is None:
        company_name = ticker

    # Fetch from all sources in parallel
    finnhub_raw, gdelt_raw = await _gather_all_news(
        ticker=ticker,
        company_name=company_name,
        days_back=days_back,
        finnhub_api_key=finnhub_api_key,
    )

    # Classify and filter
    signals: list[ControversySignal] = []

    for article in finnhub_raw:
        sig = _normalise_finnhub_article(article, ticker, company_name)
        if sig is not None:
            signals.append(sig)

    for article in gdelt_raw:
        sig = _normalise_gdelt_article(article, ticker, company_name)
        if sig is not None:
            signals.append(sig)

    # Deduplicate by headline (case-insensitive)
    seen_headlines: set[str] = set()
    deduped: list[ControversySignal] = []
    for sig in signals:
        key = sig.headline.lower().strip()
        if key not in seen_headlines:
            seen_headlines.add(key)
            deduped.append(sig)

    # Sort by date descending
    deduped.sort(
        key=lambda s: s.published_date or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    # Aggregate counts
    high_count = sum(1 for s in deduped if s.severity == "high")
    medium_count = sum(1 for s in deduped if s.severity == "medium")
    low_count = sum(1 for s in deduped if s.severity == "low")

    type_counts: dict[str, int] = {}
    for sig in deduped:
        type_counts[sig.controversy_type] = type_counts.get(sig.controversy_type, 0) + 1

    score = _compute_controversy_score(deduped)
    trend = _compute_trend(deduped, days_back)

    logger.info(
        "controversy_monitor: %s total=%d high=%d medium=%d low=%d score=%.1f trend=%s",
        ticker, len(deduped), high_count, medium_count, low_count, score, trend,
    )

    return ControversyProfile(
        ticker=ticker,
        total_controversies=len(deduped),
        high_severity=high_count,
        medium_severity=medium_count,
        low_severity=low_count,
        controversy_types=type_counts,
        recent_signals=deduped[:10],
        controversy_score=score,
        trend=trend,
        lookback_days=days_back,
    )


async def _gather_all_news(
    ticker: str,
    company_name: str,
    days_back: int,
    finnhub_api_key: Optional[str],
) -> tuple[list[dict], list[dict]]:
    """Fetch from Finnhub and GDELT in parallel."""
    import asyncio

    finnhub_task = asyncio.create_task(
        _fetch_finnhub_news(ticker, days_back, finnhub_api_key)
    )
    gdelt_task = asyncio.create_task(
        _fetch_gdelt_news(company_name, days_back)
    )

    finnhub_raw, gdelt_raw = await asyncio.gather(
        finnhub_task, gdelt_task, return_exceptions=False
    )
    return finnhub_raw, gdelt_raw
