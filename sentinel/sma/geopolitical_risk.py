"""Geopolitical risk scoring via GDELT + Claude Haiku synthesis.

Entry points:
  get_country_risk(country, lookback_days) -> CountryRiskScore
  get_geopolitical_dashboard(countries)    -> GeopoliticalDashboard
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# -- Constants ----------------------------------------------------------------

GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_COUNTRIES = [
    "United States", "China", "Russia", "Iran",
    "Saudi Arabia", "Ukraine", "Israel", "Taiwan",
]

COUNTRY_THEMES = {
    "conflict": "theme:CRISISLEX_CRISISLEXREC",
    "sanctions": "theme:ECON_BOYCOTT",
    "political_instability": "theme:WB_POLITICAL_STABILITY",
    "elections": "theme:ELECTIONS",
    "protests": "theme:PROTEST",
    "military": "theme:MILITARY",
}

_HEADERS = {"User-Agent": "SENTINEL/1.0 (research)"}
_TIMEOUT = 20.0

# -- Models -------------------------------------------------------------------


class GeoRiskEvent(BaseModel):
    title: str
    source: str
    date: str
    theme: str
    url: str


class GeoRiskTimeline(BaseModel):
    date: str
    event_volume: int
    avg_tone: Optional[float] = None


class CountryRiskScore(BaseModel):
    country: str
    overall_score: float
    conflict_score: float
    political_score: float
    economic_score: float
    trend: str
    recent_events: list[GeoRiskEvent]
    timeline_30d: list[GeoRiskTimeline]
    narrative: str
    top_themes: list[str]
    as_of: str
    warnings: list[str] = Field(default_factory=list)


class GeopoliticalDashboard(BaseModel):
    regions: list[CountryRiskScore]
    global_risk_index: float
    highest_risk: str
    lowest_risk: str
    key_flashpoints: list[str]
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# -- Tone → score ------------------------------------------------------------


def _tone_to_score(tone: float) -> float:
    """Map GDELT average tone to a 0-10 risk score (lower tone = higher risk)."""
    if tone >= 0.0:
        return 1.0
    if tone >= -2.0:
        return 2.0 + ((-tone) / 2.0)          # 2-3
    if tone >= -5.0:
        return 4.0 + ((-tone - 2.0) / 3.0) * 2.0  # 4-6
    if tone >= -10.0:
        return 7.0 + ((-tone - 5.0) / 5.0)    # 7-8
    return min(10.0, 9.0 + ((-tone - 10.0) / 5.0))


# -- GDELT HTTP helpers -------------------------------------------------------


async def _fetch_articles(
    client: "httpx.AsyncClient",  # type: ignore[name-defined]
    country: str,
    theme_key: str,
    timespan: str = "1month",
    maxrecords: int = 25,
) -> tuple[list[GeoRiskEvent], list[str]]:
    """Fetch article list from GDELT DOC API for country + theme."""
    theme_filter = COUNTRY_THEMES.get(theme_key, "")
    query = f'"{country}" {theme_filter}'
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": maxrecords,
        "format": "json",
        "timespan": timespan,
    }
    events: list[GeoRiskEvent] = []
    warnings: list[str] = []
    try:
        resp = await client.get(GDELT_DOC_BASE, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles") or []
        for art in articles:
            title = art.get("title") or ""
            url = art.get("url") or ""
            domain = art.get("domain") or ""
            seen = art.get("seendate") or ""
            if title and url:
                events.append(GeoRiskEvent(
                    title=title[:200],
                    source=domain,
                    date=seen,
                    theme=theme_key,
                    url=url,
                ))
        logger.debug("gdelt_articles", country=country, theme=theme_key, count=len(events))
    except Exception as exc:
        msg = f"GDELT artlist failed [{country}/{theme_key}]: {exc}"
        logger.warning(msg)
        warnings.append(msg)
    return events, warnings


async def _fetch_timeline(
    client: "httpx.AsyncClient",  # type: ignore[name-defined]
    country: str,
    timespan: str = "3months",
) -> tuple[list[GeoRiskTimeline], list[str]]:
    """Fetch event-volume timeline from GDELT DOC API (timelinevolume mode)."""
    query = f'"{country}" conflict OR sanctions OR military OR protest'
    params = {
        "query": query,
        "mode": "timelinevolume",
        "timespan": timespan,
        "format": "json",
    }
    timeline: list[GeoRiskTimeline] = []
    warnings: list[str] = []
    try:
        resp = await client.get(GDELT_DOC_BASE, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # GDELT returns {"timeline": [{"data": [{"date": ..., "value": ...}]}]}
        series_list = data.get("timeline") or []
        for series in series_list:
            for pt in series.get("data") or []:
                d = pt.get("date") or ""
                v = pt.get("value")
                if d and v is not None:
                    timeline.append(GeoRiskTimeline(
                        date=str(d),
                        event_volume=int(v),
                    ))
        logger.debug("gdelt_timeline", country=country, points=len(timeline))
    except Exception as exc:
        msg = f"GDELT timeline failed [{country}]: {exc}"
        logger.warning(msg)
        warnings.append(msg)
    return timeline, warnings


async def _fetch_tone(
    client: "httpx.AsyncClient",  # type: ignore[name-defined]
    country: str,
    timespan: str = "1month",
) -> tuple[Optional[float], list[str]]:
    """Fetch average tone from GDELT tonechart mode."""
    query = f'"{country}" conflict OR sanctions OR military OR protest'
    params = {
        "query": query,
        "mode": "tonechart",
        "timespan": timespan,
        "format": "json",
    }
    warnings: list[str] = []
    try:
        resp = await client.get(GDELT_DOC_BASE, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        tones: list[float] = []
        for series in data.get("timeline") or []:
            for pt in series.get("data") or []:
                val = pt.get("value")
                if val is not None:
                    tones.append(float(val))
        if tones:
            avg = sum(tones) / len(tones)
            logger.debug("gdelt_tone", country=country, avg_tone=round(avg, 3))
            return avg, warnings
    except Exception as exc:
        msg = f"GDELT tone failed [{country}]: {exc}"
        logger.warning(msg)
        warnings.append(msg)
    return None, warnings


# -- Theme scores & trend ----------------------------------------------------


def _derive_theme_scores(
    events_by_theme: dict[str, list[GeoRiskEvent]],
    base_tone_score: float,
) -> tuple[float, float, float]:
    """Compute conflict, political, economic sub-scores from event counts + base tone."""
    def _count(*keys: str) -> int:
        return sum(len(events_by_theme.get(k, [])) for k in keys)

    conflict_events = _count("conflict", "military")
    political_events = _count("political_instability", "elections", "protests")
    economic_events = _count("sanctions")

    def _blend(event_count: int) -> float:
        # Scale event count: 0 events → base, 20+ events → add up to +3
        boost = min(event_count / 20.0, 1.0) * 3.0
        return round(min(base_tone_score + boost, 10.0), 2)

    return (
        _blend(conflict_events),
        _blend(political_events),
        _blend(economic_events),
    )


def _derive_trend(timeline: list[GeoRiskTimeline]) -> str:
    """Compare first-half vs second-half volume to determine rising/falling/stable."""
    if len(timeline) < 4:
        return "unknown"
    volumes = [t.event_volume for t in timeline]
    mid = len(volumes) // 2
    first_half_avg = sum(volumes[:mid]) / mid
    second_half_avg = sum(volumes[mid:]) / (len(volumes) - mid)
    if first_half_avg == 0:
        return "stable"
    change_pct = (second_half_avg - first_half_avg) / first_half_avg
    if change_pct > 0.15:
        return "rising"
    if change_pct < -0.15:
        return "falling"
    return "stable"


# -- Claude Haiku synthesis ---------------------------------------------------


async def _claude_narrative(country: str, events: list[GeoRiskEvent], score: float) -> str:
    """Generate 3-sentence risk narrative via Claude Haiku. Falls back to template."""
    import anthropic  # lazy import

    events_text = "\n".join(
        f"- [{e.theme}] {e.title} ({e.source}, {e.date[:10] if e.date else 'n/a'})"
        for e in events[:20]
    ) or "No specific events found."

    prompt = (
        f"Summarize the geopolitical risk for {country} based on these recent events:\n"
        f"{events_text}\n"
        f"Current risk score: {score:.1f}/10.\n"
        f"Provide: 1) Key risk factors 2) Trend (rising/stable/falling) "
        f"3) Investment implications. Be concise (3 sentences max)."
    )
    try:
        client = anthropic.AsyncAnthropic()
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        logger.warning("claude_narrative_failed", country=country, error=str(exc))
        return (
            f"Geopolitical risk for {country}: score {score:.1f}/10. "
            f"{len(events)} recent events detected. Data from GDELT."
        )


async def _claude_flashpoints(regions: list[CountryRiskScore]) -> list[str]:
    """Identify top 3 global flashpoints via Claude Haiku. Falls back to top-3 countries."""
    import anthropic  # lazy import

    summary = "\n".join(
        f"- {r.country}: score={r.overall_score:.1f}, trend={r.trend}, "
        f"top_themes={','.join(r.top_themes[:3])}"
        for r in sorted(regions, key=lambda x: x.overall_score, reverse=True)[:8]
    )
    prompt = (
        f"Given these country risk scores:\n{summary}\n"
        "Identify the top 3 global geopolitical flashpoints as concise one-line phrases. "
        "Return exactly 3 bullet points, one per line, starting with '- '."
    )
    try:
        client = anthropic.AsyncAnthropic()
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = [
            ln.lstrip("- ").strip()
            for ln in msg.content[0].text.strip().splitlines()
            if ln.strip().startswith("-")
        ]
        return lines[:3] if lines else _fallback_flashpoints(regions)
    except Exception as exc:
        logger.warning("claude_flashpoints_failed", error=str(exc))
        return _fallback_flashpoints(regions)


def _fallback_flashpoints(regions: list[CountryRiskScore]) -> list[str]:
    top = sorted(regions, key=lambda x: x.overall_score, reverse=True)[:3]
    return [f"Elevated risk: {r.country} (score {r.overall_score:.1f})" for r in top]


# -- Entry points -------------------------------------------------------------


async def get_country_risk(
    country: str,
    lookback_days: int = 30,
) -> CountryRiskScore:
    """Score geopolitical risk for a single country using GDELT + Claude Haiku."""
    import httpx  # lazy import

    timespan = f"{lookback_days}days" if lookback_days <= 90 else "3months"
    all_warnings: list[str] = []
    all_events: list[GeoRiskEvent] = []
    events_by_theme: dict[str, list[GeoRiskEvent]] = {}

    async with httpx.AsyncClient() as client:
        # Fetch articles for all themes and timeline/tone concurrently
        theme_tasks = {
            theme: _fetch_articles(client, country, theme, timespan)
            for theme in COUNTRY_THEMES
        }
        timeline_task = _fetch_timeline(client, country, "3months")
        tone_task = _fetch_tone(client, country, timespan)

        results = await asyncio.gather(
            *theme_tasks.values(),
            timeline_task,
            tone_task,
            return_exceptions=False,
        )

    # Unpack theme results
    theme_keys = list(theme_tasks.keys())
    for i, key in enumerate(theme_keys):
        evts, warns = results[i]
        events_by_theme[key] = evts
        all_events.extend(evts)
        all_warnings.extend(warns)

    timeline_result: tuple[list[GeoRiskTimeline], list[str]] = results[len(theme_keys)]
    tone_result: tuple[Optional[float], list[str]] = results[len(theme_keys) + 1]

    timeline, tl_warns = timeline_result
    avg_tone, tone_warns = tone_result
    all_warnings.extend(tl_warns)
    all_warnings.extend(tone_warns)

    # Compute base risk score from tone
    if avg_tone is not None:
        base_score = _tone_to_score(avg_tone)
    elif all_events:
        # Fallback: event volume heuristic
        base_score = min(1.0 + len(all_events) / 10.0, 8.0)
    else:
        base_score = 1.0
        all_warnings.append(f"Insufficient GDELT data for {country}; score set to 1.0.")

    conflict_score, political_score, economic_score = _derive_theme_scores(
        events_by_theme, base_score
    )
    overall_score = round(
        (base_score * 0.5 + conflict_score * 0.25 + political_score * 0.15 + economic_score * 0.10),
        2,
    )

    trend = _derive_trend(timeline)

    # Top themes by event count
    top_themes = sorted(
        COUNTRY_THEMES.keys(),
        key=lambda k: len(events_by_theme.get(k, [])),
        reverse=True,
    )[:4]

    # Deduplicate and limit events
    seen_urls: set[str] = set()
    unique_events: list[GeoRiskEvent] = []
    for evt in all_events:
        if evt.url not in seen_urls:
            seen_urls.add(evt.url)
            unique_events.append(evt)
    unique_events = unique_events[:30]

    # Timeline: last 30 days only
    timeline_30d = timeline[-30:] if len(timeline) > 30 else timeline

    narrative = await _claude_narrative(country, unique_events, overall_score)

    as_of = datetime.now(timezone.utc).isoformat()
    logger.info(
        "country_risk_scored",
        country=country,
        overall=overall_score,
        trend=trend,
        events=len(unique_events),
    )
    return CountryRiskScore(
        country=country,
        overall_score=overall_score,
        conflict_score=conflict_score,
        political_score=political_score,
        economic_score=economic_score,
        trend=trend,
        recent_events=unique_events,
        timeline_30d=timeline_30d,
        narrative=narrative,
        top_themes=top_themes,
        as_of=as_of,
        warnings=all_warnings,
    )


async def get_geopolitical_dashboard(
    countries: list[str] | None = None,
) -> GeopoliticalDashboard:
    """Multi-country geopolitical risk dashboard.

    Fetches all countries in parallel via asyncio.gather. Calls Claude Haiku
    once to identify the top global flashpoints.

    Default countries: US, CN, RU, IR, SA, UA, IL, TW.
    """
    targets = countries if countries is not None else DEFAULT_COUNTRIES

    logger.info("dashboard_start", countries=targets)
    scores: list[CountryRiskScore] = list(
        await asyncio.gather(*[get_country_risk(c) for c in targets])
    )

    valid = [s for s in scores if s.overall_score > 0]
    global_risk = round(sum(s.overall_score for s in valid) / len(valid), 2) if valid else 0.0
    highest = max(scores, key=lambda s: s.overall_score).country
    lowest = min(scores, key=lambda s: s.overall_score).country

    flashpoints = await _claude_flashpoints(scores)
    all_warnings = [w for s in scores for w in s.warnings]

    as_of = datetime.now(timezone.utc).isoformat()
    logger.info(
        "dashboard_complete",
        countries=len(scores),
        global_risk=global_risk,
        highest=highest,
    )
    return GeopoliticalDashboard(
        regions=scores,
        global_risk_index=global_risk,
        highest_risk=highest,
        lowest_risk=lowest,
        key_flashpoints=flashpoints,
        as_of=as_of,
        warnings=all_warnings,
    )
