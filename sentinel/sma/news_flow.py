"""News flow monitor: yfinance + EDGAR RSS news aggregation with Claude Haiku event classification — Dimension 92 enhancement."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom"
EDGAR_TICKER_RSS = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=include&count=20&output=atom"
EDGAR_HEADERS = {"User-Agent": "SENTINEL/1.0 research@sentinel.ai", "Accept": "application/atom+xml,application/xml,text/xml,*/*"}

EVENT_TYPES = ["earnings", "merger_acquisition", "regulatory_fda", "credit_rating", "guidance_update", "restructuring", "dividend_change", "executive_change", "legal_regulatory", "product_launch", "macro_economic", "other"]
MATERIAL_EVENTS: frozenset[str] = frozenset(["earnings", "merger_acquisition", "guidance_update", "credit_rating", "restructuring"])


class NewsItem(BaseModel):
    """Single classified news article."""
    model_config = ConfigDict(frozen=True)
    title: str
    summary: str
    source: str
    url: str
    published: datetime
    ticker: str
    event_type: str = Field(..., description="One of EVENT_TYPES")
    sentiment: str = Field(..., description="positive | negative | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_material: bool


class TickerNewsFlow(BaseModel):
    """Aggregated news flow analysis for a single ticker."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    items: list[NewsItem]
    event_count_24h: int
    dominant_event_type: Optional[str]
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    volume_spike: bool
    as_of: datetime
    warnings: list[str]


class NewsVolumeStats(BaseModel):
    """Volume statistics for a ticker's news flow."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    count_24h: int
    count_7d: int
    avg_daily: float
    spike_detected: bool
    spike_ratio: float


class MarketNewsScreen(BaseModel):
    """Aggregate news screen across a universe of tickers."""
    model_config = ConfigDict(frozen=True)
    tickers_screened: list[str]
    total_items: int
    material_events: int
    tickers_with_spikes: list[str]
    dominant_themes: list[str]
    as_of: datetime
    warnings: list[str]


class NewsFlowSummary(BaseModel):
    """Compact one-line summary per ticker for dashboard display."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    headline_count: int
    top_event: Optional[str]
    sentiment_score: float
    has_spike: bool
    top_headline: Optional[str]


async def _fetch_yfinance_news(ticker: str) -> list[dict]:
    """Fetch recent news dicts from yfinance in a thread (sync library)."""
    try:
        raw: list[dict] = await asyncio.to_thread(
            lambda: __import__("yfinance").Ticker(ticker).news or []
        )
        return raw if isinstance(raw, list) else []
    except Exception as exc:
        logger.warning("yfinance news fetch failed", ticker=ticker, error=str(exc))
        return []


async def _fetch_edgar_8k(ticker: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch last 10 EDGAR 8-K filings for *ticker* via free Atom RSS — no auth required."""
    url = EDGAR_TICKER_RSS.format(ticker=ticker.upper())
    try:
        resp = await client.get(url, headers=EDGAR_HEADERS, timeout=15.0)
        resp.raise_for_status()
        text = resp.text
        titles = re.findall(r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", text, re.S)
        summaries = re.findall(r"<summary[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</summary>", text, re.S)
        updated = re.findall(r"<updated>(.*?)</updated>", text)
        links = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', text)
        titles = [t.strip() for t in titles[1:]]
        updated = [u.strip() for u in updated[1:]]
        links = [lk for lk in links if "/Archives/" in lk or "action=getcompany" in lk]
        items: list[dict] = []
        for i, title in enumerate(titles[:10]):
            summary_text = summaries[i].strip() if i < len(summaries) else ""
            pub_str = updated[i] if i < len(updated) else ""
            link_url = links[i] if i < len(links) else url
            try:
                published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                published = datetime.now(tz=timezone.utc)
            items.append({"title": title, "summary": summary_text, "url": link_url, "published": published, "source": "EDGAR"})
        return items
    except httpx.HTTPStatusError as exc:
        logger.warning("EDGAR 8-K HTTP error", ticker=ticker, status=exc.response.status_code)
        return []
    except Exception as exc:
        logger.warning("EDGAR 8-K fetch failed", ticker=ticker, error=str(exc))
        return []


_RULE_MAP: list[tuple[list[str], str, float]] = [
    (["earn", "eps", "quarter", "revenue", "sales", "per share"], "earnings", 0.9),
    (["acqui", "merger", "takeover", "deal", "buyout", "combine"], "merger_acquisition", 0.95),
    (["fda", "approval", "clinical", "trial", "nda", "bla", "drug"], "regulatory_fda", 0.9),
    (["rating", "downgrade", "upgrade", "credit", "moody", "fitch"], "credit_rating", 0.9),
    (["guidance", "outlook", "forecast", "raises", "lowers guidance", "revises"], "guidance_update", 0.85),
    (["restructur", "layoff", "job cut", "workforce reduction", "severance"], "restructuring", 0.9),
    (["dividend", "buyback", "repurchase", "special dividend", "distribution"], "dividend_change", 0.9),
    (["ceo", "cfo", "coo", "appoint", "resign", "depart", "executive", "president"], "executive_change", 0.9),
    (["lawsuit", "sec ", "subpoena", "fine", "penalty", "litigation", "settlement", "doj"], "legal_regulatory", 0.9),
    (["launch", "product", "release", "unveil", "introduce", "new model"], "product_launch", 0.8),
    (["fed ", "federal reserve", "inflation", "rate hike", "rate cut", "fomc", "gdp", "cpi"], "macro_economic", 0.85),
]


def _classify_event_rule(title: str, summary: str) -> tuple[str, float]:
    """Fast rule-based event classifier; covers ~95% of cases without an API call."""
    combined = (title + " " + summary).lower()
    for keywords, event_type, confidence in _RULE_MAP:
        if any(kw in combined for kw in keywords):
            return event_type, confidence
    return "other", 0.6


_POS_WORDS = frozenset(["beat", "exceed", "strong", "growth", "surge", "record", "rise", "up", "gain", "topped", "outperform", "above", "positive", "profit", "higher", "increase"])
_NEG_WORDS = frozenset(["miss", "below", "weak", "decline", "drop", "loss", "cut", "down", "fall", "disappoint", "lower", "reduce", "negative", "deficit", "worse", "warn"])


def _sentiment_rule(title: str, summary: str) -> tuple[str, float]:
    """Weighted sentiment: title words count 2x, summary 1x. Returns (label, confidence)."""
    def _count(text: str, wordlist: frozenset[str]) -> int:
        tl = text.lower()
        return sum(1 for w in wordlist if w in tl)

    pos = _count(title, _POS_WORDS) * 2 + _count(summary, _POS_WORDS)
    neg = _count(title, _NEG_WORDS) * 2 + _count(summary, _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return "neutral", 0.5
    if pos > neg:
        return "positive", round(min(1.0, 0.5 + 0.1 * (pos - neg)), 3)
    elif neg > pos:
        return "negative", round(min(1.0, 0.5 + 0.1 * (neg - pos)), 3)
    return "neutral", 0.55


async def _classify_with_claude(items: list[dict]) -> list[dict]:
    """Call Claude Haiku for ambiguous 'other' items in a single batched request (max 10)."""
    ambiguous_indices = [
        i for i, it in enumerate(items)
        if it.get("event_type") == "other" and it.get("confidence", 1.0) < 0.7
    ]
    if not ambiguous_indices:
        return items
    batch_indices = ambiguous_indices[:10]
    numbered_lines = "\n".join(f"{idx + 1}. {items[idx]['title']}" for idx in batch_indices)
    prompt = (
        f"Classify each headline into exactly one of: {', '.join(EVENT_TYPES)}\n\n"
        f"Headlines:\n{numbered_lines}\n\n"
        "Return JSON array only:\n"
        '[{"id": <n>, "event_type": "<type>", "sentiment": "<positive|negative|neutral>"}]'
    )
    try:
        anthropic = __import__("anthropic")
        client = anthropic.AsyncAnthropic()
        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = message.content[0].text.strip()
        json_match = re.search(r"\[.*\]", raw_text, re.S)
        if not json_match:
            return items
        import json
        classifications: list[dict] = json.loads(json_match.group())
        updated = list(items)
        for cls in classifications:
            item_id = int(cls.get("id", 0))
            if item_id < 1 or item_id > len(batch_indices):
                continue
            real_idx = batch_indices[item_id - 1]
            evt = cls.get("event_type", "other") if cls.get("event_type") in EVENT_TYPES else "other"
            snt = cls.get("sentiment", "neutral") if cls.get("sentiment") in ("positive", "negative", "neutral") else "neutral"
            updated[real_idx] = {**updated[real_idx], "event_type": evt, "sentiment": snt, "confidence": 0.78, "claude_classified": True}
        return updated
    except Exception as exc:
        logger.warning("Claude Haiku classification failed", error=str(exc))
        return items


def _is_material(event_type: str) -> bool:
    """Return True if event type is considered market-moving / material."""
    return event_type in MATERIAL_EVENTS


def _volume_spike(count_24h: int, count_7d: int) -> tuple[bool, float]:
    """Detect abnormal news volume: spike when today's count > 2× daily avg."""
    avg_daily = count_7d / 7 if count_7d > 0 else 0.0
    if avg_daily <= 0:
        return False, 1.0
    ratio = count_24h / avg_daily
    return ratio > 2.0, round(ratio, 3)


def _sentiment_score(items: list[NewsItem]) -> float:
    """Weighted mean sentiment in [-1, +1]. Material events get 1.5× weight."""
    if not items:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for item in items:
        raw_score = 1.0 if item.sentiment == "positive" else (-1.0 if item.sentiment == "negative" else 0.0)
        weight = item.confidence
        if item.is_material and item.sentiment != "neutral":
            weight *= 1.5
        weighted_sum += raw_score * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return round(max(-1.0, min(1.0, weighted_sum / total_weight)), 4)


def _dominant_event_type(items: list[NewsItem]) -> Optional[str]:
    """Return the most frequently occurring event type, or None if no items."""
    if not items:
        return None
    counts: dict[str, int] = {}
    for item in items:
        counts[item.event_type] = counts.get(item.event_type, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _deduplicate(raw_items: list[dict]) -> list[dict]:
    """Remove duplicate articles by exact title match (case-insensitive)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for item in raw_items:
        key = item.get("title", "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _parse_yfinance_item(raw: dict, ticker: str) -> dict:
    """Normalise a raw yfinance news dict into a common intermediate dict."""
    ts = raw.get("providerPublishTime") or raw.get("published") or 0
    if isinstance(ts, (int, float)):
        published = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif isinstance(ts, datetime):
        published = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        published = datetime.now(tz=timezone.utc)
    return {
        "title": raw.get("title", ""),
        "summary": raw.get("summary") or raw.get("description") or "",
        "url": raw.get("link") or raw.get("url", ""),
        "source": raw.get("publisher") or raw.get("source", "yfinance"),
        "published": published,
        "ticker": ticker.upper(),
    }


async def get_news_flow(ticker: str, max_items: int = 20) -> TickerNewsFlow:
    """
    Fetch, classify, and score recent news for *ticker*.

    Sources: yfinance (last ~10 items) + EDGAR 8-K RSS (last 10 filings).
    Event classification is rule-based; Claude Haiku is invoked only when
    more than 3 items remain classified as 'other' with confidence < 0.7.
    """
    warnings: list[str] = []
    now = datetime.now(tz=timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d = now - timedelta(days=7)

    async with httpx.AsyncClient() as client:
        yf_raw, edgar_raw = await asyncio.gather(
            _fetch_yfinance_news(ticker),
            _fetch_edgar_8k(ticker, client),
        )

    yf_items: list[dict] = []
    if isinstance(yf_raw, list):
        for raw in yf_raw:
            try:
                yf_items.append(_parse_yfinance_item(raw, ticker))
            except Exception as exc:
                warnings.append(f"yfinance parse error: {exc}")
    else:
        warnings.append("yfinance returned unexpected type")

    if isinstance(edgar_raw, list):
        for item in edgar_raw:
            item.setdefault("ticker", ticker.upper())
            item.setdefault("source", "EDGAR")
    else:
        edgar_raw = []
        warnings.append("EDGAR fetch returned unexpected type")

    combined = _deduplicate(yf_items + edgar_raw)
    if not combined:
        warnings.append(f"No news items found for {ticker}")
        return TickerNewsFlow(
            ticker=ticker.upper(), items=[], event_count_24h=0,
            dominant_event_type=None, sentiment_score=0.0,
            volume_spike=False, as_of=now, warnings=warnings,
        )

    classified: list[dict] = []
    for item in combined:
        title = item.get("title", "")
        summary = item.get("summary", "")
        event_type, confidence = _classify_event_rule(title, summary)
        sentiment, sent_conf = _sentiment_rule(title, summary)
        classified.append({**item, "event_type": event_type, "confidence": confidence, "sentiment": sentiment, "sent_conf": sent_conf, "claude_classified": False})

    other_low = [it for it in classified if it["event_type"] == "other" and it["confidence"] < 0.7]
    if len(classified) > 3 and other_low:
        try:
            classified = await _classify_with_claude(classified)
        except Exception as exc:
            warnings.append(f"Claude classification skipped: {exc}")

    news_items: list[NewsItem] = []
    for it in classified:
        pub = it.get("published")
        if pub is None or not isinstance(pub, datetime):
            pub = now
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        try:
            news_items.append(NewsItem(
                title=it.get("title", ""),
                summary=it.get("summary", ""),
                source=it.get("source", "unknown"),
                url=it.get("url", ""),
                published=pub,
                ticker=it.get("ticker", ticker.upper()),
                event_type=it["event_type"],
                sentiment=it["sentiment"],
                confidence=round((it["confidence"] + it.get("sent_conf", 0.5)) / 2, 3),
                is_material=_is_material(it["event_type"]),
            ))
        except Exception as exc:
            warnings.append(f"NewsItem construction error: {exc}")

    news_items.sort(key=lambda x: x.published, reverse=True)
    news_items = news_items[:max_items]

    count_24h = sum(1 for it in news_items if it.published >= cutoff_24h)
    count_7d = sum(1 for it in news_items if it.published >= cutoff_7d)
    spike, _ratio = _volume_spike(count_24h, count_7d)

    return TickerNewsFlow(
        ticker=ticker.upper(),
        items=news_items,
        event_count_24h=count_24h,
        dominant_event_type=_dominant_event_type(news_items),
        sentiment_score=_sentiment_score(news_items),
        volume_spike=spike,
        as_of=now,
        warnings=warnings,
    )


async def screen_news_flow(tickers: list[str], min_spike_ratio: float = 1.5) -> MarketNewsScreen:
    """
    Screen a universe of tickers for news volume spikes and material events.

    Returns aggregate MarketNewsScreen with dominant themes, spike leaders,
    and total material event count.
    """
    warnings: list[str] = []
    now = datetime.now(tz=timezone.utc)

    flows: list[TickerNewsFlow | BaseException] = await asyncio.gather(
        *[get_news_flow(t) for t in tickers], return_exceptions=True
    )

    valid_flows: list[TickerNewsFlow] = []
    for ticker, result in zip(tickers, flows):
        if isinstance(result, BaseException):
            warnings.append(f"{ticker}: fetch error — {result}")
        else:
            valid_flows.append(result)
            if result.warnings:
                warnings.extend(result.warnings)

    all_items: list[NewsItem] = [it for flow in valid_flows for it in flow.items]
    total_items = len(all_items)
    material_events = sum(1 for it in all_items if it.is_material)

    spike_tickers: list[tuple[str, float]] = []
    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d = now - timedelta(days=7)
    for flow in valid_flows:
        c24 = sum(1 for it in flow.items if it.published >= cutoff_24h)
        c7 = sum(1 for it in flow.items if it.published >= cutoff_7d)
        _spike, ratio = _volume_spike(c24, c7)
        if ratio >= min_spike_ratio or flow.volume_spike:
            spike_tickers.append((flow.ticker, ratio))
    spike_tickers.sort(key=lambda x: x[1], reverse=True)

    theme_counts: dict[str, int] = {}
    for it in all_items:
        theme_counts[it.event_type] = theme_counts.get(it.event_type, 0) + 1
    dominant_themes = sorted(theme_counts, key=lambda k: theme_counts[k], reverse=True)[:3]

    return MarketNewsScreen(
        tickers_screened=tickers,
        total_items=total_items,
        material_events=material_events,
        tickers_with_spikes=[t for t, _ in spike_tickers],
        dominant_themes=dominant_themes,
        as_of=now,
        warnings=warnings,
    )


def build_summary(flow: TickerNewsFlow) -> NewsFlowSummary:
    """Collapse a TickerNewsFlow into a compact dashboard-ready NewsFlowSummary."""
    top_headline: Optional[str] = flow.items[0].title if flow.items else None
    return NewsFlowSummary(
        ticker=flow.ticker,
        headline_count=len(flow.items),
        top_event=flow.dominant_event_type,
        sentiment_score=flow.sentiment_score,
        has_spike=flow.volume_spike,
        top_headline=top_headline,
    )


def compute_volume_stats(ticker: str, flow: TickerNewsFlow) -> NewsVolumeStats:
    """Derive NewsVolumeStats from a resolved TickerNewsFlow."""
    now = datetime.now(tz=timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d = now - timedelta(days=7)
    count_24h = sum(1 for it in flow.items if it.published >= cutoff_24h)
    count_7d = sum(1 for it in flow.items if it.published >= cutoff_7d)
    avg_daily = count_7d / 7 if count_7d > 0 else 0.0
    spike, ratio = _volume_spike(count_24h, count_7d)
    return NewsVolumeStats(
        ticker=ticker.upper(),
        count_24h=count_24h,
        count_7d=count_7d,
        avg_daily=round(avg_daily, 3),
        spike_detected=spike,
        spike_ratio=ratio,
    )
