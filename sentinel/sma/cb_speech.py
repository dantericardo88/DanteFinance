"""
Central bank speech tone analysis — hawkish / dovish signal.

Dimension #45 in the SENTINEL competitive matrix (score 1 → target 9).

Data sources (all free):
  - Fed:  https://www.federalreserve.gov/json/ne-speeches.json
  - ECB:  https://www.ecb.europa.eu/rss/speeches.rss
  - BoE:  https://www.bankofengland.co.uk/rss/speeches

Tone scoring uses a dual-lexicon keyword approach.
FinBERT is applied to each speech summary via sentinel.sil.sentiment.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from datetime import date, datetime
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.core.logging import get_logger
from sentinel.sil.sentiment import score_sentiment

logger = get_logger(__name__)

# ── Keyword lexicons ──────────────────────────────────────────────────────────

HAWKISH_TERMS: list[str] = [
    "inflation", "tighten", "rate hike", "restrictive", "overshoot",
    "price stability", "above target", "vigilant", "determined", "firm",
    "higher for longer", "not yet", "premature", "persistent",
]

DOVISH_TERMS: list[str] = [
    "cut", "easing", "accommodative", "supportive", "below target",
    "transitory", "downside risk", "labor market", "growth concern",
    "patient", "gradual", "data dependent", "monitor", "flexible",
]

# ── HTTP settings ─────────────────────────────────────────────────────────────

_HEADERS = {"User-Agent": "SENTINEL/1.0 (research; contact@sentinel.dev)"}
_TIMEOUT = 15.0

FED_JSON_URL = "https://www.federalreserve.gov/json/ne-speeches.json"
ECB_RSS_URL = "https://www.ecb.europa.eu/rss/speeches.rss"
BOE_RSS_URL = "https://www.bankofengland.co.uk/rss/speeches"


# ── Pydantic models ───────────────────────────────────────────────────────────

class CentralBankSpeech(BaseModel):
    model_config = ConfigDict(frozen=True)

    bank: str                           # "FED" | "ECB" | "BOE" | "BOJ"
    speaker: str
    title: str
    speech_date: date
    url: str
    summary: Optional[str]             # first 500 chars of extracted text
    hawkish_score: float               # 0-1, hawkish hits / total signal hits
    dovish_score: float                # 0-1, dovish hits / total signal hits
    net_tone: float                    # hawkish_score - dovish_score
    tone_label: str                    # "very_hawkish"|"hawkish"|"neutral"|"dovish"|"very_dovish"
    finbert_sentiment: Optional[str]   # "positive"|"negative"|"neutral"
    finbert_confidence: Optional[float]


class CBSpeechSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    bank: str
    as_of: date
    recent_speeches: list[CentralBankSpeech]
    avg_net_tone: float                # rolling average of net_tone
    tone_trend: str                    # "hawkish_turn"|"dovish_turn"|"stable"
    policy_signal: str                 # "hike"|"hold"|"cut"
    key_phrases: list[str]             # most frequent hawkish/dovish terms found


# ── Tone scoring ──────────────────────────────────────────────────────────────

def score_tone(text: str) -> tuple[float, float, float, str]:
    """Count hawkish vs dovish terms (case-insensitive).

    Returns (hawkish_score, dovish_score, net_tone, label).
    Scores are proportional fractions of total signal hits (0-1).
    net_tone > 0.15 → hawkish; < -0.15 → dovish; else neutral.
    Very hawkish/dovish at ±0.35.
    """
    lower = text.lower()
    h_hits = sum(lower.count(t) for t in HAWKISH_TERMS)
    d_hits = sum(lower.count(t) for t in DOVISH_TERMS)
    total = h_hits + d_hits

    if total == 0:
        return 0.0, 0.0, 0.0, "neutral"

    h_score = h_hits / total
    d_score = d_hits / total
    net = round(h_score - d_score, 4)

    if net >= 0.35:
        label = "very_hawkish"
    elif net >= 0.15:
        label = "hawkish"
    elif net <= -0.35:
        label = "very_dovish"
    elif net <= -0.15:
        label = "dovish"
    else:
        label = "neutral"

    return round(h_score, 4), round(d_score, 4), net, label


def _extract_key_phrases(speeches: list[CentralBankSpeech]) -> list[str]:
    """Return the most frequently matched hawkish/dovish terms across all speeches."""
    counter: Counter[str] = Counter()
    all_terms = HAWKISH_TERMS + DOVISH_TERMS
    for sp in speeches:
        text = ((sp.summary or "") + " " + sp.title).lower()
        for term in all_terms:
            hits = text.count(term)
            if hits:
                counter[term] += hits
    return [term for term, _ in counter.most_common(10)]


def _compute_trend(speeches: list[CentralBankSpeech]) -> str:
    """Compare avg net_tone of the last 3 vs prior 3 speeches."""
    if len(speeches) < 4:
        return "stable"
    recent = speeches[:3]
    prior = speeches[3:6]
    avg_recent = sum(s.net_tone for s in recent) / len(recent)
    avg_prior = sum(s.net_tone for s in prior) / len(prior)
    delta = avg_recent - avg_prior
    if delta >= 0.08:
        return "hawkish_turn"
    if delta <= -0.08:
        return "dovish_turn"
    return "stable"


def _policy_signal(avg_net_tone: float) -> str:
    """Derive policy signal from average net tone."""
    if avg_net_tone >= 0.1:
        return "hike"
    if avg_net_tone <= -0.1:
        return "cut"
    return "hold"


# ── HTML text extraction ──────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
async def extract_speech_text(url: str, max_chars: int = 2000) -> Optional[str]:
    """Fetch a speech page and strip HTML tags with regex.

    Returns first max_chars of plain text, or None on failure.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.debug("extract_speech_text: fetch failed", url=url, error=str(exc))
        return None

    # Strip scripts and styles first
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] if text else None


# ── FinBERT enrichment ────────────────────────────────────────────────────────

async def analyze_speech(speech: CentralBankSpeech, fetch_full_text: bool = True) -> CentralBankSpeech:
    """Optionally fetch full text, re-score tone, and run FinBERT on summary."""
    summary = speech.summary

    if fetch_full_text and not summary:
        raw = await extract_speech_text(speech.url, max_chars=2000)
        summary = raw[:500] if raw else None

    # Re-score tone on summary + title if we now have text
    score_text = (summary or "") + " " + speech.title
    h_score, d_score, net, label = score_tone(score_text)

    # FinBERT — run on summary if available, else title
    finbert_label: Optional[str] = None
    finbert_conf: Optional[float] = None
    bert_text = summary or speech.title
    if bert_text:
        try:
            result = await score_sentiment(bert_text[:512])
            finbert_label = result.label
            finbert_conf = round(result.score, 4)
        except Exception as exc:
            logger.warning("FinBERT failed for speech", url=speech.url, error=str(exc))

    return speech.model_copy(update={
        "summary": summary,
        "hawkish_score": h_score,
        "dovish_score": d_score,
        "net_tone": net,
        "tone_label": label,
        "finbert_sentiment": finbert_label,
        "finbert_confidence": finbert_conf,
    })


# ── Date parsing helpers ──────────────────────────────────────────────────────

def _parse_date(value: str) -> date:
    """Parse various date string formats into a date. Falls back to today."""
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except (ValueError, AttributeError):
            pass
    # struct_time from time.strptime / email.utils
    try:
        import time as _time
        return date.fromtimestamp(_time.mktime(value))  # type: ignore[arg-type]
    except Exception:
        pass
    return date.today()


# ── Fed speeches ──────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch_fed_speeches(limit: int = 10) -> list[CentralBankSpeech]:
    """Fetch from the Fed JSON endpoint and return CentralBankSpeech objects."""
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(FED_JSON_URL)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("fetch_fed_speeches: failed", error=str(exc))
        return []

    speeches: list[CentralBankSpeech] = []
    items = data if isinstance(data, list) else data.get("speeches", [])
    for item in items[:limit]:
        try:
            raw_date = item.get("d", "") or item.get("date", "")
            speech_date = _parse_date(raw_date)
            title = item.get("t", "") or item.get("title", "Unknown")
            speaker = item.get("s", "") or item.get("speaker", "Federal Reserve")
            url = item.get("l", "") or item.get("url", "")
            if url and not url.startswith("http"):
                url = "https://www.federalreserve.gov" + url

            h, d, net, label = score_tone(title)
            speeches.append(CentralBankSpeech(
                bank="FED",
                speaker=speaker,
                title=title,
                speech_date=speech_date,
                url=url,
                summary=None,
                hawkish_score=h,
                dovish_score=d,
                net_tone=net,
                tone_label=label,
                finbert_sentiment=None,
                finbert_confidence=None,
            ))
        except Exception as exc:
            logger.debug("fetch_fed_speeches: skipping item", error=str(exc))

    logger.info("fetch_fed_speeches: fetched", count=len(speeches))
    return speeches


# ── ECB speeches ──────────────────────────────────────────────────────────────

async def fetch_ecb_speeches(limit: int = 10) -> list[CentralBankSpeech]:
    """Fetch from the ECB RSS feed."""
    return await _fetch_rss_speeches("ECB", ECB_RSS_URL, limit)


# ── BoE speeches ──────────────────────────────────────────────────────────────

async def fetch_boe_speeches(limit: int = 10) -> list[CentralBankSpeech]:
    """Fetch from the BoE RSS feed."""
    return await _fetch_rss_speeches("BOE", BOE_RSS_URL, limit)


def _rss_text(tag: str, xml: str) -> str:
    """Extract the first occurrence of <tag>...</tag> from RSS XML, stripping CDATA."""
    m = re.search(rf"<{tag}[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>", xml, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_rss_items(xml: str, limit: int) -> list[dict[str, str]]:
    """Extract <item> blocks from RSS 2.0 XML and return key fields as plain dicts."""
    items: list[dict[str, str]] = []
    for block in re.findall(r"<item>(.*?)</item>", xml, re.DOTALL)[:limit]:
        items.append({
            "title": _rss_text("title", block),
            "link": _rss_text("link", block),
            "author": _rss_text("author", block) or _rss_text("dc:creator", block),
            "pubDate": _rss_text("pubDate", block) or _rss_text("dc:date", block),
            "description": _rss_text("description", block),
        })
    return items


async def _fetch_rss_speeches(bank: str, url: str, limit: int) -> list[CentralBankSpeech]:
    """Generic RSS → CentralBankSpeech parser. Uses stdlib re only — no feedparser."""
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw_xml = resp.text
    except Exception as exc:
        logger.warning("_fetch_rss_speeches: fetch failed", bank=bank, url=url, error=str(exc))
        return []

    speeches: list[CentralBankSpeech] = []
    for item in _parse_rss_items(raw_xml, limit):
        try:
            title = item["title"] or "Untitled"
            link = item["link"]
            author = item["author"] or f"{bank} Official"
            speech_date = _parse_date(item["pubDate"]) if item["pubDate"] else date.today()

            desc = item["description"]
            if desc:
                plain = re.sub(r"<[^>]+>", " ", desc)
                plain = re.sub(r"\s+", " ", plain).strip()
                summary_text: Optional[str] = plain[:500] or None
            else:
                summary_text = None

            score_text = (summary_text or "") + " " + title
            h, d, net, label = score_tone(score_text)

            speeches.append(CentralBankSpeech(
                bank=bank,
                speaker=author,
                title=title,
                speech_date=speech_date,
                url=link,
                summary=summary_text,
                hawkish_score=h,
                dovish_score=d,
                net_tone=net,
                tone_label=label,
                finbert_sentiment=None,
                finbert_confidence=None,
            ))
        except Exception as exc:
            logger.debug("_fetch_rss_speeches: skipping item", bank=bank, error=str(exc))

    logger.info("_fetch_rss_speeches: fetched", bank=bank, count=len(speeches))
    return speeches


# ── Pipeline orchestration ────────────────────────────────────────────────────

async def get_cb_summary(bank: str = "FED", limit: int = 10) -> CBSpeechSummary:
    """Full pipeline: fetch → analyze → aggregate tone trend → policy signal.

    Fetches speeches for the given bank, runs extract_speech_text + FinBERT
    on each, then aggregates into a CBSpeechSummary.

    Trend is computed by comparing avg_net_tone of the last 3 vs prior 3.
    """
    bank = bank.upper()

    # 1. Fetch raw speeches
    fetchers = {
        "FED": fetch_fed_speeches,
        "ECB": fetch_ecb_speeches,
        "BOE": fetch_boe_speeches,
    }
    fetcher = fetchers.get(bank)
    if fetcher is None:
        logger.warning("get_cb_summary: unsupported bank, returning empty", bank=bank)
        return _empty_summary(bank)

    raw_speeches = await fetcher(limit)
    if not raw_speeches:
        logger.warning("get_cb_summary: no speeches returned", bank=bank)
        return _empty_summary(bank)

    # 2. Analyze each speech concurrently (fetch full text + FinBERT)
    analyzed = await asyncio.gather(
        *[analyze_speech(sp, fetch_full_text=True) for sp in raw_speeches],
        return_exceptions=True,
    )

    speeches: list[CentralBankSpeech] = []
    for item in analyzed:
        if isinstance(item, CentralBankSpeech):
            speeches.append(item)
        elif isinstance(item, Exception):
            logger.warning("get_cb_summary: analyze_speech error", error=str(item))

    if not speeches:
        return _empty_summary(bank)

    # Sort newest first for trend calculation
    speeches.sort(key=lambda s: s.speech_date, reverse=True)

    avg_net_tone = round(sum(s.net_tone for s in speeches) / len(speeches), 4)
    trend = _compute_trend(speeches)
    signal = _policy_signal(avg_net_tone)
    key_phrases = _extract_key_phrases(speeches)

    logger.info(
        "get_cb_summary: complete",
        bank=bank,
        speeches=len(speeches),
        avg_net_tone=avg_net_tone,
        trend=trend,
        signal=signal,
    )

    return CBSpeechSummary(
        bank=bank,
        as_of=date.today(),
        recent_speeches=speeches,
        avg_net_tone=avg_net_tone,
        tone_trend=trend,
        policy_signal=signal,
        key_phrases=key_phrases,
    )


async def get_all_banks_summary(limit_per_bank: int = 5) -> dict[str, CBSpeechSummary]:
    """Run get_cb_summary for FED, ECB, BOE concurrently via asyncio.gather."""
    banks = ["FED", "ECB", "BOE"]
    results = await asyncio.gather(
        *[get_cb_summary(bank, limit_per_bank) for bank in banks],
        return_exceptions=True,
    )
    summaries: dict[str, CBSpeechSummary] = {}
    for bank, result in zip(banks, results):
        if isinstance(result, CBSpeechSummary):
            summaries[bank] = result
        else:
            logger.error("get_all_banks_summary: error for bank", bank=bank, error=str(result))
            summaries[bank] = _empty_summary(bank)
    return summaries


# ── Fallback helper ───────────────────────────────────────────────────────────

def _empty_summary(bank: str) -> CBSpeechSummary:
    return CBSpeechSummary(
        bank=bank,
        as_of=date.today(),
        recent_speeches=[],
        avg_net_tone=0.0,
        tone_trend="stable",
        policy_signal="hold",
        key_phrases=[],
    )
