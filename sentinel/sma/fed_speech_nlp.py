"""
Central bank speech NLP — parse Fed speeches, FOMC minutes, and ECB/BoE transcripts
to extract hawkish/dovish signals, rate-path guidance, and key themes.

Dimension #45 in the SENTINEL competitive matrix — target score 9.

Data sources (all free):
  - Fed speeches JSON:  https://www.federalreserve.gov/json/ne-speeches.json
  - Fed FOMC minutes:   https://www.federalreserve.gov/monetarypolicy/fomcminutes{YYYYMMDD}.htm
  - Fed FOMC calendar:  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  - ECB speeches RSS:   https://www.ecb.europa.eu/rss/speeches.rss
  - BoE speeches RSS:   https://www.bankofengland.co.uk/rss/speeches

NLP approach:
  - Dual-lexicon keyword counting (hawkish / dovish term lists)
  - Negation-aware context window (checks preceding 8 tokens for negators)
  - Sentence-level extraction for key passages
  - Theme detection via multi-keyword coverage
  - Rate signal inference from explicit signal phrases, with tone fallback
  - No external ML dependencies — pure Python regex + stdlib
"""
from __future__ import annotations

import asyncio
import re
import statistics
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FED_BASE = "https://www.federalreserve.gov"
FED_SPEECHES_JSON = "https://www.federalreserve.gov/json/ne-speeches.json"
FED_FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
ECB_RSS_URL = "https://www.ecb.europa.eu/rss/speeches.rss"
BOE_RSS_URL = "https://www.bankofengland.co.uk/rss/speeches"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/xhtml+xml,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT = 30.0

# Negation tokens — if any appear within 8 tokens before a signal term, invert
_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor", "without", "hardly", "barely",
    "scarcely", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "wouldn't", "couldn't", "shouldn't", "won't", "don't", "didn't",
    "cannot", "can't", "less", "unlikely", "insufficient",
})

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

HAWKISH_TERMS: list[tuple[str, float]] = [
    # (phrase, weight)
    ("inflation remains elevated", 2.0),
    ("inflation is too high", 2.0),
    ("unacceptably high inflation", 2.0),
    ("persistent inflation", 1.8),
    ("elevated inflation", 1.5),
    ("above target", 1.5),
    ("price stability", 1.2),
    ("restrictive", 1.5),
    ("higher for longer", 2.0),
    ("not yet confident", 1.8),
    ("tighten", 1.5),
    ("tightening", 1.5),
    ("rate increase", 1.8),
    ("rate hike", 1.8),
    ("raise rates", 1.8),
    ("further increases", 1.8),
    ("additional firming", 1.8),
    ("policy firming", 1.5),
    ("reduce balance sheet", 1.3),
    ("quantitative tightening", 1.5),
    ("qt", 1.0),
    ("remain vigilant", 1.3),
    ("determined to restore", 1.5),
    ("premature to cut", 2.0),
    ("premature to ease", 2.0),
    ("not the time to cut", 2.0),
    ("too soon to cut", 2.0),
    ("inflation expectations", 1.0),
    ("upside risk", 1.2),
    ("overshoot", 1.3),
    ("overshooting", 1.3),
    ("persistent", 1.0),
    ("sticky", 1.2),
    ("supply-side pressure", 0.8),
    ("wage growth", 0.8),
    ("labor market tight", 1.3),
    ("tight labor market", 1.3),
    ("strong labor market", 1.0),
]

DOVISH_TERMS: list[tuple[str, float]] = [
    ("cut rates", 2.0),
    ("rate cut", 2.0),
    ("lower rates", 1.8),
    ("reduce rates", 1.8),
    ("ease", 1.3),
    ("easing", 1.3),
    ("accommodative", 1.5),
    ("below target", 1.5),
    ("disinflation", 1.5),
    ("softer inflation", 1.5),
    ("inflation cooling", 1.5),
    ("inflation declining", 1.5),
    ("downside risk", 1.3),
    ("growth concern", 1.3),
    ("slowing growth", 1.3),
    ("recession risk", 1.5),
    ("labor market softening", 1.5),
    ("labor market cooling", 1.5),
    ("cooling labor", 1.3),
    ("unemployment rising", 1.3),
    ("support growth", 1.2),
    ("patient", 1.0),
    ("gradual", 0.8),
    ("data dependent", 1.0),
    ("data-dependent", 1.0),
    ("monitor", 0.6),
    ("flexible", 0.7),
    ("pivot", 1.5),
    ("policy normalization", 1.0),
    ("inflation well-anchored", 1.8),
    ("expectations anchored", 1.5),
    ("anchor inflation expectations", 1.5),
    ("supply chain improvement", 0.8),
    ("balance sheet reduction", 0.8),  # neutral-ish but slightly dovish direction
    ("reducing balance sheet at appropriate pace", 0.5),
    ("pause", 1.2),
    ("hold rates", 1.0),
    ("keep rates steady", 1.0),
    ("transitory", 1.0),
]

# Theme detection
THEME_KEYWORDS: dict[str, list[str]] = {
    "inflation": [
        "inflation", "price", "CPI", "PCE", "disinflation", "deflation",
        "price stability", "consumer price", "core inflation", "headline",
    ],
    "employment": [
        "employment", "jobs", "labor", "labour", "unemployment", "payroll",
        "job openings", "JOLTS", "hiring", "layoffs", "wage",
    ],
    "growth": [
        "growth", "GDP", "output", "recession", "expansion", "contraction",
        "economic activity", "productivity", "real GDP",
    ],
    "financial_stability": [
        "financial stability", "banking", "credit", "liquidity", "stress",
        "systemic risk", "bank failure", "deposit", "financial conditions",
    ],
    "global": [
        "global", "international", "trade", "geopolitical", "China",
        "Europe", "emerging market", "supply chain", "dollar",
    ],
    "housing": [
        "housing", "mortgage", "real estate", "home price", "rent",
    ],
    "monetary_policy": [
        "federal funds rate", "fed funds", "target range", "balance sheet",
        "forward guidance", "dot plot", "interest rate", "policy rate",
        "quantitative", "open market",
    ],
}

# ---------------------------------------------------------------------------
# Literal types
# ---------------------------------------------------------------------------

ToneLabel = Literal["very_hawkish", "hawkish", "neutral", "dovish", "very_dovish"]
RateSignal = Literal["hike", "hold", "cut", "data_dependent", "unknown"]
Institution = Literal["fed", "ecb", "boe", "bis", "other"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SpeechMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker: str
    title: str
    date: date
    url: str
    institution: Institution
    event: Optional[str] = None


class SpeechAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    meta: SpeechMeta
    # Tone scores
    hawkish_score: float                   # 0–10 weighted score
    dovish_score: float                    # 0–10 weighted score
    net_score: float                       # hawkish_score − dovish_score; positive = hawkish
    tone: ToneLabel
    # Key passages
    hawkish_passages: list[str]            # top hawkish sentences
    dovish_passages: list[str]             # top dovish sentences
    # Themes and rate signal
    key_themes: list[str]
    rate_path_signal: RateSignal
    # Statistics
    word_count: int
    sentence_count: int
    # Extracted numbers
    rate_mentioned: Optional[float] = None         # specific rate figure, e.g. 5.25
    inflation_mentioned: Optional[float] = None    # CPI/PCE figure mentioned
    gdp_mentioned: Optional[float] = None


class FOMCMinutesAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    meeting_date: date
    release_date: date
    minutes_url: str
    consensus_view: str           # "hawkish" | "dovish" | "neutral"
    hawkish_score: float
    dovish_score: float
    net_score: float
    tone: ToneLabel
    dissents_count: int
    data_dependencies: list[str]  # explicit conditions for future moves
    rate_guidance: str            # forward guidance extracted
    key_themes: list[str]
    # Section-level tones
    staff_economic_projection: Optional[str] = None
    committee_discussion_tone: str
    participants_view: str        # "many participants", "some participants", etc.
    word_count: int


class CentralBankMonitor(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    fed_tone: ToneLabel
    fed_net_score: float
    ecb_tone: ToneLabel
    ecb_net_score: float
    boe_tone: ToneLabel
    boe_net_score: float
    global_tone: ToneLabel        # aggregate of three CBs
    global_net_score: float
    recent_speeches: list[SpeechAnalysis]
    tone_trend: Literal["more_hawkish", "more_dovish", "stable"]
    implied_next_move: RateSignal


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"&lt;", "<", html)
    html = re.sub(r"&gt;", ">", html)
    html = re.sub(r"&#\d+;", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Strategy: split on '. '|'? '|'! ' where the next char is uppercase or a digit.
    Handles abbreviations imperfectly — good enough for scoring purposes.
    """
    # Protect known abbreviations
    protected = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|U\.S|U\.K|e\.g|i\.e)\.", r"\1<DOT>", text)
    # Split on sentence-ending punctuation
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"])", protected)
    # Restore dots
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return sentences


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z'-]+", text.lower())


def _is_negated(tokens: list[str], match_start_token: int, window: int = 8) -> bool:
    """Check whether any negator appears in the window before match_start_token."""
    start = max(0, match_start_token - window)
    preceding = tokens[start:match_start_token]
    return any(t in _NEGATORS for t in preceding)


def _count_weighted_terms(
    text: str,
    terms: list[tuple[str, float]],
    negate_weight: float = -0.5,
) -> tuple[float, list[tuple[str, int]]]:  # (total_score, [(term, count)])
    """
    Count weighted occurrences of multi-word terms in text with negation handling.

    For each match:
      - Tokenize the surrounding context.
      - If a negator is within 8 tokens before the match, apply negate_weight
        (partially reduces the score rather than fully inverting it, because
        "we are not yet confident inflation is tamed" is still hawkish context).

    Returns: (total_weighted_score, [(matched_term, hit_count), ...])
    """
    lower = text.lower()
    tokens = _tokenize(lower)
    total_score = 0.0
    hits: list[tuple[str, int]] = []

    for phrase, weight in terms:
        phrase_lower = phrase.lower()
        count = 0
        pos = 0
        while True:
            idx = lower.find(phrase_lower, pos)
            if idx == -1:
                break
            # Estimate token position (rough: count tokens before idx)
            token_pos = len(_tokenize(lower[:idx]))
            neg = _is_negated(tokens, token_pos, window=8)
            effective_weight = negate_weight if neg else weight
            total_score += effective_weight
            count += 1
            pos = idx + len(phrase_lower)
        if count:
            hits.append((phrase, count))

    return max(0.0, total_score), hits


# ---------------------------------------------------------------------------
# Core NLP functions
# ---------------------------------------------------------------------------

def compute_tone(
    text: str,
) -> tuple[float, float, ToneLabel]:
    """
    Compute hawkish and dovish weighted scores and derive a tone label.

    Scale: raw weighted counts are capped at 10.
    net_score = hawkish_score − dovish_score.
    Thresholds (empirically calibrated):
      ≥  3.0 → very_hawkish
      ≥  1.0 → hawkish
      ≤ −3.0 → very_dovish
      ≤ −1.0 → dovish
      else   → neutral
    """
    if not text or not text.strip():
        return 0.0, 0.0, "neutral"

    h_score, _ = _count_weighted_terms(text, HAWKISH_TERMS)
    d_score, _ = _count_weighted_terms(text, DOVISH_TERMS)

    # Cap at 10 to produce a bounded 0–10 scale
    h_capped = round(min(h_score, 10.0), 3)
    d_capped = round(min(d_score, 10.0), 3)
    net = round(h_capped - d_capped, 3)

    if net >= 3.0:
        tone: ToneLabel = "very_hawkish"
    elif net >= 1.0:
        tone = "hawkish"
    elif net <= -3.0:
        tone = "very_dovish"
    elif net <= -1.0:
        tone = "dovish"
    else:
        tone = "neutral"

    return h_capped, d_capped, tone


def extract_key_passages(
    text: str,
    keywords: list[str],
    n_sentences: int = 3,
) -> list[str]:
    """
    Find sentences containing any of the keywords and return the top n_sentences
    ranked by total keyword hit count (most hits first).
    """
    sentences = _split_sentences(text)
    scored: list[tuple[int, str]] = []
    for sent in sentences:
        sent_lower = sent.lower()
        hits = sum(sent_lower.count(kw.lower()) for kw in keywords)
        if hits > 0:
            scored.append((hits, sent.strip()))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[tuple[int, str]] = []
    for score, sent in sorted(scored, key=lambda x: x[0], reverse=True):
        norm = " ".join(sent.split())
        if norm not in seen and len(sent) > 20:
            seen.add(norm)
            unique.append((score, sent))

    return [s for _, s in unique[:n_sentences]]


def identify_themes(text: str) -> list[str]:
    """Return themes where at least 2 distinct keywords appear in the text."""
    lower = text.lower()
    themes: list[str] = []
    for theme, keywords in THEME_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in lower)
        if hits >= 2:
            themes.append(theme)
    return themes


def extract_rate_signal(text: str, tone: ToneLabel) -> RateSignal:
    """
    Extract explicit rate-path signal from text.
    Priority: explicit phrase > tone-derived fallback.
    """
    lower = text.lower()

    # Explicit hike signals
    hike_phrases = [
        "further increases", "additional firming", "rate increase",
        "raise rates", "rate hike", "tighten further", "additional tightening",
        "need to raise", "may be appropriate to raise",
    ]
    cut_phrases = [
        "rate cut", "cut rates", "lower rates", "reduce rates", "ease policy",
        "begin to ease", "begin easing", "pivot to easing", "rate reduction",
        "we will cut", "appropriate to cut",
    ]
    hold_phrases = [
        "hold rates", "keep rates", "maintain rates", "rates on hold",
        "pause", "patient", "hold steady", "no change", "unchanged",
        "keep policy rate", "appropriate to hold",
    ]
    data_dep_phrases = [
        "data dependent", "data-dependent", "meeting by meeting", "meeting-by-meeting",
        "incoming data", "based on incoming", "depend on data", "depends on the data",
        "monitor incoming", "monitor the data",
    ]

    # Count matches (data-dependent is a modifier, not override)
    hike_hits = sum(1 for p in hike_phrases if p in lower)
    cut_hits = sum(1 for p in cut_phrases if p in lower)
    hold_hits = sum(1 for p in hold_phrases if p in lower)
    dd_hits = sum(1 for p in data_dep_phrases if p in lower)

    # Explicit signals take priority
    max_explicit = max(hike_hits, cut_hits, hold_hits)
    if max_explicit > 0:
        if hike_hits == max_explicit:
            return "hike"
        if cut_hits == max_explicit:
            return "cut"
        if hold_hits == max_explicit:
            return "hold"

    # Data-dependent override if no explicit directional signal
    if dd_hits >= 1:
        return "data_dependent"

    # Fallback to tone
    if tone in ("very_hawkish", "hawkish"):
        return "hike"
    if tone in ("very_dovish", "dovish"):
        return "cut"
    return "data_dependent"


def _extract_numbers(text: str, pattern: str) -> Optional[float]:
    """Extract the first floating-point number following a regex pattern."""
    m = re.search(pattern + r"[^\d]*(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_data_dependencies(text: str) -> list[str]:
    """
    Extract sentences that describe conditions for future policy moves.
    e.g. "if inflation returns to 2%, we could consider cutting rates"
    """
    trigger_patterns = [
        r"if\s+\w+", r"should\s+inflation", r"provided that", r"conditional",
        r"depending on", r"in the event", r"when\s+inflation", r"once\s+\w+",
        r"before\s+we", r"until\s+\w+",
    ]
    sentences = _split_sentences(text)
    deps: list[str] = []
    for sent in sentences:
        lower_sent = sent.lower()
        if any(re.search(p, lower_sent) for p in trigger_patterns):
            # Only include if it also mentions policy/rates
            if any(kw in lower_sent for kw in ["rate", "cut", "hike", "ease", "tighten", "policy"]):
                deps.append(sent.strip())
    return deps[:5]


def _rss_extract(tag: str, xml: str) -> str:
    """Extract content from an RSS/XML tag, stripping CDATA wrappers."""
    m = re.search(rf"<{tag}[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>", xml, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_date_str(s: str) -> date:
    """Parse a date string in various formats, falling back to today."""
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%Y/%m/%d", "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S GMT", "%d %b %Y"):
        try:
            return datetime.strptime(s.strip()[:30], fmt).date()
        except (ValueError, AttributeError):
            continue
    return date.today()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_text(client: httpx.AsyncClient, url: str, max_chars: int = 50_000) -> str:
    """GET a URL and return stripped plain text (HTML stripped)."""
    resp = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    return _strip_html(resp.text)[:max_chars]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_raw(client: httpx.AsyncClient, url: str) -> str:
    """GET a URL and return raw response text."""
    resp = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# FedSpeechNLP
# ---------------------------------------------------------------------------

class FedSpeechNLP:
    """
    Comprehensive NLP engine for central bank speech and FOMC minutes analysis.

    All methods are async; text fetching uses httpx with retry logic.
    No external ML dependencies — pure keyword + regex NLP.
    """

    def __init__(self, timeout: float = 30.0, max_text_chars: int = 60_000):
        self._timeout = timeout
        self._max_chars = max_text_chars
        self._client_kwargs = {
            "headers": _HEADERS,
            "follow_redirects": True,
            "timeout": timeout,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(**self._client_kwargs)

    def _analyze_text(
        self,
        text: str,
        meta: SpeechMeta,
    ) -> SpeechAnalysis:
        """Run full NLP pipeline on speech text and return SpeechAnalysis."""
        h_score, d_score, tone = compute_tone(text)
        net_score = round(h_score - d_score, 3)

        # Key passage extraction
        h_kws = [p for p, _ in HAWKISH_TERMS]
        d_kws = [p for p, _ in DOVISH_TERMS]
        hawkish_passages = extract_key_passages(text, h_kws, n_sentences=3)
        dovish_passages = extract_key_passages(text, d_kws, n_sentences=3)

        themes = identify_themes(text)
        rate_signal = extract_rate_signal(text, tone)

        sentences = _split_sentences(text)
        words = text.split()

        # Extract numeric references
        rate_mentioned = _extract_numbers(
            text, r"(?:federal funds|policy|interest)\s+rate\s+(?:of|at|to|is)?"
        )
        inflation_mentioned = _extract_numbers(
            text, r"(?:inflation|CPI|PCE)\s+(?:of|at|is|was|reached|hit)?\s*(?:around|about|near|approximately)?"
        )
        gdp_mentioned = _extract_numbers(
            text, r"(?:GDP|growth)\s+(?:of|at|is|was|grew|expanded|contracted)?\s*(?:around|about|approximately)?"
        )

        return SpeechAnalysis(
            meta=meta,
            hawkish_score=h_score,
            dovish_score=d_score,
            net_score=net_score,
            tone=tone,
            hawkish_passages=hawkish_passages,
            dovish_passages=dovish_passages,
            key_themes=themes,
            rate_path_signal=rate_signal,
            word_count=len(words),
            sentence_count=len(sentences),
            rate_mentioned=rate_mentioned,
            inflation_mentioned=inflation_mentioned,
            gdp_mentioned=gdp_mentioned,
        )

    # ------------------------------------------------------------------
    # Public NLP primitives
    # ------------------------------------------------------------------

    def compute_tone(self, text: str) -> tuple[float, float, str]:
        """
        Compute tone from raw text.
        Returns (hawkish_score, dovish_score, tone_label).
        """
        h, d, tone = compute_tone(text)
        return h, d, tone

    def extract_key_passages(
        self,
        text: str,
        keywords: list[str],
        n_sentences: int = 3,
    ) -> list[str]:
        return extract_key_passages(text, keywords, n_sentences)

    def identify_themes(self, text: str) -> list[str]:
        return identify_themes(text)

    def extract_rate_signal(self, text: str) -> str:
        _, _, tone = compute_tone(text)
        return extract_rate_signal(text, tone)

    # ------------------------------------------------------------------
    # Fed speech fetching
    # ------------------------------------------------------------------

    async def get_recent_fed_speeches(self, n: int = 10) -> list[SpeechMeta]:
        """Fetch Fed speeches index JSON and return the last n SpeechMeta objects."""
        try:
            async with self._make_client() as client:
                resp = await client.get(FED_SPEECHES_JSON, timeout=_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("get_recent_fed_speeches: failed", error=str(exc))
            return []

        items = data if isinstance(data, list) else data.get("speeches", [])
        speeches: list[SpeechMeta] = []
        for item in items[:n]:
            try:
                raw_date = item.get("d", "") or item.get("date", "")
                speech_date = _parse_date_str(raw_date)
                title = item.get("t", "") or item.get("title", "Untitled")
                speaker = item.get("s", "") or item.get("speaker", "Federal Reserve")
                url = item.get("l", "") or item.get("url", "")
                if url and not url.startswith("http"):
                    url = FED_BASE + url
                event = item.get("e") or item.get("event") or None
                speeches.append(SpeechMeta(
                    speaker=speaker,
                    title=title,
                    date=speech_date,
                    url=url,
                    institution="fed",
                    event=event,
                ))
            except Exception as exc:
                logger.debug("get_recent_fed_speeches: skipping item", error=str(exc))

        logger.info("get_recent_fed_speeches: fetched", count=len(speeches))
        return speeches

    async def analyze_speech(
        self,
        url: str,
        speaker: str,
        date_: date,
        institution: Institution = "fed",
        title: str = "",
        event: Optional[str] = None,
    ) -> SpeechAnalysis:
        """
        Download a speech page, strip HTML, run full NLP pipeline.
        Falls back to title-only scoring if fetch fails.
        """
        meta = SpeechMeta(
            speaker=speaker,
            title=title,
            date=date_,
            url=url,
            institution=institution,
            event=event,
        )

        text = title  # fallback
        try:
            async with self._make_client() as client:
                fetched = await _fetch_text(client, url, max_chars=self._max_chars)
            if fetched and len(fetched) > 200:
                text = fetched
            else:
                logger.debug("analyze_speech: short/empty fetch, using title", url=url)
        except Exception as exc:
            logger.warning("analyze_speech: fetch failed", url=url, error=str(exc))

        return self._analyze_text(text, meta)

    async def analyze_speech_from_meta(self, meta: SpeechMeta) -> SpeechAnalysis:
        """Convenience: analyze a speech from a SpeechMeta object."""
        return await self.analyze_speech(
            url=meta.url,
            speaker=meta.speaker,
            date_=meta.date,
            institution=meta.institution,
            title=meta.title,
            event=meta.event,
        )

    # ------------------------------------------------------------------
    # FOMC Minutes
    # ------------------------------------------------------------------

    def _fomc_minutes_url(self, meeting_date: date) -> str:
        """Construct the FOMC minutes URL for a given meeting date."""
        date_str = meeting_date.strftime("%Y%m%d")
        return f"{FED_BASE}/monetarypolicy/fomcminutes{date_str}.htm"

    async def get_fomc_minutes(self, meeting_date: date) -> FOMCMinutesAnalysis:
        """
        Download and analyze FOMC minutes for a specific meeting date.
        Minutes are typically released 3 weeks after the meeting.
        """
        url = self._fomc_minutes_url(meeting_date)
        release_date = meeting_date + timedelta(days=21)  # approximate

        try:
            async with self._make_client() as client:
                text = await _fetch_text(client, url, max_chars=self._max_chars)
        except Exception as exc:
            logger.warning("get_fomc_minutes: fetch failed", meeting_date=str(meeting_date), error=str(exc))
            text = ""

        h_score, d_score, tone = compute_tone(text)
        net_score = round(h_score - d_score, 3)

        # Consensus view
        if net_score >= 1.0:
            consensus_view = "hawkish"
        elif net_score <= -1.0:
            consensus_view = "dovish"
        else:
            consensus_view = "neutral"

        # Count dissent mentions
        dissents_count = max(
            len(re.findall(r"\bdissent\w*\b", text, re.IGNORECASE)),
            len(re.findall(r"\bvoted\s+against\b", text, re.IGNORECASE)),
        )

        # Data dependencies
        data_deps = _extract_data_dependencies(text)

        # Forward guidance extraction (first sentence with "will" or "expect")
        sentences = _split_sentences(text)
        guidance_sentences: list[str] = []
        for sent in sentences:
            lower = sent.lower()
            if any(kw in lower for kw in ["expect", "anticipate", "will be", "would be", "likely to"]):
                if any(kw in lower for kw in ["rate", "policy", "cut", "hike", "ease", "tighten"]):
                    guidance_sentences.append(sent.strip())
        rate_guidance = guidance_sentences[0] if guidance_sentences else "No explicit guidance extracted."

        # Participant view (most common quantifier phrase)
        quantifier_pattern = re.compile(
            r"(many participants?|most participants?|some participants?|several participants?|"
            r"a few participants?|all participants?|the committee)", re.IGNORECASE
        )
        quantifier_matches = quantifier_pattern.findall(text)
        if quantifier_matches:
            participants_view = Counter(quantifier_matches).most_common(1)[0][0]
        else:
            participants_view = "participants"

        # Section tones
        # Staff economic projection section (usually near beginning)
        staff_proj: Optional[str] = None
        staff_match = re.search(
            r"staff\s+(?:economic\s+)?(?:projection|forecast|review)(.*?)\n\n",
            text, re.IGNORECASE | re.DOTALL
        )
        if staff_match:
            staff_text = staff_match.group(1)[:500]
            _, _, staff_tone = compute_tone(staff_text)
            staff_proj = staff_tone

        # Committee discussion section
        comm_match = re.search(
            r"committee\s+(?:discussion|policy\s+action|deliberation)(.*?)\n\n",
            text, re.IGNORECASE | re.DOTALL
        )
        if comm_match:
            comm_text = comm_match.group(1)[:2000]
            _, _, comm_tone = compute_tone(comm_text)
            committee_discussion_tone = comm_tone
        else:
            committee_discussion_tone = tone  # fallback to overall

        themes = identify_themes(text)

        logger.info(
            "get_fomc_minutes: analyzed",
            meeting_date=str(meeting_date),
            tone=tone,
            net_score=net_score,
            dissents=dissents_count,
            word_count=len(text.split()),
        )

        return FOMCMinutesAnalysis(
            meeting_date=meeting_date,
            release_date=release_date,
            minutes_url=url,
            consensus_view=consensus_view,
            hawkish_score=h_score,
            dovish_score=d_score,
            net_score=net_score,
            tone=tone,
            dissents_count=dissents_count,
            data_dependencies=data_deps,
            rate_guidance=rate_guidance,
            key_themes=themes,
            staff_economic_projection=staff_proj,
            committee_discussion_tone=committee_discussion_tone,
            participants_view=participants_view,
            word_count=len(text.split()),
        )

    async def get_recent_fomc_analyses(self, n: int = 4) -> list[FOMCMinutesAnalysis]:
        """
        Analyze the last n FOMC minutes.
        Uses hardcoded FOMC meeting dates from economic_calendar_enhanced.
        """
        from sentinel.sma.economic_calendar_enhanced import FOMC_DATES

        all_dates: list[date] = []
        today = date.today()
        for year_dates in FOMC_DATES.values():
            all_dates.extend(year_dates)

        # Keep only past meetings (minutes already published ~3 weeks after)
        past_meetings = sorted(
            [d for d in all_dates if d + timedelta(days=21) <= today],
            reverse=True,
        )[:n]

        if not past_meetings:
            logger.warning("get_recent_fomc_analyses: no past meetings found")
            return []

        analyses = await asyncio.gather(
            *[self.get_fomc_minutes(d) for d in past_meetings],
            return_exceptions=True,
        )

        results: list[FOMCMinutesAnalysis] = []
        for meeting_d, result in zip(past_meetings, analyses):
            if isinstance(result, FOMCMinutesAnalysis):
                results.append(result)
            else:
                logger.warning(
                    "get_recent_fomc_analyses: failed for meeting",
                    meeting_date=str(meeting_d),
                    error=str(result),
                )

        return results

    # ------------------------------------------------------------------
    # ECB / BoE via RSS
    # ------------------------------------------------------------------

    async def _fetch_rss_speeches(
        self,
        institution: Institution,
        rss_url: str,
        n: int = 10,
    ) -> list[SpeechMeta]:
        """Parse an RSS feed and extract speech metadata."""
        try:
            async with self._make_client() as client:
                raw = await _fetch_raw(client, rss_url)
        except Exception as exc:
            logger.warning("_fetch_rss_speeches: failed", institution=institution, error=str(exc))
            return []

        speeches: list[SpeechMeta] = []
        for block in re.findall(r"<item>(.*?)</item>", raw, re.DOTALL)[:n]:
            try:
                title = _rss_extract("title", block) or "Untitled"
                link = _rss_extract("link", block)
                author = (
                    _rss_extract("author", block)
                    or _rss_extract("dc:creator", block)
                    or f"{institution.upper()} Official"
                )
                pub_date_str = _rss_extract("pubDate", block) or _rss_extract("dc:date", block)
                speech_date = _parse_date_str(pub_date_str) if pub_date_str else date.today()
                speeches.append(SpeechMeta(
                    speaker=author,
                    title=title,
                    date=speech_date,
                    url=link,
                    institution=institution,
                ))
            except Exception as exc:
                logger.debug("_fetch_rss_speeches: skipping item", institution=institution, error=str(exc))

        logger.info("_fetch_rss_speeches: fetched", institution=institution, count=len(speeches))
        return speeches

    # ------------------------------------------------------------------
    # Central bank monitor
    # ------------------------------------------------------------------

    async def monitor_central_banks(
        self,
        n_speeches_per_bank: int = 5,
        fetch_full_text: bool = True,
    ) -> CentralBankMonitor:
        """
        Fetch and analyze recent speeches from Fed, ECB, and BoE concurrently.
        Aggregate tone signals into a CentralBankMonitor.
        """
        today = date.today()

        # Fetch speech metadata concurrently
        fed_meta_task = asyncio.create_task(self.get_recent_fed_speeches(n_speeches_per_bank))
        ecb_meta_task = asyncio.create_task(
            self._fetch_rss_speeches("ecb", ECB_RSS_URL, n_speeches_per_bank)
        )
        boe_meta_task = asyncio.create_task(
            self._fetch_rss_speeches("boe", BOE_RSS_URL, n_speeches_per_bank)
        )

        fed_meta, ecb_meta, boe_meta = await asyncio.gather(
            fed_meta_task, ecb_meta_task, boe_meta_task, return_exceptions=True
        )
        fed_meta = fed_meta if isinstance(fed_meta, list) else []
        ecb_meta = ecb_meta if isinstance(ecb_meta, list) else []
        boe_meta = boe_meta if isinstance(boe_meta, list) else []

        all_meta = fed_meta + ecb_meta + boe_meta

        if not all_meta:
            logger.warning("monitor_central_banks: no speeches retrieved")
            return self._empty_monitor(today)

        # Analyze speeches concurrently (fetch full text for richer scoring)
        if fetch_full_text:
            analyses_raw = await asyncio.gather(
                *[self.analyze_speech_from_meta(m) for m in all_meta],
                return_exceptions=True,
            )
        else:
            # Title-only scoring (fast path)
            analyses_raw = [
                self._analyze_text(m.title, m)
                for m in all_meta
            ]

        analyses: list[SpeechAnalysis] = [
            a for a in analyses_raw if isinstance(a, SpeechAnalysis)
        ]

        if not analyses:
            return self._empty_monitor(today)

        # Split by institution
        fed_analyses = [a for a in analyses if a.meta.institution == "fed"]
        ecb_analyses = [a for a in analyses if a.meta.institution == "ecb"]
        boe_analyses = [a for a in analyses if a.meta.institution == "boe"]

        def _agg_tone(bank_analyses: list[SpeechAnalysis]) -> tuple[float, ToneLabel]:
            if not bank_analyses:
                return 0.0, "neutral"
            avg_net = statistics.mean(a.net_score for a in bank_analyses)
            _, _, tone = compute_tone("")  # just to reuse threshold logic
            if avg_net >= 3.0:
                t: ToneLabel = "very_hawkish"
            elif avg_net >= 1.0:
                t = "hawkish"
            elif avg_net <= -3.0:
                t = "very_dovish"
            elif avg_net <= -1.0:
                t = "dovish"
            else:
                t = "neutral"
            return round(avg_net, 3), t

        fed_net, fed_tone = _agg_tone(fed_analyses)
        ecb_net, ecb_tone = _agg_tone(ecb_analyses)
        boe_net, boe_tone = _agg_tone(boe_analyses)

        # Global aggregate
        all_nets = [a.net_score for a in analyses]
        global_net = round(statistics.mean(all_nets), 3) if all_nets else 0.0
        if global_net >= 3.0:
            global_tone: ToneLabel = "very_hawkish"
        elif global_net >= 1.0:
            global_tone = "hawkish"
        elif global_net <= -3.0:
            global_tone = "very_dovish"
        elif global_net <= -1.0:
            global_tone = "dovish"
        else:
            global_tone = "neutral"

        # Tone trend: compare most recent half vs earlier half
        analyses_sorted = sorted(analyses, key=lambda a: a.meta.date, reverse=True)
        half = max(1, len(analyses_sorted) // 2)
        recent_nets = [a.net_score for a in analyses_sorted[:half]]
        older_nets = [a.net_score for a in analyses_sorted[half:]]
        avg_recent = statistics.mean(recent_nets) if recent_nets else 0.0
        avg_older = statistics.mean(older_nets) if older_nets else 0.0
        delta = avg_recent - avg_older
        if delta > 0.5:
            trend: Literal["more_hawkish", "more_dovish", "stable"] = "more_hawkish"
        elif delta < -0.5:
            trend = "more_dovish"
        else:
            trend = "stable"

        # Implied next move from Fed (most important)
        if fed_analyses:
            fed_signals = [a.rate_path_signal for a in fed_analyses]
            signal_counter = Counter(fed_signals)
            implied_next: RateSignal = signal_counter.most_common(1)[0][0]  # type: ignore[assignment]
        else:
            implied_next = "data_dependent"

        logger.info(
            "monitor_central_banks: complete",
            fed_tone=fed_tone,
            ecb_tone=ecb_tone,
            boe_tone=boe_tone,
            global_tone=global_tone,
            trend=trend,
            implied_next=implied_next,
            total_speeches=len(analyses),
        )

        return CentralBankMonitor(
            as_of=today,
            fed_tone=fed_tone,
            fed_net_score=fed_net,
            ecb_tone=ecb_tone,
            ecb_net_score=ecb_net,
            boe_tone=boe_tone,
            boe_net_score=boe_net,
            global_tone=global_tone,
            global_net_score=global_net,
            recent_speeches=analyses,
            tone_trend=trend,
            implied_next_move=implied_next,
        )

    def _empty_monitor(self, as_of: date) -> CentralBankMonitor:
        return CentralBankMonitor(
            as_of=as_of,
            fed_tone="neutral",
            fed_net_score=0.0,
            ecb_tone="neutral",
            ecb_net_score=0.0,
            boe_tone="neutral",
            boe_net_score=0.0,
            global_tone="neutral",
            global_net_score=0.0,
            recent_speeches=[],
            tone_trend="stable",
            implied_next_move="data_dependent",
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def monitor_central_banks(
    n_speeches_per_bank: int = 5,
    fetch_full_text: bool = True,
) -> CentralBankMonitor:
    """Run the full central bank monitor pipeline."""
    engine = FedSpeechNLP()
    return await engine.monitor_central_banks(
        n_speeches_per_bank=n_speeches_per_bank,
        fetch_full_text=fetch_full_text,
    )


async def analyze_latest_speech() -> SpeechAnalysis:
    """Fetch and analyze the most recent Fed speech."""
    engine = FedSpeechNLP()
    meta_list = await engine.get_recent_fed_speeches(n=1)
    if not meta_list:
        raise RuntimeError("No Fed speeches available")
    return await engine.analyze_speech_from_meta(meta_list[0])


async def fomc_minutes(meeting_date: date) -> FOMCMinutesAnalysis:
    """Download and analyze FOMC minutes for a specific meeting date."""
    engine = FedSpeechNLP()
    return await engine.get_fomc_minutes(meeting_date)


async def quick_tone_check(text: str) -> dict:
    """
    Quick convenience: score a block of text for hawkish/dovish tone.
    Returns a dict with all scoring details — useful for interactive analysis.
    """
    h_score, d_score, tone = compute_tone(text)
    net = round(h_score - d_score, 3)
    themes = identify_themes(text)
    rate_signal = extract_rate_signal(text, tone)
    hawkish_passages = extract_key_passages(text, [p for p, _ in HAWKISH_TERMS], n_sentences=3)
    dovish_passages = extract_key_passages(text, [p for p, _ in DOVISH_TERMS], n_sentences=3)

    return {
        "hawkish_score": h_score,
        "dovish_score": d_score,
        "net_score": net,
        "tone": tone,
        "rate_signal": rate_signal,
        "key_themes": themes,
        "hawkish_passages": hawkish_passages,
        "dovish_passages": dovish_passages,
        "word_count": len(text.split()),
    }
