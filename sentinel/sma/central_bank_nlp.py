"""
Multi-central-bank speech NLP: Fed, ECB, BOE, BOJ, BOC, RBA.
Hawkish/dovish scoring, stance change detection, policy rate prediction.
Free data: Fed website, ECB press releases, BIS speeches.

Significantly enhanced version of fed_speech_nlp.py — adds:
  - 6 central banks (Fed, ECB, BOE, BOJ, BOC, RBA)
  - 150+ hawkish and 150+ dovish terms with calibrated weights
  - Speaker credibility weighting (voting member > non-voter)
  - Uncertainty / modal-verb language detection
  - 90-day rolling StanceChangeDetector with pivot signals
  - Deep FOMCMinutesParser with quantifier-count weighting
  - PolicyRatePredictor via FRED FEDTARMD + hawk-score calibration
  - CrossCBAnalysis: FX divergence, synchronization index
  - FastAPI router with 7 endpoints
"""
from __future__ import annotations

import math
import re
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import feedparser
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Logging (reuse sentinel pattern if available, else stdlib)
# ---------------------------------------------------------------------------
try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & HTTP
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent.parent / ".danteforge" / "cb_nlp_cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/xhtml+xml,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Central bank data source URLs
# ---------------------------------------------------------------------------


class CentralBankSources:
    """URL registry for all supported central bank speech data sources."""

    FED_SPEECHES_JSON = "https://www.federalreserve.gov/json/ne-speeches.json"
    FED_SPEECHES_RSS = "https://www.federalreserve.gov/feeds/speeches.xml"
    FED_FOMC_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    FED_BASE = "https://www.federalreserve.gov"

    ECB_SPEECHES_RSS = "https://www.ecb.europa.eu/rss/speeches.rss"
    ECB_SPEECHES_PAGE = "https://www.ecb.europa.eu/press/key/html/index.en.html"

    BOE_SPEECHES_RSS = "https://www.bankofengland.co.uk/rss/speeches"
    BOE_SPEECHES_PAGE = "https://www.bankofengland.co.uk/news/speeches"

    BOJ_SPEECHES_PAGE = "https://www.boj.or.jp/en/announcements/press/index.htm"

    BOC_SPEECHES_PAGE = "https://www.bankofcanada.ca/research/speeches/"
    BOC_RSS = "https://www.bankofcanada.ca/feed/?cat=8"

    RBA_SPEECHES_PAGE = "https://www.rba.gov.au/speeches/"
    RBA_RSS = "https://www.rba.gov.au/rss/rss-cb-speeches.xml"

    BIS_SPEECHES = "https://www.bis.org/cbspeeches/index.htm"

    # FRED free CSV endpoints for policy rates
    FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    FRED_FEDTARMD = f"{FRED_BASE}?id=FEDTARMD"      # FOMC target rate midpoint
    FRED_FEDTARL = f"{FRED_BASE}?id=FEDTARL"        # target rate lower bound
    FRED_FEDTARU = f"{FRED_BASE}?id=FEDTARU"        # target rate upper bound
    FRED_ECBDFR = f"{FRED_BASE}?id=ECBDFR"          # ECB deposit facility rate
    FRED_BOEBR = f"{FRED_BASE}?id=BOEBR"            # BOE bank rate
    FRED_IRSTCB01JPM = f"{FRED_BASE}?id=IRSTCB01JPM156N"  # BOJ call rate

    @classmethod
    def policy_rate_url(cls, bank: str) -> str:
        mapping = {
            "fed": cls.FRED_FEDTARMD,
            "ecb": cls.FRED_ECBDFR,
            "boe": cls.FRED_BOEBR,
            "boj": cls.FRED_IRSTCB01JPM,
        }
        return mapping.get(bank.lower(), cls.FRED_FEDTARMD)


# ---------------------------------------------------------------------------
# Expanded Lexicons — 150+ terms each
# ---------------------------------------------------------------------------

# (phrase, weight) — weight > 0 for signal direction
HAWK_TERMS: dict[str, float] = {
    # Strong explicit hiking language
    "inflation remains elevated": 2.5,
    "inflation is too high": 2.5,
    "unacceptably high inflation": 2.5,
    "inflation well above target": 2.5,
    "inflation persistently above": 2.3,
    "inflation far from target": 2.2,
    "persistent inflation": 2.0,
    "elevated inflation": 1.8,
    "sticky inflation": 1.8,
    "entrenched inflation": 2.0,
    "inflation not yet defeated": 2.2,
    "premature to cut": 2.5,
    "premature to ease": 2.5,
    "not the time to cut": 2.5,
    "too soon to cut": 2.5,
    "not appropriate to cut": 2.3,
    "too early to declare victory": 2.2,
    "higher for longer": 2.5,
    "rates will remain elevated": 2.3,
    "keep rates elevated": 2.2,
    "maintain restrictive policy": 2.2,
    "additional tightening": 2.0,
    "further tightening": 2.0,
    "further increases": 2.0,
    "additional firming": 1.8,
    "additional increases": 1.8,
    "rate increase": 1.8,
    "rate hike": 1.8,
    "raise rates": 1.8,
    "increase rates": 1.8,
    "policy firming": 1.5,
    "tighten policy": 1.5,
    "tightening": 1.5,
    "restrictive": 1.5,
    "sufficiently restrictive": 2.0,
    "restrictive stance": 1.8,
    "above neutral": 1.5,
    "above the neutral rate": 1.5,
    "above equilibrium": 1.3,
    "above r-star": 1.5,
    "quantitative tightening": 1.5,
    "balance sheet reduction": 1.3,
    "reduce balance sheet": 1.3,
    "shrink balance sheet": 1.3,
    "runoff": 1.0,
    "qt": 0.8,
    "not yet confident": 2.0,
    "confidence not yet achieved": 2.0,
    "lack confidence": 1.8,
    "not confident": 1.8,
    "price stability is paramount": 2.0,
    "price stability mandate": 1.5,
    "determined to restore price stability": 2.0,
    "committed to returning inflation": 1.8,
    "return inflation to target": 1.5,
    "anchoring inflation expectations": 1.5,
    "unanchored expectations": 2.0,
    "second-round effects": 1.8,
    "wage-price spiral": 2.0,
    "wage growth excessive": 1.8,
    "wage pressure": 1.3,
    "wage inflation": 1.3,
    "tight labor market": 1.5,
    "labor market tight": 1.5,
    "strong labor market": 1.0,
    "robust employment": 0.8,
    "full employment achieved": 0.8,
    "overheating": 1.8,
    "economy overheating": 2.0,
    "demand outpacing supply": 1.5,
    "excess demand": 1.5,
    "above-potential growth": 1.3,
    "above potential": 1.2,
    "above trend": 1.0,
    "upside risk to inflation": 1.8,
    "upside risk": 1.3,
    "inflation expectations unanchored": 2.2,
    "deanchoring": 2.0,
    "deanchored expectations": 2.2,
    "inflation expectations drifting": 2.0,
    "vigilant": 1.3,
    "remain vigilant": 1.5,
    "highly attentive to inflation": 1.5,
    "combat inflation": 1.8,
    "fight inflation": 1.8,
    "reduce inflation": 1.3,
    "subdue inflation": 1.5,
    "control inflation": 1.3,
    "above the 2 percent": 1.5,
    "above 2%": 1.5,
    "above 2 percent": 1.5,
    "3 percent inflation": 1.5,
    "4 percent inflation": 2.0,
    "core inflation elevated": 1.8,
    "core pce elevated": 1.8,
    "core cpi elevated": 1.8,
    "services inflation persistent": 1.8,
    "supercore inflation": 1.5,
    "supply-side pressure": 0.8,
    "supply shock": 1.0,
    "commodity price pressure": 0.8,
    "energy price pressure": 0.8,
    "shelter inflation": 0.8,
    "rent inflation": 0.8,
    "financial conditions tightening": 1.3,
    "credit tightening": 1.0,
    "credit conditions tighter": 1.0,
    "lending standards tighter": 0.8,
    "monetary transmission": 0.5,
    "need to act": 1.5,
    "must act": 1.5,
    "need to tighten": 1.8,
    "raise the policy rate": 1.8,
    "increase the policy rate": 1.8,
    "policy rate increase": 1.8,
    "25 basis point": 0.5,
    "50 basis point": 1.0,
    "75 basis point": 1.5,
    "100 basis point": 2.0,
    "positive output gap": 1.2,
    "output above potential": 1.2,
    "neutral rate": 0.5,
    "real rates positive": 1.0,
    "real interest rates": 0.5,
    "terminal rate": 0.8,
    "peak rate higher": 1.5,
    "higher terminal rate": 1.8,
    "rate path higher": 1.5,
    "dot plot higher": 1.5,
    "median dot": 0.5,
    "disinflation progress stalling": 2.0,
    "disinflation has stalled": 2.0,
    "inflation progress insufficient": 1.8,
    "lack of progress on inflation": 1.8,
    "inflation not falling fast enough": 1.8,
    "hawkish": 1.0,
    "taper": 1.0,
    "tapering": 1.0,
    "normalization": 0.5,
    "policy normalization": 0.8,
    "rate normalization": 0.8,
    "lift off": 0.8,
    "liftoff": 0.8,
    "interest rate lift": 0.8,
    "hike": 1.5,
    "hiking": 1.5,
    "overshoot": 1.5,
    "overshooting target": 1.8,
    "overshoot target": 1.8,
    "tolerate higher inflation risk": 0.8,
    "inflation risks skewed upward": 1.5,
    "inflation risks to the upside": 1.5,
    "balance of risks upside inflation": 1.5,
    "labor costs rising": 1.0,
    "unit labor costs elevated": 1.2,
    "productivity insufficient to offset": 1.0,
    "inflation not transitory": 2.0,
    "not transitory": 1.8,
    "inflation is not temporary": 1.8,
    "price pressures broad-based": 1.5,
    "broad-based inflation": 1.5,
    "widespread price increases": 1.5,
    "geopolitical risk upside inflation": 1.0,
    "oil price upside": 0.8,
    "energy shock": 1.0,
    "supply disruption": 0.8,
}

DOVE_TERMS: dict[str, float] = {
    # Strong explicit cutting / easing language
    "cut rates": 2.5,
    "rate cut": 2.5,
    "lower rates": 2.2,
    "reduce rates": 2.2,
    "decrease rates": 2.0,
    "ease policy": 2.0,
    "easing": 1.8,
    "monetary easing": 2.0,
    "policy easing": 2.0,
    "begin to ease": 2.2,
    "begin easing": 2.2,
    "ease monetary policy": 2.2,
    "accommodate": 1.5,
    "accommodative": 1.8,
    "remain accommodative": 2.0,
    "keep policy accommodative": 2.0,
    "policy accommodation": 1.8,
    "support the economy": 1.5,
    "support growth": 1.5,
    "support employment": 1.5,
    "support demand": 1.3,
    "provide support": 1.0,
    "economic support": 1.0,
    "policy support": 1.0,
    "stimulus": 1.5,
    "fiscal and monetary support": 1.3,
    "quantitative easing": 1.5,
    "asset purchase": 1.3,
    "asset purchase program": 1.5,
    "bond buying": 1.3,
    "expand balance sheet": 1.5,
    "expand asset purchases": 1.8,
    "qe": 1.0,
    "yield curve control": 1.5,
    "ycc": 1.2,
    "forward guidance": 0.8,
    "below target": 2.0,
    "inflation below target": 2.2,
    "below 2 percent": 1.8,
    "below our 2%": 1.8,
    "below our goal": 1.8,
    "below mandate": 1.8,
    "below the target": 1.8,
    "undershoot": 1.5,
    "undershooting": 1.5,
    "disinflation": 1.8,
    "disinflation underway": 2.0,
    "inflation declining": 1.8,
    "inflation falling": 1.8,
    "inflation cooling": 1.8,
    "inflation coming down": 1.8,
    "inflation is easing": 1.8,
    "inflation moderating": 1.8,
    "inflation normalizing": 1.8,
    "inflation progress": 1.5,
    "good progress on inflation": 1.8,
    "confident inflation is declining": 2.0,
    "confident disinflation": 2.0,
    "inflation well-anchored": 2.0,
    "expectations well-anchored": 1.8,
    "expectations anchored": 1.8,
    "anchor inflation expectations": 1.5,
    "inflation expectations stable": 1.5,
    "inflation expectations consistent with target": 1.8,
    "downside risk to inflation": 1.8,
    "downside risk": 1.3,
    "downside risk to growth": 1.5,
    "growth slowing": 1.5,
    "slowing growth": 1.5,
    "economic slowdown": 1.5,
    "growth concerns": 1.5,
    "recession risk": 2.0,
    "recession concerns": 1.8,
    "growth below potential": 1.5,
    "below trend growth": 1.5,
    "below potential": 1.3,
    "output gap negative": 1.8,
    "negative output gap": 1.8,
    "slack in the economy": 1.8,
    "economic slack": 1.8,
    "spare capacity": 1.8,
    "labor market softening": 2.0,
    "labor market cooling": 2.0,
    "cooling labor": 1.8,
    "loosening labor market": 1.8,
    "labor market loosening": 1.8,
    "unemployment rising": 2.0,
    "unemployment increased": 1.8,
    "rising unemployment": 2.0,
    "job losses": 1.8,
    "layoffs increasing": 1.8,
    "payrolls weakening": 1.8,
    "soft labor market": 1.8,
    "labor demand falling": 1.5,
    "job openings declining": 1.5,
    "below full employment": 1.8,
    "employment below mandate": 1.8,
    "maximum employment": 0.5,
    "patient": 1.2,
    "patience": 1.0,
    "gradual": 0.8,
    "cautious": 0.8,
    "careful": 0.5,
    "measured": 0.8,
    "gradual approach": 1.0,
    "proceed carefully": 1.0,
    "data dependent": 1.0,
    "data-dependent": 1.0,
    "meeting by meeting": 1.0,
    "meeting-by-meeting": 1.0,
    "incoming data": 0.8,
    "monitor": 0.5,
    "monitoring": 0.5,
    "flexible": 0.7,
    "balance risks": 0.5,
    "two-sided risks": 0.5,
    "dual mandate": 0.5,
    "employment side of mandate": 1.0,
    "employment mandate": 0.8,
    "maximum employment goal": 0.8,
    "pivot": 1.8,
    "policy pivot": 2.0,
    "dovish pivot": 2.2,
    "rate reduction": 1.5,
    "policy rate reduction": 1.8,
    "hold rates": 0.8,
    "keep rates": 0.8,
    "rates on hold": 0.8,
    "pause": 1.3,
    "pausing": 1.3,
    "hold steady": 0.8,
    "no change to rates": 0.8,
    "unchanged policy rate": 0.8,
    "transitory": 1.0,
    "temporary inflation": 1.2,
    "supply chain improvement": 0.8,
    "supply chain normalization": 1.0,
    "supply chain easing": 1.0,
    "supply conditions improving": 0.8,
    "supply side resolved": 1.0,
    "real wages declining": 0.5,
    "consumer spending weakening": 1.3,
    "consumer spending slowing": 1.2,
    "retail sales declining": 1.0,
    "housing market weakening": 1.0,
    "housing slowdown": 1.0,
    "credit growth slowing": 0.8,
    "bank lending falling": 0.8,
    "credit contraction": 1.2,
    "financial conditions easing": 1.0,
    "financial conditions loosening": 1.0,
    "tightening financial conditions hurting": 1.3,
    "policy already sufficiently tight": 1.5,
    "rates already high enough": 1.5,
    "no need for further hikes": 2.0,
    "done hiking": 2.0,
    "at peak": 1.5,
    "at the peak": 1.5,
    "peak rate achieved": 1.8,
    "reached sufficient restriction": 1.8,
    "risk of overtightening": 1.8,
    "overtightening risk": 1.8,
    "avoid overtightening": 1.8,
    "not overtighten": 2.0,
    "global headwinds": 1.0,
    "external headwinds": 1.0,
    "global uncertainty": 0.8,
    "banking stress": 1.2,
    "banking sector concerns": 1.2,
    "financial stability concerns": 1.2,
    "dovish": 1.0,
    "below-target": 1.8,
    "well below target": 2.0,
    "inflation too low": 2.0,
    "deflationary": 2.0,
    "deflation risk": 2.2,
    "low inflation": 1.5,
    "low inflation environment": 1.5,
}

NEUTRAL_TERMS: set[str] = {
    "data-dependent", "data dependent", "monitor", "monitoring",
    "balanced", "two-sided", "flexible", "gradual", "measured",
    "appropriate", "as appropriate", "if warranted", "if needed",
    "depending on data", "based on incoming data", "conditional",
    "incoming information", "evolving outlook", "uncertainty",
    "uncertain", "watch carefully", "remain watchful",
}

MODAL_VERBS: set[str] = {"may", "might", "could", "would", "should", "can", "ought"}

UNCERTAINTY_PHRASES: list[str] = [
    "if appropriate", "if warranted", "if needed", "as warranted",
    "depending on the data", "data dependent", "data-dependent",
    "meeting by meeting", "based on incoming", "incoming data",
    "remain flexible", "if conditions warrant", "subject to",
    "uncertain outlook", "considerable uncertainty", "elevated uncertainty",
    "high uncertainty", "significant uncertainty",
    "may be appropriate", "might be appropriate", "could be appropriate",
    "could consider", "might consider", "may need",
]

# Negation tokens
_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor", "without", "hardly", "barely",
    "scarcely", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "wouldn't", "couldn't", "shouldn't", "won't", "don't", "didn't",
    "cannot", "can't", "less", "unlikely", "insufficient", "fail", "failed",
    "failing", "absence", "absent", "lack", "lacking", "little", "limited",
})

# ---------------------------------------------------------------------------
# Speaker credibility weights
# ---------------------------------------------------------------------------

FOMC_VOTING_MEMBERS_2024: set[str] = {
    # Chair + governors always vote; 4 rotating regional presidents
    "powell", "jefferson", "cook", "kugler", "waller", "bowman",
    "williams", "bostic", "barkin", "daly", "kashkari",
}

FOMC_REGIONAL_NON_VOTERS_2024: set[str] = {
    "mester", "harker", "bullard", "george", "evans",
    "rosengren", "kaplan", "barr", "logan",
}

ECB_GC_MEMBERS: set[str] = {
    "lagarde", "lane", "guindos", "schnabel", "nagel", "villeroy",
    "wunsch", "centeno", "rehn", "kazaks", "vasle", "simkus",
    "holzmann", "stournaras", "panetta", "visco",
}


def speaker_credibility(speaker: str, institution: str) -> float:
    """Return credibility weight 0.6–1.2 based on speaker seniority."""
    name_lower = speaker.lower()
    if institution == "fed":
        # Chair
        if "powell" in name_lower:
            return 1.2
        # Governors / Vice Chair
        if any(n in name_lower for n in FOMC_VOTING_MEMBERS_2024):
            return 1.1
        if any(n in name_lower for n in FOMC_REGIONAL_NON_VOTERS_2024):
            return 0.8
        return 0.75  # unknown regional
    if institution == "ecb":
        if "lagarde" in name_lower:
            return 1.2
        if "lane" in name_lower or "guindos" in name_lower:
            return 1.1
        if any(n in name_lower for n in ECB_GC_MEMBERS):
            return 1.0
        return 0.8
    if institution in ("boe", "boj", "boc", "rba"):
        # Governor
        if any(k in name_lower for k in ["bailey", "ueda", "macklem", "bullock"]):
            return 1.2
        return 0.9
    return 0.85


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Strip HTML tags, expand entities, collapse whitespace."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    for entity, repl in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                          ("&gt;", ">"), ("&#160;", " ")]:
        html = html.replace(entity, repl)
    html = re.sub(r"&#\d+;", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|U\.S|U\.K|e\.g|i\.e|approx|est|Fig|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.",
        r"\1<DOT>",
        text,
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if len(p.strip()) > 10]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z'-]+", text.lower())


def _is_negated(tokens: list[str], match_start_token: int, window: int = 8) -> bool:
    start = max(0, match_start_token - window)
    return any(t in _NEGATORS for t in tokens[start:match_start_token])


def _count_weighted_terms(
    text: str,
    terms: dict[str, float],
    negate_weight: float = -0.4,
) -> tuple[float, list[tuple[str, int]]]:
    """
    Count weighted occurrences of multi-word terms with negation detection.
    Returns (total_weighted_score, [(term, count)]).
    """
    lower = text.lower()
    tokens = _tokenize(lower)
    total_score = 0.0
    hits: list[tuple[str, int]] = []

    for phrase, weight in terms.items():
        phrase_lower = phrase.lower()
        count = 0
        pos = 0
        while True:
            idx = lower.find(phrase_lower, pos)
            if idx == -1:
                break
            token_pos = len(_tokenize(lower[:idx]))
            neg = _is_negated(tokens, token_pos, window=8)
            effective_weight = negate_weight if neg else weight
            total_score += effective_weight
            count += 1
            pos = idx + len(phrase_lower)
        if count:
            hits.append((phrase, count))

    return max(0.0, total_score), hits


def _modal_uncertainty_score(text: str) -> float:
    """Score 0–3 for uncertainty language in the text."""
    lower = text.lower()
    tokens = _tokenize(lower)
    modal_count = sum(1 for t in tokens if t in MODAL_VERBS)
    unc_phrase_count = sum(1 for p in UNCERTAINTY_PHRASES if p in lower)
    # Normalised: modal_count per 100 words + unc phrase density
    word_count = max(1, len(text.split()))
    score = (modal_count / word_count * 100) * 0.5 + unc_phrase_count * 0.3
    return round(min(score, 3.0), 3)


def _rss_extract(tag: str, xml: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>", xml, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_date_str(s: str) -> date:
    for fmt in (
        "%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%Y/%m/%d",
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT",
        "%d %b %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(s.strip()[:30], fmt).date()
        except (ValueError, AttributeError):
            continue
    return date.today()


def _fetch_text(url: str, max_chars: int = 80_000) -> str:
    """Fetch URL, strip HTML, return plain text."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return _strip_html(resp.text)[:max_chars]
    except Exception as exc:
        logger.warning(f"_fetch_text failed: {url} — {exc}")
        return ""


def _fetch_raw(url: str) -> str:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning(f"_fetch_raw failed: {url} — {exc}")
        return ""


# ---------------------------------------------------------------------------
# SQLite cache helpers
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cb_nlp_cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            ts REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hawk_score_history (
            bank TEXT,
            score_date TEXT,
            hawk_score REAL,
            dove_score REAL,
            net_score REAL,
            credibility_weight REAL,
            speech_count INTEGER,
            PRIMARY KEY (bank, score_date)
        )
    """)
    conn.commit()
    return conn


def _cache_get(key: str, ttl: float = 3600.0) -> Optional[str]:
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT value, ts FROM cb_nlp_cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row and (time.time() - row[1]) < ttl:
            return row[0]
    except Exception:
        pass
    return None


def _cache_set(key: str, value: str) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO cb_nlp_cache (key, value, ts) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _store_hawk_score(
    bank: str,
    score_date: str,
    hawk: float,
    dove: float,
    net: float,
    cred: float,
    count: int,
) -> None:
    try:
        conn = _get_db()
        conn.execute(
            """INSERT OR REPLACE INTO hawk_score_history
               (bank, score_date, hawk_score, dove_score, net_score, credibility_weight, speech_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (bank, score_date, hawk, dove, net, cred, count),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _load_hawk_history(bank: str, days: int = 200) -> pd.DataFrame:
    try:
        conn = _get_db()
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM hawk_score_history WHERE bank = ? AND score_date >= ? ORDER BY score_date",
            conn,
            params=(bank, cutoff),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

Institution = Literal["fed", "ecb", "boe", "boj", "boc", "rba", "bis", "other"]
ToneLabel = Literal["very_hawkish", "hawkish", "neutral", "dovish", "very_dovish"]
RateSignal = Literal["hike", "hold", "cut", "data_dependent", "unknown"]
PivotSignal = Literal["pivot_hawkish", "pivot_dovish", "stable", "watch"]


class SpeechMeta(BaseModel):
    model_config = ConfigDict(frozen=True)
    speaker: str
    title: str
    date: date
    url: str
    institution: Institution
    event: Optional[str] = None
    is_voting_member: bool = False


class SpeechAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)
    meta: SpeechMeta
    hawkish_score: float
    dovish_score: float
    net_score: float
    credibility_adjusted_net: float
    tone: ToneLabel
    uncertainty_score: float
    hawkish_passages: list[str]
    dovish_passages: list[str]
    key_themes: list[str]
    rate_path_signal: RateSignal
    word_count: int
    sentence_count: int
    rate_mentioned: Optional[float] = None
    inflation_mentioned: Optional[float] = None
    gdp_mentioned: Optional[float] = None


class FOMCMinutesDetail(BaseModel):
    model_config = ConfigDict(frozen=True)
    meeting_date: date
    release_date: date
    minutes_url: str
    consensus_view: str
    hawkish_score: float
    dovish_score: float
    net_score: float
    tone: ToneLabel
    dissents_count: int
    data_dependencies: list[str]
    rate_guidance: str
    key_themes: list[str]
    participant_counts: dict[str, int]  # {"many": 3, "most": 2, "some": 5, "few": 1}
    quantifier_weighted_score: float
    committee_discussion_tone: str
    participants_view: str
    forward_guidance_phrases: list[str]
    staff_economic_projection: Optional[str] = None
    word_count: int


class StanceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    bank: Institution
    as_of: date
    current_net_score: float
    prior_90d_net_score: float
    rolling_90d_net_score: float
    pivot_signal: PivotSignal
    taper_tantrum_warning: bool
    language_shift_phrases: list[str]
    trend_description: str


class RatePrediction(BaseModel):
    model_config = ConfigDict(frozen=True)
    bank: Institution
    as_of: date
    current_rate: float
    market_implied_next: float
    prob_hike: float
    prob_hold: float
    prob_cut: float
    hawk_implied_direction: RateSignal
    terminal_rate_estimate: float
    next_meeting_date: Optional[date] = None


class CrossCBDivergence(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    pair: str               # e.g. "FED_ECB"
    fed_net: float
    ecb_net: float
    divergence: float       # fed_net - ecb_net
    fx_implication: str
    rate_differential: float
    synchronization_index: float  # 0–1, 1 = fully in sync


class CentralBankMonitorResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    bank_scores: dict[str, float]
    bank_tones: dict[str, ToneLabel]
    global_net_score: float
    global_tone: ToneLabel
    tone_trend: Literal["more_hawkish", "more_dovish", "stable"]
    implied_next_move: RateSignal
    recent_speeches: list[SpeechAnalysis]
    pivot_signals: list[StanceSnapshot]


# ---------------------------------------------------------------------------
# THEME_KEYWORDS (expanded)
# ---------------------------------------------------------------------------

THEME_KEYWORDS: dict[str, list[str]] = {
    "inflation": [
        "inflation", "price", "cpi", "pce", "disinflation", "deflation",
        "price stability", "consumer price", "core inflation", "headline",
        "price level", "price pressures", "inflationary", "cost of living",
    ],
    "employment": [
        "employment", "jobs", "labor", "labour", "unemployment", "payroll",
        "job openings", "jolts", "hiring", "layoffs", "wage", "workers",
        "workforce", "job market", "labor market", "labour market",
    ],
    "growth": [
        "growth", "gdp", "output", "recession", "expansion", "contraction",
        "economic activity", "productivity", "real gdp", "economic growth",
        "growth outlook", "economic outlook",
    ],
    "financial_stability": [
        "financial stability", "banking", "credit", "liquidity", "stress",
        "systemic risk", "bank failure", "deposit", "financial conditions",
        "credit crunch", "banking sector", "financial system",
    ],
    "global": [
        "global", "international", "trade", "geopolitical", "china",
        "europe", "emerging market", "supply chain", "dollar", "global economy",
        "global growth", "trade war", "tariffs",
    ],
    "housing": [
        "housing", "mortgage", "real estate", "home price", "rent",
        "shelter", "housing market", "residential investment",
    ],
    "monetary_policy": [
        "federal funds rate", "fed funds", "target range", "balance sheet",
        "forward guidance", "dot plot", "interest rate", "policy rate",
        "quantitative", "open market", "policy stance", "monetary policy",
    ],
    "exchange_rate": [
        "exchange rate", "currency", "dollar", "euro", "yen", "sterling",
        "fx", "foreign exchange", "appreciation", "depreciation", "weaker currency",
    ],
    "credit": [
        "credit", "lending", "bank loans", "spreads", "high yield",
        "investment grade", "credit conditions", "credit growth",
    ],
}


# ---------------------------------------------------------------------------
# Core NLP: compute_tone
# ---------------------------------------------------------------------------

def compute_tone(text: str) -> tuple[float, float, ToneLabel]:
    """
    Compute hawkish and dovish weighted scores and derive a tone label.
    Returns (hawkish_score, dovish_score, tone_label).
    """
    if not text or not text.strip():
        return 0.0, 0.0, "neutral"

    h_score, _ = _count_weighted_terms(text, HAWK_TERMS)
    d_score, _ = _count_weighted_terms(text, DOVE_TERMS)

    h_capped = round(min(h_score, 10.0), 3)
    d_capped = round(min(d_score, 10.0), 3)
    net = round(h_capped - d_capped, 3)

    if net >= 3.5:
        tone: ToneLabel = "very_hawkish"
    elif net >= 1.0:
        tone = "hawkish"
    elif net <= -3.5:
        tone = "very_dovish"
    elif net <= -1.0:
        tone = "dovish"
    else:
        tone = "neutral"

    return h_capped, d_capped, tone


def extract_key_passages(text: str, keywords: list[str], n_sentences: int = 3) -> list[str]:
    sentences = _split_sentences(text)
    scored: list[tuple[int, str]] = []
    for sent in sentences:
        sent_lower = sent.lower()
        hits = sum(sent_lower.count(kw.lower()) for kw in keywords)
        if hits > 0:
            scored.append((hits, sent.strip()))
    seen: set[str] = set()
    unique: list[str] = []
    for _, sent in sorted(scored, key=lambda x: x[0], reverse=True):
        norm = " ".join(sent.split())
        if norm not in seen and len(sent) > 20:
            seen.add(norm)
            unique.append(sent)
            if len(unique) >= n_sentences:
                break
    return unique


def identify_themes(text: str) -> list[str]:
    lower = text.lower()
    return [
        theme for theme, kws in THEME_KEYWORDS.items()
        if sum(1 for kw in kws if kw.lower() in lower) >= 2
    ]


def extract_rate_signal(text: str, tone: ToneLabel) -> RateSignal:
    lower = text.lower()
    hike_phrases = [
        "further increases", "additional firming", "rate increase", "raise rates",
        "rate hike", "tighten further", "additional tightening", "need to raise",
        "may be appropriate to raise", "hike rates", "increase the policy rate",
        "raise the policy rate", "further tightening warranted",
    ]
    cut_phrases = [
        "rate cut", "cut rates", "lower rates", "reduce rates", "ease policy",
        "begin to ease", "begin easing", "pivot to easing", "rate reduction",
        "we will cut", "appropriate to cut", "lower the policy rate",
        "reduce the policy rate", "cutting rates", "easing cycle",
    ]
    hold_phrases = [
        "hold rates", "keep rates", "maintain rates", "rates on hold",
        "pause", "hold steady", "no change", "unchanged", "keep policy rate",
        "appropriate to hold", "leave rates unchanged", "rates unchanged",
    ]
    data_dep_phrases = [
        "data dependent", "data-dependent", "meeting by meeting",
        "incoming data", "based on incoming", "depend on data",
        "depends on the data", "monitor incoming", "monitor the data",
    ]

    hike_hits = sum(1 for p in hike_phrases if p in lower)
    cut_hits = sum(1 for p in cut_phrases if p in lower)
    hold_hits = sum(1 for p in hold_phrases if p in lower)
    dd_hits = sum(1 for p in data_dep_phrases if p in lower)

    max_explicit = max(hike_hits, cut_hits, hold_hits)
    if max_explicit > 0:
        if hike_hits == max_explicit and hike_hits >= cut_hits:
            return "hike"
        if cut_hits == max_explicit and cut_hits >= hike_hits:
            return "cut"
        if hold_hits == max_explicit:
            return "hold"

    if dd_hits >= 1:
        return "data_dependent"

    if tone in ("very_hawkish", "hawkish"):
        return "hike"
    if tone in ("very_dovish", "dovish"):
        return "cut"
    return "data_dependent"


def _extract_numbers(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern + r"[^\d]*(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# HawkDoveClassifier
# ---------------------------------------------------------------------------

class HawkDoveClassifier:
    """
    Sentence-level hawkish/dovish scoring with:
    - Negation detection (8-token context window)
    - Speaker credibility adjustment
    - Uncertainty language detection
    - Multi-level tone classification
    """

    def __init__(self, speaker_credibility_weight: float = 1.0):
        self.credibility = speaker_credibility_weight

    def classify_text(self, text: str) -> dict[str, Any]:
        """Full classification of a text block."""
        h_score, h_hits = _count_weighted_terms(text, HAWK_TERMS)
        d_score, d_hits = _count_weighted_terms(text, DOVE_TERMS)
        h_capped = min(h_score, 10.0)
        d_capped = min(d_score, 10.0)
        net = h_capped - d_capped
        net_adjusted = net * self.credibility
        uncertainty = _modal_uncertainty_score(text)

        if net >= 3.5:
            tone: ToneLabel = "very_hawkish"
        elif net >= 1.0:
            tone = "hawkish"
        elif net <= -3.5:
            tone = "very_dovish"
        elif net <= -1.0:
            tone = "dovish"
        else:
            tone = "neutral"

        return {
            "hawkish_score": round(h_capped, 3),
            "dovish_score": round(d_capped, 3),
            "net_score": round(net, 3),
            "net_adjusted": round(net_adjusted, 3),
            "tone": tone,
            "uncertainty_score": uncertainty,
            "top_hawk_hits": sorted(h_hits, key=lambda x: x[1], reverse=True)[:5],
            "top_dove_hits": sorted(d_hits, key=lambda x: x[1], reverse=True)[:5],
            "word_count": len(text.split()),
        }

    def classify_sentences(self, text: str) -> list[dict[str, Any]]:
        """Classify each sentence individually, return sorted by |net_score|."""
        results = []
        for sent in _split_sentences(text):
            if len(sent.split()) < 5:
                continue
            cls = self.classify_text(sent)
            cls["sentence"] = sent
            results.append(cls)
        return sorted(results, key=lambda x: abs(x["net_score"]), reverse=True)

    def extract_key_signals(self, text: str, top_n: int = 5) -> dict[str, list[str]]:
        """Return the top hawkish and dovish sentences."""
        sentences = self.classify_sentences(text)
        hawk_sents = [s["sentence"] for s in sentences if s["net_score"] > 0][:top_n]
        dove_sents = [s["sentence"] for s in sentences if s["net_score"] < 0][:top_n]
        return {"hawkish": hawk_sents, "dovish": dove_sents}


# ---------------------------------------------------------------------------
# StanceChangeDetector
# ---------------------------------------------------------------------------

TAPER_TANTRUM_PHRASES: list[str] = [
    "reduce asset purchases", "tapering asset purchases", "taper",
    "reduce pace of purchases", "wind down", "wind down asset purchases",
    "begin to reduce", "gradually reduce", "scale back purchases",
    "QE tapering", "end asset purchases", "halt asset purchases",
    "reduce monthly purchases", "reduce the pace of asset purchases",
]


class StanceChangeDetector:
    """
    Detect pivots in central bank communication using 90-day rolling windows.
    """

    def __init__(self, bank: Institution):
        self.bank = bank

    def detect_pivot(self, speech_analyses: list[SpeechAnalysis]) -> StanceSnapshot:
        """
        Compare rolling 90-day hawk score vs prior 90 days.
        Classify pivot signal.
        """
        today = date.today()
        cutoff_recent = today - timedelta(days=90)
        cutoff_prior = today - timedelta(days=180)

        recent = [s for s in speech_analyses if s.meta.date >= cutoff_recent]
        prior = [
            s for s in speech_analyses
            if cutoff_prior <= s.meta.date < cutoff_recent
        ]

        recent_avg = (
            statistics.mean(s.credibility_adjusted_net for s in recent)
            if recent else 0.0
        )
        prior_avg = (
            statistics.mean(s.credibility_adjusted_net for s in prior)
            if prior else 0.0
        )

        all_90d = [
            s for s in speech_analyses
            if s.meta.date >= cutoff_recent
        ]
        rolling_avg = (
            statistics.mean(s.credibility_adjusted_net for s in all_90d)
            if all_90d else 0.0
        )

        delta = recent_avg - prior_avg
        if delta >= 0.3:
            pivot: PivotSignal = "pivot_hawkish"
            trend_desc = f"Stance shifted hawkish by {delta:.2f} pts over 90 days"
        elif delta <= -0.3:
            pivot = "pivot_dovish"
            trend_desc = f"Stance shifted dovish by {abs(delta):.2f} pts over 90 days"
        elif abs(delta) >= 0.1:
            pivot = "watch"
            trend_desc = f"Small shift of {delta:.2f} pts — watch for follow-through"
        else:
            pivot = "stable"
            trend_desc = f"Stance stable, 90d change: {delta:.2f} pts"

        # Taper tantrum warning
        all_texts = " ".join(
            s.meta.title for s in recent
        ).lower()
        taper_warning = any(p.lower() in all_texts for p in TAPER_TANTRUM_PHRASES)

        # Language shift: phrases appearing more in recent vs prior
        recent_titles = " ".join(s.meta.title for s in recent).lower()
        prior_titles = " ".join(s.meta.title for s in prior).lower()
        shift_phrases: list[str] = []
        for phrase in list(HAWK_TERMS.keys())[:20] + list(DOVE_TERMS.keys())[:20]:
            recent_count = recent_titles.count(phrase.lower())
            prior_count = prior_titles.count(phrase.lower())
            if recent_count > prior_count + 1:
                shift_phrases.append(phrase)

        return StanceSnapshot(
            bank=self.bank,
            as_of=today,
            current_net_score=round(recent_avg, 3),
            prior_90d_net_score=round(prior_avg, 3),
            rolling_90d_net_score=round(rolling_avg, 3),
            pivot_signal=pivot,
            taper_tantrum_warning=taper_warning,
            language_shift_phrases=shift_phrases[:10],
            trend_description=trend_desc,
        )

    def detect_from_db(self) -> StanceSnapshot:
        """Use stored hawk score history to detect stance changes."""
        df = _load_hawk_history(self.bank, days=200)
        if df.empty:
            return StanceSnapshot(
                bank=self.bank,
                as_of=date.today(),
                current_net_score=0.0,
                prior_90d_net_score=0.0,
                rolling_90d_net_score=0.0,
                pivot_signal="stable",
                taper_tantrum_warning=False,
                language_shift_phrases=[],
                trend_description="No history available",
            )

        df["score_date"] = pd.to_datetime(df["score_date"])
        today = pd.Timestamp(date.today())
        cutoff_recent = today - pd.Timedelta(days=90)
        cutoff_prior = today - pd.Timedelta(days=180)

        recent = df[df["score_date"] >= cutoff_recent]
        prior = df[(df["score_date"] >= cutoff_prior) & (df["score_date"] < cutoff_recent)]

        recent_avg = float(recent["net_score"].mean()) if not recent.empty else 0.0
        prior_avg = float(prior["net_score"].mean()) if not prior.empty else 0.0
        delta = recent_avg - prior_avg

        if delta >= 0.3:
            pivot: PivotSignal = "pivot_hawkish"
            trend_desc = f"DB: hawkish shift {delta:.2f} pts"
        elif delta <= -0.3:
            pivot = "pivot_dovish"
            trend_desc = f"DB: dovish shift {abs(delta):.2f} pts"
        else:
            pivot = "stable"
            trend_desc = f"DB: stable, delta={delta:.2f}"

        return StanceSnapshot(
            bank=self.bank,
            as_of=date.today(),
            current_net_score=round(recent_avg, 3),
            prior_90d_net_score=round(prior_avg, 3),
            rolling_90d_net_score=round(recent_avg, 3),
            pivot_signal=pivot,
            taper_tantrum_warning=False,
            language_shift_phrases=[],
            trend_description=trend_desc,
        )


# ---------------------------------------------------------------------------
# FOMCMinutesParser
# ---------------------------------------------------------------------------

QUANTIFIER_WEIGHTS: dict[str, float] = {
    "all participants": 1.0,
    "most participants": 0.85,
    "many participants": 0.70,
    "several participants": 0.55,
    "some participants": 0.45,
    "a few participants": 0.30,
    "a couple of participants": 0.20,
    "one participant": 0.15,
    "all members": 1.0,
    "most members": 0.85,
    "many members": 0.70,
    "several members": 0.55,
    "some members": 0.45,
    "a few members": 0.30,
}

QUANTIFIER_PATTERN = re.compile(
    r"(all participants?|most participants?|many participants?|"
    r"several participants?|some participants?|a few participants?|"
    r"a couple of participants?|one participant|"
    r"all members?|most members?|many members?|"
    r"several members?|some members?|a few members?)",
    re.IGNORECASE,
)

FORWARD_GUIDANCE_TRIGGERS = [
    "expect", "anticipate", "will be", "would be", "likely to",
    "projected", "forecast", "path of", "rate path",
    "over the coming", "in the near term", "over the next",
]


class FOMCMinutesParser:
    """
    Deep parser for FOMC minutes documents.
    """

    def __init__(self):
        self._base_url = CentralBankSources.FED_BASE

    def minutes_url(self, meeting_date: date) -> str:
        return f"{self._base_url}/monetarypolicy/fomcminutes{meeting_date.strftime('%Y%m%d')}.htm"

    def fetch_minutes(self, meeting_date: date) -> str:
        url = self.minutes_url(meeting_date)
        cached = _cache_get(f"fomc_minutes_{meeting_date.isoformat()}", ttl=86400)
        if cached:
            return cached
        text = _fetch_text(url, max_chars=150_000)
        if text:
            _cache_set(f"fomc_minutes_{meeting_date.isoformat()}", text)
        return text

    def _extract_participant_counts(self, text: str) -> dict[str, int]:
        """Count occurrences of each quantifier phrase."""
        matches = QUANTIFIER_PATTERN.findall(text)
        counts: dict[str, int] = Counter(
            m.lower().strip() for m in matches
        )
        return dict(counts)

    def _compute_quantifier_weighted_score(
        self, text: str, participant_counts: dict[str, int]
    ) -> float:
        """
        Weight hawk/dove passages by quantifier strength.
        'Many participants noted inflation elevated' → high weight.
        """
        sentences = _split_sentences(text)
        total_weighted = 0.0
        total_weight = 0.0
        for sent in sentences:
            sent_lower = sent.lower()
            # Find max quantifier weight for this sentence
            q_weight = 0.5  # default (no quantifier)
            for q, w in QUANTIFIER_WEIGHTS.items():
                if q in sent_lower:
                    q_weight = max(q_weight, w)
            h, d, _ = compute_tone(sent)
            net = h - d
            total_weighted += net * q_weight
            total_weight += q_weight
        if total_weight == 0:
            return 0.0
        return round(total_weighted / total_weight, 3)

    def _extract_forward_guidance(self, text: str) -> list[str]:
        sentences = _split_sentences(text)
        guidance: list[str] = []
        for sent in sentences:
            lower = sent.lower()
            has_trigger = any(t in lower for t in FORWARD_GUIDANCE_TRIGGERS)
            has_policy = any(
                k in lower for k in ["rate", "cut", "hike", "ease", "tighten", "policy"]
            )
            if has_trigger and has_policy and len(sent) > 30:
                guidance.append(sent.strip())
        return guidance[:5]

    def _extract_data_dependencies(self, text: str) -> list[str]:
        trigger_patterns = [
            r"if\s+\w+", r"should\s+inflation", r"provided that", r"conditional",
            r"depending on", r"in the event", r"when\s+inflation",
            r"once\s+\w+", r"before\s+we", r"until\s+\w+", r"as long as",
        ]
        sentences = _split_sentences(text)
        deps: list[str] = []
        for sent in sentences:
            lower = sent.lower()
            if any(re.search(p, lower) for p in trigger_patterns):
                if any(kw in lower for kw in ["rate", "cut", "hike", "ease", "tighten", "policy"]):
                    deps.append(sent.strip())
        return deps[:5]

    def parse(self, meeting_date: date) -> FOMCMinutesDetail:
        text = self.fetch_minutes(meeting_date)
        release_date = meeting_date + timedelta(days=21)
        url = self.minutes_url(meeting_date)

        if not text:
            return FOMCMinutesDetail(
                meeting_date=meeting_date,
                release_date=release_date,
                minutes_url=url,
                consensus_view="unknown",
                hawkish_score=0.0,
                dovish_score=0.0,
                net_score=0.0,
                tone="neutral",
                dissents_count=0,
                data_dependencies=[],
                rate_guidance="Not available — fetch failed",
                key_themes=[],
                participant_counts={},
                quantifier_weighted_score=0.0,
                committee_discussion_tone="neutral",
                participants_view="unknown",
                forward_guidance_phrases=[],
                staff_economic_projection=None,
                word_count=0,
            )

        h_score, d_score, tone = compute_tone(text)
        net_score = round(h_score - d_score, 3)

        consensus_view = (
            "hawkish" if net_score >= 1.0
            else "dovish" if net_score <= -1.0
            else "neutral"
        )

        dissents_count = max(
            len(re.findall(r"\bdissent\w*\b", text, re.IGNORECASE)),
            len(re.findall(r"\bvoted\s+against\b", text, re.IGNORECASE)),
        )

        participant_counts = self._extract_participant_counts(text)
        q_score = self._compute_quantifier_weighted_score(text, participant_counts)
        data_deps = self._extract_data_dependencies(text)
        fwd_guidance = self._extract_forward_guidance(text)

        rate_guidance = fwd_guidance[0] if fwd_guidance else "No explicit guidance extracted."

        # Dominant quantifier phrase
        qcounter: Counter = Counter(participant_counts)
        participants_view = (
            qcounter.most_common(1)[0][0] if qcounter else "participants"
        )

        # Staff projection section
        staff_proj: Optional[str] = None
        staff_match = re.search(
            r"staff\s+(?:economic\s+)?(?:projection|forecast|review)(.*?)\n\n",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if staff_match:
            _, _, staff_tone = compute_tone(staff_match.group(1)[:800])
            staff_proj = staff_tone

        # Committee discussion section
        comm_match = re.search(
            r"committee\s+(?:discussion|policy\s+action|deliberation)(.*?)\n\n",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        committee_discussion_tone: str = tone
        if comm_match:
            _, _, ct = compute_tone(comm_match.group(1)[:3000])
            committee_discussion_tone = ct

        themes = identify_themes(text)

        return FOMCMinutesDetail(
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
            participant_counts=participant_counts,
            quantifier_weighted_score=q_score,
            committee_discussion_tone=committee_discussion_tone,
            participants_view=participants_view,
            forward_guidance_phrases=fwd_guidance,
            staff_economic_projection=staff_proj,
            word_count=len(text.split()),
        )


# ---------------------------------------------------------------------------
# PolicyRatePredictor
# ---------------------------------------------------------------------------

# Historical calibration: hawk net score → P(hike) at next meeting
# Based on rough empirical mapping from 2015-2024 FOMC cycles
_HAWK_TO_HIKE_PROB: list[tuple[float, float]] = [
    # (net_score, prob_hike)
    (-4.0, 0.01),
    (-3.0, 0.02),
    (-2.0, 0.05),
    (-1.0, 0.10),
    (0.0, 0.20),
    (0.5, 0.30),
    (1.0, 0.45),
    (1.5, 0.55),
    (2.0, 0.65),
    (2.5, 0.75),
    (3.0, 0.85),
    (4.0, 0.92),
    (5.0, 0.97),
]

_HAWK_TO_CUT_PROB: list[tuple[float, float]] = [
    (-5.0, 0.95),
    (-3.0, 0.85),
    (-2.0, 0.70),
    (-1.5, 0.55),
    (-1.0, 0.40),
    (-0.5, 0.25),
    (0.0, 0.15),
    (1.0, 0.07),
    (2.0, 0.03),
    (3.0, 0.01),
    (5.0, 0.005),
]


def _interp_prob(calibration: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation from calibration table."""
    if x <= calibration[0][0]:
        return calibration[0][1]
    if x >= calibration[-1][0]:
        return calibration[-1][1]
    for i in range(len(calibration) - 1):
        x0, y0 = calibration[i]
        x1, y1 = calibration[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 4)
    return 0.2


class PolicyRatePredictor:
    """
    Near-term policy rate prediction for major central banks.
    Uses FRED free CSV for current policy rates.
    Calibrated hawk score → rate probability mapping.
    """

    def __init__(self):
        self._fred_base = CentralBankSources.FRED_BASE

    def _fetch_fred_series(self, series_id: str, days: int = 30) -> Optional[float]:
        """Fetch the most recent value of a FRED series."""
        cache_key = f"fred_{series_id}_{date.today().isoformat()}"
        cached = _cache_get(cache_key, ttl=3600)
        if cached:
            try:
                return float(cached)
            except ValueError:
                pass

        url = f"{self._fred_base}?id={series_id}"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            lines = [l for l in resp.text.strip().splitlines() if not l.startswith("DATE")]
            if not lines:
                return None
            # Most recent non-empty value
            for line in reversed(lines):
                parts = line.split(",")
                if len(parts) == 2 and parts[1].strip() not in (".", ""):
                    val = float(parts[1].strip())
                    _cache_set(cache_key, str(val))
                    return val
        except Exception as exc:
            logger.warning(f"_fetch_fred_series {series_id} failed: {exc}")
        return None

    def _current_rate(self, bank: Institution) -> float:
        series_map = {
            "fed": "FEDTARMD",
            "ecb": "ECBDFR",
            "boe": "BOEBR",
            "boj": "IRSTCB01JPM156N",
            "boc": "IRSTCB01CAM156N",
            "rba": "IRSTCB01AUM156N",
        }
        series = series_map.get(bank, "FEDTARMD")
        val = self._fetch_fred_series(series)
        return val if val is not None else 0.0

    def _next_fomc_date(self) -> Optional[date]:
        """Estimate next FOMC meeting from calendar scrape or hardcoded schedule."""
        # Known 2025-2026 FOMC dates (hardcoded fallback)
        fomc_schedule_2025 = [
            date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
            date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
            date(2025, 10, 29), date(2025, 12, 10),
        ]
        fomc_schedule_2026 = [
            date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
            date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
            date(2026, 10, 28), date(2026, 12, 9),
        ]
        today = date.today()
        for d in fomc_schedule_2025 + fomc_schedule_2026:
            if d >= today:
                return d
        return None

    def predict(
        self,
        bank: Institution,
        hawk_net_score: float,
        uncertainty_score: float = 0.5,
    ) -> RatePrediction:
        """
        Generate rate probability prediction.
        uncertainty_score (0–3) dampens the probabilities toward 50/50.
        """
        current_rate = self._current_rate(bank)

        p_hike_raw = _interp_prob(_HAWK_TO_HIKE_PROB, hawk_net_score)
        p_cut_raw = _interp_prob(_HAWK_TO_CUT_PROB, hawk_net_score)

        # Dampen for uncertainty
        dampen = min(uncertainty_score / 6.0, 0.4)
        p_hike = p_hike_raw * (1 - dampen) + 0.20 * dampen
        p_cut = p_cut_raw * (1 - dampen) + 0.20 * dampen

        # Hold is residual
        p_hold = max(0.0, 1.0 - p_hike - p_cut)

        # Renormalize
        total = p_hike + p_hold + p_cut
        if total > 0:
            p_hike /= total
            p_hold /= total
            p_cut /= total

        # Direction
        if p_hike >= 0.5:
            direction: RateSignal = "hike"
        elif p_cut >= 0.5:
            direction = "cut"
        elif p_hold >= 0.4:
            direction = "hold"
        else:
            direction = "data_dependent"

        # Terminal rate: crude estimate
        if hawk_net_score > 2:
            terminal = current_rate + 0.5
        elif hawk_net_score < -2:
            terminal = max(0.0, current_rate - 1.0)
        else:
            terminal = current_rate

        next_meeting = self._next_fomc_date() if bank == "fed" else None

        # Market implied: derived from hawk score as proxy when CME unavailable
        market_implied = current_rate + (hawk_net_score * 0.05)

        return RatePrediction(
            bank=bank,
            as_of=date.today(),
            current_rate=round(current_rate, 4),
            market_implied_next=round(market_implied, 4),
            prob_hike=round(p_hike, 4),
            prob_hold=round(p_hold, 4),
            prob_cut=round(p_cut, 4),
            hawk_implied_direction=direction,
            terminal_rate_estimate=round(terminal, 4),
            next_meeting_date=next_meeting,
        )


# ---------------------------------------------------------------------------
# CrossCBAnalysis
# ---------------------------------------------------------------------------

class CrossCBAnalysis:
    """
    Cross-central-bank comparison: divergence, FX implications, sync index.
    """

    FX_PAIRS: dict[str, str] = {
        "FED_ECB": "EURUSD (more hawkish Fed → stronger USD → lower EURUSD)",
        "FED_BOE": "GBPUSD (more hawkish Fed → stronger USD → lower GBPUSD)",
        "FED_BOJ": "USDJPY (more hawkish Fed → higher USDJPY)",
        "FED_BOC": "USDCAD (more hawkish Fed → lower USDCAD)",
        "FED_RBA": "AUDUSD (more hawkish Fed → stronger USD → lower AUDUSD)",
        "ECB_BOE": "EURGBP (more hawkish ECB → stronger EUR → higher EURGBP)",
    }

    def compare_pair(
        self,
        bank_a: Institution,
        score_a: float,
        bank_b: Institution,
        score_b: float,
        rate_a: float,
        rate_b: float,
    ) -> CrossCBDivergence:
        pair_key = f"{bank_a.upper()}_{bank_b.upper()}"
        divergence = score_a - score_b

        fx_desc = self.FX_PAIRS.get(pair_key, "")
        if not fx_desc:
            if abs(divergence) < 0.3:
                fx_desc = "Minimal divergence — limited FX implication"
            elif divergence > 0:
                fx_desc = f"{bank_a.upper()} more hawkish → {bank_a.upper()} currency favored"
            else:
                fx_desc = f"{bank_b.upper()} more hawkish → {bank_b.upper()} currency favored"

        rate_diff = rate_a - rate_b

        # Synchronization index: 1 = same sign, same magnitude; 0 = fully opposed
        if score_a == 0.0 and score_b == 0.0:
            sync = 1.0
        else:
            # Cosine similarity in 1D
            max_abs = max(abs(score_a), abs(score_b), 0.001)
            sync = 1.0 - abs(divergence) / (2 * max_abs)
            sync = max(0.0, min(1.0, sync))

        return CrossCBDivergence(
            as_of=date.today(),
            pair=pair_key,
            fed_net=round(score_a, 3),
            ecb_net=round(score_b, 3),
            divergence=round(divergence, 3),
            fx_implication=fx_desc,
            rate_differential=round(rate_diff, 4),
            synchronization_index=round(sync, 4),
        )

    def global_sync_index(self, bank_scores: dict[str, float]) -> float:
        """Compute average pairwise synchronization index."""
        values = list(bank_scores.values())
        if len(values) < 2:
            return 1.0
        pairs: list[float] = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                a, b = values[i], values[j]
                mx = max(abs(a), abs(b), 0.001)
                sync = 1.0 - abs(a - b) / (2 * mx)
                pairs.append(max(0.0, min(1.0, sync)))
        return round(statistics.mean(pairs), 4) if pairs else 1.0


# ---------------------------------------------------------------------------
# CentralBankNLP — main orchestrator
# ---------------------------------------------------------------------------

class CentralBankNLP:
    """
    Multi-central-bank speech NLP orchestrator.
    Combines all components: fetch, classify, detect pivots, predict rates, compare.
    """

    def __init__(self):
        self.classifier = HawkDoveClassifier()
        self.fomc_parser = FOMCMinutesParser()
        self.rate_predictor = PolicyRatePredictor()
        self.cross_cb = CrossCBAnalysis()
        self._stance_detectors: dict[str, StanceChangeDetector] = {}

    def _get_stance_detector(self, bank: Institution) -> StanceChangeDetector:
        if bank not in self._stance_detectors:
            self._stance_detectors[bank] = StanceChangeDetector(bank)
        return self._stance_detectors[bank]

    def _analyze_text(self, text: str, meta: SpeechMeta) -> SpeechAnalysis:
        """Run full NLP pipeline on speech text."""
        cred = speaker_credibility(meta.speaker, meta.institution)
        classifier = HawkDoveClassifier(speaker_credibility_weight=cred)
        result = classifier.classify_text(text)

        h_score = result["hawkish_score"]
        d_score = result["dovish_score"]
        net = result["net_score"]
        tone: ToneLabel = result["tone"]
        uncertainty = result["uncertainty_score"]

        h_kws = list(HAWK_TERMS.keys())
        d_kws = list(DOVE_TERMS.keys())
        hawk_passages = extract_key_passages(text, h_kws, n_sentences=3)
        dove_passages = extract_key_passages(text, d_kws, n_sentences=3)
        themes = identify_themes(text)
        rate_signal = extract_rate_signal(text, tone)

        is_voting = (
            meta.institution == "fed"
            and any(n in meta.speaker.lower() for n in FOMC_VOTING_MEMBERS_2024)
        )

        return SpeechAnalysis(
            meta=SpeechMeta(
                speaker=meta.speaker,
                title=meta.title,
                date=meta.date,
                url=meta.url,
                institution=meta.institution,
                event=meta.event,
                is_voting_member=is_voting,
            ),
            hawkish_score=h_score,
            dovish_score=d_score,
            net_score=net,
            credibility_adjusted_net=round(net * cred, 3),
            tone=tone,
            uncertainty_score=uncertainty,
            hawkish_passages=hawk_passages,
            dovish_passages=dove_passages,
            key_themes=themes,
            rate_path_signal=rate_signal,
            word_count=len(text.split()),
            sentence_count=len(_split_sentences(text)),
            rate_mentioned=_extract_numbers(
                text, r"(?:federal funds|policy|interest)\s+rate\s+(?:of|at|to|is)?"
            ),
            inflation_mentioned=_extract_numbers(
                text, r"(?:inflation|cpi|pce)\s+(?:of|at|is|was|reached|hit)?\s*(?:around|about|near)?"
            ),
            gdp_mentioned=_extract_numbers(
                text, r"(?:gdp|growth)\s+(?:of|at|is|was|grew|expanded|contracted)?\s*(?:around|about)?"
            ),
        )

    # ------------------------------------------------------------------
    # Speech fetching methods
    # ------------------------------------------------------------------

    def fetch_fed_speeches(self, n: int = 15) -> list[SpeechMeta]:
        """Fetch Fed speeches from JSON index."""
        cached = _cache_get("fed_speeches_meta", ttl=1800)
        if cached:
            import json
            items = json.loads(cached)
            return [SpeechMeta(**i) for i in items[:n]]

        try:
            resp = requests.get(
                CentralBankSources.FED_SPEECHES_JSON,
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"fetch_fed_speeches: {exc}")
            return []

        items = data if isinstance(data, list) else data.get("speeches", [])
        speeches: list[SpeechMeta] = []
        for item in items[:n]:
            try:
                raw_date = item.get("d", "") or item.get("date", "")
                url = item.get("l", "") or item.get("url", "")
                if url and not url.startswith("http"):
                    url = CentralBankSources.FED_BASE + url
                speeches.append(SpeechMeta(
                    speaker=item.get("s", "") or item.get("speaker", "Federal Reserve"),
                    title=item.get("t", "") or item.get("title", "Untitled"),
                    date=_parse_date_str(raw_date),
                    url=url,
                    institution="fed",
                    event=item.get("e") or item.get("event"),
                ))
            except Exception:
                pass

        if speeches:
            import json
            _cache_set("fed_speeches_meta", json.dumps([s.model_dump(mode="json") for s in speeches]))

        return speeches

    def fetch_rss_speeches(
        self,
        institution: Institution,
        rss_url: str,
        n: int = 10,
    ) -> list[SpeechMeta]:
        """Fetch speech metadata from any RSS feed."""
        cache_key = f"rss_{institution}_{date.today().isoformat()}"
        cached = _cache_get(cache_key, ttl=1800)
        if cached:
            import json
            items = json.loads(cached)
            return [SpeechMeta(**i) for i in items[:n]]

        raw = _fetch_raw(rss_url)
        if not raw:
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
                speeches.append(SpeechMeta(
                    speaker=author,
                    title=title,
                    date=_parse_date_str(pub_date_str) if pub_date_str else date.today(),
                    url=link,
                    institution=institution,
                ))
            except Exception:
                pass

        if speeches:
            import json
            _cache_set(cache_key, json.dumps([s.model_dump(mode="json") for s in speeches]))

        return speeches

    def fetch_bis_speeches(self, n: int = 20) -> list[SpeechMeta]:
        """Scrape BIS speeches index for multi-CB coverage."""
        cache_key = f"bis_speeches_{date.today().isoformat()}"
        cached = _cache_get(cache_key, ttl=3600)
        if cached:
            import json
            return [SpeechMeta(**i) for i in json.loads(cached)[:n]]

        url = CentralBankSources.BIS_SPEECHES
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            logger.warning(f"fetch_bis_speeches: {exc}")
            return []

        speeches: list[SpeechMeta] = []
        # BIS speeches list has rows with date, speaker, title, CB
        for row in soup.select("tr.item, div.item, li.item")[:n]:
            try:
                link_tag = row.find("a")
                if not link_tag:
                    continue
                title = link_tag.get_text(strip=True) or "Untitled"
                href = link_tag.get("href", "")
                if href and not href.startswith("http"):
                    href = "https://www.bis.org" + href
                date_tag = row.find(class_=re.compile(r"date|time"))
                date_str = date_tag.get_text(strip=True) if date_tag else ""
                author_tag = row.find(class_=re.compile(r"author|speaker"))
                author = author_tag.get_text(strip=True) if author_tag else "BIS Speaker"
                speeches.append(SpeechMeta(
                    speaker=author,
                    title=title,
                    date=_parse_date_str(date_str) if date_str else date.today(),
                    url=href,
                    institution="bis",
                ))
            except Exception:
                pass

        if speeches:
            import json
            _cache_set(cache_key, json.dumps([s.model_dump(mode="json") for s in speeches]))

        return speeches[:n]

    def fetch_all_speeches(self, n_per_bank: int = 10) -> list[SpeechMeta]:
        """Fetch speeches from all supported central banks."""
        all_speeches: list[SpeechMeta] = []

        all_speeches += self.fetch_fed_speeches(n_per_bank)
        all_speeches += self.fetch_rss_speeches("ecb", CentralBankSources.ECB_SPEECHES_RSS, n_per_bank)
        all_speeches += self.fetch_rss_speeches("boe", CentralBankSources.BOE_SPEECHES_RSS, n_per_bank)
        all_speeches += self.fetch_rss_speeches("boc", CentralBankSources.BOC_RSS, n_per_bank)
        all_speeches += self.fetch_rss_speeches("rba", CentralBankSources.RBA_RSS, n_per_bank)
        all_speeches += self.fetch_bis_speeches(n_per_bank)

        return sorted(all_speeches, key=lambda s: s.date, reverse=True)

    def analyze_speech(
        self,
        meta: SpeechMeta,
        fetch_full_text: bool = True,
    ) -> SpeechAnalysis:
        """Analyze a single speech."""
        text = meta.title  # fallback
        if fetch_full_text and meta.url:
            cached = _cache_get(f"speech_{meta.url}", ttl=86400)
            if cached:
                text = cached
            else:
                fetched = _fetch_text(meta.url, max_chars=80_000)
                if len(fetched) > 200:
                    text = fetched
                    _cache_set(f"speech_{meta.url}", text)
        return self._analyze_text(text, meta)

    def analyze_all_speeches(
        self,
        speeches: list[SpeechMeta],
        fetch_full_text: bool = True,
    ) -> list[SpeechAnalysis]:
        """Analyze a list of speeches."""
        return [self.analyze_speech(m, fetch_full_text) for m in speeches]

    # ------------------------------------------------------------------
    # FOMC methods
    # ------------------------------------------------------------------

    def parse_fomc_minutes(self, meeting_date: date) -> FOMCMinutesDetail:
        return self.fomc_parser.parse(meeting_date)

    def latest_fomc_minutes(self) -> FOMCMinutesDetail:
        """Parse the most recent published FOMC minutes."""
        fomc_dates = [
            date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
            date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
            date(2025, 10, 29), date(2025, 12, 10),
            date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
        ]
        today = date.today()
        # Minutes published ~21 days after meeting
        published = sorted(
            [d for d in fomc_dates if d + timedelta(days=21) <= today],
            reverse=True,
        )
        if not published:
            # Fallback to last known
            return self.fomc_parser.parse(date(2025, 5, 7))
        return self.fomc_parser.parse(published[0])

    # ------------------------------------------------------------------
    # Stance and prediction
    # ------------------------------------------------------------------

    def get_stance(self, bank: Institution, analyses: list[SpeechAnalysis]) -> StanceSnapshot:
        detector = self._get_stance_detector(bank)
        bank_analyses = [a for a in analyses if a.meta.institution == bank]
        if bank_analyses:
            return detector.detect_pivot(bank_analyses)
        return detector.detect_from_db()

    def predict_rate(self, bank: Institution, analyses: list[SpeechAnalysis]) -> RatePrediction:
        bank_analyses = [a for a in analyses if a.meta.institution == bank]
        if bank_analyses:
            avg_net = statistics.mean(a.credibility_adjusted_net for a in bank_analyses)
            avg_unc = statistics.mean(a.uncertainty_score for a in bank_analyses)
        else:
            avg_net = 0.0
            avg_unc = 1.0
        return self.rate_predictor.predict(bank, avg_net, avg_unc)

    def divergence_analysis(
        self,
        bank_scores: dict[str, float],
        bank_rates: dict[str, float],
    ) -> list[CrossCBDivergence]:
        """Generate all pairwise CB divergence analyses."""
        pairs = [
            ("fed", "ecb"), ("fed", "boe"), ("fed", "boj"),
            ("fed", "boc"), ("fed", "rba"), ("ecb", "boe"),
        ]
        results: list[CrossCBDivergence] = []
        for a, b in pairs:
            if a in bank_scores and b in bank_scores:
                results.append(
                    self.cross_cb.compare_pair(
                        a, bank_scores.get(a, 0.0),
                        b, bank_scores.get(b, 0.0),
                        bank_rates.get(a, 0.0),
                        bank_rates.get(b, 0.0),
                    )
                )
        return results

    def full_monitor(self, n_speeches: int = 10, fetch_full_text: bool = True) -> CentralBankMonitorResult:
        """
        Run the full multi-CB NLP pipeline:
        1. Fetch speeches from all banks
        2. Analyze each speech
        3. Aggregate by bank
        4. Detect stance changes
        5. Return monitor result
        """
        speeches = self.fetch_all_speeches(n_speeches)
        analyses = self.analyze_all_speeches(speeches, fetch_full_text)

        banks: list[Institution] = ["fed", "ecb", "boe", "boj", "boc", "rba"]
        bank_scores: dict[str, float] = {}
        bank_tones: dict[str, ToneLabel] = {}
        pivot_signals: list[StanceSnapshot] = []

        for bank in banks:
            bank_analyses = [a for a in analyses if a.meta.institution == bank]
            if bank_analyses:
                avg_net = statistics.mean(a.credibility_adjusted_net for a in bank_analyses)
                # Store to DB
                avg_h = statistics.mean(a.hawkish_score for a in bank_analyses)
                avg_d = statistics.mean(a.dovish_score for a in bank_analyses)
                _store_hawk_score(
                    bank, date.today().isoformat(),
                    avg_h, avg_d, avg_net, 1.0, len(bank_analyses),
                )
            else:
                avg_net = 0.0
            bank_scores[bank] = round(avg_net, 3)

            if avg_net >= 3.5:
                t: ToneLabel = "very_hawkish"
            elif avg_net >= 1.0:
                t = "hawkish"
            elif avg_net <= -3.5:
                t = "very_dovish"
            elif avg_net <= -1.0:
                t = "dovish"
            else:
                t = "neutral"
            bank_tones[bank] = t

            pivot_signals.append(self.get_stance(bank, analyses))

        global_net = statistics.mean(bank_scores.values()) if bank_scores else 0.0
        if global_net >= 3.5:
            global_tone: ToneLabel = "very_hawkish"
        elif global_net >= 1.0:
            global_tone = "hawkish"
        elif global_net <= -3.5:
            global_tone = "very_dovish"
        elif global_net <= -1.0:
            global_tone = "dovish"
        else:
            global_tone = "neutral"

        # Trend: recent half vs older half
        sorted_analyses = sorted(analyses, key=lambda a: a.meta.date, reverse=True)
        half = max(1, len(sorted_analyses) // 2)
        recent_nets = [a.credibility_adjusted_net for a in sorted_analyses[:half]]
        older_nets = [a.credibility_adjusted_net for a in sorted_analyses[half:]]
        avg_r = statistics.mean(recent_nets) if recent_nets else 0.0
        avg_o = statistics.mean(older_nets) if older_nets else 0.0
        delta = avg_r - avg_o
        if delta > 0.5:
            trend: Literal["more_hawkish", "more_dovish", "stable"] = "more_hawkish"
        elif delta < -0.5:
            trend = "more_dovish"
        else:
            trend = "stable"

        # Implied next move from Fed
        fed_analyses = [a for a in analyses if a.meta.institution == "fed"]
        if fed_analyses:
            implied: RateSignal = Counter(
                a.rate_path_signal for a in fed_analyses
            ).most_common(1)[0][0]
        else:
            implied = "data_dependent"

        return CentralBankMonitorResult(
            as_of=date.today(),
            bank_scores=bank_scores,
            bank_tones=bank_tones,
            global_net_score=round(global_net, 3),
            global_tone=global_tone,
            tone_trend=trend,
            implied_next_move=implied,
            recent_speeches=analyses[:20],
            pivot_signals=pivot_signals,
        )

    def hawk_score_summary(self) -> dict[str, Any]:
        """Return hawk score summary for all banks from DB history."""
        banks: list[Institution] = ["fed", "ecb", "boe", "boj", "boc", "rba"]
        summary: dict[str, Any] = {}
        for bank in banks:
            df = _load_hawk_history(bank, days=90)
            if not df.empty:
                summary[bank] = {
                    "current": float(df.iloc[-1]["net_score"]),
                    "30d_avg": float(df["net_score"].tail(30).mean()),
                    "90d_avg": float(df["net_score"].mean()),
                    "trend": (
                        "more_hawkish"
                        if float(df["net_score"].tail(15).mean()) > float(df["net_score"].head(15).mean())
                        else "more_dovish"
                    ),
                }
            else:
                summary[bank] = {"current": 0.0, "30d_avg": 0.0, "90d_avg": 0.0, "trend": "stable"}
        return summary

    def quick_text_score(self, text: str, speaker: str = "Unknown", institution: str = "other") -> dict[str, Any]:
        """Score any text block for CB sentiment. Convenience method."""
        cred = speaker_credibility(speaker, institution)
        clf = HawkDoveClassifier(speaker_credibility_weight=cred)
        result = clf.classify_text(text)
        result["rate_signal"] = extract_rate_signal(text, result["tone"])
        result["themes"] = identify_themes(text)
        result["uncertainty"] = _modal_uncertainty_score(text)
        result["key_passages"] = clf.extract_key_signals(text)
        return result


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

cb_nlp_router = APIRouter(prefix="/cb", tags=["central-bank-nlp"])

_nlp_engine = CentralBankNLP()


@cb_nlp_router.get("/speeches/latest")
def get_latest_speeches(
    bank: str = Query(default="all", description="Bank code: fed/ecb/boe/boj/boc/rba/all"),
    n: int = Query(default=10, ge=1, le=50),
    full_text: bool = Query(default=False, description="Fetch and analyze full speech text"),
) -> dict[str, Any]:
    """Fetch and analyze the latest speeches from central banks."""
    if bank == "all":
        metas = _nlp_engine.fetch_all_speeches(n_per_bank=n)
    elif bank == "fed":
        metas = _nlp_engine.fetch_fed_speeches(n)
    elif bank == "ecb":
        metas = _nlp_engine.fetch_rss_speeches("ecb", CentralBankSources.ECB_SPEECHES_RSS, n)
    elif bank == "boe":
        metas = _nlp_engine.fetch_rss_speeches("boe", CentralBankSources.BOE_SPEECHES_RSS, n)
    elif bank == "boc":
        metas = _nlp_engine.fetch_rss_speeches("boc", CentralBankSources.BOC_RSS, n)
    elif bank == "rba":
        metas = _nlp_engine.fetch_rss_speeches("rba", CentralBankSources.RBA_RSS, n)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown bank: {bank}")

    analyses = _nlp_engine.analyze_all_speeches(metas, fetch_full_text=full_text)
    return {
        "bank": bank,
        "count": len(analyses),
        "speeches": [a.model_dump(mode="json") for a in analyses],
    }


@cb_nlp_router.get("/stance/{bank}")
def get_stance(bank: str) -> dict[str, Any]:
    """Get current stance and pivot signal for a central bank."""
    valid_banks: list[Institution] = ["fed", "ecb", "boe", "boj", "boc", "rba"]
    if bank not in valid_banks:
        raise HTTPException(status_code=400, detail=f"Unknown bank: {bank}")
    institution: Institution = bank  # type: ignore[assignment]
    snap = _nlp_engine.get_stance(institution, [])
    return snap.model_dump(mode="json")


@cb_nlp_router.get("/hawk-score")
def get_hawk_scores() -> dict[str, Any]:
    """Get hawk score summary for all central banks from history."""
    return {
        "as_of": date.today().isoformat(),
        "scores": _nlp_engine.hawk_score_summary(),
        "global_sync": _nlp_engine.cross_cb.global_sync_index(
            {k: v.get("current", 0.0) for k, v in _nlp_engine.hawk_score_summary().items()}
        ),
    }


@cb_nlp_router.get("/pivot-signal")
def get_pivot_signals() -> dict[str, Any]:
    """Get pivot signals for all central banks."""
    banks: list[Institution] = ["fed", "ecb", "boe", "boj", "boc", "rba"]
    signals = [_nlp_engine.get_stance(b, []).model_dump(mode="json") for b in banks]
    return {
        "as_of": date.today().isoformat(),
        "signals": signals,
    }


@cb_nlp_router.get("/fomc-minutes/latest")
def get_latest_fomc_minutes() -> dict[str, Any]:
    """Parse and return the most recently published FOMC minutes."""
    try:
        result = _nlp_engine.latest_fomc_minutes()
        return result.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@cb_nlp_router.get("/rate-prediction")
def get_rate_predictions(
    bank: str = Query(default="fed"),
) -> dict[str, Any]:
    """Get near-term rate prediction for a central bank."""
    valid: list[Institution] = ["fed", "ecb", "boe", "boj", "boc", "rba"]
    if bank not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown bank: {bank}")
    institution: Institution = bank  # type: ignore[assignment]

    # Quick fetch of recent speeches for scoring
    metas = (
        _nlp_engine.fetch_fed_speeches(5) if bank == "fed"
        else _nlp_engine.fetch_rss_speeches(
            institution,
            getattr(CentralBankSources, f"{bank.upper()}_RSS", CentralBankSources.ECB_SPEECHES_RSS),
            5,
        )
    )
    analyses = _nlp_engine.analyze_all_speeches(metas, fetch_full_text=False)
    prediction = _nlp_engine.predict_rate(institution, analyses)
    return prediction.model_dump(mode="json")


@cb_nlp_router.get("/divergence")
def get_divergence(
    pair: str = Query(default="FED_ECB", description="CB pair e.g. FED_ECB, FED_BOE"),
) -> dict[str, Any]:
    """Get cross-CB hawk/dove divergence and FX implication."""
    part = pair.upper().split("_")
    if len(part) != 2:
        raise HTTPException(status_code=400, detail="pair must be BANK_BANK format e.g. FED_ECB")

    bank_a_str, bank_b_str = part[0].lower(), part[1].lower()
    valid: list[Institution] = ["fed", "ecb", "boe", "boj", "boc", "rba"]
    if bank_a_str not in valid or bank_b_str not in valid:
        raise HTTPException(status_code=400, detail="Unknown banks in pair")

    bank_a: Institution = bank_a_str  # type: ignore[assignment]
    bank_b: Institution = bank_b_str  # type: ignore[assignment]

    scores = _nlp_engine.hawk_score_summary()
    score_a = scores.get(bank_a_str, {}).get("current", 0.0)
    score_b = scores.get(bank_b_str, {}).get("current", 0.0)
    rate_a = _nlp_engine.rate_predictor._current_rate(bank_a)
    rate_b = _nlp_engine.rate_predictor._current_rate(bank_b)

    result = _nlp_engine.cross_cb.compare_pair(
        bank_a, score_a, bank_b, score_b, rate_a, rate_b
    )
    return result.model_dump(mode="json")


@cb_nlp_router.post("/score-text")
def score_text(payload: dict[str, str]) -> dict[str, Any]:
    """Score any text block for CB sentiment. Useful for interactive analysis."""
    text = payload.get("text", "")
    speaker = payload.get("speaker", "Unknown")
    institution = payload.get("institution", "other")
    if not text:
        raise HTTPException(status_code=400, detail="text field required")
    return _nlp_engine.quick_text_score(text, speaker, institution)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def monitor_all_banks(n_speeches: int = 8, fetch_full_text: bool = False) -> CentralBankMonitorResult:
    """Run the full multi-CB monitor pipeline."""
    engine = CentralBankNLP()
    return engine.full_monitor(n_speeches, fetch_full_text)


def quick_tone_check(text: str) -> dict[str, Any]:
    """Score a text block — convenience wrapper."""
    engine = CentralBankNLP()
    return engine.quick_text_score(text)


def parse_fomc_minutes(meeting_date: date) -> FOMCMinutesDetail:
    """Parse FOMC minutes for a given meeting date."""
    parser = FOMCMinutesParser()
    return parser.parse(meeting_date)


def get_rate_prediction(bank: Institution = "fed") -> RatePrediction:
    """Get rate prediction for a bank."""
    predictor = PolicyRatePredictor()
    return predictor.predict(bank, 0.0)
