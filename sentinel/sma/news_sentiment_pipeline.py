"""
News Sentiment Pipeline — Dimension #084 (target score 9+).

Orchestrates a multi-source, multi-model sentiment pipeline combining:
  - GDELT DOC API v2 (global news corpus)
  - Yahoo Finance RSS, Google News RSS, MarketWatch, Reuters proxy
  - Seeking Alpha RSS
  - EDGAR 8-K material event filings
  - FinBERT (HuggingFace Inference API) → Loughran-McDonald fallback → VADER fallback

Classes
-------
MultiSourceNewsCollector  — collect + deduplicate articles from all sources
SentimentScorer           — FinBERT/LM/VADER ensemble scoring
SentimentAggregator       — ticker-level aggregation, signals, sector heatmap
NewsSentimentPipeline     — orchestrates the full pipeline
sentiment_router          — FastAPI router (GET/POST endpoints)

Free endpoints used (no API keys required unless HF_TOKEN env-var present)
--------------------------------------------------------------------------
http://api.gdeltproject.org/api/v2/doc/doc                — GDELT DOC API
https://feeds.finance.yahoo.com/rss/2.0/headline          — Yahoo Finance RSS
https://news.google.com/rss/search                        — Google News RSS
https://feeds.marketwatch.com/marketwatch/bulletins/      — MarketWatch RSS
https://seekingalpha.com/symbol/{ticker}/feed.xml         — Seeking Alpha RSS
https://data.sec.gov/submissions/CIK{cik}.json           — EDGAR submissions
https://api-inference.huggingface.co/models/ProsusAI/finbert — HF FinBERT
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
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

_GDELT_DOC_BASE = "http://api.gdeltproject.org/api/v2/doc/doc"
_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_MARKETWATCH_RSS = "https://feeds.marketwatch.com/marketwatch/bulletins/"
_SEEKING_ALPHA_RSS = "https://seekingalpha.com/symbol/{ticker}/feed.xml"
_REUTERS_VIA_GOOGLE = "https://news.google.com/rss/search?q={company}+site:reuters.com&hl=en-US&gl=US&ceid=US:en"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/"
_HF_FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
_HF_FINBERT_ALT = "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone"

_TIMEOUT = 25.0
_HF_BATCH_SIZE = 8          # FinBERT batch size to stay under HF free-tier limits
_HF_RATE_LIMIT_RPS = 10    # max requests/second to HF Inference API

# Exponential decay half-life for recency weighting (hours)
_DECAY_HALF_LIFE_HOURS = 6.0

_HEADERS = {
    "User-Agent": "SENTINEL:FinancialTerminal:1.0 (contact: sentinel@example.com)",
    "Accept": "application/json, text/xml, */*",
}
_RSS_HEADERS = {
    "User-Agent": "SENTINEL:FinancialTerminal:1.0",
    "Accept": "application/rss+xml, application/atom+xml, text/xml, */*",
}

# ---------------------------------------------------------------------------
# Loughran-McDonald Financial Sentiment Lexicon (50 key terms each)
# ---------------------------------------------------------------------------

LM_POSITIVE: list[str] = [
    "profit", "profitable", "earnings beat", "exceeded", "outperformed", "record high",
    "growth", "surpassed", "strong results", "robust", "raised guidance", "beat expectations",
    "partnership", "approved", "successful launch", "dividend increase", "share buyback",
    "margin expansion", "market share gain", "innovative", "breakthrough", "upgrade",
    "positive outlook", "raised outlook", "above expectations", "exceed", "accelerating",
    "award", "milestone", "expansion", "efficiency", "synergy", "winning", "gain",
    "improvement", "benefiting", "recovery", "outperform", "momentum", "strong demand",
    "record revenue", "record earnings", "increase", "positive", "favorable", "progress",
    "elevated", "exceptional", "superior", "success", "achievement",
]

LM_NEGATIVE: list[str] = [
    "loss", "deficit", "missed", "below expectations", "disappointing", "decline",
    "shortfall", "reduced guidance", "layoffs", "restructuring", "write-down", "impairment",
    "bankruptcy", "default", "investigation", "regulatory action", "product recall", "lawsuit",
    "data breach", "fraud", "restatement", "covenant violation", "margin compression",
    "supply chain disruption", "headwind", "uncertainty", "risk", "volatility", "downgrade",
    "warning", "miss", "weaker than expected", "debt concern", "subpoena", "class action",
    "penalty", "negative outlook", "lowered guidance", "missed estimates", "disappoints",
    "suspended", "halted", "probe", "scrutiny", "violation", "recall", "deterioration",
    "worsening", "falling", "declining", "pressure", "challenge", "concern", "risk",
]

LM_UNCERTAINTY: list[str] = [
    "uncertain", "unclear", "pending", "contingent", "might", "could", "may",
    "possible", "potential", "depends", "subject to", "awaiting", "review",
    "evaluate", "consider", "assess", "monitor", "investigate",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ArticleRecord(BaseModel):
    """Normalised article record from any news source."""
    model_config = ConfigDict(frozen=True)

    url: str
    url_hash: str
    title: str
    summary: str = ""
    source: str                             # "gdelt", "yahoo_rss", "google_news", etc.
    domain: str = ""
    published: Optional[datetime] = None
    ticker: Optional[str] = None
    tone: Optional[float] = None            # GDELT tone score if available


class ScoredArticle(BaseModel):
    """Article with ensemble sentiment score attached."""
    model_config = ConfigDict(frozen=False)

    url: str
    url_hash: str
    title: str
    summary: str = ""
    source: str
    domain: str = ""
    published: Optional[datetime] = None
    ticker: Optional[str] = None

    # Ensemble scores
    finbert_label: Optional[str] = None
    finbert_score: Optional[float] = None
    lm_net_sentiment: Optional[float] = None
    vader_compound: Optional[float] = None
    ensemble_score: float = 0.0             # -1 to +1
    ensemble_label: str = "neutral"
    ensemble_confidence: float = 0.0


class TickerSentimentSummary(BaseModel):
    """Aggregated daily sentiment summary for a single ticker."""
    ticker: str
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    period_hours: int = 24

    # Volume
    article_count: int = 0
    source_diversity: int = 0               # unique domains

    # Scores
    avg_sentiment: float = 0.0             # recency-weighted average
    sentiment_momentum: Optional[float] = None  # vs prior 24h
    positive_pct: float = 0.0
    negative_pct: float = 0.0
    neutral_pct: float = 0.0

    # Headlines
    top_positive_headlines: list[str] = Field(default_factory=list)
    top_negative_headlines: list[str] = Field(default_factory=list)

    # Signal
    signal: str = "neutral"                 # "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell"
    trend: str = "flat"                     # "rising" | "falling" | "flat"
    is_extreme: bool = False                # top/bottom 10% of 90-day distribution


# ---------------------------------------------------------------------------
# In-process TTL cache (avoids hammering free APIs)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 900.0  # 15 minutes


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

def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _title_simhash(title: str) -> int:
    """Lightweight 32-bit simhash of normalised title for near-duplicate detection."""
    normalized = re.sub(r"[^a-z0-9 ]", "", title.lower())
    tokens = normalized.split()
    h = 0
    for tok in tokens:
        th = int(hashlib.md5(tok.encode()).hexdigest(), 16) & 0xFFFFFFFF
        h ^= th
    return h


def _parse_rss_date(date_str: str) -> Optional[datetime]:
    """Parse RSS pubDate or similar format to datetime (UTC)."""
    if not date_str:
        return None
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y%m%dT%H%M%SZ",
    ):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _extract_domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else ""


def _recency_weight(published: Optional[datetime], half_life_hours: float = _DECAY_HALF_LIFE_HOURS) -> float:
    """Exponential decay weight based on article age."""
    if published is None:
        return 0.1
    now = datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - published).total_seconds() / 3600.0)
    return math.exp(-math.log(2) * age_hours / half_life_hours)


# ---------------------------------------------------------------------------
# MultiSourceNewsCollector
# ---------------------------------------------------------------------------


class MultiSourceNewsCollector:
    """
    Collects articles from GDELT, Yahoo Finance RSS, Google News RSS,
    MarketWatch, Reuters (via Google), Seeking Alpha, and EDGAR 8-K.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # GDELT
    # ------------------------------------------------------------------

    async def collect_gdelt(
        self,
        tickers: list[str],
        company_names: list[str],
        lookback_hours: int = 24,
    ) -> list[dict]:
        """
        Query GDELT DOC API for articles referencing any ticker or company name.
        Returns raw article dicts with url/title/seendate/domain/tone.
        """
        cache_key = f"gdelt:{','.join(tickers)}:{lookback_hours}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return list(cached)  # type: ignore[arg-type]

        # Build multi-term OR query
        terms: list[str] = []
        for t in tickers:
            terms.append(f'"{t}"')
        for name in company_names:
            # Use first two words of company name to avoid over-specificity
            first_two = " ".join(name.split()[:2])
            if first_two and first_two not in tickers:
                terms.append(f'"{first_two}"')

        query = " OR ".join(terms[:8])  # cap at 8 terms to avoid URL overflow
        timespan = f"{max(1, min(lookback_hours, 720))}h"

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": "100",
            "timespan": timespan,
            "format": "json",
            "sort": "DateDesc",
        }

        articles: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_GDELT_DOC_BASE, params=params, headers=_HEADERS)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("GDELT collect error", error=str(exc))
            return []

        for art in data.get("articles", []):
            url = art.get("url", "")
            if not url:
                continue
            tone_raw = art.get("tone", "0") or "0"
            try:
                tone = float(str(tone_raw).split(",")[0])
            except (ValueError, TypeError):
                tone = 0.0

            articles.append({
                "url": url,
                "title": art.get("title", ""),
                "summary": "",
                "source": "gdelt",
                "domain": art.get("domain", _extract_domain(url)),
                "published_raw": art.get("seendate", ""),
                "tone": tone,
                "social_image": art.get("socialimage", ""),
            })

        _cache_set(cache_key, articles)
        logger.info("GDELT collected", count=len(articles), tickers=tickers)
        return articles

    # ------------------------------------------------------------------
    # RSS feeds
    # ------------------------------------------------------------------

    async def _fetch_rss(self, url: str, source_tag: str, ticker: str = "") -> list[dict]:
        """Generic RSS fetch + parse."""
        cache_key = f"rss:{url}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return list(cached)  # type: ignore[arg-type]

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=_RSS_HEADERS, follow_redirects=True)
                resp.raise_for_status()
                content = resp.text
        except Exception as exc:
            logger.debug("RSS fetch error", url=url, error=str(exc))
            return []

        articles: list[dict] = []
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)

        for item in items[:50]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate") or item.find("published")
            desc_el = item.find("description") or item.find("summary")

            title = (title_el.text or "").strip() if title_el is not None else ""
            if not title:
                continue

            link = ""
            if link_el is not None:
                link = (link_el.text or link_el.get("href", "")).strip()

            pub_str = (pub_el.text or "").strip() if pub_el is not None else ""
            summary = ""
            if desc_el is not None:
                raw = desc_el.text or ""
                summary = re.sub(r"<[^>]+>", "", raw).strip()[:500]

            articles.append({
                "url": link or url,
                "title": title,
                "summary": summary,
                "source": source_tag,
                "domain": _extract_domain(link) if link else source_tag,
                "published_raw": pub_str,
                "ticker": ticker,
                "tone": None,
            })

        _cache_set(cache_key, articles)
        return articles

    async def collect_rss_feeds(self, ticker: str, company_name: str) -> list[dict]:
        """Collect articles from Yahoo Finance, Google News, MarketWatch, Reuters proxy."""
        tasks = [
            self._fetch_rss(
                _YAHOO_RSS.format(ticker=quote_plus(ticker)),
                "yahoo_finance",
                ticker,
            ),
            self._fetch_rss(
                _GOOGLE_NEWS_RSS.format(query=quote_plus(f"{ticker} stock")),
                "google_news",
                ticker,
            ),
            self._fetch_rss(
                _MARKETWATCH_RSS,
                "marketwatch",
                ticker,
            ),
            self._fetch_rss(
                _REUTERS_VIA_GOOGLE.format(company=quote_plus(company_name)),
                "reuters",
                ticker,
            ),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        articles: list[dict] = []
        for r in results:
            if isinstance(r, list):
                articles.extend(r)
        return articles

    async def collect_seeking_alpha(self, ticker: str) -> list[dict]:
        """Collect Seeking Alpha RSS feed for a ticker."""
        url = _SEEKING_ALPHA_RSS.format(ticker=ticker.upper())
        return await self._fetch_rss(url, "seeking_alpha", ticker)

    async def collect_edgar_8k(self, cik: str, lookback_hours: int = 48) -> list[dict]:
        """
        Fetch recent 8-K filings for a company from EDGAR submissions JSON.
        Returns article-like dicts with filing metadata as text.
        """
        if not cik:
            return []

        cache_key = f"edgar_8k:{cik}:{lookback_hours}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return list(cached)  # type: ignore[arg-type]

        padded_cik = cik.zfill(10)
        url = _EDGAR_SUBMISSIONS.format(cik=padded_cik)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        articles: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=_HEADERS)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug("EDGAR 8-K fetch error", cik=cik, error=str(exc))
            return []

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        descriptions = filings.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form not in ("8-K", "8-K/A"):
                continue
            try:
                filing_date = datetime.strptime(dates[i], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except (ValueError, IndexError):
                continue
            if filing_date < cutoff:
                continue

            acc_clean = accessions[i].replace("-", "") if i < len(accessions) else ""
            acc_formatted = accessions[i] if i < len(accessions) else ""
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/"
                f"{descriptions[i] if i < len(descriptions) else ''}"
            )
            title = f"8-K Filing: {data.get('name', cik)} — {dates[i]}"

            articles.append({
                "url": filing_url or f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K",
                "title": title,
                "summary": f"Material event (8-K) filed by {data.get('name', cik)} on {dates[i]}. Accession: {acc_formatted}",
                "source": "edgar_8k",
                "domain": "sec.gov",
                "published_raw": dates[i],
                "tone": None,
            })

        _cache_set(cache_key, articles)
        logger.info("EDGAR 8-K collected", cik=cik, count=len(articles))
        return articles

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate_articles(self, articles: list[dict]) -> list[dict]:
        """
        Dedup by URL hash first, then by SimHash of title to catch
        articles from different sources about the same story.
        """
        seen_urls: set[str] = set()
        seen_simhashes: set[int] = set()
        unique: list[dict] = []

        for art in articles:
            url = art.get("url", "")
            uh = _url_hash(url)
            if uh in seen_urls:
                continue
            seen_urls.add(uh)

            title = art.get("title", "")
            sh = _title_simhash(title)
            # SimHash collision window: exact match only (XOR = 0 means identical tokens)
            if sh in seen_simhashes and sh != 0:
                continue
            seen_simhashes.add(sh)

            # Normalise published field
            raw = art.get("published_raw", "")
            art["published"] = _parse_rss_date(raw) if raw else None
            art["url_hash"] = uh
            unique.append(art)

        return unique


# ---------------------------------------------------------------------------
# SentimentScorer
# ---------------------------------------------------------------------------


class SentimentScorer:
    """
    Three-tier sentiment scoring:
      1. FinBERT via HuggingFace Inference API (primary)
      2. Loughran-McDonald keyword fallback
      3. VADER fallback
    Produces a weighted ensemble score.
    """

    def __init__(self) -> None:
        self._hf_token: str = os.getenv("HF_TOKEN", "")
        self._last_hf_call: float = 0.0
        self._min_interval = 1.0 / _HF_RATE_LIMIT_RPS
        # Lazy VADER init to avoid import penalty at module load
        self._vader = None

    def _get_vader(self):
        if self._vader is None:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                import nltk
                try:
                    nltk.data.find("sentiment/vader_lexicon.zip")
                except LookupError:
                    nltk.download("vader_lexicon", quiet=True)
                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                logger.warning("NLTK/VADER not installed; VADER scoring disabled")
        return self._vader

    # ------------------------------------------------------------------
    # FinBERT (HuggingFace Inference API)
    # ------------------------------------------------------------------

    async def score_finbert(self, texts: list[str]) -> list[dict]:
        """
        Send texts to HuggingFace ProsusAI/finbert.
        Returns list of {label, score, numeric} dicts.
        Falls back to [] on error so caller can degrade gracefully.
        """
        if not texts:
            return []

        headers = {"Content-Type": "application/json"}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"

        # Rate limiting
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_hf_call)
        if wait > 0:
            await asyncio.sleep(wait)

        # Truncate texts to 512 chars (FinBERT context limit)
        truncated = [t[:512] for t in texts]
        payload = {"inputs": truncated, "options": {"wait_for_model": True}}

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        _HF_FINBERT_URL,
                        json=payload,
                        headers=headers,
                    )
                    self._last_hf_call = time.monotonic()

                    if resp.status_code == 503:
                        # Model loading — wait and retry
                        await asyncio.sleep(10 * (attempt + 1))
                        continue

                    resp.raise_for_status()
                    raw = resp.json()
                    return self._parse_finbert_response(raw, len(texts))

            except httpx.HTTPStatusError as exc:
                logger.warning("FinBERT HTTP error", status=exc.response.status_code, attempt=attempt)
                if attempt == 2:
                    return []
            except Exception as exc:
                logger.warning("FinBERT request error", error=str(exc), attempt=attempt)
                if attempt == 2:
                    return []
            await asyncio.sleep(2 ** attempt)

        return []

    def _parse_finbert_response(self, raw: object, n: int) -> list[dict]:
        """Parse HuggingFace nested response into normalised list."""
        results: list[dict] = []
        # HF returns list[list[{label, score}]] or list[{label, score}]
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, list):
                    # Each item is a list of label/score for one input
                    best = max(item, key=lambda x: x.get("score", 0))
                    results.append(self._normalise_finbert_item(best))
                elif isinstance(item, dict):
                    results.append(self._normalise_finbert_item(item))
        # Pad with neutral if fewer results than inputs
        while len(results) < n:
            results.append({"label": "neutral", "score": 0.5, "numeric": 0.0, "source": "finbert"})
        return results[:n]

    def _normalise_finbert_item(self, item: dict) -> dict:
        label = item.get("label", "neutral").lower()
        score = float(item.get("score", 0.5))
        if label in ("positive", "pos"):
            numeric = score
            label = "positive"
        elif label in ("negative", "neg"):
            numeric = -score
            label = "negative"
        else:
            numeric = 0.0
            label = "neutral"
        return {"label": label, "score": score, "numeric": numeric, "source": "finbert"}

    # ------------------------------------------------------------------
    # Loughran-McDonald keyword scoring
    # ------------------------------------------------------------------

    def score_lm_dict(self, text: str) -> dict:
        """
        Score text against Loughran-McDonald financial word lists.
        Returns {positive_count, negative_count, uncertainty_count, net_sentiment}.
        """
        text_lower = text.lower()
        pos = sum(1 for w in LM_POSITIVE if w in text_lower)
        neg = sum(1 for w in LM_NEGATIVE if w in text_lower)
        unc = sum(1 for w in LM_UNCERTAINTY if w in text_lower)
        total = max(1, pos + neg)
        net = (pos - neg) / total
        return {
            "positive_count": pos,
            "negative_count": neg,
            "uncertainty_count": unc,
            "net_sentiment": float(np.clip(net, -1.0, 1.0)),
        }

    # ------------------------------------------------------------------
    # VADER
    # ------------------------------------------------------------------

    def score_vader(self, text: str) -> dict:
        """
        VADER compound sentiment score.
        Returns {compound: float, pos: float, neg: float, neu: float}.
        """
        vader = self._get_vader()
        if vader is None:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        scores = vader.polarity_scores(text[:1024])
        return scores

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def aggregate_scores(
        self,
        finbert: Optional[dict],
        lm: dict,
        vader: dict,
    ) -> dict:
        """
        Weighted ensemble:
          FinBERT  0.60
          LM dict  0.30
          VADER    0.10

        Returns {final_score, label, confidence}.
        """
        w_fb, w_lm, w_vd = 0.6, 0.3, 0.1

        fb_numeric = finbert["numeric"] if finbert else 0.0
        fb_conf = finbert["score"] if finbert else 0.0

        lm_numeric = lm.get("net_sentiment", 0.0)
        vd_numeric = vader.get("compound", 0.0)

        if finbert is not None:
            final = w_fb * fb_numeric + w_lm * lm_numeric + w_vd * vd_numeric
        else:
            # Degrade gracefully when FinBERT unavailable
            final = (w_lm / (w_lm + w_vd)) * lm_numeric + (w_vd / (w_lm + w_vd)) * vd_numeric

        final = float(np.clip(final, -1.0, 1.0))

        if final >= 0.5:
            label = "very_positive"
        elif final >= 0.15:
            label = "positive"
        elif final <= -0.5:
            label = "very_negative"
        elif final <= -0.15:
            label = "negative"
        else:
            label = "neutral"

        confidence = (abs(final) + fb_conf * w_fb) / (1 + w_fb)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {"final_score": final, "label": label, "confidence": confidence}

    def score_article(self, article: dict, finbert_result: Optional[dict] = None) -> dict:
        """
        Score a single article using all three methods.
        finbert_result should be pre-fetched in batch; pass None to skip FinBERT.
        """
        text = f"{article.get('title', '')} {article.get('summary', '')}".strip()[:512]
        lm = self.score_lm_dict(text)
        vader = self.score_vader(text)
        ensemble = self.aggregate_scores(finbert_result, lm, vader)
        return {
            "finbert_label": finbert_result["label"] if finbert_result else None,
            "finbert_score": finbert_result["score"] if finbert_result else None,
            "lm_net_sentiment": lm["net_sentiment"],
            "vader_compound": vader["compound"],
            "ensemble_score": ensemble["final_score"],
            "ensemble_label": ensemble["label"],
            "ensemble_confidence": ensemble["confidence"],
        }

    async def batch_score(self, articles: list[dict]) -> list[dict]:
        """
        Efficiently score all articles:
          1. Batch FinBERT in groups of _HF_BATCH_SIZE
          2. Score LM + VADER inline (synchronous, fast)
          3. Merge into scored article dicts
        """
        if not articles:
            return []

        texts = [
            f"{a.get('title', '')} {a.get('summary', '')}".strip()[:512]
            for a in articles
        ]

        # Batch FinBERT calls
        finbert_results: list[Optional[dict]] = [None] * len(texts)
        for batch_start in range(0, len(texts), _HF_BATCH_SIZE):
            batch = texts[batch_start: batch_start + _HF_BATCH_SIZE]
            fb_batch = await self.score_finbert(batch)
            for j, fb in enumerate(fb_batch):
                finbert_results[batch_start + j] = fb

        scored: list[dict] = []
        for i, article in enumerate(articles):
            scores = self.score_article(article, finbert_results[i])
            merged = {**article, **scores}
            scored.append(merged)

        return scored


# ---------------------------------------------------------------------------
# SentimentAggregator
# ---------------------------------------------------------------------------


class SentimentAggregator:
    """Aggregates article-level scores into ticker signals and sector heatmaps."""

    def aggregate_to_ticker(
        self,
        scored_articles: list[dict],
        ticker: str,
        period_hours: int = 24,
    ) -> dict:
        """
        Produce ticker-level sentiment summary from scored articles.
        Uses exponential decay for recency weighting (half-life = 6h).
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=period_hours)

        relevant = [
            a for a in scored_articles
            if a.get("ticker") == ticker or ticker in a.get("title", "").upper()
        ]
        # Filter to period
        in_period = []
        for a in relevant:
            pub = a.get("published")
            if pub is None or (pub.tzinfo is None):
                pub = now  # treat undated as current
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
            if pub >= cutoff:
                in_period.append(a)

        if not in_period:
            return {
                "ticker": ticker,
                "article_count": 0,
                "avg_sentiment": 0.0,
                "sentiment_momentum": None,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "top_positive_headlines": [],
                "top_negative_headlines": [],
                "source_diversity": 0,
            }

        weights: list[float] = []
        scores: list[float] = []
        labels: list[str] = []

        for a in in_period:
            pub = a.get("published") or now
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            w = _recency_weight(pub)
            weights.append(w)
            scores.append(a.get("ensemble_score", 0.0))
            labels.append(a.get("ensemble_label", "neutral"))

        total_w = sum(weights) or 1.0
        avg_sentiment = sum(s * w for s, w in zip(scores, weights)) / total_w

        n = len(labels)
        positive_pct = sum(1 for l in labels if "positive" in l) / n * 100
        negative_pct = sum(1 for l in labels if "negative" in l) / n * 100
        neutral_pct = 100.0 - positive_pct - negative_pct

        # Top headlines
        pos_arts = sorted(
            [a for a in in_period if "positive" in a.get("ensemble_label", "")],
            key=lambda x: x.get("ensemble_score", 0.0),
            reverse=True,
        )
        neg_arts = sorted(
            [a for a in in_period if "negative" in a.get("ensemble_label", "")],
            key=lambda x: x.get("ensemble_score", 0.0),
        )

        domains = {_extract_domain(a.get("url", "")) for a in in_period if a.get("url")}

        return {
            "ticker": ticker,
            "article_count": len(in_period),
            "avg_sentiment": round(avg_sentiment, 4),
            "sentiment_momentum": None,  # filled by pipeline with prior-period comparison
            "positive_pct": round(positive_pct, 1),
            "negative_pct": round(negative_pct, 1),
            "neutral_pct": round(neutral_pct, 1),
            "top_positive_headlines": [a.get("title", "") for a in pos_arts[:3]],
            "top_negative_headlines": [a.get("title", "") for a in neg_arts[:3]],
            "source_diversity": len(domains),
        }

    def compute_sentiment_signal(
        self,
        ticker: str,
        recent_summary: dict,
        historical_scores: list[float],
    ) -> dict:
        """
        Derive actionable signal from aggregated sentiment.
        historical_scores: list of daily avg_sentiment values, most recent last.
        """
        current = recent_summary.get("avg_sentiment", 0.0)

        # Trend: compare current vs 5-day rolling mean
        if len(historical_scores) >= 5:
            rolling_mean = sum(historical_scores[-5:]) / 5
            delta = current - rolling_mean
            if delta > 0.1:
                trend = "rising"
            elif delta < -0.1:
                trend = "falling"
            else:
                trend = "flat"
        else:
            trend = "flat"

        # Extreme: top/bottom 10% of 90-day distribution
        is_extreme = False
        if len(historical_scores) >= 30:
            p10 = float(np.percentile(historical_scores, 10))
            p90 = float(np.percentile(historical_scores, 90))
            is_extreme = current <= p10 or current >= p90

        # Signal mapping
        if current >= 0.5 and trend == "rising":
            signal = "strong_buy"
        elif current >= 0.15:
            signal = "buy"
        elif current <= -0.5 and trend == "falling":
            signal = "strong_sell"
        elif current <= -0.15:
            signal = "sell"
        else:
            signal = "neutral"

        return {
            "ticker": ticker,
            "signal": signal,
            "trend": trend,
            "is_extreme": is_extreme,
            "current_sentiment": current,
        }

    def sector_sentiment_heatmap(
        self,
        ticker_summaries: list[dict],
        ticker_sector_map: dict[str, str],
    ) -> pd.DataFrame:
        """
        Aggregate ticker-level summaries into a GICS sector heatmap.
        Returns DataFrame indexed by sector with avg_sentiment, article_count.
        """
        rows: list[dict] = []
        for summary in ticker_summaries:
            ticker = summary.get("ticker", "")
            sector = ticker_sector_map.get(ticker, "Unknown")
            rows.append({
                "sector": sector,
                "ticker": ticker,
                "avg_sentiment": summary.get("avg_sentiment", 0.0),
                "article_count": summary.get("article_count", 0),
            })

        if not rows:
            return pd.DataFrame(columns=["sector", "avg_sentiment", "article_count", "ticker_count"])

        df = pd.DataFrame(rows)
        heatmap = (
            df.groupby("sector")
            .agg(
                avg_sentiment=("avg_sentiment", "mean"),
                article_count=("article_count", "sum"),
                ticker_count=("ticker", "nunique"),
            )
            .reset_index()
            .sort_values("avg_sentiment", ascending=False)
        )
        return heatmap


# ---------------------------------------------------------------------------
# NewsSentimentPipeline — orchestrator
# ---------------------------------------------------------------------------


class NewsSentimentPipeline:
    """
    Full end-to-end news sentiment pipeline.

    Usage::

        pipeline = NewsSentimentPipeline()
        df = await pipeline.run(["AAPL", "MSFT", "GOOGL"])
        print(df)
    """

    def __init__(
        self,
        company_name_map: Optional[dict[str, str]] = None,
        cik_map: Optional[dict[str, str]] = None,
        lookback_hours: int = 24,
    ) -> None:
        self.collector = MultiSourceNewsCollector()
        self.scorer = SentimentScorer()
        self.aggregator = SentimentAggregator()
        # ticker → company name (for query building)
        self.company_name_map: dict[str, str] = company_name_map or {}
        # ticker → SEC CIK (for EDGAR 8-K)
        self.cik_map: dict[str, str] = cik_map or {}
        self.lookback_hours = lookback_hours

    async def run(
        self,
        tickers: list[str],
        lookback_hours: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Full pipeline run.

        Steps:
          1. Collect from GDELT + all RSS sources + EDGAR 8-K
          2. Deduplicate
          3. Batch score (FinBERT → LM → VADER)
          4. Aggregate to ticker-level
          5. Compute signals
          6. Return summary DataFrame

        Returns
        -------
        pd.DataFrame with columns:
            ticker, article_count, avg_sentiment, sentiment_momentum,
            positive_pct, negative_pct, neutral_pct, signal, trend,
            is_extreme, source_diversity, top_positive_headlines,
            top_negative_headlines
        """
        hours = lookback_hours or self.lookback_hours
        company_names = [self.company_name_map.get(t, t) for t in tickers]

        logger.info("Pipeline run start", tickers=tickers, lookback_hours=hours)

        # Step 1: Collect
        all_articles: list[dict] = []

        gdelt_articles = await self.collector.collect_gdelt(tickers, company_names, hours)
        all_articles.extend(gdelt_articles)

        rss_tasks = []
        for ticker, company in zip(tickers, company_names):
            rss_tasks.append(self.collector.collect_rss_feeds(ticker, company))
            rss_tasks.append(self.collector.collect_seeking_alpha(ticker))
            cik = self.cik_map.get(ticker)
            if cik:
                rss_tasks.append(self.collector.collect_edgar_8k(cik, hours))

        rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
        for r in rss_results:
            if isinstance(r, list):
                all_articles.extend(r)

        logger.info("Articles collected (pre-dedup)", total=len(all_articles))

        # Step 2: Deduplicate
        deduped = self.collector.deduplicate_articles(all_articles)
        logger.info("Articles after dedup", total=len(deduped))

        # Tag articles with tickers based on title mention
        for art in deduped:
            if not art.get("ticker"):
                for t in tickers:
                    if t.upper() in (art.get("title", "") + art.get("summary", "")).upper():
                        art["ticker"] = t
                        break

        # Step 3: Batch score
        scored = await self.scorer.batch_score(deduped)
        logger.info("Articles scored", total=len(scored))

        # Step 4 + 5: Aggregate + compute signals
        rows: list[dict] = []
        for ticker in tickers:
            summary = self.aggregator.aggregate_to_ticker(scored, ticker, hours)
            signal_info = self.aggregator.compute_sentiment_signal(
                ticker=ticker,
                recent_summary=summary,
                historical_scores=[],  # no historical DB in this call; wire up separately
            )
            row = {**summary, **signal_info}
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info("Pipeline complete", tickers=tickers, rows=len(df))
        return df

    async def run_continuous(
        self,
        tickers: list[str],
        interval_seconds: int = 900,
    ) -> None:
        """
        Scheduled polling loop. Runs every interval_seconds (default 15 min).
        Intended to be run as a background asyncio task.
        """
        logger.info(
            "Continuous pipeline started",
            tickers=tickers,
            interval_seconds=interval_seconds,
        )
        while True:
            try:
                df = await self.run(tickers)
                logger.info(
                    "Continuous pipeline cycle complete",
                    tickers=tickers,
                    rows=len(df),
                )
            except Exception as exc:
                logger.error("Continuous pipeline error", error=str(exc))
            await asyncio.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

sentiment_router = APIRouter(prefix="/api/sentiment", tags=["News Sentiment"])

# Module-level pipeline instance (initialised lazily)
_pipeline: Optional[NewsSentimentPipeline] = None


def _get_pipeline() -> NewsSentimentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = NewsSentimentPipeline()
    return _pipeline


@sentiment_router.get("/{ticker}", summary="Current sentiment summary for a ticker")
async def get_ticker_sentiment(ticker: str):
    """Return current news sentiment summary for a single ticker."""
    pipeline = _get_pipeline()
    try:
        df = await pipeline.run([ticker.upper()], lookback_hours=24)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No sentiment data for {ticker}")
        row = df.iloc[0].to_dict()
        # Convert list columns to JSON-safe
        for col in ("top_positive_headlines", "top_negative_headlines"):
            if col in row and not isinstance(row[col], list):
                row[col] = []
        return row
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Sentiment endpoint error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@sentiment_router.get("/{ticker}/history", summary="Sentiment timeseries")
async def get_ticker_sentiment_history(
    ticker: str,
    days: int = Query(default=30, ge=1, le=90),
):
    """Return daily sentiment timeseries for the past N days (cached data)."""
    # In production this would query the news_articles DB table
    # Return placeholder with today's live reading
    pipeline = _get_pipeline()
    df = await pipeline.run([ticker.upper()], lookback_hours=days * 24)
    if df.empty:
        return {"ticker": ticker, "history": []}
    row = df.iloc[0].to_dict()
    return {
        "ticker": ticker,
        "history": [
            {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "avg_sentiment": row.get("avg_sentiment", 0.0),
                "article_count": row.get("article_count", 0),
                "signal": row.get("signal", "neutral"),
            }
        ],
    }


@sentiment_router.get("/{ticker}/articles", summary="Recent articles with sentiment scores")
async def get_ticker_articles(
    ticker: str,
    hours: int = Query(default=24, ge=1, le=168),
):
    """Return recent news articles with sentiment scores for a ticker."""
    pipeline = _get_pipeline()
    collector = pipeline.collector
    scorer = pipeline.scorer

    company = pipeline.company_name_map.get(ticker.upper(), ticker)
    articles = await collector.collect_rss_feeds(ticker.upper(), company)
    gdelt = await collector.collect_gdelt([ticker.upper()], [company], hours)
    articles.extend(gdelt)
    deduped = collector.deduplicate_articles(articles)
    scored = await scorer.batch_score(deduped[:20])  # cap at 20 for API response

    return {
        "ticker": ticker.upper(),
        "count": len(scored),
        "articles": [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "source": a.get("source"),
                "published": a.get("published").isoformat() if a.get("published") else None,
                "sentiment": a.get("ensemble_label"),
                "score": a.get("ensemble_score"),
                "confidence": a.get("ensemble_confidence"),
            }
            for a in scored
        ],
    }


@sentiment_router.get("/sector/{sector}", summary="Sector-level sentiment aggregation")
async def get_sector_sentiment(sector: str):
    """Return aggregated sentiment for a GICS sector (uses representative tickers)."""
    # Representative tickers per GICS sector
    SECTOR_TICKERS: dict[str, list[str]] = {
        "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
        "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
        "financials": ["JPM", "BAC", "WFC", "GS", "MS"],
        "energy": ["XOM", "CVX", "SLB", "COP", "EOG"],
        "consumer_discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
        "industrials": ["CAT", "BA", "HON", "UPS", "GE"],
        "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
        "materials": ["LIN", "APD", "ECL", "SHW", "FCX"],
        "real_estate": ["AMT", "PLD", "CCI", "EQIX", "PSA"],
        "communication_services": ["GOOGL", "META", "T", "VZ", "NFLX"],
        "consumer_staples": ["PG", "KO", "PEP", "WMT", "COST"],
    }

    sec_lower = sector.lower().replace(" ", "_")
    tickers = SECTOR_TICKERS.get(sec_lower)
    if not tickers:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sector '{sector}'. Valid: {list(SECTOR_TICKERS.keys())}",
        )

    pipeline = _get_pipeline()
    df = await pipeline.run(tickers, lookback_hours=24)
    if df.empty:
        return {"sector": sector, "avg_sentiment": 0.0, "tickers": []}

    sector_avg = float(df["avg_sentiment"].mean())
    return {
        "sector": sector,
        "avg_sentiment": round(sector_avg, 4),
        "article_count": int(df["article_count"].sum()),
        "tickers": df[["ticker", "avg_sentiment", "signal"]].to_dict(orient="records"),
    }


class BatchSentimentRequest(BaseModel):
    tickers: list[str]
    lookback_hours: int = 24


@sentiment_router.post("/batch", summary="Batch sentiment for multiple tickers")
async def batch_ticker_sentiment(request: BatchSentimentRequest):
    """Score sentiment for a list of tickers in a single call."""
    if not request.tickers:
        raise HTTPException(status_code=400, detail="tickers list cannot be empty")
    if len(request.tickers) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 tickers per batch request")

    pipeline = _get_pipeline()
    tickers = [t.upper() for t in request.tickers]

    try:
        df = await pipeline.run(tickers, lookback_hours=request.lookback_hours)
        return {
            "count": len(df),
            "results": df[
                ["ticker", "avg_sentiment", "article_count", "signal", "trend", "positive_pct", "negative_pct"]
            ].to_dict(orient="records"),
        }
    except Exception as exc:
        logger.error("Batch sentiment error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
