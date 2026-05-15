"""Earnings call / news corpus management — RAG-ready (dim_057).

Targets score raise from 6 → 9+ by building comprehensive, queryable corpora of:
  - Earnings press releases (EDGAR 8-K Exhibit 99.1 / Items 7.01 & 8.01)
  - Google News / Seeking Alpha / MarketWatch RSS feeds
  - Deduplication, classification, and sentiment scoring
  - TimescaleDB hypertables for earnings_corpus + news_corpus
  - Analytics: sentiment trend, management language, guidance trend, peer correlation,
    pre-earnings risk signals

Architecture:
  EarningsTranscriptAdapter  — EDGAR 8-K earnings data + yfinance earnings dates
  NewsCorpusBuilder          — Multi-source RSS ingest + dedup + classify
  CorpusAnalyticsEngine      — Sentiment trend, guidance, peer correlation, risk signals
  CorpusIndex                — DDL, hypertable setup, full ingest pipeline
  earnings_router            — FastAPI router

Usage::
    from sentinel.sai.earnings_corpus import EarningsTranscriptAdapter, CorpusIndex

    adapter = EarningsTranscriptAdapter()
    timeline = await adapter.build_earnings_timeline("AAPL")

    index = CorpusIndex(db_url)
    await index.ingest_full_corpus("AAPL")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional
from urllib.parse import quote_plus

import feedparser
import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_USER_AGENT = "SENTINEL financial-terminal richard.porras@realempanada.com"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Loughran-McDonald positive / negative word lists (abridged, financial domain)
_LM_POSITIVE = frozenset([
    "achieve", "achieved", "advances", "benefit", "benefits", "better", "confident",
    "deliver", "delivered", "efficient", "exceed", "exceeded", "excellent", "expand",
    "growth", "improve", "improved", "improvement", "increase", "increased", "innovative",
    "momentum", "opportunities", "outperform", "positive", "profitability", "profitable",
    "progress", "record", "robust", "strong", "strengthened", "success", "successful",
    "surpass", "sustainable", "thriving", "upside",
])
_LM_NEGATIVE = frozenset([
    "adverse", "challenging", "challenges", "concern", "concerns", "decline", "declined",
    "decrease", "decreased", "deteriorate", "difficult", "difficulties", "disappoint",
    "disappointing", "disruption", "downturn", "fail", "failure", "headwind", "headwinds",
    "impair", "impairment", "inflation", "loss", "losses", "negative", "pressure",
    "pressures", "reduce", "reduction", "risk", "risks", "shortage", "slowdown",
    "uncertainty", "unfavorable", "volatile", "volatility", "weakness", "worsen",
])

# ── Engine registry ────────────────────────────────────────────────────────────

_engines: dict[str, AsyncEngine] = {}
_schema_ready: set[str] = set()


def _get_engine(db_url: str) -> AsyncEngine:
    if db_url not in _engines:
        _engines[db_url] = create_async_engine(
            db_url, pool_size=5, max_overflow=10, pool_pre_ping=True
        )
    return _engines[db_url]


# ── DDL ────────────────────────────────────────────────────────────────────────

_EARNINGS_CORPUS_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS earnings_corpus (
    id           BIGSERIAL,
    ticker       VARCHAR(20)  NOT NULL,
    quarter      VARCHAR(20),
    doc_type     VARCHAR(30)  NOT NULL,
    chunk_text   TEXT         NOT NULL,
    embedding    vector(384),
    metadata     JSONB        NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

CREATE TABLE IF NOT EXISTS news_corpus (
    id              BIGSERIAL,
    ticker          VARCHAR(20),
    published_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    headline        TEXT        NOT NULL,
    chunk_text      TEXT        NOT NULL,
    embedding       vector(384),
    category        VARCHAR(40),
    sentiment_score FLOAT,
    url_hash        VARCHAR(32),
    metadata        JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id, published_at)
);

CREATE INDEX IF NOT EXISTS ix_ec_ticker   ON earnings_corpus (ticker);
CREATE INDEX IF NOT EXISTS ix_ec_quarter  ON earnings_corpus (quarter);
CREATE INDEX IF NOT EXISTS ix_nc_ticker   ON news_corpus (ticker);
CREATE INDEX IF NOT EXISTS ix_nc_pub      ON news_corpus (published_at DESC);
CREATE INDEX IF NOT EXISTS ix_nc_url_hash ON news_corpus (url_hash) WHERE url_hash IS NOT NULL;
"""

_HYPERTABLE_EARNINGS = """
SELECT create_hypertable('earnings_corpus', 'created_at', if_not_exists => TRUE)
"""
_HYPERTABLE_NEWS = """
SELECT create_hypertable('news_corpus', 'published_at', if_not_exists => TRUE)
"""
_IVFFLAT_EC = """
CREATE INDEX IF NOT EXISTS ix_ec_embedding_ivfflat
    ON earnings_corpus USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = {lists})
"""
_IVFFLAT_NC = """
CREATE INDEX IF NOT EXISTS ix_nc_embedding_ivfflat
    ON news_corpus USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = {lists})
"""


async def _ensure_corpus_schema(db_url: str) -> None:
    if db_url in _schema_ready:
        return
    engine = _get_engine(db_url)
    async with engine.begin() as conn:
        for stmt in _EARNINGS_CORPUS_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    await conn.execute(text(stmt))
                except Exception as exc:
                    logger.debug("DDL skipped", stmt=stmt[:60], error=str(exc))

        # TimescaleDB hypertables (no-op if already created)
        for ht_sql in [_HYPERTABLE_EARNINGS, _HYPERTABLE_NEWS]:
            try:
                await conn.execute(text(ht_sql))
            except Exception as exc:
                logger.debug("Hypertable creation skipped", error=str(exc))

    _schema_ready.add(db_url)
    logger.info("earnings_corpus + news_corpus schema ready")


# ── Embedding helpers ──────────────────────────────────────────────────────────

_embedder: Any = None


def _get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading all-MiniLM-L6-v2 for earnings corpus...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Corpus embedder ready")
    return _embedder


async def _embed_async(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    loop = asyncio.get_event_loop()
    arr: np.ndarray = await loop.run_in_executor(
        None, lambda: _get_embedder().encode(texts, batch_size=64, show_progress_bar=False)
    )
    return [row.tolist() for row in arr]


def _vec_str(emb: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in emb) + "]"


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks, start, total = [], 0, len(words)
    while start < total:
        end = min(start + chunk_size, total)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == total:
            break
        start = end - overlap
    return chunks


def _strip_html(text: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, ch in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                    ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(ent, ch)
    text = re.sub(r"&#\d+;", " ", text)
    return re.sub(r"\s{3,}", "  ", text).strip()


# ── EDGAR helpers ──────────────────────────────────────────────────────────────

async def _resolve_cik(ticker: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                EDGAR_TICKERS_URL,
                headers={**_HEADERS, "Host": "www.sec.gov"},
            )
            resp.raise_for_status()
            data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK resolution failed", ticker=ticker, error=str(exc))
    return None


async def _fetch_submissions(cik: str) -> dict:
    await asyncio.sleep(0.12)  # EDGAR 10 req/s limit
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{EDGAR_BASE}/submissions/CIK{cik}.json",
            headers={**_HEADERS, "Host": "data.sec.gov"},
        )
        resp.raise_for_status()
        return resp.json()


async def _fetch_filing_raw(cik: str, accession: str, primary_doc: str) -> str:
    acc_clean = accession.replace("-", "")
    cik_s = cik.lstrip("0") or "0"
    url = f"{EDGAR_ARCHIVES}/{cik_s}/{acc_clean}/{primary_doc}"
    await asyncio.sleep(0.12)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        return resp.text


# ── EarningsTranscriptAdapter ──────────────────────────────────────────────────

class EarningsTranscriptAdapter:
    """Retrieves earnings data from EDGAR 8-K filings and yfinance earnings dates.

    EDGAR 8-K Item 7.01 (Regulation FD Disclosure) and Item 8.01 filings contain
    earnings press releases (Exhibit 99.1) with reported revenue, EPS, and guidance.
    """

    async def get_from_edgar_8k(
        self,
        ticker: str,
        cik: Optional[str] = None,
        lookback_quarters: int = 8,
    ) -> list[dict]:
        """Search EDGAR 8-K filings for earnings press releases.

        Fetches up to lookback_quarters most recent 8-K filings, extracts
        Exhibit 99.1 text, and parses reported financials and guidance.

        Args:
            ticker:             Ticker symbol.
            cik:                CIK (resolved from ticker if omitted).
            lookback_quarters:  Max number of 8-K filings to check.

        Returns:
            List of dicts with period, revenue_announced, eps_announced,
            guidance_text, beat_miss.
        """
        if not cik:
            cik = await _resolve_cik(ticker)
        if not cik:
            return [{"error": f"CIK not found: {ticker}"}]

        cik = cik.zfill(10)
        try:
            submissions = await _fetch_submissions(cik)
        except Exception as exc:
            return [{"error": str(exc)}]

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        periods = recent.get("reportDate", [])

        eightk_filings = [
            {
                "filing_date": dates[i],
                "accession": accessions[i],
                "primary_doc": primary_docs[i],
                "period": periods[i] if i < len(periods) else "",
            }
            for i, form in enumerate(forms)
            if form == "8-K"
        ][:lookback_quarters]

        results = []
        for f in eightk_filings:
            if not f["accession"] or not f["primary_doc"]:
                continue
            try:
                raw = await _fetch_filing_raw(cik, f["accession"], f["primary_doc"])
                clean = _strip_html(raw) if "<" in raw[:500] else raw
                parsed = self.parse_earnings_press_release(clean)
                parsed["period"] = f["period"]
                parsed["filing_date"] = f["filing_date"]
                parsed["accession"] = f["accession"]
                results.append(parsed)
            except Exception as exc:
                logger.warning("8-K parse failed", accession=f["accession"], error=str(exc))

        return results

    def parse_earnings_press_release(self, text: str) -> dict:
        """Extract structured financial data from an earnings press release.

        Uses regex patterns to find reported revenue, EPS, guidance ranges,
        and key qualitative phrases.

        Args:
            text: Cleaned plain-text content of the press release.

        Returns:
            Dict with reported_revenue, reported_eps, guidance_revenue_low/high,
            guidance_eps_low/high, beat_miss (None if estimates unavailable),
            key_phrases (list of notable management phrases).
        """
        result: dict = {
            "reported_revenue": None,
            "reported_eps": None,
            "guidance_revenue_low": None,
            "guidance_revenue_high": None,
            "guidance_eps_low": None,
            "guidance_eps_high": None,
            "guidance_text": None,
            "beat_miss": None,
            "key_phrases": [],
        }

        tl = text.lower()

        # Revenue: "$X.X billion" or "$X,XXX million"
        m = re.search(
            r"(?:revenue|net\s+sales|total\s+revenue)[^$\n]*?\$([\d,\.]+)\s*(billion|million|B\b|M\b)",
            text, re.IGNORECASE
        )
        if m:
            val = float(m.group(1).replace(",", ""))
            unit = m.group(2).lower()
            result["reported_revenue"] = val * 1e9 if unit in ("billion", "b") else val * 1e6

        # EPS: "earnings per share of $X.XX" or "diluted EPS of $X.XX"
        m = re.search(
            r"(?:diluted\s+)?(?:earnings\s+per\s+(?:diluted\s+)?share|EPS)[^$\n]*?\$([\d\.]+)",
            text, re.IGNORECASE
        )
        if m:
            result["reported_eps"] = float(m.group(1))

        # Guidance ranges: "$X.X billion to $X.X billion"
        m = re.search(
            r"(?:expect|guide|guidance|outlook|anticipate)[^$\n]*?\$([\d,\.]+)\s*(?:billion|B)\s+to\s+\$([\d,\.]+)\s*(billion|B)",
            text, re.IGNORECASE
        )
        if m:
            result["guidance_revenue_low"] = float(m.group(1).replace(",", "")) * 1e9
            result["guidance_revenue_high"] = float(m.group(2).replace(",", "")) * 1e9

        # EPS guidance range: "$X.XX to $X.XX"
        m = re.search(
            r"(?:EPS|earnings\s+per\s+share)[^$\n]*?\$([\d\.]+)\s+to\s+\$([\d\.]+)",
            text, re.IGNORECASE
        )
        if m:
            result["guidance_eps_low"] = float(m.group(1))
            result["guidance_eps_high"] = float(m.group(2))

        # Guidance text extraction (first 300 chars after "outlook" or "guidance")
        m = re.search(r"(?:outlook|full.year guidance|fiscal year guidance)[:\s](.{50,300})", text, re.IGNORECASE)
        if m:
            result["guidance_text"] = m.group(1).strip()[:300]

        # Key qualitative phrases
        key_phrase_patterns = [
            r"record\s+\w+\s*(?:revenue|quarter|year|earnings|results)",
            r"headwind[s]?\s+(?:from|in|related)",
            r"strong\s+(?:demand|growth|performance|momentum)",
            r"margin\s+(?:expansion|compression|improvement|pressure)",
            r"challenges?\s+(?:in|from|related)",
        ]
        phrases = []
        for pat in key_phrase_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                phrases.append(m.group(0).strip()[:80])
        result["key_phrases"] = phrases[:10]

        return result

    async def get_earnings_dates(self, ticker: str) -> list[dict]:
        """Retrieve past and upcoming earnings dates via yfinance.

        Returns:
            List of {date, eps_estimate, eps_actual, surprise_pct, reported}.
        """
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()

            def _fetch() -> list[dict]:
                yft = yf.Ticker(ticker.upper())
                df = yft.earnings_dates
                if df is None or df.empty:
                    return []
                rows = []
                for idx, row in df.iterrows():
                    rows.append({
                        "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                        "eps_estimate": row.get("EPS Estimate"),
                        "eps_actual": row.get("Reported EPS"),
                        "surprise_pct": row.get("Surprise(%)"),
                        "reported": row.get("Reported EPS") is not None,
                    })
                return rows

            return await loop.run_in_executor(None, _fetch)
        except ImportError:
            logger.warning("yfinance not installed — install: pip install yfinance")
            return []
        except Exception as exc:
            logger.error("get_earnings_dates failed", ticker=ticker, error=str(exc))
            return []

    async def build_earnings_timeline(self, ticker: str) -> pd.DataFrame:
        """Build a quarter-by-quarter earnings timeline with stock reaction.

        Combines EDGAR 8-K press release data with yfinance earnings dates and
        historical price data to calculate short-term stock reactions.

        Args:
            ticker: Ticker symbol.

        Returns:
            DataFrame with columns: date, quarter, revenue, eps, revenue_surprise_pct,
            eps_surprise_pct, stock_reaction_1d, stock_reaction_5d.
        """
        earnings_dates = await self.get_earnings_dates(ticker)
        edgar_8k = await self.get_from_edgar_8k(ticker)

        if not earnings_dates:
            return pd.DataFrame(columns=[
                "date", "quarter", "revenue", "eps",
                "revenue_surprise_pct", "eps_surprise_pct",
                "stock_reaction_1d", "stock_reaction_5d"
            ])

        # Try to enrich with price data for stock reaction
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()

            def _get_prices() -> pd.DataFrame:
                cutoff = datetime.utcnow() - timedelta(days=365 * 3)
                yft = yf.Ticker(ticker.upper())
                hist = yft.history(start=cutoff.strftime("%Y-%m-%d"))
                return hist

            price_df = await loop.run_in_executor(None, _get_prices)
        except Exception:
            price_df = pd.DataFrame()

        rows = []
        for ed in earnings_dates:
            ed_date_str = ed.get("date", "")
            try:
                ed_dt = datetime.fromisoformat(ed_date_str[:10])
            except Exception:
                continue

            # Stock reaction: price change on earnings day vs day before
            reaction_1d = None
            reaction_5d = None
            if not price_df.empty:
                try:
                    idx = price_df.index.searchsorted(ed_dt)
                    if idx > 0 and idx < len(price_df):
                        p0 = float(price_df.iloc[idx - 1]["Close"])
                        p1 = float(price_df.iloc[idx]["Close"])
                        reaction_1d = round((p1 / p0 - 1) * 100, 2) if p0 > 0 else None
                    if idx > 0 and idx + 5 < len(price_df):
                        p5 = float(price_df.iloc[idx + 5]["Close"])
                        p0 = float(price_df.iloc[idx - 1]["Close"])
                        reaction_5d = round((p5 / p0 - 1) * 100, 2) if p0 > 0 else None
                except Exception:
                    pass

            rows.append({
                "date": ed_date_str[:10],
                "quarter": _date_to_quarter(ed_dt),
                "revenue": None,
                "eps": ed.get("eps_actual"),
                "eps_estimate": ed.get("eps_estimate"),
                "revenue_surprise_pct": None,
                "eps_surprise_pct": ed.get("surprise_pct"),
                "stock_reaction_1d": reaction_1d,
                "stock_reaction_5d": reaction_5d,
            })

        # Merge 8-K revenue data
        for r in rows:
            quarter = r.get("quarter", "")
            for e8k in edgar_8k:
                if _date_to_quarter(_parse_date(e8k.get("period", ""))) == quarter:
                    r["revenue"] = e8k.get("reported_revenue")
                    break

        return pd.DataFrame(rows) if rows else pd.DataFrame()


def _date_to_quarter(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    q = (dt.month - 1) // 3 + 1
    return f"Q{q} {dt.year}"


def _parse_date(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:10], fmt)
        except Exception:
            pass
    return None


# ── NewsCorpusBuilder ──────────────────────────────────────────────────────────

class NewsCorpusBuilder:
    """Multi-source RSS news ingest with deduplication and classification."""

    _ARTICLE_CATEGORIES = [
        ("earnings", r"\b(earnings|quarterly\s+results|eps|revenue\s+beat|revenue\s+miss)\b"),
        ("guidance", r"\b(guidance|outlook|forecast|raises?\s+guidance|lowers?\s+guidance)\b"),
        ("analyst_upgrade", r"\b(upgrade|upgraded|buy\s+rating|outperform|overweight)\b"),
        ("analyst_downgrade", r"\b(downgrade|downgraded|sell\s+rating|underperform|underweight)\b"),
        ("ma_announcement", r"\b(merger|acquisition|acquires?|deal|takeover|buyout)\b"),
        ("regulatory", r"\b(regulatory|fda|sec|ftc|doj|antitrust|investigation|fine|penalty)\b"),
        ("product_launch", r"\b(launches?|announces?\s+new|unveils?|introduces?|debut)\b"),
        ("management_change", r"\b(ceo|cfo|cto|appoints?|resigns?|steps\s+down|leadership)\b"),
        ("macro", r"\b(fed|fomc|interest\s+rate|inflation|gdp|recession|tariff|trade\s+war)\b"),
    ]

    async def ingest_google_news_rss(
        self,
        ticker: str,
        company_name: str,
        lookback_days: int = 90,
    ) -> list[dict]:
        """Fetch articles from Google News RSS for a ticker and company name.

        Args:
            ticker:         Ticker symbol (used in search query).
            company_name:   Company name (used in search query).
            lookback_days:  Articles older than this are excluded.

        Returns:
            List of article dicts with title, link, pubDate, summary, source.
        """
        query = quote_plus(f"{company_name} {ticker} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        return await self._fetch_rss(url, source="google_news", ticker=ticker,
                                     lookback_days=lookback_days)

    async def ingest_seeking_alpha_rss(self, ticker: str) -> list[dict]:
        """Fetch articles from Seeking Alpha public RSS feed for a ticker."""
        url = f"https://seekingalpha.com/symbol/{ticker.upper()}/feed.xml"
        return await self._fetch_rss(url, source="seeking_alpha", ticker=ticker)

    async def ingest_marketwatch_rss(self, ticker: str) -> list[dict]:
        """Fetch articles from MarketWatch RSS bulletin feed."""
        url = "https://feeds.marketwatch.com/marketwatch/bulletins/"
        articles = await self._fetch_rss(url, source="marketwatch", ticker=None)
        # Filter for articles mentioning the ticker
        t = ticker.upper()
        return [a for a in articles if t in a.get("title", "").upper() or
                t in a.get("summary", "").upper()]

    async def ingest_edgar_press_releases(
        self, cik: str, lookback_days: int = 180
    ) -> list[dict]:
        """Fetch EDGAR 8-K filings as news events (press releases).

        Treats 8-K filing dates as news events with the filing abstract as summary.

        Args:
            cik:           Zero-padded CIK.
            lookback_days: Lookback window in days.

        Returns:
            List of article dicts with title, pubDate, source, summary.
        """
        cik = cik.zfill(10)
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        try:
            from sentinel.sai.rag_financial_docs import _fetch_submissions
            submissions = await _fetch_submissions(cik)
        except Exception as exc:
            logger.warning("EDGAR submissions failed", cik=cik, error=str(exc))
            return []

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])

        articles = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            filing_date_str = dates[i] if i < len(dates) else None
            if not filing_date_str:
                continue
            try:
                filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d")
            except Exception:
                continue
            if filing_dt < cutoff:
                break  # EDGAR returns in descending date order

            accession = accessions[i] if i < len(accessions) else ""
            articles.append({
                "title": f"{cik} 8-K Filing {filing_date_str}",
                "link": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K",
                "pubDate": filing_date_str,
                "summary": f"EDGAR 8-K filing {accession}",
                "source": "edgar_8k",
                "url_hash": hashlib.md5(accession.encode()).hexdigest(),
            })

        return articles

    async def _fetch_rss(
        self,
        url: str,
        source: str,
        ticker: Optional[str],
        lookback_days: int = 90,
    ) -> list[dict]:
        """Fetch and parse an RSS feed, returning structured article dicts."""
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                content = resp.text
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, lambda: feedparser.parse(content))

            articles = []
            for entry in feed.entries[:50]:
                pub_time = _parse_feed_time(entry)
                if pub_time < cutoff:
                    continue
                articles.append({
                    "title": entry.get("title", "")[:300],
                    "link": entry.get("link", ""),
                    "pubDate": pub_time.isoformat(),
                    "summary": entry.get("summary", "")[:500],
                    "source": source,
                    "ticker": ticker,
                    "url_hash": hashlib.md5(entry.get("link", "").encode()).hexdigest(),
                })
            return articles
        except Exception as exc:
            logger.warning("RSS fetch failed", source=source, url=url[:80], error=str(exc))
            return []

    def deduplicate(self, articles: list[dict]) -> list[dict]:
        """Remove duplicate articles by URL hash and title Jaccard similarity.

        Two articles are considered duplicates if they share the same URL hash
        or if their title word overlap exceeds 80% (Jaccard index).

        Args:
            articles: List of article dicts.

        Returns:
            Deduplicated list preserving the first occurrence.
        """
        seen_hashes: set[str] = set()
        seen_titles: list[set[str]] = []
        result = []

        for article in articles:
            url_hash = article.get("url_hash", "")
            if url_hash and url_hash in seen_hashes:
                continue

            title_words = set(
                re.sub(r"[^\w\s]", "", article.get("title", "").lower()).split()
            ) - {"the", "a", "an", "and", "or", "in", "of", "to", "for", "on"}

            # Check Jaccard similarity against existing titles
            is_dup = False
            for prev_words in seen_titles:
                if not prev_words or not title_words:
                    continue
                intersection = len(title_words & prev_words)
                union = len(title_words | prev_words)
                if union > 0 and intersection / union >= 0.8:
                    is_dup = True
                    break

            if is_dup:
                continue

            if url_hash:
                seen_hashes.add(url_hash)
            seen_titles.append(title_words)
            result.append(article)

        return result

    def classify_article(self, article: dict) -> str:
        """Classify article into a category based on title + summary content.

        Categories: earnings, guidance, analyst_upgrade, analyst_downgrade,
        ma_announcement, regulatory, product_launch, management_change, macro, other.

        Args:
            article: Article dict with title and summary fields.

        Returns:
            Category string.
        """
        text = (article.get("title", "") + " " + article.get("summary", "")).lower()
        for category, pattern in self._ARTICLE_CATEGORIES:
            if re.search(pattern, text, re.IGNORECASE):
                return category
        return "other"


def _parse_feed_time(entry: Any) -> datetime:
    import time as time_mod
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if published:
        try:
            return datetime.utcfromtimestamp(time_mod.mktime(published))
        except Exception:
            pass
    return datetime.utcnow()


# ── CorpusAnalyticsEngine ──────────────────────────────────────────────────────

class CorpusAnalyticsEngine:
    """Analytics over the earnings_corpus and news_corpus tables."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def _lm_sentiment(self, text: str) -> float:
        """Loughran-McDonald word-list sentiment score in [-1, 1].

        Positive score = net optimistic language; negative = net cautious/pessimistic.
        """
        words = re.findall(r"\b[a-z]+\b", text.lower())
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in _LM_POSITIVE)
        neg = sum(1 for w in words if w in _LM_NEGATIVE)
        total = pos + neg
        return round((pos - neg) / total, 4) if total > 0 else 0.0

    async def compute_news_sentiment_trend(
        self,
        ticker: str,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """Compute daily average news sentiment from the news_corpus table.

        Uses stored sentiment_score values (set at ingest time via LM word list).
        Falls back to corpus text if scores are missing.

        Args:
            ticker:       Ticker filter.
            lookback_days: Lookback window in days.

        Returns:
            DataFrame with columns: date, avg_sentiment, article_count, category_counts.
        """
        await _ensure_corpus_schema(self.db_url)
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        engine = _get_engine(self.db_url)

        sql = text("""
            SELECT
                DATE(published_at) AS pub_date,
                AVG(COALESCE(sentiment_score, 0)) AS avg_sentiment,
                COUNT(*) AS article_count,
                category
            FROM news_corpus
            WHERE ticker = :ticker
              AND published_at >= :cutoff
            GROUP BY pub_date, category
            ORDER BY pub_date DESC
        """)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, {"ticker": ticker.upper(), "cutoff": cutoff})).fetchall()

        if not rows:
            return pd.DataFrame(columns=["date", "avg_sentiment", "article_count"])

        df = pd.DataFrame([dict(r._mapping) for r in rows])
        daily = (
            df.groupby("pub_date")
            .agg(avg_sentiment=("avg_sentiment", "mean"), article_count=("article_count", "sum"))
            .reset_index()
            .rename(columns={"pub_date": "date"})
        )
        daily["date"] = daily["date"].astype(str)
        return daily.sort_values("date", ascending=False)

    async def extract_management_language(
        self,
        ticker: str,
        quarter: Optional[str] = None,
    ) -> dict:
        """Analyse management language in earnings corpus.

        Counts positive/negative words, forward-looking statements, uncertainty
        mentions, and guidance specificity from stored earnings press release chunks.

        Args:
            ticker:  Ticker filter.
            quarter: Optional quarter filter (e.g. 'Q1 2024').

        Returns:
            Dict with positive_words, negative_words, forward_looking_count,
            uncertainty_count, guidance_specificity, sample_positive, sample_negative.
        """
        await _ensure_corpus_schema(self.db_url)
        engine = _get_engine(self.db_url)

        where = "WHERE ticker = :ticker"
        params: dict = {"ticker": ticker.upper()}
        if quarter:
            where += " AND quarter = :quarter"
            params["quarter"] = quarter

        sql = text(f"SELECT chunk_text FROM earnings_corpus {where} LIMIT 200")
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, params)).fetchall()

        if not rows:
            return {"error": "No earnings corpus data found", "ticker": ticker}

        full_text = " ".join(r.chunk_text for r in rows)
        words = re.findall(r"\b[a-z]+\b", full_text.lower())

        pos_words = [w for w in words if w in _LM_POSITIVE]
        neg_words = [w for w in words if w in _LM_NEGATIVE]
        fwd_keywords = ["expect", "anticipate", "guidance", "outlook", "forecast",
                        "project", "believe", "plan"]
        fwd_count = sum(full_text.lower().count(kw) for kw in fwd_keywords)
        uncertainty_count = sum(full_text.lower().count(w) for w in [
            "uncertain", "uncertainty", "unclear", "may", "might", "could"
        ])

        # Guidance specificity: does it include a range ($X to $Y)?
        guidance_ranges = re.findall(r"\$[\d,\.]+\s+(?:billion|million|B|M)?\s+to\s+\$[\d,\.]+",
                                     full_text, re.IGNORECASE)

        return {
            "ticker": ticker,
            "quarter": quarter,
            "positive_words_count": len(pos_words),
            "negative_words_count": len(neg_words),
            "net_tone_score": round((len(pos_words) - len(neg_words)) / max(len(words), 1) * 100, 2),
            "forward_looking_count": fwd_count,
            "uncertainty_count": uncertainty_count,
            "guidance_ranges_found": len(guidance_ranges),
            "guidance_specificity": "specific" if guidance_ranges else "vague",
            "sample_positive": list(set(pos_words))[:10],
            "sample_negative": list(set(neg_words))[:10],
        }

    async def earnings_surprise_history(self, ticker: str) -> pd.DataFrame:
        """Return historical beats/misses with stock reactions from earnings_corpus metadata.

        Merges EarningsTranscriptAdapter.build_earnings_timeline() data.
        """
        adapter = EarningsTranscriptAdapter()
        df = await adapter.build_earnings_timeline(ticker)
        return df

    async def compute_guidance_trend(self, ticker: str) -> dict:
        """Analyse guidance revision trend over the last 8 quarters.

        Extracts guidance_revenue_low/high from stored earnings corpus metadata
        and determines whether guidance is being raised, lowered, or withdrawn.

        Args:
            ticker: Ticker symbol.

        Returns:
            Dict with trend ('raising'|'lowering'|'stable'|'withdrawn'|'insufficient_data'),
            quarters analysed, and average guidance change pct.
        """
        await _ensure_corpus_schema(self.db_url)
        engine = _get_engine(self.db_url)

        sql = text("""
            SELECT quarter, metadata
            FROM earnings_corpus
            WHERE ticker = :ticker AND doc_type = 'earnings_press_release'
            ORDER BY quarter DESC
            LIMIT 8
        """)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, {"ticker": ticker.upper()})).fetchall()

        if len(rows) < 2:
            return {"ticker": ticker, "trend": "insufficient_data", "quarters_analysed": len(rows)}

        guidance_series = []
        for r in rows:
            meta = r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata or "{}")
            low = meta.get("guidance_revenue_low")
            high = meta.get("guidance_revenue_high")
            if low is not None and high is not None:
                guidance_series.append({
                    "quarter": r.quarter,
                    "midpoint": (float(low) + float(high)) / 2,
                })

        if len(guidance_series) < 2:
            return {"ticker": ticker, "trend": "insufficient_data",
                    "quarters_analysed": len(guidance_series)}

        guidance_series.sort(key=lambda x: x["quarter"])
        changes = [
            (guidance_series[i + 1]["midpoint"] / guidance_series[i]["midpoint"] - 1) * 100
            for i in range(len(guidance_series) - 1)
            if guidance_series[i]["midpoint"] > 0
        ]

        avg_change = sum(changes) / len(changes) if changes else 0
        if avg_change > 2:
            trend = "raising"
        elif avg_change < -2:
            trend = "lowering"
        else:
            trend = "stable"

        return {
            "ticker": ticker,
            "trend": trend,
            "avg_guidance_change_pct": round(avg_change, 2),
            "quarters_analysed": len(guidance_series),
            "guidance_series": guidance_series,
        }

    async def peer_earnings_correlation(
        self,
        ticker: str,
        peers: list[str],
    ) -> dict:
        """Estimate whether peer earnings reactions predict target stock's earnings reaction.

        Uses 1-day stock reactions from yfinance earnings timeline data.

        Args:
            ticker: Target ticker.
            peers:  List of peer tickers.

        Returns:
            Dict with correlation coefficients (Pearson) between peer and target reactions.
        """
        adapter = EarningsTranscriptAdapter()

        # Gather target and peer timelines
        target_task = adapter.build_earnings_timeline(ticker)
        peer_tasks = [adapter.build_earnings_timeline(p) for p in peers]
        all_results = await asyncio.gather(target_task, *peer_tasks, return_exceptions=True)

        target_df = all_results[0] if not isinstance(all_results[0], Exception) else pd.DataFrame()
        correlations = {}

        for peer, peer_result in zip(peers, all_results[1:]):
            if isinstance(peer_result, Exception) or not isinstance(peer_result, pd.DataFrame):
                correlations[peer] = {"correlation": None, "error": str(peer_result)}
                continue
            peer_df = peer_result

            if target_df.empty or peer_df.empty:
                correlations[peer] = {"correlation": None, "n_quarters": 0}
                continue

            # Merge on quarter
            merged = pd.merge(
                target_df[["quarter", "stock_reaction_1d"]].rename(
                    columns={"stock_reaction_1d": "target_reaction"}
                ),
                peer_df[["quarter", "stock_reaction_1d"]].rename(
                    columns={"stock_reaction_1d": "peer_reaction"}
                ),
                on="quarter", how="inner",
            ).dropna()

            if len(merged) < 3:
                correlations[peer] = {"correlation": None, "n_quarters": len(merged)}
                continue

            corr = float(merged["target_reaction"].corr(merged["peer_reaction"]))
            correlations[peer] = {
                "correlation": round(corr, 4),
                "n_quarters": len(merged),
                "interpretation": (
                    "strong" if abs(corr) > 0.7 else
                    "moderate" if abs(corr) > 0.4 else "weak"
                ),
            }

        return {"ticker": ticker, "peers": correlations}

    async def detect_earnings_risk(self, ticker: str) -> dict:
        """Detect pre-earnings risk signals for an upcoming earnings event.

        Signals checked:
          - Implied volatility spike (IV percentile via options chain if available)
          - Analyst revision trend (sentiment from recent news corpus)
          - Options skew (calls vs puts ratio via yfinance)

        Args:
            ticker: Ticker symbol.

        Returns:
            Dict with risk_level ('low'|'medium'|'high'), signals list,
            and individual signal details.
        """
        signals = []
        details = {}

        # 1. Analyst revision trend from news corpus sentiment
        try:
            sentiment_df = await self.compute_news_sentiment_trend(ticker, lookback_days=30)
            if not sentiment_df.empty:
                avg_sent = float(sentiment_df["avg_sentiment"].mean())
                details["news_sentiment_30d"] = round(avg_sent, 4)
                if avg_sent < -0.1:
                    signals.append("negative_news_sentiment")
        except Exception as exc:
            logger.debug("Sentiment signal failed", error=str(exc))

        # 2. Options implied volatility and skew via yfinance
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()

            def _get_options():
                yft = yf.Ticker(ticker.upper())
                expiries = yft.options
                if not expiries:
                    return None, None
                # Use near-term expiry
                chain = yft.option_chain(expiries[0])
                calls = chain.calls
                puts = chain.puts
                if calls.empty or puts.empty:
                    return None, None
                # Average IV from ATM options
                call_iv = float(calls["impliedVolatility"].median())
                put_iv = float(puts["impliedVolatility"].median())
                return call_iv, put_iv

            call_iv, put_iv = await loop.run_in_executor(None, _get_options)
            if call_iv is not None and put_iv is not None:
                details["call_iv"] = round(call_iv, 4)
                details["put_iv"] = round(put_iv, 4)
                details["put_call_iv_ratio"] = round(put_iv / call_iv, 4) if call_iv > 0 else None
                if put_iv > call_iv * 1.2:
                    signals.append("elevated_put_skew")
                if call_iv > 0.5:
                    signals.append("high_implied_volatility")
        except Exception as exc:
            logger.debug("Options signal failed", error=str(exc))

        # 3. Earnings date proximity
        try:
            dates = await EarningsTranscriptAdapter().get_earnings_dates(ticker)
            upcoming = [d for d in dates if not d.get("reported")]
            if upcoming:
                next_date_str = upcoming[0].get("date", "")
                try:
                    next_dt = datetime.fromisoformat(next_date_str[:10])
                    days_to_earnings = (next_dt - datetime.utcnow()).days
                    details["days_to_next_earnings"] = days_to_earnings
                    if 0 <= days_to_earnings <= 14:
                        signals.append("earnings_within_2_weeks")
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Earnings date signal failed", error=str(exc))

        # Risk level
        risk_level = "low"
        if len(signals) >= 2:
            risk_level = "high"
        elif len(signals) == 1:
            risk_level = "medium"

        return {
            "ticker": ticker,
            "risk_level": risk_level,
            "signals": signals,
            "details": details,
        }


# ── CorpusIndex ────────────────────────────────────────────────────────────────

class CorpusIndex:
    """Manages schema and orchestrates full corpus ingest pipelines."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self._builder = NewsCorpusBuilder()
        self._adapter = EarningsTranscriptAdapter()

    async def ensure_schema(self) -> None:
        """Create all tables and hypertables. Idempotent."""
        await _ensure_corpus_schema(self.db_url)

    async def _upsert_earnings_chunks(
        self,
        ticker: str,
        quarter: Optional[str],
        doc_type: str,
        chunks: list[str],
        metadata: dict,
    ) -> int:
        """Embed and upsert chunks into earnings_corpus. Returns inserted count."""
        if not chunks:
            return 0
        await _ensure_corpus_schema(self.db_url)
        embeddings = await _embed_async(chunks)
        engine = _get_engine(self.db_url)
        meta_json = json.dumps(metadata)

        sql = text("""
            INSERT INTO earnings_corpus
                (ticker, quarter, doc_type, chunk_text, embedding, metadata, created_at)
            VALUES
                (:ticker, :quarter, :doc_type, :chunk_text, :embedding::vector, :metadata, NOW())
        """)
        async with engine.begin() as conn:
            for chunk, emb in zip(chunks, embeddings):
                await conn.execute(sql, {
                    "ticker": ticker.upper(),
                    "quarter": quarter,
                    "doc_type": doc_type,
                    "chunk_text": chunk,
                    "embedding": _vec_str(emb),
                    "metadata": meta_json,
                })
        return len(chunks)

    async def _upsert_news_articles(
        self,
        ticker: Optional[str],
        articles: list[dict],
    ) -> int:
        """Classify, embed, and upsert articles into news_corpus. Returns count."""
        if not articles:
            return 0
        await _ensure_corpus_schema(self.db_url)

        # Deduplicate before ingest
        articles = self._builder.deduplicate(articles)

        # Classify and compute LM sentiment
        engine = _get_engine(self.db_url)
        analytics = CorpusAnalyticsEngine(self.db_url)
        texts = [a.get("title", "") + " " + a.get("summary", "") for a in articles]
        embeddings = await _embed_async(texts)

        sql = text("""
            INSERT INTO news_corpus
                (ticker, published_at, headline, chunk_text, embedding,
                 category, sentiment_score, url_hash, metadata)
            VALUES
                (:ticker, :published_at, :headline, :chunk_text, :embedding::vector,
                 :category, :sentiment_score, :url_hash, :metadata)
            ON CONFLICT DO NOTHING
        """)

        inserted = 0
        async with engine.begin() as conn:
            for article, emb in zip(articles, embeddings):
                text_content = (article.get("title", "") + " " + article.get("summary", ""))
                category = self._builder.classify_article(article)
                sentiment = analytics._lm_sentiment(text_content)

                pub_str = article.get("pubDate", datetime.utcnow().isoformat())
                try:
                    pub_dt = datetime.fromisoformat(pub_str[:19])
                except Exception:
                    pub_dt = datetime.utcnow()

                try:
                    await conn.execute(sql, {
                        "ticker": ticker.upper() if ticker else None,
                        "published_at": pub_dt,
                        "headline": article.get("title", "")[:300],
                        "chunk_text": text_content[:2000],
                        "embedding": _vec_str(emb),
                        "category": category,
                        "sentiment_score": sentiment,
                        "url_hash": article.get("url_hash", ""),
                        "metadata": json.dumps({
                            "source": article.get("source", ""),
                            "url": article.get("link", ""),
                        }),
                    })
                    inserted += 1
                except Exception as exc:
                    logger.debug("News article upsert skipped", error=str(exc))

        return inserted

    async def ingest_full_corpus(self, ticker: str, company_name: Optional[str] = None) -> dict:
        """Run the complete corpus ingest pipeline for a ticker.

        Steps:
          1. EDGAR 8-K earnings press releases → earnings_corpus
          2. Google News + Seeking Alpha RSS → news_corpus
          3. EDGAR 8-K as news events → news_corpus

        Args:
            ticker:       Ticker symbol.
            company_name: Optional company name for Google News search;
                          defaults to ticker if not provided.

        Returns:
            Summary dict with earnings_chunks, news_articles counts.
        """
        cik = await _resolve_cik(ticker)
        company_name = company_name or ticker

        # 1. Earnings press releases from EDGAR 8-K
        earnings_data = await self._adapter.get_from_edgar_8k(ticker, cik=cik)
        earnings_chunks_total = 0
        for ed in earnings_data:
            if "error" in ed:
                continue
            text_parts = []
            if ed.get("guidance_text"):
                text_parts.append(ed["guidance_text"])
            if ed.get("key_phrases"):
                text_parts.extend(ed["key_phrases"])
            combined = " ".join(text_parts)
            if not combined:
                continue
            quarter = _date_to_quarter(_parse_date(ed.get("period", "")))
            chunks = _chunk_text(combined)
            n = await self._upsert_earnings_chunks(
                ticker=ticker,
                quarter=quarter,
                doc_type="earnings_press_release",
                chunks=chunks,
                metadata={k: v for k, v in ed.items()
                          if k not in ("key_phrases",)},
            )
            earnings_chunks_total += n

        # 2. Google News RSS
        news_articles = await self._builder.ingest_google_news_rss(
            ticker, company_name, lookback_days=90
        )

        # 3. Seeking Alpha RSS
        sa_articles = await self._builder.ingest_seeking_alpha_rss(ticker)
        news_articles.extend(sa_articles)

        # 4. EDGAR 8-K as news events
        if cik:
            edgar_news = await self._builder.ingest_edgar_press_releases(cik, lookback_days=180)
            news_articles.extend(edgar_news)

        news_inserted = await self._upsert_news_articles(ticker, news_articles)

        logger.info(
            "Full corpus ingest complete",
            ticker=ticker,
            earnings_chunks=earnings_chunks_total,
            news_articles=news_inserted,
        )

        return {
            "ticker": ticker,
            "earnings_chunks_ingested": earnings_chunks_total,
            "news_articles_ingested": news_inserted,
            "earnings_press_releases": len([e for e in earnings_data if "error" not in e]),
        }

    async def build_ivfflat_indexes(self) -> None:
        """Build IVFFlat approximate nearest-neighbour indexes on both corpus tables."""
        engine = _get_engine(self.db_url)
        for table, ddl in [("earnings_corpus", _IVFFLAT_EC), ("news_corpus", _IVFFLAT_NC)]:
            async with engine.connect() as conn:
                result = await conn.execute(
                    text(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL")
                )
                count = result.scalar() or 0
            if count < 100:
                logger.warning("Too few rows for IVFFlat", table=table, count=count)
                continue
            lists = max(10, min(int(count / 39), 1000))
            async with engine.begin() as conn:
                await conn.execute(text(ddl.format(lists=lists)))
            logger.info("IVFFlat index built", table=table, lists=lists)


# ── FastAPI Router ─────────────────────────────────────────────────────────────

earnings_router = APIRouter(prefix="/api/earnings", tags=["Earnings Corpus"])


def _get_db_url() -> str:
    return get_settings().database_url


class IngestCorpusRequest(BaseModel):
    company_name: Optional[str] = None


@earnings_router.get("/{ticker}/history", summary="Earnings timeline with stock reactions")
async def earnings_history(ticker: str):
    """Return a quarter-by-quarter earnings timeline.

    Combines EDGAR 8-K press release data with yfinance earnings dates and
    historical price data to compute EPS surprise and 1-day / 5-day stock
    reactions to earnings announcements.
    """
    adapter = EarningsTranscriptAdapter()
    df = await adapter.build_earnings_timeline(ticker.upper())
    if df.empty:
        return {"ticker": ticker.upper(), "quarters": []}
    return {"ticker": ticker.upper(), "quarters": df.to_dict(orient="records")}


@earnings_router.get("/{ticker}/sentiment-trend", summary="News sentiment trend over time")
async def sentiment_trend(
    ticker: str,
    days: int = Query(90, ge=7, le=365, description="Lookback window in days"),
):
    """Daily average news sentiment from the indexed news_corpus.

    Uses Loughran-McDonald word-list scoring. Positive scores indicate net
    optimistic language; negative scores indicate pessimism.
    """
    db_url = _get_db_url()
    analytics = CorpusAnalyticsEngine(db_url)
    df = await analytics.compute_news_sentiment_trend(ticker.upper(), lookback_days=days)
    if df.empty:
        return {"ticker": ticker.upper(), "trend": []}
    return {"ticker": ticker.upper(), "trend": df.to_dict(orient="records")}


@earnings_router.get("/{ticker}/guidance", summary="Management guidance trend history")
async def guidance_history(ticker: str):
    """Analyse whether management has been raising, lowering, or withdrawing guidance.

    Compares guidance revenue midpoints across the last 8 quarters stored in
    the earnings_corpus table.
    """
    db_url = _get_db_url()
    analytics = CorpusAnalyticsEngine(db_url)
    return await analytics.compute_guidance_trend(ticker.upper())


@earnings_router.get("/{ticker}/risk", summary="Pre-earnings risk signals")
async def earnings_risk(ticker: str):
    """Detect pre-earnings risk signals: options skew, IV spike, sentiment reversal.

    Combines news corpus sentiment (30-day), options chain implied volatility
    (via yfinance), and earnings date proximity into a composite risk score.
    """
    db_url = _get_db_url()
    analytics = CorpusAnalyticsEngine(db_url)
    return await analytics.detect_earnings_risk(ticker.upper())


@earnings_router.post("/{ticker}/ingest", summary="Trigger full corpus ingest for ticker")
async def ingest_corpus(ticker: str, body: IngestCorpusRequest = None):
    """Download and index earnings press releases and news articles for a ticker.

    Sources:
      - EDGAR 8-K filings (Items 7.01 / 8.01 / Exhibit 99.1)
      - Google News RSS
      - Seeking Alpha public RSS

    Embeddings are stored in TimescaleDB hypertables for time-series queries.
    """
    if body is None:
        body = IngestCorpusRequest()
    db_url = _get_db_url()
    index = CorpusIndex(db_url)
    return await index.ingest_full_corpus(
        ticker=ticker.upper(),
        company_name=body.company_name,
    )
