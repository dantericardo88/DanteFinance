"""
SENTINEL Research Synthesis — Intelligence Layer (SIL).

Multi-source AI research synthesis using Claude. Targets dimension 89 of the
110-dimension competitive matrix (AI research synthesis: ~3/10 → target 8+/10).
"""
from __future__ import annotations

import asyncio
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 15.0
_DEFAULT_MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
_FINNHUB_BASE = "https://finnhub.io/api/v1"
_FINNHUB_DEMO_TOKEN = os.environ.get("FINNHUB_API_KEY", "d")

_POSITIVE_WORDS = frozenset({
    "beat", "beats", "exceeded", "raised", "growth", "grew", "record",
    "strong", "momentum", "upgrade", "outperform", "buy", "positive",
    "profit", "expansion", "gained", "surpassed", "upbeat", "optimistic",
    "accelerate", "boost", "recovery", "breakthrough", "launch",
})
_NEGATIVE_WORDS = frozenset({
    "miss", "missed", "decline", "fell", "concern", "concerns", "weak",
    "warning", "downgrade", "underperform", "sell", "negative", "loss",
    "contraction", "investigation", "lawsuit", "fine", "penalty", "downside",
    "risk", "volatile", "uncertainty", "cut", "layoff", "restatement",
    "fraud", "recall", "shortage",
})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResearchSource(BaseModel):
    source_type: str      # "filing" | "news" | "fundamental" | "insider" | "market"
    title: str
    summary: str
    sentiment: str        # "positive" | "negative" | "neutral"
    relevance: float      # 0-1


class ResearchSynthesis(BaseModel):
    ticker: str
    query: str | None
    synthesis: str           # markdown — the Claude-generated synthesis paragraph(s)
    bull_case: str           # 2-3 bullet points
    bear_case: str           # 2-3 bullet points
    key_risks: list[str]
    catalyst_watch: list[str]
    sources_used: list[ResearchSource]
    confidence: str          # "high" | "medium" | "low"
    as_of: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_sentiment(text: str) -> str:
    """Keyword-score sentiment: positive / negative / neutral."""
    if not text:
        return "neutral"
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    return "positive" if pos > neg else ("negative" if neg > pos else "neutral")


def _relevance(text: str, ticker: str) -> float:
    """0-1 relevance based on ticker mention frequency."""
    if not text:
        return 0.3
    count = text.lower().count(ticker.lower())
    return round(min(count * 0.2 + 0.2, 1.0), 2)


def _extract_section(text: str, name: str) -> str:
    m = re.search(rf"###\s*{re.escape(name)}\s*\n(.*?)(?=###|\Z)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_bullets(text: str) -> list[str]:
    return [l.lstrip("- ").strip() for l in text.splitlines() if l.strip().startswith("-") and l.strip() != "-"]


# ---------------------------------------------------------------------------
# Data gathering — EDGAR filings
# ---------------------------------------------------------------------------

async def _gather_filings(ticker: str) -> list[ResearchSource]:
    from sentinel.sil.edgar_search import search_edgar
    response = await search_edgar(
        query=ticker, form_types=["10-K", "10-Q", "8-K"],
        days_back=180, ticker=ticker, limit=5,
    )
    sources: list[ResearchSource] = []
    for r in response.results:
        excerpt = " ".join(r.relevant_excerpts[:2]) if r.relevant_excerpts else ""
        summary = f"{r.description or r.form_type} | {excerpt[:300]}".strip(" |")
        if not summary:
            summary = f"{r.form_type} filed {r.filed_date.isoformat()}"
        title = f"{r.form_type} — {r.company_name or ticker} ({r.filed_date.isoformat()})"
        sources.append(ResearchSource(
            source_type="filing", title=title, summary=summary[:500],
            sentiment=_classify_sentiment(summary), relevance=_relevance(summary, ticker),
        ))
    return sources


# ---------------------------------------------------------------------------
# Data gathering — News
# ---------------------------------------------------------------------------

async def _gather_news(ticker: str, days_back: int = 14) -> list[ResearchSource]:
    """Try news_rag_bridge first; fall back to Finnhub free-tier."""
    sources: list[ResearchSource] = []

    # Attempt 1: news_rag_bridge
    try:
        from sentinel.sds.adapters.news_rag_bridge import get_recent_news  # type: ignore
        items = await get_recent_news(ticker, days_back=days_back) or []
        for item in items[:8]:
            headline = item.get("headline", item.get("title", ""))
            summary = item.get("summary", item.get("description", headline))
            sources.append(ResearchSource(
                source_type="news", title=headline[:200], summary=summary[:400],
                sentiment=_classify_sentiment(f"{headline} {summary}"),
                relevance=_relevance(f"{headline} {summary}", ticker),
            ))
        if sources:
            return sources
    except ImportError:
        logger.debug("research_synthesis: news_rag_bridge not available")
    except Exception as exc:
        logger.warning("research_synthesis: news_rag_bridge error: %s", exc)

    # Attempt 2: Finnhub demo
    to_date = date.today()
    from_date = to_date - timedelta(days=days_back)
    url = (
        f"{_FINNHUB_BASE}/company-news?symbol={ticker}"
        f"&from={from_date.isoformat()}&to={to_date.isoformat()}"
        f"&token={_FINNHUB_DEMO_TOKEN}"
    )
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        articles = resp.json() if isinstance(resp.json(), list) else []

    for article in articles[:8]:
        headline = article.get("headline", "")
        summary = article.get("summary", headline)
        sources.append(ResearchSource(
            source_type="news", title=headline[:200], summary=summary[:400],
            sentiment=_classify_sentiment(f"{headline} {summary}"),
            relevance=_relevance(f"{headline} {summary}", ticker),
        ))
    return sources


# ---------------------------------------------------------------------------
# Data gathering — Fundamentals
# ---------------------------------------------------------------------------

def _fundamental_sentiment(metric: Any) -> str:
    pos, neg = 0, 0
    for attr, thresh in [("revenue_growth_yoy", 0.05), ("eps_growth_yoy", 0.05)]:
        v = getattr(metric, attr, None)
        if v is not None:
            pos += (v > thresh); neg += (v < -thresh)
    op = getattr(metric, "operating_margin", None)
    if op is not None:
        pos += (op > 0.10); neg += (op < 0)
    return "positive" if pos > neg else ("negative" if neg > pos else "neutral")


async def _gather_fundamentals(ticker: str) -> list[ResearchSource]:
    from sentinel.sfe.comps_table import get_comps_table
    table = await get_comps_table(ticker)
    sources: list[ResearchSource] = []
    if not table.peers:
        return sources

    target = next((p for p in table.peers if p.ticker.upper() == ticker.upper()), table.peers[0])
    parts: list[str] = []
    for attr, label, scale, fmt in [
        ("pe_ratio", "P/E", 1, "{:.1f}x"),
        ("ev_ebitda", "EV/EBITDA", 1, "{:.1f}x"),
        ("price_to_sales", "P/S", 1, "{:.1f}x"),
        ("revenue_growth_yoy", "Rev growth YoY", 100, "{:.1f}%"),
        ("gross_margin", "Gross margin", 100, "{:.1f}%"),
        ("operating_margin", "Op margin", 100, "{:.1f}%"),
        ("net_margin", "Net margin", 100, "{:.1f}%"),
    ]:
        v = getattr(target, attr, None)
        if v is not None:
            parts.append(f"{label}: {fmt.format(v * scale)}")
    if (mc := getattr(target, "market_cap_usd", None)) is not None:
        parts.append(f"Market cap: ${mc/1e9:.1f}B")

    summary = "; ".join(parts) if parts else "Fundamentals data available"
    sources.append(ResearchSource(
        source_type="fundamental",
        title=f"Fundamentals — {target.company_name or ticker} ({ticker})",
        summary=summary, sentiment=_fundamental_sentiment(target), relevance=0.9,
    ))

    # Peer valuation comparison
    peers = [p for p in table.peers if p.ticker.upper() != ticker.upper()]
    if peers and getattr(target, "pe_ratio", None):
        peer_pes = [p.pe_ratio for p in peers if p.pe_ratio]
        if peer_pes:
            avg_pe = sum(peer_pes) / len(peer_pes)
            premium = (target.pe_ratio - avg_pe) / avg_pe * 100 if avg_pe else 0
            direction = "premium" if premium > 0 else "discount"
            sources.append(ResearchSource(
                source_type="fundamental",
                title=f"Peer Comparison — {ticker} vs sector",
                summary=(
                    f"Trading at {abs(premium):.0f}% {direction} to peer average P/E "
                    f"of {avg_pe:.1f}x. Peers: {', '.join(p.ticker for p in peers[:4])}"
                ),
                sentiment="neutral", relevance=0.75,
            ))
    return sources


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    ticker: str,
    query: str | None,
    filing_sources: list[ResearchSource],
    news_sources: list[ResearchSource],
    fundamental_sources: list[ResearchSource],
) -> str:
    lines: list[str] = [
        f"You are a senior equity research analyst. Synthesize all available data on "
        f"**{ticker}** into a concise, actionable research memo.",
    ]
    if query:
        lines.append(f"\n**Focus question:** {query}")
    if fundamental_sources:
        lines += ["\n## Fundamental Data"] + [f"- **{s.title}**: {s.summary}" for s in fundamental_sources]
    if filing_sources:
        lines += ["\n## Recent SEC Filings"] + [f"- **{s.title}**: {s.summary}" for s in filing_sources]
    if news_sources:
        lines += ["\n## Recent News"] + [
            f"- [{s.sentiment.upper()}] {s.title}: {s.summary}" for s in news_sources[:6]
        ]
    if not (fundamental_sources or filing_sources or news_sources):
        lines.append(f"\nNo live data was retrieved for {ticker}. Provide a framework-based analysis.")

    lines.append("""
## Required Output Format

Respond ONLY in this exact format (use section headers verbatim):

### SYNTHESIS
[2-4 paragraphs of integrated research synthesis in markdown]

### BULL CASE
- [bullet 1]
- [bullet 2]
- [optional bullet 3]

### BEAR CASE
- [bullet 1]
- [bullet 2]
- [optional bullet 3]

### KEY RISKS
- [risk 1]
- [risk 2]
- [risk 3]

### CATALYST WATCH
- [catalyst 1]
- [catalyst 2]
- [catalyst 3]

### CONFIDENCE
[high | medium | low] — [one-sentence rationale]
""")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_claude_response(text: str) -> dict:
    bull_raw = _extract_section(text, "BULL CASE")
    bear_raw = _extract_section(text, "BEAR CASE")
    risks_raw = _extract_section(text, "KEY RISKS")
    catalysts_raw = _extract_section(text, "CATALYST WATCH")
    confidence_raw = _extract_section(text, "CONFIDENCE").lower()

    def _fmt_bullets(raw: str) -> str:
        bullets = _parse_bullets(raw)
        return "\n".join(f"- {b}" for b in bullets) if bullets else raw

    confidence = "high" if "high" in confidence_raw else ("low" if "low" in confidence_raw else "medium")

    return {
        "synthesis": _extract_section(text, "SYNTHESIS") or text,
        "bull_case": _fmt_bullets(bull_raw),
        "bear_case": _fmt_bullets(bear_raw),
        "key_risks": _parse_bullets(risks_raw) or ["Data insufficient for risk assessment"],
        "catalyst_watch": _parse_bullets(catalysts_raw) or ["Monitor upcoming earnings"],
        "confidence": confidence,
    }


def _fallback_synthesis(ticker: str, query: str | None, sources: list[ResearchSource]) -> dict:
    pos = [s for s in sources if s.sentiment == "positive"]
    neg = [s for s in sources if s.sentiment == "negative"]
    lines = [f"## Research Summary — {ticker}"]
    if query:
        lines.append(f"\n*Focus: {query}*")
    lines.append(
        f"\nAnalysis based on {len(sources)} data points "
        f"({len(pos)} positive, {len(neg)} negative signals)."
    )
    for s in sources[:5]:
        lines.append(f"\n**{s.title}**: {s.summary}")
    return {
        "synthesis": "\n".join(lines),
        "bull_case": "\n".join(f"- {s.title}" for s in pos[:3]) or "- Insufficient data",
        "bear_case": "\n".join(f"- {s.title}" for s in neg[:3]) or "- Insufficient data",
        "key_risks": ["AI synthesis unavailable — manual review required"],
        "catalyst_watch": ["Monitor upcoming earnings and filings"],
        "confidence": "low",
    }


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

async def _run_claude_synthesis(prompt: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=_DEFAULT_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.content and hasattr(response.content[0], "text"):
        return response.content[0].text
    return ""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def synthesize_research(
    ticker: str,
    query: str | None = None,
    include_filings: bool = True,
    include_news: bool = True,
    include_fundamentals: bool = True,
    include_insider: bool = True,
    max_tokens: int = 2048,
) -> ResearchSynthesis:
    """
    Multi-source AI research synthesis for a ticker.

    Gathers EDGAR filings, recent news, and fundamental metrics in parallel,
    then synthesises a structured investment memo via Claude. Degrades
    gracefully if any source or Claude itself fails.

    Args:
        ticker:               Equity ticker symbol (e.g. "AAPL").
        query:                Optional focus question to steer the synthesis.
        include_filings:      Include SEC filing data.
        include_news:         Include news data.
        include_fundamentals: Include fundamental/valuation data.
        include_insider:      Reserved for future insider trade data inclusion.
        max_tokens:           Max tokens for Claude response.

    Returns:
        ResearchSynthesis with synthesis, bull/bear cases, risks, and catalysts.
    """
    ticker = ticker.upper().strip()
    warnings: list[str] = []
    as_of = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Phase 1: gather all sources in parallel
    tasks: list[Any] = []
    labels: list[str] = []
    if include_filings:
        tasks.append(_gather_filings(ticker)); labels.append("filings")
    if include_news:
        tasks.append(_gather_news(ticker)); labels.append("news")
    if include_fundamentals:
        tasks.append(_gather_fundamentals(ticker)); labels.append("fundamentals")

    results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

    filing_sources: list[ResearchSource] = []
    news_sources: list[ResearchSource] = []
    fundamental_sources: list[ResearchSource] = []
    bucket_map = {"filings": filing_sources, "news": news_sources, "fundamentals": fundamental_sources}

    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            warnings.append(f"{label} data unavailable: {result}")
            logger.warning("research_synthesis: %s gather error: %s", label, result)
        elif isinstance(result, list):
            bucket_map[label].extend(result)

    all_sources = filing_sources + news_sources + fundamental_sources

    # Phase 2: Claude synthesis
    parsed: dict
    try:
        prompt = _build_prompt(ticker, query, filing_sources, news_sources, fundamental_sources)
        raw = await _run_claude_synthesis(prompt, max_tokens)
        parsed = _parse_claude_response(raw) if raw else _fallback_synthesis(ticker, query, all_sources)
        if not raw:
            warnings.append("Claude returned empty response — using fallback synthesis")
    except Exception as exc:
        warnings.append(f"Claude synthesis failed ({exc}) — using fallback synthesis")
        parsed = _fallback_synthesis(ticker, query, all_sources)

    # Phase 3: resolve confidence
    claude_failed = any("Claude" in w for w in warnings)
    if claude_failed:
        confidence = "low"
    else:
        confidence = parsed.get("confidence", "medium")
        if len(all_sources) < 3 and confidence == "high":
            confidence = "medium"

    return ResearchSynthesis(
        ticker=ticker,
        query=query,
        synthesis=parsed.get("synthesis", ""),
        bull_case=parsed.get("bull_case", ""),
        bear_case=parsed.get("bear_case", ""),
        key_risks=parsed.get("key_risks", []),
        catalyst_watch=parsed.get("catalyst_watch", []),
        sources_used=all_sources,
        confidence=confidence,
        as_of=as_of,
        warnings=warnings,
    )
