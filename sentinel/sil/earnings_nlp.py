"""
SENTINEL Earnings NLP — Intelligence Layer (SIL).

Analyzes earnings-related SEC 8-K filings using EDGAR + Claude Haiku.
Extracts management tone, forward guidance, key themes, and sentiment trend.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
_HEADERS = {"User-Agent": "SENTINEL/1.0 research@sentinel.ai", "Accept-Encoding": "gzip, deflate"}
_TIMEOUT = 20.0
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TEXT_CHARS = 8_000
_MIN_TEXT_CHARS = 200


class EarningsFilingAnalysis(BaseModel):
    ticker: str
    file_date: str
    period_of_report: Optional[str] = None
    filing_url: str
    tone: str               # "confident" | "cautious" | "negative" | "mixed" | "unknown"
    guidance_signal: str    # "raised" | "lowered" | "maintained" | "none" | "unknown"
    revenue_signal: str     # "beat" | "miss" | "in-line" | "unknown"
    profit_signal: str      # "improving" | "declining" | "stable" | "unknown"
    key_themes: list[str]
    risks: list[str]
    catalysts: list[str]
    sentiment_score: float  # -5 to +5
    text_excerpt: str       # first 500 chars of cleaned filing text
    warnings: list[str] = Field(default_factory=list)


class EarningsTrend(BaseModel):
    ticker: str
    analyses: list[EarningsFilingAnalysis]
    sentiment_trend: str    # "improving" | "deteriorating" | "stable"
    avg_sentiment: float
    latest_tone: str
    latest_guidance: str
    guidance_changes: list[str]  # e.g. ["Q2 2024: raised", "Q1 2024: maintained"]
    as_of: str
    warnings: list[str] = Field(default_factory=list)


async def _resolve_cik(ticker: str, client: httpx.AsyncClient) -> str:
    """Resolve ticker to zero-padded 10-digit CIK via EFTS search."""
    url = (
        f"{_EFTS_BASE}?q={quote(chr(34) + ticker + chr(34))}"
        f"&forms=10-K&hits.hits._source=entity_id,display_names"
    )
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        if hits:
            entity_id = hits[0].get("_source", {}).get("entity_id", "")
            if entity_id:
                return str(entity_id).zfill(10)
    except Exception as exc:
        logger.warning("earnings_nlp: CIK resolution failed for %s: %s", ticker, exc)
    raise ValueError(f"Could not resolve CIK for ticker '{ticker}'")


async def _find_8k_filings(
    ticker: str, cik: str, limit: int, client: httpx.AsyncClient,
) -> list[dict]:
    """Return list of 8-K filing metadata dicts, newest first."""
    cik_plain = cik.lstrip("0") or cik
    filings: list[dict] = []

    # Primary: submissions JSON from data.sec.gov
    try:
        resp = await client.get(
            f"{_SUBMISSIONS_BASE}/CIK{cik}.json", headers=_HEADERS, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        periods = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if form not in ("8-K", "8-K/A"):
                continue
            acc = accessions[i] if i < len(accessions) else ""
            acc_nodash = acc.replace("-", "")
            pdoc = primary_docs[i] if i < len(primary_docs) else ""
            filing_url = (
                f"{_ARCHIVE_BASE}/{cik_plain}/{acc_nodash}/{pdoc}"
                if pdoc else
                f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_plain}&type=8-K"
            )
            filings.append({
                "file_date": dates[i] if i < len(dates) else "",
                "accession_no": acc,
                "acc_nodash": acc_nodash,
                "primary_doc": pdoc,
                "period_of_report": periods[i] if i < len(periods) else None,
                "filing_url": filing_url,
                "cik_plain": cik_plain,
            })
            if len(filings) >= limit:
                break
    except Exception as exc:
        logger.warning("earnings_nlp: submissions fetch failed for CIK %s: %s", cik, exc)

    # Fallback: EFTS search
    if not filings:
        start_dt = (date.today() - timedelta(days=365)).isoformat()
        efts_url = (
            f"{_EFTS_BASE}?q={quote(chr(34) + ticker + chr(34))}"
            f"&forms=8-K&dateRange=custom&startdt={start_dt}&enddt={date.today().isoformat()}"
        )
        try:
            resp = await client.get(efts_url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            for hit in resp.json().get("hits", {}).get("hits", [])[:limit]:
                src = hit.get("_source", {})
                acc = src.get("accession_no", "")
                acc_nodash = acc.replace("-", "")
                file_name = src.get("file_name", "")
                cik_hit = str(src.get("cik", cik_plain)).lstrip("0") or cik_plain
                filings.append({
                    "file_date": src.get("file_date", "")[:10],
                    "accession_no": acc,
                    "acc_nodash": acc_nodash,
                    "primary_doc": file_name,
                    "period_of_report": src.get("period_of_report"),
                    "filing_url": f"{_ARCHIVE_BASE}/{cik_hit}/{acc_nodash}/{file_name}" if file_name else "",
                    "cik_plain": cik_hit,
                })
        except Exception as exc2:
            logger.warning("earnings_nlp: EFTS fallback failed for %s: %s", ticker, exc2)

    filings.sort(key=lambda f: f.get("file_date", ""), reverse=True)
    return filings[:limit]


def _strip_html(html_content: str) -> str:
    """Strip HTML tags and unescape entities using stdlib only."""
    import html as _html
    text = re.sub(r"<[^>]+>", " ", html_content)
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


async def _fetch_filing_text(filing: dict, client: httpx.AsyncClient) -> tuple[str, str]:
    """Fetch and clean the primary 8-K document. Returns (clean_text, url_used)."""
    cik_plain = filing.get("cik_plain", "")
    acc_nodash = filing.get("acc_nodash", "")
    urls_to_try = [filing.get("filing_url", "")]
    if cik_plain and acc_nodash:
        acc_dashes = filing.get("accession_no", acc_nodash)
        urls_to_try.append(
            f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/{acc_nodash}/{acc_dashes}-index.htm"
        )
    for url in filter(None, urls_to_try):
        try:
            resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
            if resp.status_code == 200:
                text = _strip_html(resp.text)
                if len(text) >= _MIN_TEXT_CHARS:
                    return text, url
        except Exception as exc:
            logger.debug("earnings_nlp: fetch attempt failed for %s: %s", url, exc)
    return "", filing.get("filing_url", "")


def _build_analysis_prompt(ticker: str, file_date: str, text: str) -> str:
    return f"""Analyze this earnings filing for {ticker}.

Filing date: {file_date}
Text:
{text[:_MAX_TEXT_CHARS]}

Extract:
1. TONE: overall management tone (confident/cautious/negative/mixed)
2. GUIDANCE: any forward guidance mentioned (raised/lowered/maintained/none)
3. KEY_THEMES: top 5 themes mentioned (comma-separated)
4. REVENUE_SIGNAL: revenue trend signal (beat/miss/in-line/not_mentioned)
5. PROFIT_SIGNAL: profit/margin signal (improving/declining/stable/not_mentioned)
6. RISKS: top 3 risks mentioned (comma-separated)
7. CATALYSTS: top 2 positive catalysts mentioned (comma-separated)
8. SENTIMENT_SCORE: numeric -5 to +5 (very negative to very positive)

Format each on its own line: FIELD: value"""


def _parse_haiku_response(raw: str) -> dict:
    result: dict = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().upper(), value.strip()
        if key and value:
            result[key] = value
    return result


def _coerce_analysis(parsed: dict, warnings: list[str]) -> dict:
    """Normalize raw Claude output to canonical field values."""
    def _csv(key: str, n: int) -> list[str]:
        return [x.strip() for x in parsed.get(key, "").split(",") if x.strip()][:n]

    tone_raw = parsed.get("TONE", "").lower()
    tone = next((t for t in ("confident", "cautious", "negative", "mixed") if t in tone_raw), "mixed")

    guidance_raw = parsed.get("GUIDANCE", "").lower()
    if "raise" in guidance_raw:
        guidance = "raised"
    elif any(k in guidance_raw for k in ("lower", "reduc", "cut")):
        guidance = "lowered"
    elif any(k in guidance_raw for k in ("maintain", "reaffirm", "reiterat")):
        guidance = "maintained"
    elif any(k in guidance_raw for k in ("no guidance", "none", "not")):
        guidance = "none"
    else:
        guidance = "unknown"

    rev_raw = parsed.get("REVENUE_SIGNAL", "").lower()
    revenue = "beat" if "beat" in rev_raw else "miss" if "miss" in rev_raw else \
              "in-line" if any(k in rev_raw for k in ("in-line", "inline", "in line")) else "unknown"

    profit_raw = parsed.get("PROFIT_SIGNAL", "").lower()
    profit = "improving" if "improv" in profit_raw else \
             "declining" if any(k in profit_raw for k in ("declin", "deteriorat")) else \
             "stable" if any(k in profit_raw for k in ("stable", "flat")) else "unknown"

    try:
        score_raw = parsed.get("SENTIMENT_SCORE", "0")
        score = float(re.search(r"-?\d+\.?\d*", score_raw).group())  # type: ignore[union-attr]
        score = max(-5.0, min(5.0, score))
    except Exception:
        score = 0.0
        warnings.append("Could not parse SENTIMENT_SCORE — defaulting to 0")

    return {
        "tone": tone,
        "guidance_signal": guidance,
        "revenue_signal": revenue,
        "profit_signal": profit,
        "key_themes": _csv("KEY_THEMES", 5),
        "risks": _csv("RISKS", 3),
        "catalysts": _csv("CATALYSTS", 2),
        "sentiment_score": score,
    }


async def _call_haiku(ticker: str, file_date: str, text: str) -> dict:
    """Lazy-import anthropic, call Haiku, return parsed structured dict."""
    import anthropic  # lazy — only imported when analysis is run
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=_DEFAULT_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": _build_analysis_prompt(ticker, file_date, text)}],
    )
    raw = response.content[0].text if response.content and hasattr(response.content[0], "text") else ""
    return _parse_haiku_response(raw)


def _compute_trend(scores: list[float]) -> str:
    """Compute sentiment direction from oldest→newest scores."""
    if len(scores) < 2:
        return "stable"
    mid = len(scores) // 2
    delta = sum(scores[mid:]) / len(scores[mid:]) - sum(scores[:mid]) / len(scores[:mid])
    return "improving" if delta > 0.5 else "deteriorating" if delta < -0.5 else "stable"


async def _analyze_single_filing(
    ticker: str, filing: dict, client: httpx.AsyncClient,
) -> EarningsFilingAnalysis:
    """Fetch text and call Claude for one 8-K; return EarningsFilingAnalysis."""
    warnings: list[str] = []
    file_date = filing.get("file_date", "")
    period = filing.get("period_of_report")

    text, final_url = await _fetch_filing_text(filing, client)
    if not text:
        warnings.append("Could not fetch filing text — analysis based on empty document")
    elif len(text) < _MIN_TEXT_CHARS:
        warnings.append(f"Filing text very short ({len(text)} chars) — analysis may be limited")

    coerced: dict = {}
    try:
        if text:
            parsed = await _call_haiku(ticker, file_date, text)
            coerced = _coerce_analysis(parsed, warnings)
        else:
            warnings.append("Skipped Claude analysis — no text available")
    except Exception as exc:
        warnings.append(f"Claude analysis failed: {exc}")
        logger.warning("earnings_nlp: Haiku call failed for %s %s: %s", ticker, file_date, exc)

    return EarningsFilingAnalysis(
        ticker=ticker,
        file_date=file_date,
        period_of_report=str(period) if period else None,
        filing_url=final_url or filing.get("filing_url", ""),
        tone=coerced.get("tone", "unknown"),
        guidance_signal=coerced.get("guidance_signal", "unknown"),
        revenue_signal=coerced.get("revenue_signal", "unknown"),
        profit_signal=coerced.get("profit_signal", "unknown"),
        key_themes=coerced.get("key_themes", []),
        risks=coerced.get("risks", []),
        catalysts=coerced.get("catalysts", []),
        sentiment_score=coerced.get("sentiment_score", 0.0),
        text_excerpt=text[:500],
        warnings=warnings,
    )


async def analyze_earnings_filing(
    ticker: str,
    filing_date: str | None = None,
) -> EarningsFilingAnalysis:
    """Analyze the most recent (or a specific date) earnings 8-K filing for a ticker.

    Args:
        ticker:      Equity ticker symbol (e.g. "AAPL").
        filing_date: Optional "YYYY-MM-DD"; if None, the most recent 8-K is used.

    Raises:
        ValueError: If no 8-K filings are found for the ticker.
    """
    ticker = ticker.upper().strip()
    logger.info("earnings_nlp: analyze_earnings_filing", ticker=ticker, filing_date=filing_date)

    async with httpx.AsyncClient() as client:
        cik = await _resolve_cik(ticker, client)
        filings = await _find_8k_filings(ticker, cik, limit=20, client=client)

    if not filings:
        raise ValueError(f"No 8-K filings found for ticker '{ticker}'")

    target = filings[0]
    if filing_date:
        matched = [f for f in filings if f.get("file_date", "").startswith(filing_date)]
        if matched:
            target = matched[0]
        else:
            logger.warning("earnings_nlp: no filing found for %s, using latest", filing_date)

    async with httpx.AsyncClient() as client:
        analysis = await _analyze_single_filing(ticker, target, client)

    logger.info(
        "earnings_nlp: analyze_earnings_filing complete",
        ticker=ticker,
        file_date=analysis.file_date,
        tone=analysis.tone,
        score=analysis.sentiment_score,
    )
    return analysis


async def get_earnings_trend(
    ticker: str,
    quarters: int = 4,
) -> EarningsTrend:
    """Analyze the last N quarterly earnings 8-K filings and compute sentiment trend.

    Args:
        ticker:   Equity ticker symbol (e.g. "MSFT").
        quarters: Number of quarterly filings to analyze (default 4).

    Raises:
        ValueError: If no 8-K filings are found for the ticker.
    """
    ticker = ticker.upper().strip()
    logger.info("earnings_nlp: get_earnings_trend", ticker=ticker, quarters=quarters)
    warnings: list[str] = []
    as_of = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with httpx.AsyncClient() as client:
        cik = await _resolve_cik(ticker, client)
        filings = await _find_8k_filings(ticker, cik, limit=quarters * 3, client=client)

    if not filings:
        raise ValueError(f"No 8-K filings found for ticker '{ticker}'")

    # Deduplicate: one filing per fiscal quarter
    deduplicated: list[dict] = []
    seen_quarters: set[str] = set()
    for f in filings:
        fd = f.get("file_date", "")
        try:
            dt = date.fromisoformat(fd[:10])
            qkey = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except ValueError:
            qkey = fd[:7]
        if qkey not in seen_quarters:
            seen_quarters.add(qkey)
            deduplicated.append(f)
        if len(deduplicated) >= quarters:
            break

    if len(deduplicated) < quarters:
        warnings.append(
            f"Only {len(deduplicated)} filing(s) available (requested {quarters})"
        )

    # Analyze sequentially to respect rate limits
    analyses: list[EarningsFilingAnalysis] = []
    async with httpx.AsyncClient() as client:
        for filing in deduplicated:
            try:
                analyses.append(await _analyze_single_filing(ticker, filing, client))
            except Exception as exc:
                warnings.append(f"Filing {filing.get('file_date', '?')} failed: {exc}")

    if not analyses:
        raise ValueError(f"All filing analyses failed for ticker '{ticker}'")

    # Oldest→newest for trend computation
    analyses_sorted = sorted(analyses, key=lambda a: a.file_date)
    scores = [a.sentiment_score for a in analyses_sorted]
    trend = _compute_trend(scores)
    avg_sentiment = round(sum(scores) / len(scores), 2)
    latest = analyses_sorted[-1]

    # Guidance change log (newest first)
    guidance_changes: list[str] = []
    for a in reversed(analyses_sorted):
        try:
            dt = date.fromisoformat(a.file_date[:10])
            label = f"Q{(dt.month - 1) // 3 + 1} {dt.year}"
        except ValueError:
            label = a.file_date
        if a.guidance_signal != "unknown":
            guidance_changes.append(f"{label}: {a.guidance_signal}")

    # Bubble per-filing warnings to trend level
    for a in analyses:
        for w in a.warnings:
            entry = f"[{a.file_date}] {w}"
            if entry not in warnings:
                warnings.append(entry)

    result = EarningsTrend(
        ticker=ticker,
        analyses=analyses,
        sentiment_trend=trend,
        avg_sentiment=avg_sentiment,
        latest_tone=latest.tone,
        latest_guidance=latest.guidance_signal,
        guidance_changes=guidance_changes,
        as_of=as_of,
        warnings=warnings,
    )
    logger.info(
        "earnings_nlp: get_earnings_trend complete",
        ticker=ticker,
        filings_analyzed=len(analyses),
        trend=trend,
        avg_sentiment=avg_sentiment,
    )
    return result
