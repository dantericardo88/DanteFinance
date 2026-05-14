"""governance.py — Corporate governance scoring from EDGAR DEF 14A proxy filings.

Score 0-10: board independence, CEO duality, audit committee, say-on-pay,
board diversity, classified board, poison pill, pay-for-performance.
"""
from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@sentinel.ai",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 30.0
_EDGAR_ATOM = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&company={ticker}&type=DEF+14A"
    "&dateb=&owner=include&count=5&search_text=&output=atom"
)
_EDGAR_CIK_ATOM = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type=DEF+14A"
    "&dateb=&owner=include&count=3&search_text=&output=atom"
)
_EFTS_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{ticker}%22&dateRange=custom&startdt={startdt}"
    "&forms=DEF+14A&hits.hits.total.value=true"
)



class GovernanceProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: Optional[str] = None
    filing_date: Optional[str] = None
    filing_url: Optional[str] = None
    board_size: Optional[int] = None
    board_independence_pct: Optional[float] = None
    ceo_duality: Optional[bool] = None
    audit_committee_independent: Optional[bool] = None
    say_on_pay_pct: Optional[float] = None
    board_has_female_director: Optional[bool] = None
    classified_board: Optional[bool] = None
    poison_pill: Optional[bool] = None
    pay_for_performance: Optional[bool] = None
    governance_score: float
    score_components: dict[str, float]
    verdict: str
    as_of: str
    warnings: list[str]


class GovernanceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    governance_score: float
    verdict: str
    ceo_duality: Optional[bool] = None
    board_independence_pct: Optional[float] = None
    say_on_pay_pct: Optional[float] = None


class GovernanceScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    min_score_filter: float
    results: list[GovernanceSummary]
    best_governance: Optional[str] = None
    worst_governance: Optional[str] = None
    avg_score: Optional[float] = None
    strong_governance: list[str]
    weak_governance: list[str]
    as_of: str
    warnings: list[str]


def _extract_cik_from_atom(text: str) -> Optional[str]:
    """Pull CIK out of an EDGAR atom feed."""
    m = re.search(r"CIK=(\d+)", text)
    if m:
        return m.group(1).lstrip("0") or m.group(1)
    m = re.search(r"<category[^>]*term=\"(\d{10})\"", text)
    if m:
        return m.group(1).lstrip("0") or m.group(1)
    return None


async def _resolve_cik(ticker: str, client: httpx.AsyncClient) -> Optional[str]:
    url = _EDGAR_ATOM.format(ticker=ticker)
    try:
        r = await client.get(url)
        r.raise_for_status()
        cik = _extract_cik_from_atom(r.text)
        if cik:
            return cik
    except Exception as exc:
        logger.warning("CIK atom lookup failed", ticker=ticker, error=str(exc))

    # Fallback: EFTS full-text search
    startdt = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    url2 = _EFTS_SEARCH.format(ticker=ticker, startdt=startdt)
    try:
        r2 = await client.get(url2)
        r2.raise_for_status()
        data = r2.json()
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            cik_val = str(src.get("entity_id", "") or src.get("cik", ""))
            if cik_val.isdigit():
                return cik_val.lstrip("0") or cik_val
    except Exception as exc:
        logger.warning("CIK EFTS fallback failed", ticker=ticker, error=str(exc))

    return None


async def _get_latest_def14a(cik: str, client: httpx.AsyncClient) -> tuple[Optional[str], Optional[str]]:
    try:
        r = await client.get(_EDGAR_CIK_ATOM.format(cik=cik))
        r.raise_for_status()
        txt = r.text
        date_m = re.search(r"<updated>(\d{4}-\d{2}-\d{2})", txt)
        link_m = re.search(r'<link[^>]*href="([^"]+/Archives/edgar/data/[^"]+)"', txt)
        if not link_m:
            link_m = re.search(r"https://www\.sec\.gov/Archives/edgar/data/\S+", txt)
        return (date_m.group(1) if date_m else None), (link_m.group(1) if link_m else None)
    except Exception as exc:
        logger.warning("DEF 14A filing lookup failed", cik=cik, error=str(exc))
        return None, None


async def _fetch_filing_text(filing_index_url: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        idx_r = await client.get(filing_index_url)
        idx_r.raise_for_status()
        idx_text = idx_r.text[:100_000]
        doc_m = re.search(
            r'href="(/Archives/edgar/data/[^"]+\.htm)"[^>]*>(?:[^<]*DEF 14A|[^<]*proxy)',
            idx_text, re.IGNORECASE,
        ) or re.search(r'href="(/Archives/edgar/data/[^"]+\.htm)"', idx_text)
        doc_url = ("https://www.sec.gov" + doc_m.group(1)) if doc_m else filing_index_url
        doc_r = await client.get(doc_url)
        doc_r.raise_for_status()
        return doc_r.text[:500_000]
    except Exception as exc:
        logger.warning("Filing text fetch failed", url=filing_index_url, error=str(exc))
        return None


def _clean(text: str) -> str:
    """Strip HTML tags and unescape entities for regex parsing."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]{1,200}>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_board_size(text: str) -> Optional[int]:
    # Look for explicit "Board consists of N directors" language
    m = re.search(
        r"board\s+(?:of\s+directors\s+)?(?:consists?\s+of|is\s+comprised?\s+of|"
        r"has|currently\s+has)\s+(\d{1,2})\s+(?:members?|directors?)",
        text, re.IGNORECASE
    )
    if m:
        n = int(m.group(1))
        if 3 <= n <= 30:
            return n
    # Count "(Director)" type mentions as a rough proxy
    count = len(re.findall(r"\b(?:Director|Board\s+Member)\b", text[:50_000]))
    if 4 <= count <= 60:
        return min(count // 2, 20)  # rough heuristic
    return None


def _parse_board_independence(text: str) -> Optional[float]:
    _IND = (
        r"(\d{2,3})\s*%\s*(?:of\s+(?:our\s+)?(?:board|directors?).*?)?independent|"
        r"independent\s+directors?\s+(?:represent|comprise|make\s+up)\s+(\d{2,3})\s*%"
    )
    m = re.search(_IND, text, re.IGNORECASE)
    if m:
        raw = next(g for g in m.groups() if g is not None)
        val = float(raw)
        if 0 < val <= 100:
            return val
    m2 = re.search(
        r"(\d{1,2})\s+of\s+(?:our\s+)?(\d{1,2})\s+directors?\s+(?:are\s+)?independent",
        text, re.IGNORECASE,
    )
    if m2:
        n, total = int(m2.group(1)), int(m2.group(2))
        if total > 0 and n <= total:
            return round(100.0 * n / total, 1)
    return None


def _parse_ceo_duality(text: str) -> Optional[bool]:
    if re.search(
        r"chairman\s+and\s+chief\s+executive|chief\s+executive\s+officer\s+and\s+chairman|"
        r"serves\s+as\s+both\s+(?:chairman|ceo)|ceo\s+and\s+chairman|"
        r"combined\s+(?:role|position)\s+of\s+(?:chairman|ceo)",
        text, re.IGNORECASE,
    ):
        return True
    if re.search(
        r"independent\s+chairman|non.executive\s+chairman|"
        r"separate[sd]?\s+(?:the\s+roles?\s+of\s+)?(?:chairman|ceo)|"
        r"chairman\s+and\s+(?:chief\s+executive|ceo)\s+(?:roles?\s+)?are\s+(?:held\s+by\s+)?separate",
        text, re.IGNORECASE,
    ):
        return False
    return None


def _parse_audit_committee_independent(text: str) -> Optional[bool]:
    audit_section = re.search(
        r"audit\s+committee.{0,3000}",
        text, re.IGNORECASE | re.DOTALL
    )
    if not audit_section:
        return None
    chunk = audit_section.group(0)[:2000]
    if re.search(
        r"all\s+members?\s+(?:of\s+(?:the\s+)?audit\s+committee\s+)?(?:are\s+)?independent|"
        r"audit\s+committee\s+(?:consists?\s+of\s+|is\s+composed\s+of\s+)?(?:all\s+)?independent",
        chunk, re.IGNORECASE
    ):
        return True
    if re.search(r"audit\s+committee\s+(?:member[^s]|director).{0,100}not\s+independent", chunk, re.IGNORECASE):
        return False
    return None


def _parse_say_on_pay(text: str) -> Optional[float]:
    _SOP = (
        r"(\d{2,3}(?:\.\d+)?)\s*%\s*(?:of\s+(?:votes?\s+)?(?:cast\s+)?)?(?:in\s+favor\s+of\s+)?"
        r"(?:the\s+)?(?:say.on.pay|executive\s+compensation\s+(?:vote|proposal))|"
        r"say.on.pay\s+(?:vote|proposal|resolution).{0,200}(\d{2,3}(?:\.\d+)?)\s*%|"
        r"advisory\s+vote\s+on\s+executive\s+compensation.{0,300}(\d{2,3}(?:\.\d+)?)\s*%|"
        r"(\d{2,3}(?:\.\d+)?)\s*%\s*(?:approval|approved).{0,100}(?:executive\s+compensation|say.on.pay)"
    )
    m = re.search(_SOP, text, re.IGNORECASE)
    if m:
        raw = next(g for g in m.groups() if g is not None)
        val = float(raw)
        if 20 <= val <= 100:
            return val
    return None


def _parse_board_female_director(text: str) -> Optional[bool]:
    if re.search(
        r"\bwom[ae]n\b.{0,100}(?:director|board)|"
        r"(?:director|board).{0,100}\bwom[ae]n\b|"
        r"\bfemale\s+director|\bdirectors?\s+(?:who\s+)?(?:are\s+)?female|"
        r"\bgender\s+diversity\b.{0,200}\bboard\b|"
        r"\b(?:she|her)\b.{0,50}\bboard\b",
        text, re.IGNORECASE
    ):
        return True
    return None


def _parse_classified_board(text: str) -> Optional[bool]:
    if re.search(
        r"staggered\s+board|classified\s+board|three\s+classes?\s+of\s+directors?|"
        r"elected\s+(?:for\s+)?(?:a\s+)?three.year\s+term",
        text, re.IGNORECASE
    ):
        return True
    if re.search(
        r"annual(?:ly)?\s+elected.{0,50}board|all\s+directors?\s+(?:are\s+)?elected\s+annually",
        text, re.IGNORECASE
    ):
        return False
    return None


def _parse_poison_pill(text: str) -> Optional[bool]:
    if re.search(
        r"shareholder\s+rights\s+plan|poison\s+pill|rights\s+agreement|"
        r"rights\s+plan\b(?!\s+has\s+(?:expired|terminated|been\s+terminated))",
        text, re.IGNORECASE
    ):
        # Check if it was terminated
        if re.search(
            r"rights\s+plan.{0,200}(?:expired|terminated|no\s+longer\s+in\s+effect)|"
            r"(?:expired|terminated).{0,100}rights\s+plan",
            text, re.IGNORECASE
        ):
            return False
        return True
    return False


def _parse_pay_for_performance(text: str) -> Optional[bool]:
    if re.search(
        r"pay.for.performance|pay\s+for\s+performance|"
        r"compensation\s+aligned?\s+with\s+performance|"
        r"performance.based\s+(?:compensation|pay|awards?)|"
        r"link(?:ing|ed)?\s+(?:executive\s+)?pay\s+to\s+performance",
        text, re.IGNORECASE
    ):
        return True
    return None


def _compute_score(
    board_independence_pct: Optional[float],
    ceo_duality: Optional[bool],
    audit_committee_independent: Optional[bool],
    say_on_pay_pct: Optional[float],
    board_has_female_director: Optional[bool],
    classified_board: Optional[bool],
    poison_pill: Optional[bool],
    pay_for_performance: Optional[bool],
) -> tuple[float, dict[str, float]]:
    sop = 0.0
    if say_on_pay_pct is not None:
        sop = 1.0 if say_on_pay_pct > 90 else (0.5 if say_on_pay_pct > 80 else 0.0)
    c: dict[str, float] = {
        "board_independence":         2.0 if (board_independence_pct or 0) >= 70 else 0.0,
        "no_ceo_duality":             0.0 if ceo_duality else (1.5 if ceo_duality is False else 0.0),
        "audit_committee_independent": 1.5 if audit_committee_independent is True else 0.0,
        "say_on_pay":                 sop,
        "board_diversity":            1.5 if board_has_female_director else 0.0,
        "pay_for_performance":        0.5 if pay_for_performance else 0.0,
        "classified_board_penalty":   -1.5 if classified_board else 0.0,
        "poison_pill_penalty":        -1.0 if poison_pill else 0.0,
    }
    return float(np.clip(sum(c.values()), 0.0, 10.0)), c


def _verdict(score: float) -> str:
    if score >= 7.0:
        return "strong"
    if score >= 5.0:
        return "adequate"
    if score >= 3.0:
        return "weak"
    return "poor"


def _null_profile(ticker: str, as_of: str, warnings: list[str], **kw: object) -> GovernanceProfile:
    """Return a zero-score profile when data is unavailable."""
    _, comps = _compute_score(None, None, None, None, None, None, None, None)
    return GovernanceProfile(
        ticker=ticker, cik=kw.get("cik"), filing_date=kw.get("filing_date"),  # type: ignore[arg-type]
        filing_url=kw.get("filing_url"), board_size=None, board_independence_pct=None,  # type: ignore[arg-type]
        ceo_duality=None, audit_committee_independent=None, say_on_pay_pct=None,
        board_has_female_director=None, classified_board=None, poison_pill=None,
        pay_for_performance=None, governance_score=0.0, score_components=comps,
        verdict="insufficient data", as_of=as_of, warnings=warnings,
    )


async def get_governance_profile(ticker: str) -> GovernanceProfile:
    """Fetch and score corporate governance from the latest DEF 14A filing."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []
    ticker = ticker.upper().strip()

    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
        cik = await _resolve_cik(ticker, client)
        if not cik:
            warnings.append(f"Could not resolve CIK for {ticker}")
            return _null_profile(ticker, as_of, warnings)

        filing_date, filing_url = await _get_latest_def14a(cik, client)
        if not filing_url:
            warnings.append(f"No DEF 14A filing found for CIK {cik}")
            return _null_profile(ticker, as_of, warnings, cik=cik, filing_date=filing_date)

        raw_text = await _fetch_filing_text(filing_url, client)
        if not raw_text:
            warnings.append(f"Could not fetch filing document from {filing_url}")
            return _null_profile(ticker, as_of, warnings, cik=cik, filing_date=filing_date, filing_url=filing_url)

    text = _clean(raw_text)
    board_size = _parse_board_size(text)
    board_independence_pct = _parse_board_independence(text)
    ceo_duality = _parse_ceo_duality(text)
    audit_committee_independent = _parse_audit_committee_independent(text)
    say_on_pay_pct = _parse_say_on_pay(text)
    board_has_female_director = _parse_board_female_director(text)
    classified_board = _parse_classified_board(text)
    poison_pill = _parse_poison_pill(text)
    pay_for_performance = _parse_pay_for_performance(text)

    if board_independence_pct is None:
        warnings.append("Board independence percentage not found in filing")
    if say_on_pay_pct is None:
        warnings.append("Say-on-pay vote result not found in filing")

    score, comps = _compute_score(
        board_independence_pct, ceo_duality, audit_committee_independent,
        say_on_pay_pct, board_has_female_director, classified_board,
        poison_pill, pay_for_performance,
    )
    logger.info("Governance profile built", ticker=ticker, cik=cik, score=score, filing_date=filing_date)

    return GovernanceProfile(
        ticker=ticker,
        cik=cik,
        filing_date=filing_date,
        filing_url=filing_url,
        board_size=board_size,
        board_independence_pct=board_independence_pct,
        ceo_duality=ceo_duality,
        audit_committee_independent=audit_committee_independent,
        say_on_pay_pct=say_on_pay_pct,
        board_has_female_director=board_has_female_director,
        classified_board=classified_board,
        poison_pill=poison_pill,
        pay_for_performance=pay_for_performance,
        governance_score=score,
        score_components=comps,
        verdict=_verdict(score),
        as_of=as_of,
        warnings=warnings,
    )


async def screen_governance(
    tickers: list[str],
    min_score: float = 5.0,
) -> GovernanceScreen:
    """Screen multiple tickers by governance score in parallel."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    screen_warnings: list[str] = []

    async def _safe_profile(t: str) -> GovernanceProfile:
        try:
            return await get_governance_profile(t)
        except Exception as exc:
            logger.error("Governance fetch failed", ticker=t, error=str(exc))
            return _null_profile(t.upper(), as_of, [f"Fetch error: {exc}"])

    profiles = await asyncio.gather(*[_safe_profile(t) for t in tickers])

    all_summaries: list[GovernanceSummary] = []
    for p in profiles:
        screen_warnings.extend(p.warnings)
        all_summaries.append(GovernanceSummary(
            ticker=p.ticker, governance_score=p.governance_score, verdict=p.verdict,
            ceo_duality=p.ceo_duality, board_independence_pct=p.board_independence_pct,
            say_on_pay_pct=p.say_on_pay_pct,
        ))

    filtered = [s for s in all_summaries if s.governance_score >= min_score]
    filtered.sort(key=lambda s: s.governance_score, reverse=True)

    scores = [s.governance_score for s in all_summaries]
    best = max(all_summaries, key=lambda s: s.governance_score, default=None)
    worst = min(all_summaries, key=lambda s: s.governance_score, default=None)
    avg_score = float(np.mean(scores)) if scores else None

    strong = [s.ticker for s in all_summaries if s.governance_score >= 7.0]
    weak = [s.ticker for s in all_summaries if s.governance_score <= 4.0]

    logger.info(
        "Governance screen complete",
        tickers=len(tickers), passing=len(filtered), avg_score=avg_score,
    )

    return GovernanceScreen(
        tickers_screened=len(tickers),
        min_score_filter=min_score,
        results=filtered,
        best_governance=best.ticker if best else None,
        worst_governance=worst.ticker if worst else None,
        avg_score=avg_score,
        strong_governance=strong,
        weak_governance=weak,
        as_of=as_of,
        warnings=screen_warnings,
    )
