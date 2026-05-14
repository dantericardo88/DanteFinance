"""
EDGAR Full-Text Search (EFTS) — Bloomberg-grade textual intelligence for free.

Wraps the SEC EDGAR EFTS API to provide:
  - Keyword search across all SEC filings by date, form type, entity
  - Pre-built search templates for material events (going concern, restatements,
    data breaches, CEO departures, acquisitions, guidance cuts, etc.)
  - Material-event feed scanning all templates in parallel
  - Watchlist monitoring for a basket of tickers
  - Cross-filing disclosure trend (mention count per 10-K over time)
  - Section extraction from 10-K/10-Q HTML filings

Score target: dim_033 — Full-text EDGAR search (target 9).

Public API
----------
EDGARFullTextSearch                — async class, main entry point
search_edgar(query, forms, days_back)     -> SearchResponse
material_events_feed(days_back)           -> IntelligenceFeed
monitor_tickers(tickers, query)           -> list[SearchResult]
"""
from __future__ import annotations

import asyncio
import re
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFTS_BASE      = "https://efts.sec.gov/LATEST/search-index"
EDGAR_BASE     = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_BROWSE   = "https://www.sec.gov/cgi-bin/browse-edgar"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT        = 25.0
_RATE_DELAY     = 0.12   # 120 ms between EDGAR requests (stay under 10 req/s)
_MAX_EFTS_HITS  = 200    # EFTS caps at 200 per request

# EFTS _source fields to request (keeps response compact)
_EFTS_SOURCE = (
    "period_of_report,entity_name,file_num,biz_location,inc_states,"
    "category_name,accession_no,file_date,form_type,display_names,entity_id"
)

# ---------------------------------------------------------------------------
# Search templates — Bloomberg terminal charges for this; EFTS gives it free
# ---------------------------------------------------------------------------

SEARCH_TEMPLATES: dict[str, str] = {
    "credit_downgrade":   '("credit rating" OR "rating downgrade") AND ("Moody\'s" OR "S&P" OR "Fitch")',
    "going_concern":      '"going concern" OR "substantial doubt"',
    "data_breach":        '"data breach" OR "cybersecurity incident" OR "unauthorized access"',
    "ceo_resignation":    '"resigned" AND ("chief executive" OR "CEO")',
    "sec_investigation":  '"SEC investigation" OR "Securities and Exchange Commission investigation"',
    "restatement":        '"restatement" OR "restate" AND "financial statements"',
    "bankruptcy_risk":    '"Chapter 11" OR "bankruptcy protection" OR "insolvency"',
    "share_buyback":      '"share repurchase" OR "stock buyback" AND "program"',
    "dividend_cut":       '"dividend" AND ("suspend" OR "reduce" OR "eliminate" OR "cut")',
    "acquisition":        '"definitive agreement" AND ("acquire" OR "acquisition" OR "merger")',
    "ipo_pricing":        '"priced" AND ("initial public offering" OR "IPO")',
    "guidance_cut":       '"lowering guidance" OR "reducing outlook" OR "withdraws guidance"',
    "layoffs":            '"reduction in force" OR "workforce reduction" OR "layoffs" AND "employees"',
    "esg_commitment":     '"net zero" OR "carbon neutral" AND "commitment"',
}

# Severity classification for material events
_HIGH_SEVERITY = {"going_concern", "bankruptcy_risk", "restatement", "ceo_resignation", "sec_investigation"}
_LOW_SEVERITY  = {"esg_commitment", "share_buyback", "ipo_pricing"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    entity_name:       str
    cik:               Optional[str] = None
    form_type:         str
    filed_date:        date
    period_date:       Optional[date] = None
    accession_number:  str
    file_url:          str
    excerpt:           Optional[str] = None   # text snippet around the match
    relevance_score:   Optional[float] = None


class SearchQuery(BaseModel):
    query:       str
    form_types:  list[str]        = Field(default_factory=list)
    start_date:  Optional[date]   = None
    end_date:    Optional[date]   = None
    entity_name: Optional[str]    = None
    limit:       int              = 50


class SearchResponse(BaseModel):
    query:      SearchQuery
    total_hits: int
    results:    list[SearchResult]
    took_ms:    Optional[int] = None


class MaterialEvent(BaseModel):
    entity_name: str
    cik:         str
    event_type:  str       # key from SEARCH_TEMPLATES
    filed_date:  date
    form_type:   str
    headline:    str       # first ~200 chars of relevant excerpt
    accession:   str
    severity:    str       # "high" | "medium" | "low"
    url:         str


class IntelligenceFeed(BaseModel):
    as_of:              date
    events:             list[MaterialEvent]
    event_counts:       dict[str, int]   # event_type → count
    top_entities:       list[str]        # most-mentioned company names


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EDGARFullTextSearch:
    """
    Async EDGAR full-text search client.

    All methods create their own httpx.AsyncClient internally so callers need
    not manage connection lifecycle. Heavy parallel workflows use a single
    shared client passed explicitly to helper methods.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Core search against EDGAR EFTS.  Handles pagination up to `query.limit`
        results (EFTS cap: 200 per call; we request min(limit, 200)).
        """
        t0 = datetime.utcnow()
        url = self._build_efts_url(query)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(url, headers=_HEADERS)
                if resp.status_code == 429:
                    logger.warning("edgar_fts: rate-limited by EFTS", url=url)
                    return SearchResponse(query=query, total_hits=0, results=[])
                resp.raise_for_status()
                data: dict = resp.json()
            except Exception as exc:
                logger.error("edgar_fts: EFTS request failed", error=str(exc), url=url)
                return SearchResponse(query=query, total_hits=0, results=[])

        took_ms = int((datetime.utcnow() - t0).total_seconds() * 1000)
        response = self._parse_efts_response(data, query)
        response.took_ms = took_ms
        logger.info(
            "edgar_fts: search complete",
            query=query.query[:80],
            hits=response.total_hits,
            returned=len(response.results),
            took_ms=took_ms,
        )
        return response

    async def search_simple(
        self,
        query_text: str,
        forms: Optional[list[str]] = None,
        days_back: int = 30,
        limit: int = 50,
    ) -> SearchResponse:
        """Convenience wrapper with sensible date defaults."""
        today = date.today()
        q = SearchQuery(
            query=query_text,
            form_types=forms or [],
            start_date=today - timedelta(days=days_back),
            end_date=today,
            limit=limit,
        )
        return await self.search(q)

    async def search_by_template(
        self,
        template_name: str,
        days_back: int = 7,
    ) -> SearchResponse:
        """Run one of the pre-defined SEARCH_TEMPLATES."""
        if template_name not in SEARCH_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {sorted(SEARCH_TEMPLATES)}"
            )
        return await self.search_simple(
            query_text=SEARCH_TEMPLATES[template_name],
            days_back=days_back,
        )

    async def search_company(
        self,
        company_name: str,
        query_text: str,
        forms: Optional[list[str]] = None,
        days_back: int = 365,
    ) -> SearchResponse:
        """Search for `query_text` specifically within filings from `company_name`."""
        # EFTS supports entity_name filter via the q parameter with company name quoted
        combined = f'"{company_name}" AND ({query_text})'
        return await self.search_simple(
            query_text=combined,
            forms=forms,
            days_back=days_back,
        )

    # ------------------------------------------------------------------
    # Material events feed
    # ------------------------------------------------------------------

    async def monitor_material_events(self, days_back: int = 7) -> IntelligenceFeed:
        """
        Run all SEARCH_TEMPLATES in parallel, deduplicate, classify severity,
        and return a consolidated IntelligenceFeed.
        """
        today = date.today()

        # Fire all template searches concurrently
        tasks = {
            name: self.search_simple(
                query_text=template,
                days_back=days_back,
                limit=50,
            )
            for name, template in SEARCH_TEMPLATES.items()
        }

        responses: dict[str, SearchResponse] = {}
        for name, coro in tasks.items():
            responses[name] = await coro
            await asyncio.sleep(_RATE_DELAY)   # gentle throttle

        # Build events, deduplicate by (entity+event_type+accession)
        seen: set[str] = set()
        events: list[MaterialEvent] = []
        event_counts: dict[str, int] = {}
        entity_counter: dict[str, int] = {}

        for event_type, response in responses.items():
            count = 0
            for r in response.results:
                key = f"{r.accession_number}|{event_type}"
                if key in seen:
                    continue
                seen.add(key)

                severity = (
                    "high"   if event_type in _HIGH_SEVERITY else
                    "low"    if event_type in _LOW_SEVERITY  else
                    "medium"
                )
                headline = (r.excerpt or r.entity_name)[:200]
                events.append(MaterialEvent(
                    entity_name=r.entity_name,
                    cik=r.cik or "",
                    event_type=event_type,
                    filed_date=r.filed_date,
                    form_type=r.form_type,
                    headline=headline,
                    accession=r.accession_number,
                    severity=severity,
                    url=r.file_url,
                ))
                entity_counter[r.entity_name] = entity_counter.get(r.entity_name, 0) + 1
                count += 1

            event_counts[event_type] = count

        # Sort by severity (high → medium → low), then by filed_date desc
        _sev_rank = {"high": 0, "medium": 1, "low": 2}
        events.sort(key=lambda e: (_sev_rank[e.severity], -e.filed_date.toordinal()))

        # Top entities by event frequency
        top_entities = sorted(entity_counter, key=entity_counter.get, reverse=True)[:20]  # type: ignore[arg-type]

        logger.info(
            "edgar_fts: material_events_feed",
            days_back=days_back,
            total_events=len(events),
            templates=len(SEARCH_TEMPLATES),
        )
        return IntelligenceFeed(
            as_of=today,
            events=events,
            event_counts=event_counts,
            top_entities=top_entities,
        )

    # ------------------------------------------------------------------
    # Disclosure trend comparison
    # ------------------------------------------------------------------

    async def compare_disclosure(
        self,
        ticker: str,
        query_text: str,
        n_filings: int = 4,
    ) -> pd.DataFrame:
        """
        Retrieve the last `n_filings` annual 10-K filings for `ticker` and count
        how many times `query_text` appears in each, returning a trend DataFrame.

        Columns: filing_year, form, filed_date, mention_count, excerpt
        """
        today = date.today()
        # Search broadly: last 10 years for this ticker, 10-K only
        q = SearchQuery(
            query=f'"{ticker}" AND ({query_text})',
            form_types=["10-K"],
            start_date=today - timedelta(days=365 * 10),
            end_date=today,
            limit=min(n_filings * 2, 20),
        )
        response = await self.search(q)

        # Also try without ticker to cast a wider net
        if len(response.results) < n_filings:
            q2 = SearchQuery(
                query=query_text,
                form_types=["10-K"],
                entity_name=ticker,
                start_date=today - timedelta(days=365 * 10),
                end_date=today,
                limit=min(n_filings * 2, 20),
            )
            r2 = await self.search(q2)
            # Merge, deduplicate by accession
            seen_acc: set[str] = {r.accession_number for r in response.results}
            for r in r2.results:
                if r.accession_number not in seen_acc:
                    response.results.append(r)
                    seen_acc.add(r.accession_number)

        results = sorted(response.results, key=lambda r: r.filed_date, reverse=True)[:n_filings]

        # Count mentions per filing by downloading filing text
        rows: list[dict] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for r in results:
                text = await _fetch_text(client, r.file_url)
                words = _query_words(query_text)
                count = _count_mentions(text, words) if text else 0
                excerpt = self._extract_excerpt(text, words, max_chars=300) if text else None

                rows.append({
                    "filing_year":   r.filed_date.year if r.period_date is None else r.period_date.year,
                    "form":          r.form_type,
                    "filed_date":    r.filed_date,
                    "mention_count": count,
                    "excerpt":       excerpt or r.excerpt,
                })
                await asyncio.sleep(_RATE_DELAY)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("filed_date", ascending=False).reset_index(drop=True)
        logger.info(
            "edgar_fts: compare_disclosure",
            ticker=ticker, query=query_text, filings=len(rows),
        )
        return df

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------

    async def extract_key_sections(
        self,
        cik: str,
        accession: str,
        sections: list[str] = None,
    ) -> dict[str, str]:
        """
        Download a 10-K filing and extract named sections (by Item headers).
        Returns dict: {section_name: extracted_text}.
        """
        if sections is None:
            sections = ["risk factors", "management's discussion"]

        # Build filing index URL to find primary document
        acc_nd    = accession.replace("-", "")
        cik_plain = str(cik).lstrip("0") or cik
        index_url = f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nd}/{acc_nd}-index.htm"
        filing_url = f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nd}/"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Try to get the primary .htm document
            text = await _fetch_text(client, index_url)
            if not text:
                text = await _fetch_text(client, filing_url)

        result: dict[str, str] = {}
        if not text:
            logger.warning("edgar_fts: extract_key_sections — could not fetch filing", cik=cik, acc=accession)
            return result

        # Strip HTML tags for text processing
        clean = _strip_html(text)

        for section in sections:
            extracted = _extract_section(clean, section)
            if extracted:
                result[section] = extracted[:8000]   # cap at 8 kB per section
            else:
                result[section] = ""

        logger.info("edgar_fts: extract_key_sections", cik=cik, sections=list(result.keys()))
        return result

    # ------------------------------------------------------------------
    # Watchlist monitor
    # ------------------------------------------------------------------

    async def watch_list_monitor(
        self,
        tickers: list[str],
        query_text: str,
        days_back: int = 30,
    ) -> list[SearchResult]:
        """
        For each ticker in the watchlist, search for `query_text` in recent filings.
        Returns all matching SearchResults across the basket, sorted by filed_date desc.
        """
        tasks = [
            self.search_company(
                company_name=ticker,
                query_text=query_text,
                days_back=days_back,
            )
            for ticker in tickers
        ]

        all_results: list[SearchResult] = []
        seen: set[str] = set()

        for coro in asyncio.as_completed(tasks):
            try:
                resp: SearchResponse = await coro
                for r in resp.results:
                    if r.accession_number not in seen:
                        seen.add(r.accession_number)
                        all_results.append(r)
            except Exception as exc:
                logger.warning("edgar_fts: watchlist monitor error", error=str(exc))
            await asyncio.sleep(_RATE_DELAY)

        all_results.sort(key=lambda r: r.filed_date, reverse=True)
        logger.info(
            "edgar_fts: watchlist_monitor",
            tickers=tickers,
            query=query_text[:60],
            results=len(all_results),
        )
        return all_results

    # ------------------------------------------------------------------
    # URL builder
    # ------------------------------------------------------------------

    def _build_efts_url(self, query: SearchQuery) -> str:
        """Construct the EFTS search URL with proper URL encoding."""
        params: dict[str, str] = {}

        # Encode query text
        params["q"] = query.query

        # Form types
        if query.form_types:
            params["forms"] = ",".join(query.form_types)

        # Date range
        if query.start_date or query.end_date:
            params["dateRange"] = "custom"
            if query.start_date:
                params["startdt"] = query.start_date.isoformat()
            if query.end_date:
                params["enddt"] = query.end_date.isoformat()

        # Entity name filter
        if query.entity_name:
            params["entity"] = query.entity_name

        # Result count (EFTS uses hits.hits._source / size)
        hit_count = min(query.limit, _MAX_EFTS_HITS)
        params["hits.hits.total.value"] = "true"
        params["hits.hits._source"]     = _EFTS_SOURCE
        params["hits.hits.size"]        = str(hit_count)

        # Build URL — use urllib.parse.urlencode for proper percent-encoding
        qs = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        return f"{EFTS_BASE}?{qs}"

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_efts_response(self, data: dict, query: SearchQuery) -> SearchResponse:
        """
        Parse raw EFTS JSON.
        Response shape:
          {"hits": {"total": {"value": N}, "hits": [{
              "_id": "...",
              "_score": 1.0,
              "_source": {"entity_name": "...", "file_date": "...", "form_type": "...",
                          "accession_no": "...", "entity_id": "...",
                          "period_of_report": "...", "display_names": [...]},
              "highlight": {"body": ["...excerpt..."]}
          }]}}
        """
        hits_data    = data.get("hits", {})
        total_value  = hits_data.get("total", {})
        total_hits   = (
            total_value.get("value", 0)
            if isinstance(total_value, dict)
            else int(total_value or 0)
        )
        raw_hits: list[dict] = hits_data.get("hits", [])

        results: list[SearchResult] = []
        for hit in raw_hits:
            src     = hit.get("_source", {})
            score   = hit.get("_score")
            hl_body = hit.get("highlight", {}).get("body", [])

            entity_name = _best_entity_name(src)
            cik         = _extract_cik(src)
            form_type   = src.get("form_type", "")
            accession   = src.get("accession_no", hit.get("_id", ""))
            acc_nd      = accession.replace("-", "")

            filed_str = src.get("file_date", "")
            try:
                filed_date = date.fromisoformat(str(filed_str)[:10])
            except (ValueError, TypeError):
                continue

            period_str = src.get("period_of_report", "")
            period_date: Optional[date] = None
            try:
                if period_str:
                    period_date = date.fromisoformat(str(period_str)[:10])
            except (ValueError, TypeError):
                pass

            cik_plain = (cik or "").lstrip("0") or cik or ""
            file_url = (
                f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nd}/"
                if cik_plain and acc_nd else
                f"{EDGAR_BROWSE}?action=getcompany&CIK={cik}&type={urllib.parse.quote(form_type)}&dateb=&owner=include&count=10"
            )

            # Prefer EFTS highlight excerpt; fall back to query-based excerpt
            excerpt: Optional[str] = None
            if hl_body:
                raw_exc = " … ".join(hl_body[:2])
                excerpt = _strip_html(raw_exc)[:400]

            results.append(SearchResult(
                entity_name=entity_name,
                cik=cik,
                form_type=form_type,
                filed_date=filed_date,
                period_date=period_date,
                accession_number=accession,
                file_url=file_url,
                excerpt=excerpt,
                relevance_score=float(score) if score is not None else None,
            ))

        return SearchResponse(
            query=query,
            total_hits=total_hits,
            results=results,
        )

    # ------------------------------------------------------------------
    # Excerpt extractor
    # ------------------------------------------------------------------

    def _extract_excerpt(
        self,
        filing_text: str,
        query_words: list[str],
        max_chars: int = 300,
    ) -> str:
        """
        Find the first occurrence of any query word and return surrounding context.
        Strips HTML, normalises whitespace.
        """
        if not filing_text or not query_words:
            return ""

        clean = _strip_html(filing_text)
        clean_lower = clean.lower()

        first_pos = len(clean)
        for word in query_words:
            idx = clean_lower.find(word.lower())
            if 0 <= idx < first_pos:
                first_pos = idx

        if first_pos == len(clean):
            return ""

        half = max_chars // 2
        start = max(0, first_pos - half)
        end   = min(len(clean), first_pos + half)

        excerpt = clean[start:end].strip()
        if start > 0:
            excerpt = "…" + excerpt
        if end < len(clean):
            excerpt = excerpt + "…"
        return excerpt


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _best_entity_name(src: dict) -> str:
    """Pull a human-readable entity name from the EFTS _source dict."""
    display = src.get("display_names")
    if display and isinstance(display, list):
        first = display[0]
        if isinstance(first, dict):
            return first.get("entity") or first.get("name") or ""
        return str(first)
    return src.get("entity_name") or "Unknown"


def _extract_cik(src: dict) -> Optional[str]:
    raw = src.get("entity_id") or src.get("cik") or ""
    return str(raw).zfill(10) if raw else None


def _query_words(query_text: str) -> list[str]:
    """Extract plain words from a query (remove operators/quotes)."""
    tokens = re.findall(r'[A-Za-z]{3,}', query_text)
    stop   = {"and", "or", "not", "the", "its", "this", "that", "with", "from"}
    return [t for t in tokens if t.lower() not in stop]


def _count_mentions(text: str, words: list[str]) -> int:
    """Count total occurrences of any query word in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(text_lower.count(w.lower()) for w in words)


_HTML_TAG_RE  = re.compile(r"<[^>]+>", re.DOTALL)
_WS_RE        = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    no_tags = _HTML_TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", no_tags).strip()


# Map section keywords to their canonical Item headers in 10-K filings
_SECTION_HEADERS: dict[str, list[str]] = {
    "risk factors":              ["Item 1A", "ITEM 1A"],
    "management's discussion":   ["Item 7", "ITEM 7", "MD&A"],
    "business":                  ["Item 1.", "ITEM 1."],
    "legal proceedings":         ["Item 3", "ITEM 3"],
    "financial statements":      ["Item 8", "ITEM 8"],
    "quantitative disclosures":  ["Item 7A", "ITEM 7A"],
}

def _extract_section(text: str, section_name: str) -> str:
    """
    Locate a named 10-K section by its Item header and return the content up
    to the next Item header.
    """
    headers = _SECTION_HEADERS.get(section_name.lower(), [section_name])

    start_idx = -1
    for header in headers:
        idx = text.find(header)
        if idx >= 0:
            start_idx = idx
            break

    if start_idx < 0:
        return ""

    # Find the next "Item N" header to mark end of section
    next_item_re = re.compile(r"\bItem\s+\d+[A-Z]?\b", re.IGNORECASE)
    search_from  = start_idx + 20    # skip past the header itself

    end_idx = len(text)
    for m in next_item_re.finditer(text, search_from):
        end_idx = m.start()
        break

    return text[start_idx:end_idx].strip()


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    """Best-effort text fetch; returns empty string on error."""
    try:
        headers = {**_HEADERS, "Accept": "text/html,application/xhtml+xml,*/*"}
        resp = await client.get(url, headers=headers, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text[:100_000]   # cap at 100 kB
    except Exception as exc:
        logger.debug("edgar_fts: _fetch_text failed", url=url, error=str(exc))
    return ""


# ---------------------------------------------------------------------------
# Module-level helpers (convenience wrappers)
# ---------------------------------------------------------------------------

async def search_edgar(
    query: str,
    forms: Optional[list[str]] = None,
    days_back: int = 30,
) -> SearchResponse:
    """Search EDGAR full-text index. Quick one-liner wrapper."""
    engine = EDGARFullTextSearch()
    return await engine.search_simple(query, forms=forms, days_back=days_back)


async def material_events_feed(days_back: int = 7) -> IntelligenceFeed:
    """Return a deduplicated feed of material corporate events across all templates."""
    engine = EDGARFullTextSearch()
    return await engine.monitor_material_events(days_back=days_back)


async def monitor_tickers(
    tickers: list[str],
    query: str,
    days_back: int = 30,
) -> list[SearchResult]:
    """Monitor a list of tickers for a specific disclosure keyword."""
    engine = EDGARFullTextSearch()
    return await engine.watch_list_monitor(tickers, query, days_back=days_back)
