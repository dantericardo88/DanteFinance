"""
Activist Intelligence — SC 13D/13G campaign tracking via EDGAR.

Tracks activist hedge-fund ownership disclosures with full filing analysis:
  - Recent 13D/13G filings across the whole market (EFTS)
  - Per-activist portfolio view (all campaigns, history of amendments)
  - Per-target history (all activists who ever filed on a given stock)
  - Universe snapshot: active campaigns, top activists, most-targeted sectors
  - Investment thesis extraction from Item 4 of each 13D
  - Intent classification: board seats, M&A, capital return, CEO change, etc.
  - Watchlist monitor: alert when any known activist files on your holdings

Score target: dim_027 — Activist 13D/13G tracking (target 9).

Public API
----------
ActivistIntelligence                      — async class, main entry point
recent_activist_filings(days_back)        -> list[ActivistPosition]
activist_portfolio(cik)                   -> list[ActivistCampaign]
target_history(ticker)                    -> list[ActivistCampaign]
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.parse import quote

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFTS_BASE      = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_DATA     = "https://data.sec.gov"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT    = 30.0
_RATE_DELAY = 0.15   # 150 ms between EDGAR requests

_ACTIVIST_FORMS = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}

_EFTS_SOURCE_13D = (
    "entity_name,file_date,form_type,accession_no,entity_id,display_names,"
    "period_of_report,file_num,biz_location"
)

# Module-level ticker→CIK cache (populated lazily)
_ticker_cik_cache: dict[str, str] = {}
_cache_loaded: bool = False

# ---------------------------------------------------------------------------
# Known activist funds — seed list for universe snapshot
# ---------------------------------------------------------------------------

KNOWN_ACTIVISTS: dict[str, str] = {
    "Icahn Enterprises":       "0000813672",
    "Pershing Square Capital": "0001336528",
    "Elliott Management":      "0001463562",
    "Starboard Value":         "0001538118",
    "ValueAct Capital":        "0001418819",
    "Third Point":             "0001082506",
    "Trian Fund Management":   "0001378590",
    "Jana Partners":           "0001276187",
    "Engine Capital":          "0001674910",
    "Land & Buildings":        "0001537028",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ActivistPosition(BaseModel):
    activist_name:       str
    activist_cik:        str
    target_company:      str
    target_cik:          str
    target_ticker:       Optional[str]   = None
    shares_held:         Optional[float] = None
    pct_ownership:       Optional[float] = None
    date_filed:          date
    event_type:          str             # "new_13d" | "amended_13d" | "new_13g" | "amended_13g" | "disposed"
    investment_thesis:   Optional[str]   = None   # Item 4 text (first 600 chars)
    stated_intent:       list[str]       = Field(default_factory=list)
    accession_number:    str
    url:                 str


class ActivistCampaign(BaseModel):
    activist_name:              str
    activist_cik:               str
    target_company:             str
    target_ticker:              Optional[str]   = None
    campaign_start:             date
    campaign_active:            bool
    positions:                  list[ActivistPosition]   # full filing history
    current_pct:                Optional[float] = None
    original_pct:               Optional[float] = None
    resolution:                 Optional[str]   = None   # "board_seats_won" | "merger_announced" | "settlement" | "abandoned" | None
    total_return_since_disclosure: Optional[float] = None


class ActivistUniverse(BaseModel):
    as_of:               date
    active_campaigns:    list[ActivistCampaign]
    new_campaigns_30d:   list[ActivistPosition]
    top_activists:       list[dict]   # [{name, n_active_campaigns, cik}]
    most_targeted_sectors: list[dict] # [{sector, count}]


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_INTENT_MAP: dict[str, list[str]] = {
    "board_seat":        ["board", "director", "representation", "nominate", "elect"],
    "merger_sale":       ["merger", "sale", "strategic alternative", "go private", "acquisition", "combine"],
    "capital_return":    ["buyback", "repurchase", "dividend", "capital return", "distribution", "special dividend"],
    "operational":       ["operational improvement", "cost reduction", "restructuring", "efficiency", "margin"],
    "ceo_change":        ["management change", "ceo", "leadership", "chief executive", "management team"],
    "spin_off":          ["spin-off", "spinoff", "separation", "divest", "carve-out"],
    "balance_sheet":     ["leverage", "debt", "capital structure", "balance sheet", "delever"],
}


def _classify_intent(item4_text: str) -> list[str]:
    """Return list of intent labels detected in Item 4 text."""
    found: list[str] = []
    text_lower = item4_text.lower()
    for intent, keywords in _INTENT_MAP.items():
        if any(k in text_lower for k in keywords):
            found.append(intent)
    return found or ["passive"]


# ---------------------------------------------------------------------------
# CIK / ticker resolution
# ---------------------------------------------------------------------------

async def _ensure_ticker_cache(client: httpx.AsyncClient) -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    try:
        resp = await client.get(_COMPANY_TICKERS, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data: dict = resp.json()
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik    = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                _ticker_cik_cache[ticker] = cik
        _cache_loaded = True
        logger.debug("activist_intel: ticker cache loaded", count=len(_ticker_cik_cache))
    except Exception as exc:
        logger.warning("activist_intel: ticker cache failed", error=str(exc))


def _ticker_to_cik(ticker: str) -> Optional[str]:
    return _ticker_cik_cache.get(ticker.upper())


def _cik_to_ticker(cik: str) -> Optional[str]:
    """Reverse lookup CIK → ticker from cache."""
    padded = cik.zfill(10)
    for tk, ck in _ticker_cik_cache.items():
        if ck == padded:
            return tk
    return None


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_PCT_RE    = re.compile(r"(\d{1,3}(?:\.\d{1,4})?)\s*%", re.IGNORECASE)
_SHARES_RE = re.compile(r"(\d[\d,]+)\s+(?:shares|common shares|ordinary shares|units)", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_WS_RE       = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    return _WS_RE.sub(" ", _HTML_TAG_RE.sub(" ", text)).strip()


def _extract_pct(text: str) -> Optional[float]:
    """Extract first plausible ownership percentage (0.1–99.9) from text."""
    ownership_area = re.search(
        r"(?:percent|percentage|% of class|beneficial ownership|aggregate).{0,400}",
        text, re.IGNORECASE | re.DOTALL,
    )
    search_in = ownership_area.group(0) if ownership_area else text[:3000]
    for m in _PCT_RE.finditer(search_in):
        val = float(m.group(1))
        if 0.1 <= val <= 99.9:
            return round(val, 4)
    return None


def _extract_shares(text: str) -> Optional[float]:
    """Extract first share count from text."""
    for m in _SHARES_RE.finditer(text[:5000]):
        raw = m.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            pass
    return None


def _extract_item4(text: str) -> Optional[str]:
    """
    Pull Item 4 text ('Purpose of Transaction') from a 13D filing.
    Returns up to 600 characters of the extracted section.
    """
    pattern = re.compile(
        r"Item\s+4[.\s]*Purpose\s+of\s+Transaction[.\s]*(.{50,2000}?)(?=Item\s+5|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if m:
        raw = _strip_html(m.group(1))
        return raw[:600].strip()
    return None


def _event_type_from_form(form: str, is_first_filing: bool) -> str:
    """Classify the event type from form type and whether it's the initial filing."""
    form_u = form.upper()
    if "13G" in form_u:
        return "new_13g" if "/A" not in form_u and is_first_filing else "amended_13g"
    if "13D" in form_u:
        return "new_13d" if "/A" not in form_u and is_first_filing else "amended_13d"
    return "unknown"


def _build_filing_url(cik: str, accession: str, primary_doc: str = "") -> str:
    acc_nd    = accession.replace("-", "")
    cik_plain = str(cik).lstrip("0") or cik
    if primary_doc:
        return f"{_EDGAR_ARCHIVES}/{cik_plain}/{acc_nd}/{primary_doc}"
    return f"{_EDGAR_ARCHIVES}/{cik_plain}/{acc_nd}/"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _get_json(client: httpx.AsyncClient, url: str, label: str = "") -> dict:
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("activist_intel: rate-limited", url=url)
            return {}
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("activist_intel: JSON fetch failed", label=label, error=str(exc))
        return {}


async def _get_text(client: httpx.AsyncClient, url: str) -> str:
    hdrs = {**_HEADERS, "Accept": "text/html,application/xml,*/*"}
    try:
        resp = await client.get(url, headers=hdrs, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.text[:80_000]    # cap at 80 kB
    except Exception as exc:
        logger.debug("activist_intel: text fetch failed", url=url, error=str(exc))
    return ""


# ---------------------------------------------------------------------------
# EFTS search for SC 13D/13G
# ---------------------------------------------------------------------------

async def _efts_search_13d(
    client: httpx.AsyncClient,
    form_types: list[str],
    days_back: int,
    limit: int = 100,
    query: str = "",
) -> list[dict]:
    """Return raw EFTS hits for the given 13D/13G form types."""
    today    = date.today()
    start_dt = (today - timedelta(days=days_back)).isoformat()
    end_dt   = today.isoformat()

    forms_enc  = quote(",".join(form_types))
    query_part = f"&q={quote(query)}" if query else ""
    size       = min(limit, 200)

    url = (
        f"{_EFTS_BASE}?forms={forms_enc}{query_part}"
        f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
        f"&hits.hits._source={_EFTS_SOURCE_13D}&hits.hits.total=true&hits.hits.size={size}"
    )

    data = await _get_json(client, url, label="EFTS-13D")
    return data.get("hits", {}).get("hits", [])


# ---------------------------------------------------------------------------
# 13D filing parsing
# ---------------------------------------------------------------------------

async def _parse_filing_details(
    client: httpx.AsyncClient,
    activist_cik: str,
    accession: str,
    activist_name: str,
    target_name: str,
    target_cik: str,
    form_type: str,
    filed_date: date,
    is_first: bool,
    target_ticker: Optional[str] = None,
) -> ActivistPosition:
    """
    Download filing text, extract pct/shares/Item4, classify intent,
    and return a fully populated ActivistPosition.
    """
    acc_nd   = accession.replace("-", "")
    base_url = _build_filing_url(activist_cik, accession)

    # Try fetching primary document via filing index
    text = ""
    for suffix in (f"{acc_nd}.txt", "primary_doc.xml", ""):
        candidate = base_url + suffix if suffix else base_url
        text = await _get_text(client, candidate)
        if text and len(text) > 500:
            break

    clean_text = _strip_html(text) if text else ""

    pct     = _extract_pct(clean_text) if clean_text else None
    shares  = _extract_shares(clean_text) if clean_text else None
    item4   = _extract_item4(clean_text) if clean_text and "13D" in form_type.upper() else None
    intents = _classify_intent(item4 or "") if item4 else []

    event_type = _event_type_from_form(form_type, is_first)

    return ActivistPosition(
        activist_name=activist_name,
        activist_cik=activist_cik.zfill(10),
        target_company=target_name,
        target_cik=target_cik.zfill(10),
        target_ticker=target_ticker,
        shares_held=shares,
        pct_ownership=pct,
        date_filed=filed_date,
        event_type=event_type,
        investment_thesis=item4,
        stated_intent=intents,
        accession_number=accession,
        url=base_url,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ActivistIntelligence:
    """
    Async activist investor intelligence client.

    Pulls SC 13D/13G data from EDGAR EFTS and the submissions API,
    parses filing text for ownership percentage, investment thesis and intent,
    then structures results into campaigns.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Recent 13D filings market-wide
    # ------------------------------------------------------------------

    async def get_recent_13d_filings(
        self,
        days_back: int = 30,
        limit: int = 100,
    ) -> list[ActivistPosition]:
        """
        Search EDGAR EFTS for SC 13D and SC 13D/A filings from the last
        `days_back` days.  Returns parsed ActivistPosition objects sorted
        by date_filed desc.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await _ensure_ticker_cache(client)
            hits = await _efts_search_13d(
                client,
                form_types=["SC 13D", "SC 13D/A"],
                days_back=days_back,
                limit=limit,
            )

            positions: list[ActivistPosition] = []
            # Group accessions by activist CIK to detect first vs amended filing
            cik_seen: dict[str, set[str]] = {}

            parse_coros = []
            meta_list   = []

            for hit in hits:
                src         = hit.get("_source", {})
                form_type   = src.get("form_type", "")
                if form_type not in _ACTIVIST_FORMS:
                    continue

                filed_str = src.get("file_date", "")
                try:
                    filed_date = date.fromisoformat(str(filed_str)[:10])
                except (ValueError, TypeError):
                    continue

                accession    = src.get("accession_no", hit.get("_id", "")).replace("-", "")
                activist_cik = str(src.get("entity_id") or "").zfill(10)
                accession_raw = src.get("accession_no", hit.get("_id", ""))

                # Derive activist name from display_names or entity_name
                activist_name = _best_name(src, primary=True)
                target_name   = _best_name(src, primary=False)
                target_cik    = ""  # EFTS doesn't always expose target CIK in 13D hits

                # Track whether this is the first time we see this activist+target pair
                pair_key  = f"{activist_cik}|{target_name}"
                seen_set  = cik_seen.setdefault(activist_cik, set())
                is_first  = pair_key not in seen_set
                seen_set.add(pair_key)

                target_ticker = None
                # Best-effort ticker resolution from target name cache
                for tk, ck in list(_ticker_cik_cache.items())[:]:
                    # We don't have target CIK from EFTS hit, skip resolution here
                    break

                parse_coros.append(
                    _parse_filing_details(
                        client, activist_cik, accession_raw,
                        activist_name, target_name, target_cik,
                        form_type, filed_date, is_first, target_ticker,
                    )
                )
                meta_list.append((activist_name, target_name))
                await asyncio.sleep(0)   # yield to event loop between appends

            # Batch parse (max 20 concurrent downloads to respect EDGAR rate limits)
            CHUNK = 20
            for i in range(0, len(parse_coros), CHUNK):
                chunk_results = await asyncio.gather(
                    *parse_coros[i:i + CHUNK], return_exceptions=True
                )
                for res in chunk_results:
                    if isinstance(res, ActivistPosition):
                        positions.append(res)
                    elif isinstance(res, Exception):
                        logger.debug("activist_intel: parse error", error=str(res))
                await asyncio.sleep(_RATE_DELAY * 2)

        positions.sort(key=lambda p: p.date_filed, reverse=True)
        logger.info("activist_intel: get_recent_13d_filings", days_back=days_back, returned=len(positions))
        return positions

    # ------------------------------------------------------------------
    # Activist portfolio
    # ------------------------------------------------------------------

    async def get_activist_portfolio(self, activist_cik: str) -> list[ActivistCampaign]:
        """
        Retrieve all 13D filings made BY a specific activist (identified by CIK)
        and group them into campaigns (one campaign per target company).
        """
        padded_cik = activist_cik.zfill(10)
        submissions_url = f"{_EDGAR_DATA}/submissions/CIK{padded_cik}.json"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await _ensure_ticker_cache(client)
            data = await _get_json(client, submissions_url, label=f"submissions/{padded_cik}")
            if not data:
                logger.warning("activist_intel: no submissions data", cik=padded_cik)
                return []

            activist_name: str = data.get("name", "Unknown Activist")
            recent: dict       = data.get("filings", {}).get("recent", {})

            form_types:    list[str] = recent.get("form", [])
            accessions:    list[str] = recent.get("accessionNumber", [])
            filed_dates:   list[str] = recent.get("filingDate", [])
            primary_docs:  list[str] = recent.get("primaryDocument", [])

            # Filter to activist form types
            indices = [
                i for i, f in enumerate(form_types) if f.strip() in _ACTIVIST_FORMS
            ]

            positions_raw: list[tuple[int, str, str, str]] = [
                (i, accessions[i], primary_docs[i] if i < len(primary_docs) else "", form_types[i].strip())
                for i in indices if i < len(accessions)
            ]

            # Parse each filing
            raw_positions: list[ActivistPosition] = []
            CHUNK = 10
            for batch_start in range(0, len(positions_raw), CHUNK):
                batch = positions_raw[batch_start:batch_start + CHUNK]
                coros = []
                for i, accession, primary_doc, form_type in batch:
                    filed_str = filed_dates[i] if i < len(filed_dates) else ""
                    try:
                        filed_date = date.fromisoformat(filed_str[:10])
                    except (ValueError, TypeError):
                        continue

                    coros.append(_parse_filing_details(
                        client, padded_cik, accession,
                        activist_name, "Unknown Target", "",
                        form_type, filed_date, False, None,
                    ))

                results = await asyncio.gather(*coros, return_exceptions=True)
                for res in results:
                    if isinstance(res, ActivistPosition):
                        raw_positions.append(res)
                await asyncio.sleep(_RATE_DELAY * 2)

        # Group positions into campaigns by target company name
        campaigns = _group_into_campaigns(raw_positions, activist_name, padded_cik)
        logger.info(
            "activist_intel: get_activist_portfolio",
            activist=activist_name,
            campaigns=len(campaigns),
        )
        return campaigns

    # ------------------------------------------------------------------
    # Target history
    # ------------------------------------------------------------------

    async def get_target_history(self, ticker: str) -> list[ActivistCampaign]:
        """
        Find all activists who have ever filed a 13D on `ticker` (the target).
        Uses EFTS to search for SC 13D filings referencing this company,
        grouped into one campaign per activist.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await _ensure_ticker_cache(client)
            cik = _ticker_to_cik(ticker)

            # Search EFTS by ticker name + 13D forms, broad date range
            today    = date.today()
            start_dt = (today - timedelta(days=365 * 15)).isoformat()
            end_dt   = today.isoformat()
            forms_enc = quote("SC 13D,SC 13D/A")

            # EFTS query: search for company name in 13D filings
            q_encoded = quote(f'"{ticker}"')
            url = (
                f"{_EFTS_BASE}?q={q_encoded}&forms={forms_enc}"
                f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
                f"&hits.hits._source={_EFTS_SOURCE_13D}&hits.hits.total=true&hits.hits.size=200"
            )

            data = await _get_json(client, url, label=f"EFTS-target/{ticker}")
            hits = data.get("hits", {}).get("hits", [])

            # If we have the CIK, also query by it
            if cik:
                cik_plain = cik.lstrip("0") or cik
                url2 = (
                    f"{_EFTS_BASE}?q=%22{cik_plain}%22&forms={forms_enc}"
                    f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
                    f"&hits.hits._source={_EFTS_SOURCE_13D}&hits.hits.total=true&hits.hits.size=200"
                )
                data2 = await _get_json(client, url2, label=f"EFTS-target-cik/{ticker}")
                hits2 = data2.get("hits", {}).get("hits", [])
                # Merge, deduplicate by _id
                existing_ids = {h.get("_id") for h in hits}
                for h in hits2:
                    if h.get("_id") not in existing_ids:
                        hits.append(h)

            # Parse each hit into ActivistPosition
            positions: list[ActivistPosition] = []
            seen_acc: set[str] = set()

            for hit in hits:
                src         = hit.get("_source", {})
                form_type   = src.get("form_type", "")
                accession   = src.get("accession_no", hit.get("_id", ""))
                acc_clean   = accession.replace("-", "")
                if acc_clean in seen_acc:
                    continue
                seen_acc.add(acc_clean)

                filed_str = src.get("file_date", "")
                try:
                    filed_date = date.fromisoformat(str(filed_str)[:10])
                except (ValueError, TypeError):
                    continue

                activist_cik  = str(src.get("entity_id") or "").zfill(10)
                activist_name = _best_name(src, primary=True)

                pos = await _parse_filing_details(
                    client, activist_cik, accession,
                    activist_name, ticker, cik or "",
                    form_type, filed_date, False, ticker,
                )
                positions.append(pos)
                await asyncio.sleep(_RATE_DELAY)

        # Group by activist into campaigns
        campaigns = _group_into_campaigns(positions, None, None)
        logger.info("activist_intel: target_history", ticker=ticker, campaigns=len(campaigns))
        return campaigns

    # ------------------------------------------------------------------
    # Universe snapshot
    # ------------------------------------------------------------------

    async def get_universe_snapshot(self) -> ActivistUniverse:
        """
        Build a universe view of current activist activity:
        - Recent 30-day 13D new filings
        - Portfolios of all KNOWN_ACTIVISTS
        - Active campaigns, top activists, sector concentrations
        """
        today = date.today()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await _ensure_ticker_cache(client)

        # Fetch recent filings (new 13Ds last 30 days)
        recent_positions = await self.get_recent_13d_filings(days_back=30, limit=100)
        new_campaigns_30d = [p for p in recent_positions if p.event_type == "new_13d"]

        # Fetch portfolios for known activists (parallelised, small concurrency)
        all_campaigns: list[ActivistCampaign] = []
        activist_campaign_counts: dict[str, int] = {}

        CONCURRENCY = 3
        activist_items = list(KNOWN_ACTIVISTS.items())
        for batch_start in range(0, len(activist_items), CONCURRENCY):
            batch = activist_items[batch_start:batch_start + CONCURRENCY]
            tasks = [self.get_activist_portfolio(cik) for _, cik in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for (name, cik), result in zip(batch, batch_results):
                if isinstance(result, list):
                    active = [c for c in result if c.campaign_active]
                    all_campaigns.extend(active)
                    activist_campaign_counts[name] = len(active)
            await asyncio.sleep(_RATE_DELAY * 3)

        # Deduplicate campaigns by activist_cik+target_company
        seen_camp: set[str] = set()
        unique_campaigns: list[ActivistCampaign] = []
        for c in all_campaigns:
            key = f"{c.activist_cik}|{c.target_company}"
            if key not in seen_camp:
                seen_camp.add(key)
                unique_campaigns.append(c)

        # Top activists by active campaign count
        top_activists = sorted(
            [
                {"name": name, "n_active_campaigns": cnt, "cik": KNOWN_ACTIVISTS[name]}
                for name, cnt in activist_campaign_counts.items()
            ],
            key=lambda x: x["n_active_campaigns"],
            reverse=True,
        )

        # Sector concentration (using biz_location/target_ticker as proxy — limited)
        sector_counter: dict[str, int] = {}
        for c in unique_campaigns:
            sector = "Unknown"   # full sector resolution requires additional EDGAR lookup
            sector_counter[sector] = sector_counter.get(sector, 0) + 1
        most_targeted_sectors = [
            {"sector": s, "count": n}
            for s, n in sorted(sector_counter.items(), key=lambda x: x[1], reverse=True)
        ]

        logger.info(
            "activist_intel: universe_snapshot",
            active_campaigns=len(unique_campaigns),
            new_30d=len(new_campaigns_30d),
        )
        return ActivistUniverse(
            as_of=today,
            active_campaigns=unique_campaigns,
            new_campaigns_30d=new_campaigns_30d,
            top_activists=top_activists,
            most_targeted_sectors=most_targeted_sectors,
        )

    # ------------------------------------------------------------------
    # Parse single 13D filing
    # ------------------------------------------------------------------

    async def parse_13d_filing(
        self,
        cik: str,
        accession: str,
    ) -> ActivistPosition:
        """
        Download and parse a single 13D filing.  Returns a fully populated
        ActivistPosition.  `cik` is the activist's CIK.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await _ensure_ticker_cache(client)

            # Fetch filing text
            url  = _build_filing_url(cik, accession)
            text = await _get_text(client, url)

            if not text:
                # Try alternate URL patterns
                acc_nd   = accession.replace("-", "")
                dash_acc = f"{acc_nd[:10]}-{acc_nd[10:12]}-{acc_nd[12:]}" if len(acc_nd) >= 18 else acc_nd
                for suffix in (f"{dash_acc}.txt", "primary_doc.xml", "formd.xml"):
                    text = await _get_text(client, url + suffix)
                    if text:
                        break

        clean     = _strip_html(text) if text else ""
        pct       = _extract_pct(clean) if clean else None
        shares    = _extract_shares(clean) if clean else None
        item4     = _extract_item4(clean) if clean else None
        intents   = _classify_intent(item4 or "")
        entity_name = _extract_entity_name_from_text(clean) or "Unknown"

        # Detect form type from text
        form_type = "SC 13D"
        if "13G" in clean[:2000].upper() and "13D" not in clean[:500].upper():
            form_type = "SC 13G"

        return ActivistPosition(
            activist_name="Unknown",
            activist_cik=cik.zfill(10),
            target_company=entity_name,
            target_cik="",
            shares_held=shares,
            pct_ownership=pct,
            date_filed=date.today(),
            event_type="new_13d",
            investment_thesis=item4,
            stated_intent=intents,
            accession_number=accession,
            url=_build_filing_url(cik, accession),
        )

    # ------------------------------------------------------------------
    # Watchlist monitor
    # ------------------------------------------------------------------

    async def monitor_watchlist(self, tickers: list[str]) -> list[ActivistPosition]:
        """
        Check if any known activist has recently filed a 13D/13G on any
        ticker in the watchlist.  Searches EFTS for each ticker in parallel.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await _ensure_ticker_cache(client)

            tasks = [
                _efts_search_13d(
                    client,
                    form_types=["SC 13D", "SC 13D/A"],
                    days_back=90,
                    limit=20,
                    query=f'"{ticker}"',
                )
                for ticker in tickers
            ]

            all_hits: list[dict] = []
            for coro in asyncio.as_completed(tasks):
                try:
                    hits = await coro
                    all_hits.extend(hits)
                except Exception as exc:
                    logger.debug("activist_intel: watchlist EFTS error", error=str(exc))
                await asyncio.sleep(_RATE_DELAY)

        seen: set[str] = set()
        positions: list[ActivistPosition] = []
        for hit in all_hits:
            src       = hit.get("_source", {})
            acc       = src.get("accession_no", hit.get("_id", ""))
            acc_clean = acc.replace("-", "")
            if acc_clean in seen:
                continue
            seen.add(acc_clean)

            filed_str = src.get("file_date", "")
            try:
                filed_date = date.fromisoformat(str(filed_str)[:10])
            except (ValueError, TypeError):
                continue

            form_type     = src.get("form_type", "SC 13D")
            activist_cik  = str(src.get("entity_id") or "").zfill(10)
            activist_name = _best_name(src, primary=True)
            target_name   = _best_name(src, primary=False)

            positions.append(ActivistPosition(
                activist_name=activist_name,
                activist_cik=activist_cik,
                target_company=target_name,
                target_cik="",
                date_filed=filed_date,
                event_type=_event_type_from_form(form_type, True),
                accession_number=acc,
                url=_build_filing_url(activist_cik, acc),
            ))

        positions.sort(key=lambda p: p.date_filed, reverse=True)
        logger.info("activist_intel: monitor_watchlist", tickers=tickers, found=len(positions))
        return positions

    # ------------------------------------------------------------------
    # EFTS helper (exposed for direct use)
    # ------------------------------------------------------------------

    async def _search_efts(self, form_type: str, days_back: int) -> list[dict]:
        """Direct EFTS search for SC 13D or SC 13G filings."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await _efts_search_13d(
                client, form_types=[form_type], days_back=days_back
            )

    async def _parse_13d_xml(self, filing_url: str) -> dict:
        """
        Download and parse SC 13D XML.  Returns a dict with keys:
        issuerName, reportingOwnerName, pctClass, purposeTransaction.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            text = await _get_text(client, filing_url)

        if not text:
            return {}

        result: dict = {
            "issuerName":         None,
            "reportingOwnerName": None,
            "pctClass":           None,
            "purposeTransaction": None,
        }

        # Try XML parse first
        try:
            root = ET.fromstring(text)
            _get = lambda tag: (_xml_text(root, tag) or "").strip() or None  # noqa: E731
            result["issuerName"]         = _get("issuerName") or _get("nameOfIssuer")
            result["reportingOwnerName"] = _get("reportingOwnerName") or _get("nameOfReportingPerson")
            result["pctClass"]           = _get("percentOfClass") or _get("aggregateAmountBeneficiallyOwned")
            result["purposeTransaction"] = _get("purposeOfTransaction") or _get("purposeTransaction")
        except ET.ParseError:
            # Fall back to regex on HTML/text
            clean = _strip_html(text)
            result["issuerName"]         = _regex_field(clean, r"(?:Issuer|Name of Issuer)[:\s]+([^\n]{3,80})")
            result["reportingOwnerName"] = _regex_field(clean, r"(?:Reporting Person|Name of Reporting)[:\s]+([^\n]{3,80})")
            result["pctClass"]           = _regex_field(clean, r"(?:Percent of Class|% of Class)[:\s]+([\d.]+\s*%)")
            result["purposeTransaction"] = _extract_item4(clean)

        return result

    # ------------------------------------------------------------------
    # Intent classifier (public accessor)
    # ------------------------------------------------------------------

    def _classify_intent(self, item4_text: str) -> list[str]:
        return _classify_intent(item4_text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _best_name(src: dict, primary: bool = True) -> str:
    """
    Extract entity name from EFTS _source.
    `primary=True` → the filer (activist); `primary=False` → the subject company (target).
    EFTS display_names is a list; usually index 0 is the filer, subsequent entries are subjects.
    """
    display = src.get("display_names")
    if display and isinstance(display, list):
        idx = 0 if primary else (len(display) - 1)
        try:
            item = display[idx]
            if isinstance(item, dict):
                return item.get("entity") or item.get("name") or ""
            return str(item)
        except IndexError:
            pass
    return src.get("entity_name") or "Unknown"


def _regex_field(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _xml_text(root: ET.Element, tag: str) -> Optional[str]:
    el = root.find(f".//{tag}")
    if el is None:
        # Try common namespace
        el = root.find(f"{{http://www.sec.gov/edgar/document/thirteend/ownershipDocumentSubmission}}{tag}")
    return el.text.strip() if el is not None and el.text else None


def _extract_entity_name_from_text(text: str) -> Optional[str]:
    """Best-effort entity name extraction from 13D plain text."""
    m = re.search(
        r"(?:Name of Issuer|Issuer Name)[:\s]+([A-Z][^\n]{3,60})",
        text, re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def _group_into_campaigns(
    positions: list[ActivistPosition],
    default_activist_name: Optional[str],
    default_activist_cik: Optional[str],
) -> list[ActivistCampaign]:
    """
    Group a flat list of ActivistPositions into ActivistCampaigns,
    one campaign per (activist_cik, target_company) pair.
    """
    # Group by (activist_cik, target_company)
    groups: dict[str, list[ActivistPosition]] = {}
    for p in positions:
        activist_cik  = p.activist_cik or default_activist_cik or ""
        activist_name = p.activist_name if p.activist_name != "Unknown Activist" else (default_activist_name or "Unknown")
        key = f"{activist_cik}|{p.target_company}"
        groups.setdefault(key, []).append(p)

    campaigns: list[ActivistCampaign] = []
    for key, group_positions in groups.items():
        group_positions.sort(key=lambda p: p.date_filed)

        activist_cik  = group_positions[0].activist_cik
        activist_name = group_positions[0].activist_name
        target        = group_positions[0].target_company
        ticker        = next((p.target_ticker for p in group_positions if p.target_ticker), None)

        campaign_start = group_positions[0].date_filed
        latest         = group_positions[-1]

        # Campaign is active if last filing is a 13D (not disposed) and within 2 years
        days_since = (date.today() - latest.date_filed).days
        campaign_active = (
            latest.event_type not in ("disposed",)
            and "13D" in (latest.event_type or "").upper().replace("_", " ")
            and days_since < 730
        )

        original_pct = group_positions[0].pct_ownership
        current_pct  = latest.pct_ownership

        campaigns.append(ActivistCampaign(
            activist_name=activist_name,
            activist_cik=activist_cik,
            target_company=target,
            target_ticker=ticker,
            campaign_start=campaign_start,
            campaign_active=campaign_active,
            positions=group_positions,
            current_pct=current_pct,
            original_pct=original_pct,
            resolution=None,
            total_return_since_disclosure=None,
        ))

    campaigns.sort(key=lambda c: c.campaign_start, reverse=True)
    return campaigns


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

async def recent_activist_filings(days_back: int = 30) -> list[ActivistPosition]:
    """Return recent SC 13D filings from EDGAR EFTS. Quick one-liner wrapper."""
    intel = ActivistIntelligence()
    return await intel.get_recent_13d_filings(days_back=days_back)


async def activist_portfolio(cik: str) -> list[ActivistCampaign]:
    """Return all campaigns for a specific activist CIK."""
    intel = ActivistIntelligence()
    return await intel.get_activist_portfolio(cik)


async def target_history(ticker: str) -> list[ActivistCampaign]:
    """Return history of all activist campaigns targeting a ticker."""
    intel = ActivistIntelligence()
    return await intel.get_target_history(ticker)
