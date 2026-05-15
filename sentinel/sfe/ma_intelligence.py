"""
M&A Deal Intelligence — Dimension #100 (target 9+).

Comprehensive merger & acquisition intelligence using free EDGAR data sources.
Covers deal discovery, merger-arb spread analysis, deal completion scoring,
regulatory timeline tracking, and proxy-statement parsing.

Public API
----------
MADealScraper
    get_edgar_ma_filings(lookback_days)       -> list[dict]
    get_ma_from_8k(cik, accession)            -> dict
    parse_merger_proxy(accession_number)      -> dict
    get_pending_deals(as_of_date)             -> pd.DataFrame

MergerArbAnalyzer
    compute_arb_spread(target_ticker, offer_price, deal_close_date, ...) -> dict
    deal_completion_probability(deal)         -> float
    screen_arb_opportunities(min_spread_pct, max_days_to_close) -> pd.DataFrame
    historical_deal_performance(lookback_days) -> pd.DataFrame

MATimeline
    build_deal_timeline(target_ticker)        -> list[dict]
    get_regulatory_status(deal)               -> dict
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional
from urllib.parse import quote

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFTS_BASE          = "https://efts.sec.gov/LATEST/search-index"
EDGAR_BASE         = "https://data.sec.gov"
EDGAR_ARCHIVES     = "https://www.sec.gov/Archives/edgar/data"
EDGAR_SUBMISSIONS  = "https://data.sec.gov/submissions"
COMPANY_TICKERS    = "https://www.sec.gov/files/company_tickers.json"
COMPANY_SEARCH     = "https://efts.sec.gov/LATEST/search-index?q={name}&forms=&dateRange=custom&startdt=2000-01-01&enddt=2030-01-01&hits.hits._source=entity_name,cik"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept":     "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT       = 25.0
_RATE_DELAY    = 0.11  # 110 ms — SEC rate limit ~10 req/s

# Deal type labels
MA_DEAL_TYPES = {
    "all_cash":    "All-Cash",
    "all_stock":   "All-Stock",
    "mixed":       "Mixed (Cash + Stock)",
    "lbo":         "Leveraged Buyout",
    "hostile":     "Hostile Tender Offer",
    "merger":      "Statutory Merger",
    "asset_sale":  "Asset Sale",
    "unknown":     "Unknown",
}

# Form types that indicate M&A activity
_MA_FORMS = ["SC TO-T", "SC 13E-3", "DEFM14A", "PREM14A", "8-K"]

# EDGAR full-text queries — (phrase, form_list, deal_type_hint)
_EFTS_MA_QUERIES: list[tuple[str, list[str], str]] = [
    ("Agreement and Plan of Merger",         ["8-K", "DEFM14A", "PREM14A"], "merger"),
    ("definitive merger agreement",          ["8-K"],                        "merger"),
    ("business combination agreement",       ["8-K"],                        "merger"),
    ("Agreement and Plan of Merger",         ["SC 13E-3"],                   "merger"),
    ("tender offer",                         ["SC TO-T"],                    "all_cash"),
    ("cash merger consideration",            ["DEFM14A", "PREM14A"],         "all_cash"),
    ("stock for stock merger",               ["DEFM14A", "PREM14A"],         "all_stock"),
    ("definitive agreement to be acquired",  ["8-K"],                        "merger"),
]

# Regex patterns for deal term extraction
_PRICE_RE          = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B\b|M\b)', re.I)
_PER_SHARE_RE      = re.compile(r'\$\s*([\d.]+)\s*(?:per|each)\s+(?:share|common share)', re.I)
_PREMIUM_RE        = re.compile(r'([\d.]+)%?\s*premium', re.I)
_ACQUIRER_RE       = re.compile(
    r'(?:acquired by|merger with|acquisition by|acquirer[,\s]+)([A-Z][A-Za-z\s,\.]+?)(?:,|\.|for|\()',
    re.I,
)
_TARGET_RE         = re.compile(
    r'(?:acquisition of|acquire|merger with|target company[,\s]+)([A-Z][A-Za-z\s,\.]+?)(?:,|\.|for|\()',
    re.I,
)
_TERMINATION_FEE_RE = re.compile(r'termination fee[^$]*\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?', re.I)
_SYNERGIES_RE       = re.compile(r'(?:annual|cost|revenue)?\s*synergies[^$]*\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?', re.I)
_VOTE_DATE_RE       = re.compile(r'shareholder\s+vote\s+(?:on|scheduled|expected)?[:\s]+(\w+ \d+,? \d{4})', re.I)
_GO_SHOP_RE         = re.compile(r'go[\-\s]shop\s+period\s+of\s+(\d+)\s+days?', re.I)
_CLOSING_DATE_RE    = re.compile(r'expected\s+to\s+(?:close|complete)\s+(?:in\s+)?(?:the\s+)?([A-Za-z\s\d,]+?)(?:\.|,)', re.I)
_FAIRNESS_RE        = re.compile(r'(?:fairness opinion|financial advisor)[^.]*?by\s+([A-Z][A-Za-z\s&,\.]+?)(?:,|\.|\band\b)', re.I)
_HSR_RE             = re.compile(r'Hart[\-\s]Scott[\-\s]Rodino|HSR\s+Act', re.I)
_CFIUS_RE           = re.compile(r'CFIUS|Committee on Foreign Investment', re.I)
_EU_COMP_RE         = re.compile(r'European Commission|EU\s+merger|DG\s+COMP', re.I)

# Historical deal completion rates by deal type (empirical base rates)
_BASE_COMPLETION_RATES: dict[str, float] = {
    "merger":    0.87,
    "all_cash":  0.92,
    "all_stock": 0.82,
    "mixed":     0.85,
    "lbo":       0.78,
    "hostile":   0.55,
    "unknown":   0.80,
}

# CIK ticker cache (module-level singleton)
_cik_cache:        dict[str, str] = {}   # ticker -> CIK (zero-padded 10 digits)
_name_cik_cache:   dict[str, str] = {}   # lower company name -> CIK
_cik_cache_loaded: bool = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MADeal(BaseModel):
    """Parsed M&A deal record."""
    acquirer:               Optional[str]   = None
    target:                 Optional[str]   = None
    cik:                    Optional[str]   = None
    ticker:                 Optional[str]   = None
    deal_type:              str             = "unknown"
    deal_value_billions:    Optional[float] = None
    consideration_per_share: Optional[float] = None
    premium_pct:            Optional[float] = None
    filing_date:            str             = ""
    form_type:              str             = ""
    accession_number:       str             = ""
    edgar_url:              str             = ""
    status:                 str             = "pending"
    termination_fee_mm:     Optional[float] = None
    synergies_estimated_mm: Optional[float] = None
    shareholder_vote_date:  Optional[str]   = None
    go_shop_days:           Optional[int]   = None
    fairness_opinion_firms: list[str]       = Field(default_factory=list)
    requires_hsr:           bool            = False
    requires_cfius:         bool            = False
    requires_eu_comp:       bool            = False
    expected_close_text:    Optional[str]   = None

    def deal_to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), default=str)


class MADealSummary(BaseModel):
    deals:       list[MADeal]
    total:       int
    as_of:       str
    lookback_days: int
    warnings:    list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------

async def _get(client: httpx.AsyncClient, url: str, warnings: list[str]) -> dict | list | None:
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        warnings.append(f"HTTP {exc.response.status_code}: {url}")
    except Exception as exc:
        warnings.append(f"Request error ({type(exc).__name__}): {url}")
    return None


async def _load_cik_map(client: httpx.AsyncClient) -> None:
    global _cik_cache_loaded
    if _cik_cache_loaded:
        return
    try:
        resp = await client.get(COMPANY_TICKERS, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        for entry in resp.json().values():
            ticker = str(entry.get("ticker", "")).upper().strip()
            cik    = str(entry.get("cik_str", entry.get("cik", ""))).zfill(10)
            name   = str(entry.get("title", "")).lower().strip()
            if ticker and cik:
                _cik_cache[ticker] = cik
            if name and cik:
                _name_cik_cache[name] = cik
        _cik_cache_loaded = True
        logger.debug("ma_intelligence: loaded %d CIK entries", len(_cik_cache))
    except Exception as exc:
        logger.warning("ma_intelligence: CIK map load failed: %s", exc)


async def _resolve_ticker_to_cik(client: httpx.AsyncClient, ticker: str) -> Optional[str]:
    await _load_cik_map(client)
    return _cik_cache.get(ticker.upper())


async def _resolve_name_to_cik(client: httpx.AsyncClient, name: str) -> Optional[str]:
    await _load_cik_map(client)
    key = name.lower().strip()
    # Exact match first
    if key in _name_cik_cache:
        return _name_cik_cache[key]
    # Substring match (first hit)
    for k, cik in _name_cik_cache.items():
        if key in k or k in key:
            return cik
    return None


def _accession_to_url(accession_no: str, cik: str) -> str:
    acc_nodash = accession_no.replace("-", "")
    cik_plain  = cik.lstrip("0") or cik
    return f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nodash}/"


async def _fetch_filing_text(
    client: httpx.AsyncClient,
    cik: str,
    accession_no: str,
    warnings: list[str],
) -> str:
    """Fetch the primary document text from an EDGAR filing index."""
    acc_nodash = accession_no.replace("-", "")
    cik_plain  = cik.lstrip("0") or cik
    index_url  = f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nodash}/{accession_no}-index.htm"
    try:
        resp = await client.get(index_url, headers=_HEADERS, timeout=_TIMEOUT)
        # Find the primary document href
        links = re.findall(r'href="([^"]*\.htm[l]?)"', resp.text, re.I)
        primary = next((l for l in links if "index" not in l.lower()), None)
        if not primary:
            return resp.text[:8000]
        doc_url = f"https://www.sec.gov{primary}" if primary.startswith("/") else primary
        doc_resp = await client.get(doc_url, headers=_HEADERS, timeout=_TIMEOUT)
        # Strip HTML tags for text extraction
        text = re.sub(r'<[^>]+>', ' ', doc_resp.text)
        text = re.sub(r'\s+', ' ', text)
        return text[:40000]
    except Exception as exc:
        warnings.append(f"Filing text fetch failed for {accession_no}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_deal_value(text: str) -> Optional[float]:
    """Return deal value in billions from narrative text."""
    best: Optional[float] = None
    for m in _PRICE_RE.finditer(text):
        try:
            amount = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        val = amount if m.group(2).lower() in ("billion", "b") else amount / 1_000.0
        if best is None or val > best:
            best = val
    return round(best, 3) if best else None


def _extract_per_share(text: str) -> Optional[float]:
    m = _PER_SHARE_RE.search(text)
    if m:
        try:
            return round(float(m.group(1).replace(",", "")), 4)
        except ValueError:
            pass
    return None


def _extract_premium(text: str) -> Optional[float]:
    m = _PREMIUM_RE.search(text)
    if m:
        try:
            return round(float(m.group(1)), 2)
        except ValueError:
            pass
    return None


def _extract_termination_fee(text: str) -> Optional[float]:
    """Return termination fee in millions."""
    m = _TERMINATION_FEE_RE.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "million").lower()
            return round(val if "billion" in unit else val, 2)
        except ValueError:
            pass
    return None


def _extract_synergies(text: str) -> Optional[float]:
    """Return synergies estimate in millions."""
    m = _SYNERGIES_RE.search(text)
    if m:
        try:
            val  = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "million").lower()
            return round(val * 1000.0 if "billion" in unit else val, 2)
        except ValueError:
            pass
    return None


def _extract_fairness_firms(text: str) -> list[str]:
    return list({m.group(1).strip() for m in _FAIRNESS_RE.finditer(text)})[:4]


def _detect_deal_type(form_type: str, text: str) -> str:
    t = text.lower()
    ft = form_type.upper()
    if ft == "SC TO-T":
        if "all cash" in t or "cash consideration" in t:
            return "all_cash"
        return "all_cash"
    if "lbo" in t or "leveraged buyout" in t or "private equity" in t:
        return "lbo"
    if "hostile" in t or "unsolicited" in t:
        return "hostile"
    if "stock for stock" in t or "all stock" in t or "exchange ratio" in t:
        return "all_stock"
    if ("cash and stock" in t or "combination of cash" in t
            or ("cash consideration" in t and "exchange ratio" in t)):
        return "mixed"
    if "plan of merger" in t or "merger agreement" in t:
        return "merger"
    return "unknown"


def _detect_status(text: str, form_type: str) -> str:
    t = text.lower()
    if any(kw in t for kw in ("transaction has been completed", "merger has been completed",
                               "consummated", "merger closed", "transaction closed")):
        return "completed"
    if any(kw in t for kw in ("terminated", "withdrawn", "abandoned", "agreement terminated")):
        return "terminated"
    if form_type.upper() in ("DEFM14A", "SC TO-T"):
        return "pending"
    if "definitive agreement" in t:
        return "announced"
    return "pending"


# ---------------------------------------------------------------------------
# EFTS search
# ---------------------------------------------------------------------------

async def _efts_search(
    client:     httpx.AsyncClient,
    query:      str,
    forms:      list[str],
    start_date: date,
    end_date:   date,
    warnings:   list[str],
) -> list[dict]:
    url = (
        f"{EFTS_BASE}"
        f"?q={quote(chr(34) + query + chr(34))}"
        f"&forms={quote(','.join(forms))}"
        f"&dateRange=custom"
        f"&startdt={start_date.isoformat()}"
        f"&enddt={end_date.isoformat()}"
        f"&hits.hits.total.value=true"
    )
    data = await _get(client, url, warnings)
    if data is None:
        return []
    return data.get("hits", {}).get("hits", [])


def _hit_to_deal(hit: dict, deal_type_hint: str) -> Optional[MADeal]:
    src = hit.get("_source", {})
    accession_no = src.get("accession_no") or hit.get("_id", "")
    if not accession_no:
        return None

    cik = str(src.get("cik", "") or src.get("entity_id", "")).zfill(10)
    form_type = src.get("form_type", "8-K")

    # Company name
    entity_name = src.get("entity_name", "")
    if not entity_name:
        dn = src.get("display_names", [])
        if dn and isinstance(dn, list):
            first = dn[0]
            entity_name = first.get("entity", "") if isinstance(first, dict) else str(first)

    try:
        filing_date = date.fromisoformat(src.get("file_date", "")[:10]).isoformat()
    except (ValueError, TypeError):
        return None

    description = (src.get("description", "") + " " + src.get("biz_descs", "")).strip()
    if not description:
        description = f"{form_type} by {entity_name}"

    combined = description
    deal_type = _detect_deal_type(form_type, combined) if deal_type_hint == "unknown" else deal_type_hint
    if deal_type == "unknown":
        deal_type = _detect_deal_type(form_type, deal_type_hint)

    return MADeal(
        target           = entity_name or None,
        cik              = cik or None,
        deal_type        = deal_type,
        deal_value_billions = _extract_deal_value(description),
        consideration_per_share = _extract_per_share(description),
        premium_pct      = _extract_premium(description),
        filing_date      = filing_date,
        form_type        = form_type,
        accession_number = accession_no,
        edgar_url        = _accession_to_url(accession_no, cik) if cik else "",
        status           = _detect_status(description, form_type),
        requires_hsr     = bool(_HSR_RE.search(description)),
        requires_cfius   = bool(_CFIUS_RE.search(description)),
        requires_eu_comp = bool(_EU_COMP_RE.search(description)),
    )


# ---------------------------------------------------------------------------
# MADealScraper
# ---------------------------------------------------------------------------

class MADealScraper:
    """
    Scrapes and parses M&A deal intelligence from free EDGAR data sources.

    All methods are async-native. Call via ``asyncio.run(scraper.method())``.
    """

    def __init__(self, lookback_days: int = 90) -> None:
        self._default_lookback = lookback_days

    # ------------------------------------------------------------------
    # Public: get_edgar_ma_filings
    # ------------------------------------------------------------------

    async def get_edgar_ma_filings(self, lookback_days: int = 90) -> list[dict]:
        """
        EDGAR EFTS full-text search for merger/acquisition filings.

        Searches SC TO-T, SC 13E-3, DEFM14A, PREM14A and 8-K forms for key
        merger agreement phrases.  Returns de-duplicated parsed deal dicts.

        Args:
            lookback_days: Calendar days to look back from today (default 90).

        Returns:
            List of deal dicts (acquirer, target, deal_value, deal_type, etc.)
        """
        warnings_:  list[str] = []
        end_date   = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        seen:       set[str]  = set()
        deals:      list[dict] = []

        async with httpx.AsyncClient() as client:
            tasks = [
                _efts_search(client, phrase, forms, start_date, end_date, warnings_)
                for phrase, forms, _ in _EFTS_MA_QUERIES
            ]
            results = await asyncio.gather(*tasks)

        for (_, _, hint), hits in zip(_EFTS_MA_QUERIES, results):
            for hit in hits:
                src = hit.get("_source", {})
                acc = src.get("accession_no") or hit.get("_id", "")
                if not acc or acc in seen:
                    continue
                seen.add(acc)
                deal = _hit_to_deal(hit, hint)
                if deal:
                    deals.append(deal.deal_to_dict())

        deals.sort(key=lambda d: d.get("filing_date", ""), reverse=True)
        logger.info("ma_intelligence: get_edgar_ma_filings found=%d lookback=%d", len(deals), lookback_days)
        return deals

    # ------------------------------------------------------------------
    # Public: get_ma_from_8k
    # ------------------------------------------------------------------

    async def get_ma_from_8k(self, cik: str, accession: str) -> dict:
        """
        Parse an 8-K Item 1.01/2.01 filing for M&A deal terms.

        Args:
            cik:       Company CIK (zero-padded or plain).
            accession: Accession number (dashes optional).

        Returns:
            Dict with deal terms extracted via regex from filing text.
        """
        warnings_: list[str] = []
        cik_padded = cik.zfill(10)
        acc_fmt    = (
            accession if "-" in accession
            else f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
        )

        async with httpx.AsyncClient() as client:
            text = await _fetch_filing_text(client, cik_padded, acc_fmt, warnings_)

        if not text:
            return {"error": "Filing text unavailable", "warnings": warnings_}

        # Detect Item 1.01 or Item 2.01 section
        item_match = re.search(
            r'Item\s+(?:1\.01|2\.01)[^\n]*\n(.*?)(?:Item\s+\d|\Z)',
            text, re.DOTALL | re.I,
        )
        section = item_match.group(1) if item_match else text

        acquirer_m = _ACQUIRER_RE.search(section)
        target_m   = _TARGET_RE.search(section)

        result: dict = {
            "cik":                    cik_padded,
            "accession_number":       acc_fmt,
            "edgar_url":              _accession_to_url(acc_fmt, cik_padded),
            "acquirer":               acquirer_m.group(1).strip() if acquirer_m else None,
            "target":                 target_m.group(1).strip() if target_m else None,
            "deal_value_billions":    _extract_deal_value(section),
            "consideration_per_share": _extract_per_share(section),
            "premium_pct":            _extract_premium(section),
            "deal_type":              _detect_deal_type("8-K", section),
            "termination_fee_mm":     _extract_termination_fee(section),
            "synergies_estimated_mm": _extract_synergies(section),
            "expected_close_text":    _extract_expected_close(section),
            "requires_hsr":           bool(_HSR_RE.search(section)),
            "requires_cfius":         bool(_CFIUS_RE.search(section)),
            "requires_eu_comp":       bool(_EU_COMP_RE.search(section)),
            "status":                 _detect_status(section, "8-K"),
            "warnings":               warnings_,
        }
        logger.debug("ma_intelligence: 8-K parsed cik=%s acc=%s", cik, accession)
        return result

    # ------------------------------------------------------------------
    # Public: parse_merger_proxy
    # ------------------------------------------------------------------

    async def parse_merger_proxy(self, accession_number: str) -> dict:
        """
        Parse DEFM14A / PREM14A proxy statement for deal details.

        Extracts synergy estimates, fairness opinion firms, shareholder vote date,
        termination fee, go-shop period, and expected close timing.

        Args:
            accession_number: EDGAR accession number (with or without dashes).

        Returns:
            Dict with structured proxy deal terms.
        """
        warnings_: list[str] = []
        acc_fmt = (
            accession_number if "-" in accession_number
            else f"{accession_number[:10]}-{accession_number[10:12]}-{accession_number[12:]}"
        )

        # We need the CIK from EDGAR search — try full-text search for this accession
        async with httpx.AsyncClient() as client:
            search_url = f"{EFTS_BASE}?q=%22{acc_fmt}%22&forms=DEFM14A,PREM14A"
            data = await _get(client, search_url, warnings_)
            hits = data.get("hits", {}).get("hits", []) if data else []
            cik = "0000000000"
            if hits:
                src = hits[0].get("_source", {})
                cik = str(src.get("cik", "")).zfill(10)

            text = await _fetch_filing_text(client, cik, acc_fmt, warnings_)

        if not text:
            return {"error": "Proxy text unavailable", "accession_number": acc_fmt, "warnings": warnings_}

        vote_m      = _VOTE_DATE_RE.search(text)
        go_shop_m   = _GO_SHOP_RE.search(text)
        close_text  = _extract_expected_close(text)

        result: dict = {
            "accession_number":       acc_fmt,
            "cik":                    cik,
            "edgar_url":              _accession_to_url(acc_fmt, cik),
            "deal_type":              _detect_deal_type("DEFM14A", text),
            "deal_value_billions":    _extract_deal_value(text),
            "consideration_per_share": _extract_per_share(text),
            "premium_pct":            _extract_premium(text),
            "synergies_estimated_mm": _extract_synergies(text),
            "termination_fee_mm":     _extract_termination_fee(text),
            "fairness_opinion_firms": _extract_fairness_firms(text),
            "shareholder_vote_date":  vote_m.group(1).strip() if vote_m else None,
            "go_shop_period_days":    int(go_shop_m.group(1)) if go_shop_m else None,
            "expected_close_text":    close_text,
            "requires_hsr":           bool(_HSR_RE.search(text)),
            "requires_cfius":         bool(_CFIUS_RE.search(text)),
            "requires_eu_comp":       bool(_EU_COMP_RE.search(text)),
            "warnings":               warnings_,
        }
        logger.info("ma_intelligence: parse_merger_proxy acc=%s", accession_number)
        return result

    # ------------------------------------------------------------------
    # Public: get_pending_deals
    # ------------------------------------------------------------------

    async def get_pending_deals(self, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Return all pending M&A deals from recent SC TO-T and DEFM14A filings.

        Scans the last 180 days of filings and filters to status == pending/announced.

        Args:
            as_of_date: ISO date string (YYYY-MM-DD). Defaults to today.

        Returns:
            pd.DataFrame with one row per pending deal.
        """
        if as_of_date:
            try:
                end_date = date.fromisoformat(as_of_date)
            except ValueError:
                end_date = date.today()
        else:
            end_date = date.today()

        start_date = end_date - timedelta(days=180)
        warnings_: list[str] = []
        seen:       set[str] = set()
        rows:       list[dict] = []

        pending_queries = [
            ("Agreement and Plan of Merger", ["DEFM14A", "PREM14A"], "merger"),
            ("tender offer",                 ["SC TO-T"],             "all_cash"),
            ("business combination",         ["SC 13E-3"],            "merger"),
        ]

        async with httpx.AsyncClient() as client:
            tasks = [
                _efts_search(client, q, forms, start_date, end_date, warnings_)
                for q, forms, _ in pending_queries
            ]
            results = await asyncio.gather(*tasks)

        for (_, _, hint), hits in zip(pending_queries, results):
            for hit in hits:
                src = hit.get("_source", {})
                acc = src.get("accession_no") or hit.get("_id", "")
                if not acc or acc in seen:
                    continue
                seen.add(acc)
                deal = _hit_to_deal(hit, hint)
                if deal and deal.status in ("pending", "announced"):
                    rows.append(deal.deal_to_dict())

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "target", "cik", "deal_type", "deal_value_billions",
            "consideration_per_share", "premium_pct", "filing_date",
            "form_type", "accession_number", "edgar_url", "status",
        ])
        if not df.empty and "filing_date" in df.columns:
            df = df.sort_values("filing_date", ascending=False).reset_index(drop=True)
        logger.info("ma_intelligence: get_pending_deals found=%d as_of=%s", len(df), end_date)
        return df


def _extract_expected_close(text: str) -> Optional[str]:
    m = _CLOSING_DATE_RE.search(text)
    if m:
        return m.group(1).strip()[:80]
    return None


# ---------------------------------------------------------------------------
# MergerArbAnalyzer
# ---------------------------------------------------------------------------

class MergerArbAnalyzer:
    """
    Merger arbitrage analysis: spread computation, deal completion probability,
    opportunity screening, and historical deal performance tracking.
    """

    # ------------------------------------------------------------------
    # Public: compute_arb_spread
    # ------------------------------------------------------------------

    def compute_arb_spread(
        self,
        target_ticker:       str,
        offer_price:         float,
        deal_close_date:     str,
        target_current_price: Optional[float] = None,
        deal_type:           str = "unknown",
    ) -> dict:
        """
        Compute merger arbitrage spread metrics.

        Args:
            target_ticker:        Target company ticker symbol.
            offer_price:          Per-share consideration in the deal.
            deal_close_date:      Expected close date (ISO YYYY-MM-DD).
            target_current_price: Current market price.  If None, uses a
                                  synthetic estimate of offer_price * 0.96.
            deal_type:            One of MA_DEAL_TYPES keys.

        Returns:
            Dict with gross_spread_pct, annualized_spread_pct,
            implied_close_probability, days_to_close.
        """
        if target_current_price is None or target_current_price <= 0:
            # Default: assume market is pricing ~4% risk discount
            target_current_price = offer_price * 0.96

        today = date.today()
        try:
            close_date = date.fromisoformat(deal_close_date)
        except ValueError:
            close_date = today + timedelta(days=90)

        days_to_close = max((close_date - today).days, 1)

        if offer_price <= 0:
            return {"error": "offer_price must be positive"}

        gross_spread      = offer_price - target_current_price
        gross_spread_pct  = gross_spread / target_current_price * 100.0
        annualized_spread = gross_spread_pct * (365.0 / days_to_close)

        # Implied probability: solve spread = (1-p) * loss_if_fail + p * 0
        # Assume failure → price falls 20% below current (deal premium reversal)
        fail_drop_pct  = 0.20
        loss_if_fail   = target_current_price * fail_drop_pct
        # gross_spread = p * gross_spread - (1-p) * loss_if_fail
        # gross_spread = p * (gross_spread + loss_if_fail) - loss_if_fail
        # p = (gross_spread + loss_if_fail) / (gross_spread + loss_if_fail)
        denom = gross_spread + loss_if_fail
        implied_prob = max(0.0, min(1.0, (gross_spread + loss_if_fail) / denom)) if denom > 0 else 0.5

        base_rate = _BASE_COMPLETION_RATES.get(deal_type, 0.80)

        return {
            "target_ticker":          target_ticker.upper(),
            "deal_type":              deal_type,
            "offer_price":            round(offer_price, 4),
            "current_price":          round(target_current_price, 4),
            "gross_spread":           round(gross_spread, 4),
            "gross_spread_pct":       round(gross_spread_pct, 4),
            "annualized_spread_pct":  round(annualized_spread, 4),
            "days_to_close":          days_to_close,
            "expected_close_date":    deal_close_date,
            "implied_close_probability": round(implied_prob, 4),
            "historical_base_rate":   base_rate,
            "deal_risk_premium_pct":  round(gross_spread_pct - (base_rate * 5), 4),
        }

    # ------------------------------------------------------------------
    # Public: deal_completion_probability
    # ------------------------------------------------------------------

    def deal_completion_probability(self, deal: dict) -> float:
        """
        Score deal completion probability from 0.0 to 1.0.

        Factors: regulatory risk, financing risk, hostile bid, termination fee,
        deal type base rates.

        Args:
            deal: Dict with keys: deal_type, requires_cfius, requires_eu_comp,
                  requires_hsr, termination_fee_mm, deal_value_billions,
                  premium_pct, acquirer (optional), target (optional).

        Returns:
            Float in [0.0, 1.0].
        """
        deal_type = deal.get("deal_type", "unknown")
        score     = _BASE_COMPLETION_RATES.get(deal_type, 0.80)

        # Regulatory adjustments
        if deal.get("requires_cfius"):
            score -= 0.08   # CFIUS adds meaningful failure risk
        if deal.get("requires_eu_comp"):
            score -= 0.05   # EU DG COMP remedies risk
        if deal.get("requires_hsr"):
            score -= 0.02   # HSR review is routine but adds risk

        # Termination fee coverage
        term_fee    = deal.get("termination_fee_mm") or 0.0
        deal_val_mm = (deal.get("deal_value_billions") or 0.0) * 1_000.0
        if deal_val_mm > 0:
            fee_pct = term_fee / deal_val_mm
            # Higher termination fee → more committed parties
            if fee_pct >= 0.04:
                score += 0.03
            elif fee_pct >= 0.02:
                score += 0.01
        else:
            # No deal value info → slight penalty for uncertainty
            score -= 0.02

        # Premium signal: very high premiums occasionally signal desperation
        premium = deal.get("premium_pct") or 0.0
        if premium > 60.0:
            score -= 0.03
        elif premium > 40.0:
            score -= 0.01

        # Hostile / unsolicited
        if deal_type == "hostile":
            score -= 0.15

        # LBO financing risk
        if deal_type == "lbo":
            score -= 0.05

        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Public: screen_arb_opportunities
    # ------------------------------------------------------------------

    async def screen_arb_opportunities(
        self,
        min_spread_pct:  float = 2.0,
        max_days_to_close: int = 180,
    ) -> pd.DataFrame:
        """
        Scan pending M&A deals and return those with attractive arb spreads.

        Args:
            min_spread_pct:    Minimum gross spread percentage (default 2%).
            max_days_to_close: Maximum days until expected close (default 180).

        Returns:
            pd.DataFrame with arb metrics for qualifying opportunities.
        """
        scraper = MADealScraper()
        pending_df = await scraper.get_pending_deals()

        if pending_df.empty:
            return pd.DataFrame(columns=[
                "target", "deal_type", "gross_spread_pct", "annualized_spread_pct",
                "days_to_close", "completion_probability", "consideration_per_share",
            ])

        rows: list[dict] = []
        for _, row in pending_df.iterrows():
            offer_price = row.get("consideration_per_share")
            if not offer_price or float(offer_price) <= 0:
                continue

            # Estimate days to close from filing date + deal type average
            filing_date_str = row.get("filing_date", date.today().isoformat())
            try:
                filing_date = date.fromisoformat(str(filing_date_str)[:10])
            except ValueError:
                filing_date = date.today()

            deal_type = row.get("deal_type", "unknown")
            avg_days  = {"all_cash": 100, "merger": 150, "all_stock": 160, "lbo": 180}.get(deal_type, 120)
            close_date = (filing_date + timedelta(days=avg_days)).isoformat()

            today = date.today()
            days_to_close = max((date.fromisoformat(close_date) - today).days, 1)
            if days_to_close > max_days_to_close:
                continue

            spread_data = self.compute_arb_spread(
                target_ticker        = str(row.get("target", "UNK"))[:8],
                offer_price          = float(offer_price),
                deal_close_date      = close_date,
                deal_type            = deal_type,
            )
            if spread_data.get("gross_spread_pct", 0.0) < min_spread_pct:
                continue

            prob = self.deal_completion_probability(row.to_dict())
            rows.append({
                "target":                  row.get("target"),
                "cik":                     row.get("cik"),
                "deal_type":               deal_type,
                "filing_date":             filing_date_str,
                "offer_price":             float(offer_price),
                "gross_spread_pct":        spread_data["gross_spread_pct"],
                "annualized_spread_pct":   spread_data["annualized_spread_pct"],
                "days_to_close":           days_to_close,
                "completion_probability":  prob,
                "requires_cfius":          row.get("requires_cfius", False),
                "requires_eu_comp":        row.get("requires_eu_comp", False),
                "accession_number":        row.get("accession_number"),
                "edgar_url":               row.get("edgar_url"),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("annualized_spread_pct", ascending=False).reset_index(drop=True)
        logger.info("ma_intelligence: screen_arb_opportunities found=%d", len(df))
        return df

    # ------------------------------------------------------------------
    # Public: historical_deal_performance
    # ------------------------------------------------------------------

    async def historical_deal_performance(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Track completed and failed deals from EDGAR to compute success rates.

        Args:
            lookback_days: Calendar days to look back.

        Returns:
            pd.DataFrame with deal outcomes and aggregate success rate by type.
        """
        scraper  = MADealScraper()
        all_deals = await scraper.get_edgar_ma_filings(lookback_days=lookback_days)

        if not all_deals:
            return pd.DataFrame(columns=["deal_type", "status", "count", "success_rate"])

        df = pd.DataFrame(all_deals)

        # Aggregate by deal_type and status
        if "status" not in df.columns or "deal_type" not in df.columns:
            return df

        summary = (
            df.groupby(["deal_type", "status"])
            .size()
            .reset_index(name="count")
        )

        # Compute success rates per deal type
        rates: list[dict] = []
        for dt in df["deal_type"].unique():
            subset = df[df["deal_type"] == dt]
            total      = len(subset)
            completed  = len(subset[subset["status"] == "completed"])
            terminated = len(subset[subset["status"] == "terminated"])
            pending    = total - completed - terminated
            success_rate = completed / (completed + terminated) if (completed + terminated) > 0 else None
            rates.append({
                "deal_type":    dt,
                "total_deals":  total,
                "completed":    completed,
                "terminated":   terminated,
                "pending":      pending,
                "success_rate": round(success_rate, 4) if success_rate is not None else None,
            })

        result_df = pd.DataFrame(rates).sort_values("total_deals", ascending=False).reset_index(drop=True)
        logger.info("ma_intelligence: historical_deal_performance lookback=%d deal_types=%d", lookback_days, len(rates))
        return result_df


# ---------------------------------------------------------------------------
# MATimeline
# ---------------------------------------------------------------------------

class MATimeline:
    """
    Builds chronological deal event timelines and retrieves regulatory status
    for pending and completed M&A transactions.
    """

    # Regulatory review windows (business days approximation)
    _REG_WINDOWS = {
        "hsr":     30,   # HSR initial waiting period
        "cfius":   75,   # CFIUS average (30-day initial + 45-day investigation)
        "eu_comp": 90,   # EU Phase I (25 WD) or Phase II (~90 WD)
    }

    # ------------------------------------------------------------------
    # Public: build_deal_timeline
    # ------------------------------------------------------------------

    async def build_deal_timeline(self, target_ticker: str) -> list[dict]:
        """
        Build a chronological sequence of deal events for a target company.

        Pulls EDGAR filings for the ticker and reconstructs: announcement,
        proxy filing, regulatory filings, shareholder vote, and completion.

        Args:
            target_ticker: Exchange ticker of the acquisition target.

        Returns:
            List of dicts sorted by date with fields:
            event_type, date, description, form_type, accession_number, edgar_url.
        """
        warnings_: list[str] = []
        ticker    = target_ticker.upper()
        events:   list[dict] = []

        async with httpx.AsyncClient() as client:
            cik = await _resolve_ticker_to_cik(client, ticker)
            if cik is None:
                logger.warning("ma_intelligence: cannot resolve CIK for ticker %s", ticker)
                return [{"error": f"Could not resolve CIK for {ticker}", "warnings": warnings_}]

            # Fetch submissions index
            subs_url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
            data     = await _get(client, subs_url, warnings_)

        if not data:
            return [{"error": "Submissions unavailable", "cik": cik}]

        company_name = data.get("name", ticker)
        filings_block = data.get("filings", {}).get("recent", {})

        forms      = filings_block.get("form", [])
        dates      = filings_block.get("filingDate", [])
        accessions = filings_block.get("accessionNumber", [])
        cik_plain  = cik.lstrip("0") or cik

        _deal_form_events = {
            "8-K":      "merger_announcement",
            "8-K/A":    "merger_announcement_amendment",
            "PREM14A":  "preliminary_proxy",
            "DEFM14A":  "definitive_proxy",
            "SC TO-T":  "tender_offer_commenced",
            "SC TO-T/A":"tender_offer_amendment",
            "SC 13E-3": "going_private_filing",
            "15-12G":   "deregistration",
            "15-12B":   "deregistration",
            "425":      "prospectus_supplement",
        }

        cutoff = date.today() - timedelta(days=730)

        for i, form in enumerate(forms):
            if form.upper() not in _deal_form_events:
                continue
            try:
                filing_date = date.fromisoformat((dates[i] if i < len(dates) else "")[:10])
            except (ValueError, TypeError):
                continue
            if filing_date < cutoff:
                continue

            acc = accessions[i] if i < len(accessions) else ""
            acc_nodash = acc.replace("-", "")
            events.append({
                "event_type":       _deal_form_events[form.upper()],
                "date":             filing_date.isoformat(),
                "form_type":        form,
                "description":      f"{_deal_form_events[form.upper()].replace('_', ' ').title()} — {company_name}",
                "accession_number": acc,
                "edgar_url":        f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nodash}/",
                "company":          company_name,
                "ticker":           ticker,
            })

        # Sort chronologically
        events.sort(key=lambda e: e.get("date", ""), reverse=False)

        # Inject synthetic regulatory events if deal-relevant forms present
        has_merger = any(e["form_type"] in ("DEFM14A", "SC TO-T") for e in events)
        if has_merger and events:
            announcement_date_str = next(
                (e["date"] for e in events if e["form_type"] in ("8-K", "DEFM14A")),
                events[0]["date"],
            )
            try:
                ann_date = date.fromisoformat(announcement_date_str)
            except ValueError:
                ann_date = date.today()

            hsr_date  = ann_date + timedelta(days=5)
            events.append({
                "event_type":  "hsr_filing_expected",
                "date":        hsr_date.isoformat(),
                "form_type":   "HSR",
                "description": "Hart-Scott-Rodino antitrust filing expected (T+5 days post-announcement)",
                "accession_number": "",
                "edgar_url":   "",
                "company":     company_name,
                "ticker":      ticker,
            })

        events.sort(key=lambda e: e.get("date", ""))
        logger.info("ma_intelligence: build_deal_timeline ticker=%s events=%d", ticker, len(events))
        return events

    # ------------------------------------------------------------------
    # Public: get_regulatory_status
    # ------------------------------------------------------------------

    def get_regulatory_status(self, deal: dict) -> dict:
        """
        Determine regulatory review status and expected timelines for a deal.

        Args:
            deal: Deal dict (output of MADealScraper methods) with keys:
                  requires_hsr, requires_cfius, requires_eu_comp,
                  filing_date, deal_value_billions, acquirer, target.

        Returns:
            Dict with per-regulator status, expected clearance dates, and risk level.
        """
        filing_date_str = deal.get("filing_date", date.today().isoformat())
        try:
            announcement = date.fromisoformat(str(filing_date_str)[:10])
        except ValueError:
            announcement = date.today()

        today  = date.today()
        status: dict[str, Any] = {
            "deal_acquirer":    deal.get("acquirer"),
            "deal_target":      deal.get("target"),
            "announcement_date": announcement.isoformat(),
            "regulators":       {},
        }

        # HSR (Hart-Scott-Rodino)
        if deal.get("requires_hsr", False):
            hsr_filing_date    = announcement + timedelta(days=5)
            hsr_clearance_date = hsr_filing_date + timedelta(days=self._REG_WINDOWS["hsr"])
            status["regulators"]["hsr"] = {
                "regulator":           "DOJ / FTC (Hart-Scott-Rodino)",
                "filing_expected":     hsr_filing_date.isoformat(),
                "clearance_expected":  hsr_clearance_date.isoformat(),
                "days_remaining":      max((hsr_clearance_date - today).days, 0),
                "status":              "pending" if hsr_clearance_date > today else "cleared",
                "risk_level":          self._hsr_risk(deal),
            }

        # CFIUS (foreign investment review)
        if deal.get("requires_cfius", False):
            cfius_filing  = announcement + timedelta(days=10)
            cfius_clear   = cfius_filing + timedelta(days=self._REG_WINDOWS["cfius"])
            status["regulators"]["cfius"] = {
                "regulator":           "CFIUS (Committee on Foreign Investment in the US)",
                "filing_expected":     cfius_filing.isoformat(),
                "clearance_expected":  cfius_clear.isoformat(),
                "days_remaining":      max((cfius_clear - today).days, 0),
                "status":              "pending" if cfius_clear > today else "cleared",
                "risk_level":          "high",
                "note":                "CFIUS review applies to foreign acquirers of US businesses",
            }

        # EU DG COMP
        if deal.get("requires_eu_comp", False):
            eu_filing  = announcement + timedelta(days=15)
            eu_clear   = eu_filing + timedelta(days=self._REG_WINDOWS["eu_comp"])
            status["regulators"]["eu_dg_comp"] = {
                "regulator":          "European Commission DG COMP",
                "filing_expected":    eu_filing.isoformat(),
                "clearance_expected": eu_clear.isoformat(),
                "days_remaining":     max((eu_clear - today).days, 0),
                "status":             "pending" if eu_clear > today else "cleared",
                "risk_level":         self._eu_risk(deal),
            }

        if not status["regulators"]:
            status["regulators"]["domestic"] = {
                "regulator":   "State/local approvals only",
                "status":      "minimal_review",
                "risk_level":  "low",
            }

        # Overall risk
        risk_levels = [r.get("risk_level", "low") for r in status["regulators"].values()]
        status["overall_regulatory_risk"] = (
            "high" if "high" in risk_levels else
            "medium" if "medium" in risk_levels else
            "low"
        )
        return status

    def _hsr_risk(self, deal: dict) -> str:
        dv = deal.get("deal_value_billions") or 0.0
        if dv > 50.0:
            return "high"
        if dv > 10.0:
            return "medium"
        return "low"

    def _eu_risk(self, deal: dict) -> str:
        dv = deal.get("deal_value_billions") or 0.0
        return "high" if dv > 5.0 else "medium"


# ---------------------------------------------------------------------------
# Convenience async runner
# ---------------------------------------------------------------------------

def run_ma_intelligence(lookback_days: int = 90) -> dict:
    """
    Synchronous entry point: run full M&A intelligence sweep.

    Returns dict with: filings, pending_deals (as records), arb_opportunities.
    """
    async def _run() -> dict:
        scraper  = MADealScraper()
        analyzer = MergerArbAnalyzer()

        filings, pending_df, arb_df = await asyncio.gather(
            scraper.get_edgar_ma_filings(lookback_days),
            scraper.get_pending_deals(),
            analyzer.screen_arb_opportunities(),
        )
        return {
            "filings":          filings,
            "pending_deals":    pending_df.to_dict(orient="records") if not pending_df.empty else [],
            "arb_opportunities": arb_df.to_dict(orient="records") if not arb_df.empty else [],
            "as_of":            date.today().isoformat(),
        }

    return asyncio.run(_run())
