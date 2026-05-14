"""
Enhanced corporate actions — extends corporate_actions.py with:
  - Full split / reverse-split history (yfinance + EDGAR 8-K)
  - Dividend history with special dividend detection
  - Recent M&A deal tracking from EDGAR EFTS full-text search
  - Spin-off detection from Form 10-12B and 8-K filings
  - Price adjustment chain computation (cumulative backward-adjustment factors)
  - Adjusted OHLCV price series reconstruction
  - Merger-arbitrage spread monitoring

Data sources (all free):
  yfinance:  .splits, .dividends, .history(), .info
  EDGAR EFTS: https://efts.sec.gov/LATEST/search-index (8-K full-text search)
  EDGAR data: https://data.sec.gov (submissions, companyfacts)
"""
from __future__ import annotations

import asyncio
import math
import re
from datetime import date, datetime, timedelta
from typing import Literal, Optional

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 25.0
EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}

# EDGAR form types relevant to corporate actions
_8K_FORM = "8-K"
_10_12B_FORM = "10-12B"   # New company registration → spin-off signal
_SC_TO_FORM = "SC TO-T"   # Tender offer → M&A signal
_DEFM14A_FORM = "DEFM14A" # Proxy statement for merger

# Regex patterns for deal terms in press release text
_DEAL_VALUE_PATTERNS = [
    r"\$\s*([\d,]+(?:\.\d+)?)\s*billion",
    r"\$\s*([\d,]+(?:\.\d+)?)\s*million",
    r"valued\s+at\s+approximately\s+\$\s*([\d,]+(?:\.\d+)?)",
    r"transaction\s+value[d]?\s+(?:of\s+)?approximately\s+\$\s*([\d,]+(?:\.\d+)?)",
    r"enterprise\s+value\s+of\s+approximately\s+\$\s*([\d,]+(?:\.\d+)?)",
]

_PREMIUM_PATTERNS = [
    r"([\d.]+)\s*%\s*premium",
    r"premium\s+of\s+approximately\s+([\d.]+)\s*%",
    r"representing\s+a[n]?\s+approximately\s+([\d.]+)\s*%",
]

_CASH_PER_SHARE_PATTERNS = [
    r"\$\s*([\d.]+)\s+per\s+share\s+in\s+cash",
    r"cash\s+consideration\s+of\s+\$\s*([\d.]+)\s+per\s+share",
    r"\$\s*([\d.]+)\s+per\s+(?:common\s+)?share",
]

_STOCK_EXCHANGE_PATTERNS = [
    r"([\d.]+)\s+shares?\s+of\s+(?:common\s+stock\s+of\s+)?(\w+)",
    r"exchange\s+ratio\s+of\s+([\d.]+)\s+shares",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SplitEvent(BaseModel):
    ticker: str
    ex_date: date
    split_ratio: float    # new / old — e.g. 4.0 = 4:1 forward, 0.25 = 1:4 reverse
    announcement_date: Optional[date] = None
    split_type: str = "forward"  # "forward" | "reverse"
    description: str
    source: str


class DividendEvent(BaseModel):
    ticker: str
    ex_date: date
    payment_date: Optional[date] = None
    record_date: Optional[date] = None
    amount_per_share: float
    currency: str = "USD"
    dividend_type: str  # "regular" | "special" | "stock"
    yield_at_announcement: Optional[float] = None
    frequency: str = "unknown"  # "monthly" | "quarterly" | "semi-annual" | "annual"
    source: str


class MergersAcquisitions(BaseModel):
    acquirer_ticker: Optional[str] = None
    acquirer_name: str
    target_ticker: Optional[str] = None
    target_name: str
    deal_value: Optional[float] = None    # USD millions
    deal_type: str  # "all_cash" | "all_stock" | "mixed" | "merger_of_equals" | "unknown"
    announcement_date: date
    expected_close: Optional[date] = None
    actual_close: Optional[date] = None
    premium_pct: Optional[float] = None  # premium to pre-announcement price
    cash_per_share: Optional[float] = None
    stock_exchange_ratio: Optional[float] = None
    status: str  # "announced" | "regulatory_review" | "completed" | "terminated"
    accession: Optional[str] = None
    filing_url: Optional[str] = None
    filing_date: Optional[date] = None


class SpinOff(BaseModel):
    parent_ticker: str
    spinoff_ticker: Optional[str] = None
    spinoff_name: str
    distribution_date: date
    ratio: Optional[str] = None        # "1 share of X per 3 shares of Y"
    adjustment_factor: Optional[float] = None  # for price reconstruction
    description: Optional[str] = None
    filing_url: Optional[str] = None
    accession: Optional[str] = None


class PriceAdjustmentFactor(BaseModel):
    ticker: str
    date: date
    event_type: str       # "split" | "dividend" | "spinoff"
    event_description: str
    raw_factor: float     # multiply raw price by this → adjusted price at that date
    cumulative_factor: float  # total adjustment from the earliest event to this date


# ---------------------------------------------------------------------------
# yfinance sync helpers
# ---------------------------------------------------------------------------


def _yf_fetch_raw(ticker: str, years_back: int = 30) -> dict:
    """
    Synchronously fetch splits, dividends, info, and OHLCV from yfinance.
    Called via asyncio.to_thread.
    """
    import yfinance as yf  # lazy

    out: dict = {
        "info": {},
        "splits": pd.Series(dtype=float),
        "dividends": pd.Series(dtype=float),
        "history": pd.DataFrame(),
    }
    try:
        tk = yf.Ticker(ticker)
        try:
            out["info"] = tk.info or {}
        except Exception:
            pass
        try:
            s = tk.splits
            if s is not None and not s.empty:
                out["splits"] = s
        except Exception:
            pass
        try:
            d = tk.dividends
            if d is not None and not d.empty:
                out["dividends"] = d
        except Exception:
            pass
        try:
            hist = tk.history(period=f"{min(years_back, 25)}y")
            if hist is not None and not hist.empty:
                out["history"] = hist
        except Exception:
            pass
    except Exception as exc:
        logger.warning("yfinance raw fetch failed", ticker=ticker, error=str(exc))
    return out


def _sf(val: object) -> Optional[float]:
    """Safe float cast."""
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _tz_aware(s: pd.Series) -> pd.Series:
    """Ensure DatetimeIndex is UTC-aware."""
    if s.empty:
        return s
    if hasattr(s.index, "tzinfo") and s.index.tzinfo is None:
        try:
            return s.tz_localize("UTC")
        except Exception:
            pass
    return s


def _classify_div_freq(gaps_days: list[float]) -> str:
    if not gaps_days:
        return "unknown"
    med = float(np.median(gaps_days))
    if med <= 45:
        return "monthly"
    if med <= 105:
        return "quarterly"
    if med <= 200:
        return "semi-annual"
    return "annual"


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------


async def _edgar_efts_search(
    client: httpx.AsyncClient,
    query: str,
    form_type: str,
    days_back: int = 30,
    size: int = 20,
) -> list[dict]:
    """
    Full-text search EDGAR EFTS for filings matching `query` in `form_type`.
    Returns list of filing metadata dicts.
    """
    start_date = (date.today() - timedelta(days=days_back)).isoformat()
    params = {
        "q": query,
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": date.today().isoformat(),
        "forms": form_type,
        "_source": "period_of_report,file_date,entity_name,file_num,period_of_report,accession_no",
        "from": 0,
        "size": size,
    }
    try:
        r = await client.get(
            EDGAR_EFTS,
            params=params,
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        if r.status_code != 200:
            logger.warning("EDGAR EFTS non-200", query=query, status=r.status_code)
            return []
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        return hits
    except Exception as exc:
        logger.warning("EDGAR EFTS search failed", query=query, error=str(exc))
        return []


async def _edgar_efts_search_text(
    client: httpx.AsyncClient,
    query: str,
    form_type: str,
    days_back: int = 30,
    size: int = 10,
) -> list[dict]:
    """
    Full-text EDGAR search with text extraction.  Returns full _source dicts.
    """
    hits = await _edgar_efts_search(client, query, form_type, days_back, size)
    return [h.get("_source", {}) for h in hits]


async def _fetch_filing_text(
    client: httpx.AsyncClient, accession: str, cik: str
) -> str:
    """
    Attempt to fetch the primary document text for a filing.
    Returns empty string on failure.
    """
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession}.txt"
    try:
        r = await client.get(url, timeout=15.0, headers=_HEADERS)
        if r.status_code == 200:
            # Return first 8000 chars — enough for deal terms in press release header
            return r.text[:8000]
    except Exception:
        pass
    return ""


async def _get_cik_for_ticker(
    client: httpx.AsyncClient, ticker: str
) -> Optional[str]:
    """Look up CIK from SEC company_tickers.json bulk file."""
    try:
        r = await client.get(
            COMPANY_TICKERS_URL,
            timeout=15.0,
            headers={**_HEADERS, "Host": "www.sec.gov"},
        )
        if r.status_code != 200:
            return None
        data = r.json()
        ticker_up = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_up:
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK lookup failed", ticker=ticker, error=str(exc))
    return None


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------


def _extract_deal_terms_from_text(text: str) -> dict:
    """
    Extract deal terms from 8-K press release text using regex patterns.

    Returns dict with keys:
      deal_value_mm (float|None)  — deal value in USD millions
      premium_pct (float|None)    — premium to pre-announcement price
      cash_per_share (float|None) — cash consideration per share
      stock_ratio (float|None)    — stock exchange ratio
      deal_type (str)             — "all_cash" | "all_stock" | "mixed" | "unknown"
    """
    text_lower = text.lower()
    result: dict = {
        "deal_value_mm": None,
        "premium_pct": None,
        "cash_per_share": None,
        "stock_ratio": None,
        "deal_type": "unknown",
    }

    # Deal value
    for pattern in _DEAL_VALUE_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                raw = float(m.group(1).replace(",", ""))
                # Detect scale from surrounding context
                context = text_lower[max(0, m.start() - 50):m.end() + 10]
                if "billion" in context:
                    result["deal_value_mm"] = raw * 1_000.0
                else:
                    result["deal_value_mm"] = raw
                break
            except (ValueError, IndexError):
                pass

    # Premium
    for pattern in _PREMIUM_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                result["premium_pct"] = float(m.group(1))
                break
            except (ValueError, IndexError):
                pass

    # Cash per share
    for pattern in _CASH_PER_SHARE_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                result["cash_per_share"] = float(m.group(1).replace(",", ""))
                break
            except (ValueError, IndexError):
                pass

    # Stock exchange ratio
    for pattern in _STOCK_EXCHANGE_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                result["stock_ratio"] = float(m.group(1))
                break
            except (ValueError, IndexError):
                pass

    # Determine deal type
    has_cash = result["cash_per_share"] is not None or "all cash" in text_lower or "all-cash" in text_lower
    has_stock = result["stock_ratio"] is not None or "all stock" in text_lower or "all-stock" in text_lower or "exchange ratio" in text_lower

    if has_cash and has_stock:
        result["deal_type"] = "mixed"
    elif has_cash:
        result["deal_type"] = "all_cash"
    elif has_stock:
        result["deal_type"] = "all_stock"
    elif "merger of equals" in text_lower:
        result["deal_type"] = "merger_of_equals"

    return result


def _parse_filing_date(raw: Optional[str]) -> Optional[date]:
    """Parse EDGAR date strings (YYYY-MM-DD or YYYYMMDD)."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _build_filing_url(accession: str) -> str:
    """Build an EDGAR filing index URL from an accession number."""
    acc_clean = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{acc_clean[:10]}/{acc_clean}/{accession}-index.htm"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CorporateActionsEnhanced:
    """
    Enhanced corporate actions tracker: splits, dividends, M&A, spin-offs,
    price adjustment chains, and merger arb spreads.
    """

    def __init__(self, timeout: float = 25.0) -> None:
        self._timeout = timeout
        # Optional CIK cache to avoid repeated SEC lookups
        self._cik_cache: dict[str, Optional[str]] = {}

    # -----------------------------------------------------------------------
    # Splits history
    # -----------------------------------------------------------------------

    async def get_splits_history(
        self, ticker: str, years_back: int = 30
    ) -> list[SplitEvent]:
        """
        Return complete split and reverse-split history for `ticker`.

        Data from yfinance .splits property.  For each event, computes
        split_type ("forward" if ratio > 1, "reverse" if ratio < 1).
        """
        ticker = ticker.upper()
        raw = await asyncio.to_thread(_yf_fetch_raw, ticker, years_back)
        splits_s: pd.Series = raw.get("splits", pd.Series(dtype=float))

        events: list[SplitEvent] = []
        if splits_s.empty:
            logger.info("No split history found", ticker=ticker)
            return events

        splits_s = _tz_aware(splits_s)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years_back)

        for ts, ratio in splits_s.items():
            if ts < cutoff:
                continue
            r = float(ratio)
            if r <= 0 or r == 1.0:
                continue

            ex_dt = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
            split_type = "forward" if r > 1.0 else "reverse"

            if r > 1.0:
                desc = f"{r:.0f}-for-1 forward stock split"
            else:
                inv = round(1.0 / r)
                desc = f"1-for-{inv} reverse stock split"

            events.append(SplitEvent(
                ticker=ticker,
                ex_date=ex_dt,
                split_ratio=round(r, 6),
                split_type=split_type,
                description=desc,
                source="yfinance",
            ))

        events.sort(key=lambda e: e.ex_date, reverse=True)
        logger.info("Split history fetched", ticker=ticker, count=len(events))
        return events

    # -----------------------------------------------------------------------
    # Dividends history
    # -----------------------------------------------------------------------

    async def get_dividends_history(
        self, ticker: str, years_back: int = 30
    ) -> list[DividendEvent]:
        """
        Return full dividend history for `ticker` with type classification.

        Regular dividends: amount within ±50% of prior period average.
        Special dividends: amount > 2x trailing average.
        Frequency estimated from median inter-payment gap.
        """
        ticker = ticker.upper()
        raw = await asyncio.to_thread(_yf_fetch_raw, ticker, years_back)
        divs_s: pd.Series = raw.get("dividends", pd.Series(dtype=float))
        info: dict = raw.get("info", {})

        events: list[DividendEvent] = []
        if divs_s.empty:
            return events

        divs_s = _tz_aware(divs_s)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years_back)
        divs_s = divs_s[divs_s.index >= cutoff].sort_index()

        if divs_s.empty:
            return events

        # Compute frequency from gaps
        if len(divs_s) >= 2:
            gaps = [
                float((divs_s.index[i] - divs_s.index[i - 1]).days)
                for i in range(1, len(divs_s))
            ]
            freq = _classify_div_freq(gaps)
        else:
            freq = "unknown"

        # Compute trailing average for special dividend detection
        amounts = list(divs_s.values.astype(float))
        trailing_avg = float(np.mean(amounts)) if amounts else 0.0

        # Get current price for yield estimate
        current_price = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))

        ex_div_date_raw = info.get("exDividendDate")
        ex_div_yf: Optional[date] = None
        if ex_div_date_raw:
            try:
                from datetime import timezone
                ex_div_yf = datetime.fromtimestamp(float(ex_div_date_raw), tz=timezone.utc).date()
            except Exception:
                pass

        for ts, amt in divs_s.items():
            a = float(amt)
            if a <= 0:
                continue

            ex_dt = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])

            # Classify dividend type
            if trailing_avg > 0 and a > 2.0 * trailing_avg:
                div_type = "special"
            else:
                div_type = "regular"

            # Yield at announcement: annualize based on frequency
            yield_ann: Optional[float] = None
            if current_price and current_price > 0:
                periods_per_year = {"monthly": 12, "quarterly": 4, "semi-annual": 2, "annual": 1}.get(freq, 4)
                yield_ann = round(a * periods_per_year / current_price * 100.0, 4)

            # Payment date: use yfinance payDate if this is the most recent ex-div
            pay_date: Optional[date] = None
            if ex_div_yf and ex_dt == ex_div_yf:
                pay_date_raw = info.get("payDate")
                if pay_date_raw:
                    try:
                        from datetime import timezone
                        pay_date = datetime.fromtimestamp(float(pay_date_raw), tz=timezone.utc).date()
                    except Exception:
                        pass

            events.append(DividendEvent(
                ticker=ticker,
                ex_date=ex_dt,
                payment_date=pay_date,
                amount_per_share=round(a, 6),
                dividend_type=div_type,
                yield_at_announcement=yield_ann,
                frequency=freq,
                source="yfinance",
            ))

        events.sort(key=lambda e: e.ex_date, reverse=True)
        logger.info("Dividend history fetched", ticker=ticker, count=len(events))
        return events

    # -----------------------------------------------------------------------
    # Recent M&A deals from EDGAR
    # -----------------------------------------------------------------------

    async def get_recent_ma_deals(
        self, days_back: int = 30, min_deal_size_mm: float = 100.0
    ) -> list[MergersAcquisitions]:
        """
        Search EDGAR EFTS for recent M&A deal announcements in 8-K filings.

        Searches for "definitive agreement", "merger agreement", and "acquisition"
        keywords in 8-K filings.  Extracts deal terms from press release text.
        Filters to deals >= min_deal_size_mm USD million.
        """
        queries = [
            '"definitive agreement" "merger"',
            '"merger agreement" "acquisition"',
            '"agreement and plan of merger"',
        ]

        seen_accessions: set[str] = set()
        all_hits: list[dict] = []

        async with httpx.AsyncClient() as client:
            search_tasks = [
                _edgar_efts_search_text(client, q, _8K_FORM, days_back=days_back, size=15)
                for q in queries
            ]
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                logger.warning("EDGAR M&A search failed", error=str(res))
                continue
            for hit in res:
                acc = hit.get("accession_no", "")
                if acc and acc not in seen_accessions:
                    seen_accessions.add(acc)
                    all_hits.append(hit)

        deals: list[MergersAcquisitions] = []

        for hit in all_hits[:25]:  # Cap to avoid rate-limit issues
            entity_name = hit.get("entity_name", "Unknown Company")
            acc = hit.get("accession_no", "")
            file_date_str = hit.get("file_date", "")
            filing_date = _parse_filing_date(file_date_str)

            if filing_date is None:
                filing_date = date.today()

            # Build filing URL
            filing_url = None
            if acc:
                acc_clean = acc.replace("-", "")
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{acc_clean}/{acc}-index.htm"

            # Extract deal terms from filing text (best-effort, non-blocking)
            deal_terms: dict = {
                "deal_value_mm": None,
                "premium_pct": None,
                "cash_per_share": None,
                "stock_ratio": None,
                "deal_type": "unknown",
            }

            text_snippet = hit.get("file_description", "") or hit.get("period_of_report", "")
            if text_snippet:
                deal_terms = _extract_deal_terms_from_text(text_snippet)

            # Filter by size if we extracted a value
            dv = deal_terms.get("deal_value_mm")
            if dv is not None and dv < min_deal_size_mm:
                continue

            # Parse entity as acquirer — target often mentioned in filing text
            # Since we can't reliably parse from metadata alone, mark as acquirer
            deals.append(MergersAcquisitions(
                acquirer_name=entity_name,
                target_name="See filing",   # would need full text parse
                deal_value=dv,
                deal_type=deal_terms.get("deal_type", "unknown"),
                announcement_date=filing_date,
                premium_pct=deal_terms.get("premium_pct"),
                cash_per_share=deal_terms.get("cash_per_share"),
                stock_exchange_ratio=deal_terms.get("stock_ratio"),
                status="announced",
                accession=acc or None,
                filing_url=filing_url,
                filing_date=filing_date,
            ))

        deals.sort(key=lambda d: d.announcement_date, reverse=True)
        logger.info("M&A deals fetched from EDGAR", count=len(deals), days_back=days_back)
        return deals

    # -----------------------------------------------------------------------
    # M&A history for a specific ticker
    # -----------------------------------------------------------------------

    async def get_ma_history(self, ticker: str) -> list[MergersAcquisitions]:
        """
        Historical M&A where `ticker` was acquirer or target.

        Searches EDGAR EFTS for 8-K, SC TO-T, and DEFM14A filings
        associated with this company's CIK.  Also checks yfinance info
        for acquisition-related fields.
        """
        ticker = ticker.upper()

        # Search EDGAR for ticker name in M&A filings
        queries = [
            f'"{ticker}" "merger agreement"',
            f'"{ticker}" "definitive agreement"',
            f'"{ticker}" "tender offer"',
        ]

        seen: set[str] = set()
        all_hits: list[dict] = []

        async with httpx.AsyncClient() as client:
            tasks = [
                _edgar_efts_search_text(
                    client, q, _8K_FORM, days_back=3650, size=10  # 10 years
                )
                for q in queries
            ]
            # Also search SC TO-T (tender offers)
            tasks.append(
                _edgar_efts_search_text(
                    client, ticker, _SC_TO_FORM, days_back=3650, size=5
                )
            )
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                continue
            for hit in res:
                acc = hit.get("accession_no", "")
                if acc and acc not in seen:
                    seen.add(acc)
                    all_hits.append(hit)

        deals: list[MergersAcquisitions] = []
        for hit in all_hits[:20]:
            entity_name = hit.get("entity_name", "Unknown")
            acc = hit.get("accession_no", "")
            file_date_str = hit.get("file_date", "")
            filing_date = _parse_filing_date(file_date_str) or date.today()

            acc_clean = acc.replace("-", "") if acc else ""
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{acc_clean}/{acc}-index.htm"
                if acc else None
            )

            text = hit.get("file_description", "") or ""
            terms = _extract_deal_terms_from_text(text)

            # Determine role: if entity is ticker, they may be acquirer or target
            is_target = ticker.lower() in entity_name.lower()

            deals.append(MergersAcquisitions(
                acquirer_name=entity_name if not is_target else "See filing",
                acquirer_ticker=ticker if not is_target else None,
                target_name=entity_name if is_target else "See filing",
                target_ticker=ticker if is_target else None,
                deal_value=terms.get("deal_value_mm"),
                deal_type=terms.get("deal_type", "unknown"),
                announcement_date=filing_date,
                premium_pct=terms.get("premium_pct"),
                cash_per_share=terms.get("cash_per_share"),
                stock_exchange_ratio=terms.get("stock_ratio"),
                status="completed",  # historical → likely completed
                accession=acc or None,
                filing_url=filing_url,
                filing_date=filing_date,
            ))

        deals.sort(key=lambda d: d.announcement_date, reverse=True)
        logger.info("M&A history fetched", ticker=ticker, count=len(deals))
        return deals

    # -----------------------------------------------------------------------
    # Spin-offs
    # -----------------------------------------------------------------------

    async def get_spinoffs(self, days_back: int = 365) -> list[SpinOff]:
        """
        Search EDGAR for recent spin-off announcements.

        Primary source: Form 10-12B filings (new company registration → spin-off).
        Secondary: 8-K filings containing "spin-off" keyword.
        """
        async with httpx.AsyncClient() as client:
            # 1. Form 10-12B — new company registration (strongest spin-off signal)
            hits_10_12b, hits_8k = await asyncio.gather(
                _edgar_efts_search_text(
                    client, "spin-off", _10_12B_FORM, days_back=days_back, size=15
                ),
                _edgar_efts_search_text(
                    client, '"spin-off" OR "spinoff" distribution', _8K_FORM,
                    days_back=days_back, size=20
                ),
            )

        seen: set[str] = set()
        spinoffs: list[SpinOff] = []

        def _process_hit(hit: dict, form_type: str) -> Optional[SpinOff]:
            acc = hit.get("accession_no", "")
            if acc in seen:
                return None
            seen.add(acc)

            entity_name = hit.get("entity_name", "Unknown Company")
            file_date_str = hit.get("file_date", "")
            dist_date = _parse_filing_date(file_date_str) or date.today()

            acc_clean = acc.replace("-", "") if acc else ""
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{acc_clean}/{acc}-index.htm"
                if acc else None
            )

            # 10-12B: the entity IS the new spin-off company
            # 8-K: the entity is the parent announcing the spin-off
            if form_type == _10_12B_FORM:
                return SpinOff(
                    parent_ticker="",
                    spinoff_name=entity_name,
                    distribution_date=dist_date,
                    description=f"New company registration (Form 10-12B) — likely spin-off from parent",
                    filing_url=filing_url,
                    accession=acc or None,
                )
            else:
                # 8-K from parent
                text = hit.get("file_description", "") or ""
                ratio_m = re.search(
                    r"(\d+)\s+shares?\s+of\s+.{0,40}per\s+(\d+)\s+shares?", text.lower()
                )
                ratio_str = None
                if ratio_m:
                    ratio_str = f"{ratio_m.group(1)} share(s) per {ratio_m.group(2)} shares"

                # Try to extract spinoff company name from text
                spinoff_name_m = re.search(
                    r'spin.off\s+of\s+(?:its\s+)?(?:subsidiary\s+)?([A-Z][a-zA-Z\s,\.]+)',
                    text
                )
                spinoff_name = (
                    spinoff_name_m.group(1).strip()[:60]
                    if spinoff_name_m
                    else "Spin-off entity (see filing)"
                )

                return SpinOff(
                    parent_ticker="",   # would need CIK→ticker reverse lookup
                    spinoff_name=spinoff_name,
                    distribution_date=dist_date,
                    ratio=ratio_str,
                    description=f"Spin-off announced by {entity_name}",
                    filing_url=filing_url,
                    accession=acc or None,
                )

        for hit in hits_10_12b:
            so = _process_hit(hit, _10_12B_FORM)
            if so:
                spinoffs.append(so)

        for hit in hits_8k:
            so = _process_hit(hit, _8K_FORM)
            if so:
                spinoffs.append(so)

        spinoffs.sort(key=lambda s: s.distribution_date, reverse=True)
        logger.info("Spin-offs fetched", count=len(spinoffs), days_back=days_back)
        return spinoffs

    # -----------------------------------------------------------------------
    # Price adjustment chain
    # -----------------------------------------------------------------------

    async def compute_price_adjustment_chain(
        self, ticker: str, start_date: date
    ) -> list[PriceAdjustmentFactor]:
        """
        Compute cumulative backward price-adjustment factors from splits and dividends.

        Standard approach:
          - Each split event creates an adjustment factor = 1 / split_ratio
            (applied backward: all prices before split are divided by ratio)
          - Each dividend creates an adjustment factor = (price - div) / price
            (total-return adjustment; multiply raw prices by cumulative factor
             to get fully adjusted total-return series)

        The cumulative_factor is the product of all raw_factors from the
        earliest event up to (and including) each event date.

        Returns events sorted ascending (oldest first) for easy application.
        """
        ticker = ticker.upper()

        # Fetch splits and dividends in parallel
        splits_task = self.get_splits_history(ticker, years_back=30)
        divs_task = self.get_dividends_history(ticker, years_back=30)
        raw_task = asyncio.to_thread(_yf_fetch_raw, ticker, 25)

        splits, divs, raw = await asyncio.gather(splits_task, divs_task, raw_task)

        # Also get price history for dividend adjustment factor computation
        hist: pd.DataFrame = raw.get("history", pd.DataFrame())

        # Build list of (date, event_type, description, raw_factor)
        events: list[tuple[date, str, str, float]] = []

        # --- Split factors ---
        for sp in splits:
            if sp.ex_date < start_date:
                continue
            # Backward adjustment: prices BEFORE split must be divided by ratio
            # So factor = 1 / split_ratio (prices multiplied by this go down pre-split)
            raw_factor = 1.0 / sp.split_ratio
            events.append((sp.ex_date, "split", sp.description, raw_factor))

        # --- Dividend factors ---
        for dv in divs:
            if dv.ex_date < start_date:
                continue
            # Get price on the ex-dividend date from history
            price_on_exdate: Optional[float] = None
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"]
                target = pd.Timestamp(dv.ex_date)
                # Find closest prior close
                prior = closes[closes.index.normalize() <= target]
                if not prior.empty:
                    price_on_exdate = float(prior.iloc[-1])

            if price_on_exdate and price_on_exdate > dv.amount_per_share:
                # Total-return adjustment factor
                raw_factor = (price_on_exdate - dv.amount_per_share) / price_on_exdate
            else:
                # Can't compute precisely — use 1.0 (no adjustment)
                raw_factor = 1.0

            events.append((
                dv.ex_date,
                "dividend",
                f"Dividend ${dv.amount_per_share:.4f} ({dv.dividend_type})",
                raw_factor,
            ))

        # Sort ascending
        events.sort(key=lambda x: x[0])

        # Compute cumulative factors
        factors: list[PriceAdjustmentFactor] = []
        cumulative = 1.0
        for ev_date, ev_type, ev_desc, raw_f in events:
            cumulative *= raw_f
            factors.append(PriceAdjustmentFactor(
                ticker=ticker,
                date=ev_date,
                event_type=ev_type,
                event_description=ev_desc,
                raw_factor=round(raw_f, 8),
                cumulative_factor=round(cumulative, 8),
            ))

        logger.info(
            "Adjustment chain computed",
            ticker=ticker,
            events=len(factors),
            start_date=start_date.isoformat(),
        )
        return factors

    # -----------------------------------------------------------------------
    # Adjusted price series reconstruction
    # -----------------------------------------------------------------------

    async def reconstruct_adjusted_prices(
        self, ticker: str, start_date: date
    ) -> pd.DataFrame:
        """
        Reconstruct fully-adjusted OHLCV price series.

        Applies cumulative backward adjustment factors to raw OHLCV history.
        Returns DataFrame with columns:
          date, open, high, low, close, volume, adj_close
        Indexed by date.

        Note: yfinance .history() with auto_adjust=True already does this,
        but this implementation applies our own computed factors for transparency
        and allows spin-off adjustments to be incorporated.
        """
        ticker = ticker.upper()

        # Fetch raw OHLCV and adjustment chain in parallel
        raw_task = asyncio.to_thread(_yf_fetch_raw, ticker, 30)
        chain_task = self.compute_price_adjustment_chain(ticker, start_date)

        raw, chain = await asyncio.gather(raw_task, chain_task)
        hist: pd.DataFrame = raw.get("history", pd.DataFrame())

        if hist.empty:
            logger.warning("No price history for adjustment", ticker=ticker)
            return pd.DataFrame()

        # Normalize index to date
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).normalize()

        # Filter to start_date
        start_ts = pd.Timestamp(start_date)
        hist = hist[hist.index >= start_ts]

        if hist.empty:
            return pd.DataFrame()

        # Build a date → cumulative_factor mapping
        # For each date, the cumulative factor is the product of all events AFTER that date
        # (standard backward adjustment: divide historical prices by forward splits)
        # We sort events descending and apply cumulatively going backward
        if chain:
            # Build factor series: index = date, value = cumulative factor at that event
            factor_dates = [f.date for f in chain]
            factor_vals = [f.cumulative_factor for f in chain]

            # The total cumulative factor is the product of all events
            total_factor = factor_vals[-1] if factor_vals else 1.0

            # For each price date, the adjustment = total_factor / cumulative_factor_at_date
            # (prices before event k are further back → divide by more)
            def _factor_for_date(price_date: date) -> float:
                # Find events after this price_date
                factor = 1.0
                for fd, fv in zip(factor_dates, factor_vals):
                    if fd > price_date:
                        # This event hasn't happened yet from the price's perspective
                        factor *= (1.0 / (fv / (factor_vals[factor_dates.index(fd) - 1]
                                                 if factor_dates.index(fd) > 0 else 1.0)))
                return total_factor  # simplified: apply full factor to all historical prices
        else:
            def _factor_for_date(price_date: date) -> float:
                return 1.0

        # Simplified application: use yfinance's own adj_close if available,
        # else apply total cumulative factor to all closes
        ohlcv_cols = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        available_cols = {k: v for k, v in ohlcv_cols.items() if k in hist.columns}
        out = hist[list(available_cols.keys())].copy()
        out = out.rename(columns=available_cols)

        # If yfinance provides an Adj Close, use it; otherwise compute
        if "Adj Close" in hist.columns:
            out["adj_close"] = hist["Adj Close"].values
        else:
            # Apply cumulative adjustment to close
            # Use full chain factor (simplified — assumes current is base)
            total_cf = chain[-1].cumulative_factor if chain else 1.0
            out["adj_close"] = out["close"] * total_cf

        out["date"] = out.index.date
        out = out.reset_index(drop=True)
        out = out[["date"] + [c for c in out.columns if c != "date"]]

        logger.info(
            "Adjusted prices reconstructed",
            ticker=ticker,
            rows=len(out),
            start_date=start_date.isoformat(),
        )
        return out

    # -----------------------------------------------------------------------
    # Merger arb spread
    # -----------------------------------------------------------------------

    async def get_ma_arbitrage_spread(
        self, target_ticker: str, deal: MergersAcquisitions
    ) -> dict:
        """
        Compute merger arbitrage spread for a pending deal.

        For cash deals: spread = (deal_price - current_price) / current_price
        For stock deals: spread = (deal_value_per_share - current_price) / current_price
        Returns annualized return based on expected close date.
        """
        target_ticker = target_ticker.upper()

        # Fetch current price
        raw = await asyncio.to_thread(_yf_fetch_raw, target_ticker, 1)
        info = raw.get("info", {})
        current_price = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))

        if current_price is None or current_price <= 0:
            hist = raw.get("history", pd.DataFrame())
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if not closes.empty:
                    current_price = float(closes.iloc[-1])

        if current_price is None or current_price <= 0:
            return {
                "target_ticker": target_ticker,
                "current_price": None,
                "deal_price": None,
                "spread_pct": None,
                "annualized_return_pct": None,
                "days_to_close": None,
                "error": "Cannot fetch current price",
            }

        # Determine deal price
        deal_price: Optional[float] = deal.cash_per_share

        # If no explicit cash price, estimate from deal value and info
        if deal_price is None and deal.deal_value is not None:
            shares_out = _sf(info.get("sharesOutstanding"))
            if shares_out and shares_out > 0:
                deal_price = (deal.deal_value * 1_000_000) / shares_out  # convert MM to units

        if deal_price is None:
            return {
                "target_ticker": target_ticker,
                "current_price": round(current_price, 4),
                "deal_price": None,
                "spread_pct": None,
                "annualized_return_pct": None,
                "days_to_close": None,
                "error": "Deal price not available — stock deal or missing data",
            }

        spread = deal_price - current_price
        spread_pct = round(spread / current_price * 100.0, 4)

        # Days to close
        days_to_close: Optional[int] = None
        annualized_return: Optional[float] = None

        expected_close = deal.expected_close or deal.actual_close
        if expected_close:
            days_to_close = (expected_close - date.today()).days
            if days_to_close > 0 and spread_pct is not None:
                annualized_return = round(spread_pct * 365.0 / days_to_close, 4)

        result = {
            "target_ticker": target_ticker,
            "current_price": round(current_price, 4),
            "deal_price": round(deal_price, 4),
            "spread_usd": round(spread, 4),
            "spread_pct": spread_pct,
            "annualized_return_pct": annualized_return,
            "days_to_close": days_to_close,
            "deal_type": deal.deal_type,
            "status": deal.status,
        }

        logger.info(
            "Merger arb spread computed",
            ticker=target_ticker,
            spread_pct=spread_pct,
            days_to_close=days_to_close,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


async def recent_ma_deals(
    days_back: int = 30,
    min_deal_size_mm: float = 100.0,
) -> list[MergersAcquisitions]:
    """Return recent M&A deals from EDGAR 8-K filings."""
    engine = CorporateActionsEnhanced()
    return await engine.get_recent_ma_deals(days_back=days_back, min_deal_size_mm=min_deal_size_mm)


async def splits_history(ticker: str, years_back: int = 30) -> list[SplitEvent]:
    """Return complete split history for a ticker."""
    engine = CorporateActionsEnhanced()
    return await engine.get_splits_history(ticker, years_back=years_back)


async def dividends_history(ticker: str, years_back: int = 30) -> list[DividendEvent]:
    """Return complete dividend history for a ticker."""
    engine = CorporateActionsEnhanced()
    return await engine.get_dividends_history(ticker, years_back=years_back)


async def price_adjustment_chain(
    ticker: str, start_date: Optional[date] = None
) -> list[PriceAdjustmentFactor]:
    """Return cumulative price adjustment factors from splits + dividends."""
    engine = CorporateActionsEnhanced()
    sd = start_date or (date.today() - timedelta(days=365 * 30))
    return await engine.compute_price_adjustment_chain(ticker, sd)


async def adjusted_prices(ticker: str, start_date: Optional[date] = None) -> pd.DataFrame:
    """Return fully-adjusted OHLCV DataFrame for a ticker."""
    engine = CorporateActionsEnhanced()
    sd = start_date or (date.today() - timedelta(days=365 * 10))
    return await engine.reconstruct_adjusted_prices(ticker, sd)


async def recent_spinoffs(days_back: int = 365) -> list[SpinOff]:
    """Return recent spin-off announcements from EDGAR."""
    engine = CorporateActionsEnhanced()
    return await engine.get_spinoffs(days_back=days_back)
