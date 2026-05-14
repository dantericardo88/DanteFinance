"""ipo_intelligence.py — IPO/S-1 filing intelligence from free EDGAR sources.

Covers:
  - S-1 / S-1/A / S-11 filing discovery via EDGAR Atom feed and EFTS
  - Financial data extraction (revenue, net income, margins) via regex
  - SPAC detection and tracker
  - Lockup expiration calendar (IPO date + 180 days)
  - Underwriter identification from cover-page text
  - IPO quality scoring (red flags / green flags)
  - 424B4 prospectus tracking for pricing / shares offered
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ATOM = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}

# Atom namespace used by SEC EDGAR feeds
_ATOM_NS = "http://www.w3.org/2005/Atom"

# Major underwriters to scan for on the cover page
_MAJOR_UNDERWRITERS = [
    "Goldman Sachs",
    "Morgan Stanley",
    "JPMorgan",
    "J.P. Morgan",
    "Bank of America",
    "Merrill Lynch",
    "Citigroup",
    "Citi",
    "Credit Suisse",
    "Deutsche Bank",
    "Barclays",
    "UBS",
    "Wells Fargo",
    "RBC Capital",
    "Jefferies",
    "Cowen",
    "Piper Sandler",
    "Needham",
    "William Blair",
    "Stifel",
    "Cantor Fitzgerald",
    "Oppenheimer",
    "KeyBanc",
    "Evercore",
    "Lazard",
]

# SPAC detection keywords
_SPAC_KEYWORDS = [
    "blank check company",
    "special purpose acquisition",
    "SPAC",
    "no operating history",
    "business combination",
    "trust account",
    "founder shares",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IPOFiling(BaseModel):
    company_name: str
    cik: str
    ticker: Optional[str] = None
    form_type: str  # "S-1", "S-1/A", "S-11", "424B4", etc.
    filed_date: date
    period_date: Optional[date] = None
    accession_number: str
    sic_code: Optional[str] = None
    industry: Optional[str] = None
    state_of_incorporation: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    # Extracted financial data
    revenue_ttm: Optional[float] = None          # trailing 12m revenue (USD)
    net_income_ttm: Optional[float] = None
    gross_profit_margin: Optional[float] = None  # 0-1 fraction
    revenue_growth_yoy: Optional[float] = None   # e.g. 0.35 = 35%
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equiv: Optional[float] = None
    shares_offered: Optional[int] = None
    price_range_low: Optional[float] = None
    price_range_high: Optional[float] = None
    use_of_proceeds: Optional[str] = None
    underwriters: list[str] = Field(default_factory=list)
    risk_factors_count: Optional[int] = None
    is_spac: bool = False
    lockup_days: Optional[int] = None   # standard 180 days


class LockupExpiry(BaseModel):
    company_name: str
    ticker: Optional[str] = None
    ipo_date: date
    lockup_days: int
    lockup_expiry: date
    shares_locked: Optional[int] = None
    insiders_pct: Optional[float] = None
    days_until_expiry: int


class SPACFiling(BaseModel):
    company_name: str
    cik: str
    filed_date: date
    trust_amount: Optional[float] = None
    target_industry: Optional[str] = None
    sponsor_name: Optional[str] = None
    deadline_months: Optional[int] = None   # typical 18-24 months to close
    status: str  # "searching", "announced_target", "completed", "dissolved"


class IPOCalendar(BaseModel):
    week_of: date
    upcoming_ipos: list[dict]
    recent_filings: list[IPOFiling]
    spacs_active: list[SPACFiling]


# ---------------------------------------------------------------------------
# Internal helpers — financial regex parsing
# ---------------------------------------------------------------------------

def _parse_dollar_amount(raw: str) -> Optional[float]:
    """Convert strings like '$1.23 billion', '456,789', '1.2M' to float USD."""
    raw = raw.strip().replace(",", "")
    mult = 1.0
    lower = raw.lower()
    if "billion" in lower or lower.endswith("b"):
        mult = 1_000_000_000.0
    elif "million" in lower or lower.endswith("m"):
        mult = 1_000_000.0
    elif "thousand" in lower or lower.endswith("k"):
        mult = 1_000.0
    num_match = re.search(r"[\d]+(?:\.\d+)?", raw)
    if not num_match:
        return None
    try:
        return float(num_match.group()) * mult
    except ValueError:
        return None


def _find_dollar_near(text: str, label_pattern: str, window: int = 300) -> Optional[float]:
    """Search for a dollar amount within `window` chars after `label_pattern`."""
    m = re.search(label_pattern, text, re.IGNORECASE)
    if not m:
        return None
    snippet = text[m.end(): m.end() + window]
    # Match patterns like: $123.4 million / $1.2 billion / 1,234,567 / 123.4M
    dollar_pat = r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|[BMK])?"
    dm = re.search(dollar_pat, snippet, re.IGNORECASE)
    if not dm:
        return None
    num_str = dm.group(1).replace(",", "")
    suffix = (dm.group(2) or "").lower()
    try:
        val = float(num_str)
    except ValueError:
        return None
    if suffix in ("billion", "b"):
        val *= 1_000_000_000.0
    elif suffix in ("million", "m"):
        val *= 1_000_000.0
    elif suffix in ("thousand", "k"):
        val *= 1_000.0
    return val


def _parse_shares(text: str, label_pattern: str) -> Optional[int]:
    """Extract share count near a label."""
    m = re.search(label_pattern, text, re.IGNORECASE)
    if not m:
        return None
    snippet = text[m.end(): m.end() + 200]
    sm = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?\s*shares", snippet, re.IGNORECASE)
    if not sm:
        return None
    raw = sm.group(1).replace(",", "")
    suffix_m = re.search(r"(million|thousand)", sm.group(), re.IGNORECASE)
    mult = 1
    if suffix_m:
        suffix = suffix_m.group(1).lower()
        if suffix == "million":
            mult = 1_000_000
        elif suffix == "thousand":
            mult = 1_000
    try:
        return int(float(raw) * mult)
    except ValueError:
        return None


def _parse_price_range(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract IPO price range, e.g. '$14.00 to $16.00'."""
    pat = r"\$\s*([\d]+(?:\.\d+)?)\s*(?:to|and|-)\s*\$\s*([\d]+(?:\.\d+)?)"
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None


def _count_risk_factors(text: str) -> int:
    """Count numbered risk factor headers as a proxy for filing complexity."""
    # S-1s typically list: "1. Risk related to...", or bold headings in the risk section
    hits = re.findall(
        r"(?:^|\n)\s*(?:\d+\.|•|-)\s+(?:Risk|Our|We |The Company|Changes in|Competition|Failure)",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    # Also count standalone "RISK FACTORS" sub-headings in ALL CAPS
    all_caps = re.findall(r"\n[A-Z][A-Z\s]{15,80}\n", text)
    return len(hits) + max(0, len(all_caps) - 5)  # subtract boilerplate headings


def _extract_use_of_proceeds(text: str) -> Optional[str]:
    """Pull the 'Use of Proceeds' narrative (~first 500 chars of that section)."""
    m = re.search(r"USE OF PROCEEDS(.{30,600})", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    snippet = m.group(1).strip()
    # Clean up whitespace
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet[:500]


def _extract_sponsor(text: str) -> Optional[str]:
    """Extract SPAC sponsor name."""
    m = re.search(r"(?:our\s+)?sponsor[,\s]+([A-Z][A-Za-z\s&,\.]+(?:LLC|LP|Inc|Corp|Partners))", text)
    if m:
        return m.group(1).strip()
    return None


def _extract_trust_amount(text: str) -> Optional[float]:
    """Extract SPAC trust account amount."""
    m = re.search(
        r"trust\s+account.{0,100}\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    suffix = (m.group(2) or "").lower()
    try:
        val = float(raw)
    except ValueError:
        return None
    if suffix == "billion":
        val *= 1_000_000_000.0
    elif suffix == "million":
        val *= 1_000_000.0
    elif suffix == "thousand":
        val *= 1_000.0
    return val


def _accession_url(cik: str, accession_number: str) -> str:
    """Build the EDGAR filing index URL from CIK and accession number."""
    acc_clean = accession_number.replace("-", "")
    return f"{EDGAR_ARCHIVES}/{cik}/{acc_clean}/{accession_number}-index.htm"


def _filing_text_url(cik: str, accession_number: str, filename: str) -> str:
    acc_clean = accession_number.replace("-", "")
    return f"{EDGAR_ARCHIVES}/{cik}/{acc_clean}/{filename}"


def _parse_date_str(s: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _zero_pad_cik(cik: str) -> str:
    return cik.zfill(10)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class IPOIntelligence:
    """Async client for IPO/S-1 intelligence from free EDGAR sources."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get_recent_s1_filings(
        self,
        days_back: int = 30,
        limit: int = 50,
        form_type: str = "S-1",
    ) -> list[IPOFiling]:
        """Fetch recent S-1 / S-11 filings from the EDGAR Atom current-events feed.

        Handles pagination by fetching up to ``limit`` filings across multiple
        count=40 pages.
        """
        results: list[IPOFiling] = []
        cutoff = date.today() - timedelta(days=days_back)

        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            start = 0
            while len(results) < limit:
                batch = min(40, limit - len(results))
                url = (
                    f"{EDGAR_ATOM}?action=getcurrent&type={form_type}"
                    f"&dateb=&owner=include&count={batch}&start={start}&output=atom"
                )
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("EDGAR Atom feed error", url=url, error=str(exc))
                    break

                entries = self._parse_atom_feed(resp.text, cutoff=cutoff)
                if not entries:
                    break

                results.extend(entries)
                start += batch

                # If the feed returned fewer entries than requested, we're done
                if len(entries) < batch:
                    break

        # Deduplicate by accession number
        seen: set[str] = set()
        unique: list[IPOFiling] = []
        for f in sorted(results, key=lambda x: x.filed_date, reverse=True):
            if f.accession_number not in seen:
                seen.add(f.accession_number)
                unique.append(f)

        return unique[:limit]

    async def get_ipo_details(self, cik: str, accession_number: str) -> IPOFiling:
        """Download and enrich an S-1 filing with extracted financial data.

        Fetches the filing index, identifies the primary document, downloads it,
        then runs all extraction helpers.
        """
        cik_clean = cik.lstrip("0")
        cik_padded = _zero_pad_cik(cik)
        acc_clean = accession_number.replace("-", "")
        index_url = f"{EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{accession_number}-index.json"

        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            # 1. Fetch filing index (JSON preferred)
            filing_meta: dict = {}
            primary_doc: Optional[str] = None
            try:
                resp = await client.get(index_url)
                resp.raise_for_status()
                idx = resp.json()
                filing_meta = idx
                # Find the largest .htm document as primary
                docs = idx.get("documents", [])
                for doc in docs:
                    if doc.get("type") in ("S-1", "S-1/A", "S-11", "424B4") and doc.get("filename", "").endswith(".htm"):
                        primary_doc = doc["filename"]
                        break
                # Fallback: first .htm file
                if not primary_doc:
                    for doc in docs:
                        if doc.get("filename", "").endswith((".htm", ".html", ".txt")):
                            primary_doc = doc["filename"]
                            break
            except Exception as exc:
                logger.warning("Filing index fetch failed", cik=cik, acc=accession_number, error=str(exc))

            # 2. Download filing text
            filing_text = ""
            if primary_doc:
                doc_url = f"{EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{primary_doc}"
                try:
                    doc_resp = await client.get(doc_url)
                    doc_resp.raise_for_status()
                    filing_text = doc_resp.text
                except Exception as exc:
                    logger.warning("Filing document fetch failed", url=doc_url, error=str(exc))

            # 3. Also try submission metadata for company info
            sub_url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
            company_name = filing_meta.get("company", "Unknown")
            form_type = "S-1"
            filed_date_str = filing_meta.get("filingDate", str(date.today()))
            filed_date = _parse_date_str(filed_date_str) or date.today()
            sic_code: Optional[str] = None
            state_of_inc: Optional[str] = None
            ticker: Optional[str] = None
            fiscal_year_end: Optional[str] = None

            try:
                sub_resp = await client.get(sub_url)
                sub_resp.raise_for_status()
                sub = sub_resp.json()
                company_name = sub.get("name", company_name)
                sic_code = str(sub.get("sic", "")) or None
                state_of_inc = sub.get("stateOfIncorporation")
                fiscal_year_end = sub.get("fiscalYearEnd")
                tickers = sub.get("tickers", [])
                if tickers:
                    ticker = tickers[0]
                # Try to find the exact filing date from the filings list
                filings_data = sub.get("filings", {}).get("recent", {})
                accessions = filings_data.get("accessionNumber", [])
                acc_norm = accession_number.replace("-", "")
                if acc_norm in [a.replace("-", "") for a in accessions]:
                    idx2 = [a.replace("-", "") for a in accessions].index(acc_norm)
                    forms = filings_data.get("form", [])
                    dates = filings_data.get("filingDate", [])
                    if idx2 < len(forms):
                        form_type = forms[idx2]
                    if idx2 < len(dates):
                        filed_date = _parse_date_str(dates[idx2]) or filed_date
            except Exception as exc:
                logger.debug("Submission metadata fetch failed", cik=cik, error=str(exc))

        # 4. Run extraction helpers on filing text
        financials = self._extract_financials_from_text(filing_text)
        is_spac = self._is_spac(filing_text)
        underwriters = self._extract_underwriters(filing_text)
        risk_count = _count_risk_factors(filing_text) if filing_text else None
        use_of_proceeds = _extract_use_of_proceeds(filing_text)
        price_low, price_high = _parse_price_range(filing_text)
        shares_offered = _parse_shares(filing_text, r"(?:total\s+)?shares\s+(?:of\s+)?(?:common\s+stock\s+)?offered")
        lockup_days: Optional[int] = None
        lockup_m = re.search(r"(\d+)[-\s]day\s+lock[-\s]?up", filing_text, re.IGNORECASE)
        if lockup_m:
            lockup_days = int(lockup_m.group(1))
        else:
            lockup_days = 180 if filing_text else None

        industry = _sic_to_industry(sic_code) if sic_code else None

        return IPOFiling(
            company_name=company_name,
            cik=cik,
            ticker=ticker,
            form_type=form_type,
            filed_date=filed_date,
            accession_number=accession_number,
            sic_code=sic_code,
            industry=industry,
            state_of_incorporation=state_of_inc,
            fiscal_year_end=fiscal_year_end,
            revenue_ttm=financials.get("revenue"),
            net_income_ttm=financials.get("net_income"),
            gross_profit_margin=financials.get("gross_profit_margin"),
            revenue_growth_yoy=financials.get("revenue_growth_yoy"),
            total_assets=financials.get("total_assets"),
            total_debt=financials.get("total_debt"),
            cash_and_equiv=financials.get("cash_and_equiv"),
            shares_offered=shares_offered,
            price_range_low=price_low,
            price_range_high=price_high,
            use_of_proceeds=use_of_proceeds,
            underwriters=underwriters,
            risk_factors_count=risk_count,
            is_spac=is_spac,
            lockup_days=lockup_days,
        )

    async def search_ipos(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        industry: Optional[str] = None,
        min_revenue: Optional[float] = None,
    ) -> list[IPOFiling]:
        """Search S-1 / S-11 filings via EDGAR EFTS full-text search.

        Filters by date range. Industry and min_revenue filtering is applied
        client-side after enrichment because EFTS does not support those filters
        natively.
        """
        today = date.today()
        start_date = start_date or (today - timedelta(days=90))
        end_date = end_date or today

        params = {
            "q": '"initial public offering"',
            "forms": "S-1,S-11",
            "dateRange": "custom",
            "startdt": start_date.isoformat(),
            "enddt": end_date.isoformat(),
            "hits.hits.total.value": 1,
        }

        filings: list[IPOFiling] = []

        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            try:
                resp = await client.get(EDGAR_EFTS, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("EDGAR EFTS search failed", error=str(exc))
                # Fall back to Atom feed
                return await self.get_recent_s1_filings(
                    days_back=(today - start_date).days + 1,
                    limit=100,
                )

            hits = data.get("hits", {}).get("hits", [])
            tasks = []
            for hit in hits:
                src = hit.get("_source", {})
                cik = str(src.get("entity_id", src.get("file_num", ""))).lstrip("0") or "0"
                accession = src.get("file_date", "")
                acc = hit.get("_id", "")
                # EFTS _id format: accession-number (with dashes)
                if not acc:
                    continue
                tasks.append(self._enrich_efts_hit(client, hit))

            enriched = await asyncio.gather(*tasks, return_exceptions=True)
            for result in enriched:
                if isinstance(result, IPOFiling):
                    filings.append(result)

        # Client-side filtering
        if industry:
            filings = [f for f in filings if f.industry and industry.lower() in f.industry.lower()]
        if min_revenue is not None:
            filings = [f for f in filings if f.revenue_ttm is not None and f.revenue_ttm >= min_revenue]

        return sorted(filings, key=lambda x: x.filed_date, reverse=True)

    async def get_lockup_expirations(
        self,
        days_ahead: int = 90,
    ) -> list[LockupExpiry]:
        """Find IPOs with lockup periods expiring within the next ``days_ahead`` days.

        Strategy: fetch 424B4 (final prospectus) filings from the last 6 months,
        treat the filing date as the IPO date, and compute expiry = IPO date + 180 days.
        """
        today = date.today()
        expiry_cutoff = today + timedelta(days=days_ahead)

        # Fetch 424B4 filings — these are the final prospectuses filed on IPO day
        filings_424 = await self.get_recent_s1_filings(days_back=200, limit=200, form_type="424B4")

        expirations: list[LockupExpiry] = []
        for filing in filings_424:
            lockup_days = filing.lockup_days or 180
            expiry = filing.filed_date + timedelta(days=lockup_days)
            days_until = (expiry - today).days

            # Only include if expiry is in the future (or very recently passed)
            if expiry < today - timedelta(days=7):
                continue
            if expiry > expiry_cutoff:
                continue

            expirations.append(
                LockupExpiry(
                    company_name=filing.company_name,
                    ticker=filing.ticker,
                    ipo_date=filing.filed_date,
                    lockup_days=lockup_days,
                    lockup_expiry=expiry,
                    shares_locked=filing.shares_offered,
                    days_until_expiry=days_until,
                )
            )

        return sorted(expirations, key=lambda x: x.lockup_expiry)

    async def get_spac_tracker(self, limit: int = 50) -> list[SPACFiling]:
        """Return a list of active SPAC filings from recent S-1 submissions.

        Fetches recent S-1s, downloads each filing text, and filters to those that
        exhibit SPAC characteristics. Then enriches with trust amount and sponsor.
        """
        raw_filings = await self.get_recent_s1_filings(days_back=365, limit=limit * 3)
        spac_results: list[SPACFiling] = []

        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            tasks = [
                self._check_spac_filing(client, f)
                for f in raw_filings
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, SPACFiling):
                spac_results.append(result)
            if len(spac_results) >= limit:
                break

        return spac_results

    async def get_ipo_calendar(self) -> IPOCalendar:
        """Build a structured IPO calendar for the current week.

        Combines:
        - Recent S-1 filings (last 30 days)
        - Upcoming 424B4 prospectuses (pricing imminent)
        - Active SPAC tracker
        """
        today = date.today()
        # Snap to Monday
        week_start = today - timedelta(days=today.weekday())

        recent_task = self.get_recent_s1_filings(days_back=30, limit=50)
        upcoming_task = self.get_recent_s1_filings(days_back=14, limit=30, form_type="424B4")
        spac_task = self.get_spac_tracker(limit=20)

        recent_filings, upcoming_raw, spacs = await asyncio.gather(
            recent_task, upcoming_task, spac_task
        )

        upcoming_ipos = [
            {
                "company_name": f.company_name,
                "ticker": f.ticker,
                "filed_date": f.filed_date.isoformat(),
                "form_type": f.form_type,
                "price_range": (
                    f"${f.price_range_low:.2f} – ${f.price_range_high:.2f}"
                    if f.price_range_low and f.price_range_high
                    else "TBD"
                ),
                "underwriters": f.underwriters,
                "is_spac": f.is_spac,
            }
            for f in upcoming_raw
        ]

        return IPOCalendar(
            week_of=week_start,
            upcoming_ipos=upcoming_ipos,
            recent_filings=recent_filings,
            spacs_active=spacs,
        )

    async def analyze_ipo_quality(self, cik: str, accession: str) -> dict:
        """Parse an S-1 for quality signals and red flags.

        Returns a dict with:
          score       : 0-10 (higher = higher quality)
          red_flags   : list[str]
          green_flags : list[str]
          details     : dict of extracted metrics
        """
        filing = await self.get_ipo_details(cik, accession)

        red_flags: list[str] = []
        green_flags: list[str] = []
        score = 5  # start neutral

        # Revenue presence
        if filing.revenue_ttm is None:
            red_flags.append("No revenue found — possible pre-revenue company")
            score -= 1
        elif filing.revenue_ttm < 5_000_000:
            red_flags.append(f"Very low revenue: ${filing.revenue_ttm:,.0f}")
            score -= 1
        else:
            green_flags.append(f"Revenue: ${filing.revenue_ttm / 1e6:.1f}M")
            score += 1

        # Revenue growth
        if filing.revenue_growth_yoy is not None:
            if filing.revenue_growth_yoy > 0.50:
                green_flags.append(f"High revenue growth: {filing.revenue_growth_yoy:.0%} YoY")
                score += 1
            elif filing.revenue_growth_yoy < 0:
                red_flags.append(f"Declining revenue: {filing.revenue_growth_yoy:.0%} YoY")
                score -= 1

        # Profitability
        if filing.net_income_ttm is not None:
            if filing.net_income_ttm > 0:
                green_flags.append("Company is profitable at IPO")
                score += 2
            elif filing.net_income_ttm < -50_000_000:
                red_flags.append(f"Large losses: ${filing.net_income_ttm / 1e6:.1f}M net loss")
                score -= 1

        # Gross margin
        if filing.gross_profit_margin is not None:
            if filing.gross_profit_margin > 0.60:
                green_flags.append(f"Strong gross margin: {filing.gross_profit_margin:.0%}")
                score += 1
            elif filing.gross_profit_margin < 0.20:
                red_flags.append(f"Low gross margin: {filing.gross_profit_margin:.0%}")
                score -= 1

        # Cash runway
        if filing.cash_and_equiv is not None and filing.net_income_ttm is not None and filing.net_income_ttm < 0:
            monthly_burn = abs(filing.net_income_ttm) / 12
            if monthly_burn > 0:
                runway_months = filing.cash_and_equiv / monthly_burn
                if runway_months < 12:
                    red_flags.append(f"Short cash runway: ~{runway_months:.0f} months")
                    score -= 1
                elif runway_months > 24:
                    green_flags.append(f"Solid cash runway: ~{runway_months:.0f} months")
                    score += 1

        # Risk factor complexity
        if filing.risk_factors_count is not None:
            if filing.risk_factors_count > 60:
                red_flags.append(f"Very high risk factor count: {filing.risk_factors_count}")
                score -= 1
            elif filing.risk_factors_count > 40:
                red_flags.append(f"High risk factor count: {filing.risk_factors_count}")

        # SPAC signal
        if filing.is_spac:
            red_flags.append("SPAC structure — no operating history")
            score -= 1

        # Underwriter quality
        tier1 = {"Goldman Sachs", "Morgan Stanley", "JPMorgan", "J.P. Morgan"}
        if any(u in tier1 for u in filing.underwriters):
            green_flags.append(f"Tier-1 underwriter(s): {', '.join(u for u in filing.underwriters if u in tier1)}")
            score += 1
        elif not filing.underwriters:
            red_flags.append("No major underwriter identified")
            score -= 1

        # Insider selling proxy — look for use_of_proceeds mentioning shareholder
        if filing.use_of_proceeds and re.search(
            r"selling\s+shareholder|secondary\s+offer|existing\s+stockholder", filing.use_of_proceeds, re.IGNORECASE
        ):
            red_flags.append("Proceeds include selling shareholder / secondary component")
            score -= 1

        score = max(0, min(10, score))

        return {
            "score": score,
            "red_flags": red_flags,
            "green_flags": green_flags,
            "details": {
                "company_name": filing.company_name,
                "revenue_ttm": filing.revenue_ttm,
                "net_income_ttm": filing.net_income_ttm,
                "gross_profit_margin": filing.gross_profit_margin,
                "revenue_growth_yoy": filing.revenue_growth_yoy,
                "cash_and_equiv": filing.cash_and_equiv,
                "risk_factors_count": filing.risk_factors_count,
                "underwriters": filing.underwriters,
                "is_spac": filing.is_spac,
                "lockup_days": filing.lockup_days,
            },
        }

    # ------------------------------------------------------------------
    # Core extraction helpers
    # ------------------------------------------------------------------

    def _extract_financials_from_text(self, text: str) -> dict:
        """Regex-based extraction of financial metrics from S-1 filing text.

        Handles mixed formats: "$123.4 million", "123,456", "1.2B", table cells.
        Returns a dict with keys: revenue, net_income, gross_profit_margin,
        revenue_growth_yoy, total_assets, total_debt, cash_and_equiv.
        """
        if not text:
            return {}

        result: dict = {}

        # ---- Revenue ----
        # Try multiple label variants
        revenue_labels = [
            r"(?:total\s+)?(?:net\s+)?revenue[s]?(?:\s*\(.*?\))?\s*[\$]?",
            r"net\s+sales\s*[\$]?",
            r"total\s+sales\s*[\$]?",
        ]
        for label in revenue_labels:
            val = _find_dollar_near(text, label)
            if val and val > 0:
                result["revenue"] = val
                break

        # ---- Net income / loss ----
        net_income_labels = [
            r"net\s+(?:income|loss)(?:\s+attributable)?(?:\s*\(.*?\))?\s*[\$]?",
            r"net\s+(?:income|loss)\s+available",
        ]
        for label in net_income_labels:
            val = _find_dollar_near(text, label)
            if val is not None:
                # Detect if it's a loss (negative context)
                m = re.search(label, text, re.IGNORECASE)
                if m:
                    snippet = text[m.start(): m.end() + 300]
                    if re.search(r"\([\d,\.]+\)|net\s+loss", snippet, re.IGNORECASE):
                        val = -abs(val)
                result["net_income"] = val
                break

        # ---- Gross profit ----
        gross_profit = _find_dollar_near(text, r"gross\s+profit\s*[\$]?")
        if gross_profit and result.get("revenue") and result["revenue"] > 0:
            result["gross_profit_margin"] = gross_profit / result["revenue"]

        # ---- Revenue growth YoY ----
        # Look for explicit growth percentage mentions
        growth_m = re.search(
            r"revenue[s]?\s+(?:increased|grew|declined|decreased)\s+(?:by\s+)?([\d\.]+)%",
            text,
            re.IGNORECASE,
        )
        if growth_m:
            growth_val = float(growth_m.group(1)) / 100.0
            # Detect sign
            if re.search(r"declin|decreas", growth_m.group(), re.IGNORECASE):
                growth_val = -growth_val
            result["revenue_growth_yoy"] = growth_val
        else:
            # Try to compute from two revenue figures if they appear near "prior year"
            yoy_m = re.search(
                r"([\d,\.]+)\s*(?:million|billion)?\s*(?:for|in)\s+(?:the\s+)?(?:year|twelve|12).{0,50}"
                r"([\d,\.]+)\s*(?:million|billion)?\s*(?:for|in)\s+(?:the\s+)?(?:prior|previous|year\s+ended)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if yoy_m:
                try:
                    current = float(yoy_m.group(1).replace(",", ""))
                    prior = float(yoy_m.group(2).replace(",", ""))
                    if prior > 0:
                        result["revenue_growth_yoy"] = (current - prior) / prior
                except ValueError:
                    pass

        # ---- Total assets ----
        assets_val = _find_dollar_near(text, r"total\s+assets\s*[\$]?")
        if assets_val:
            result["total_assets"] = assets_val

        # ---- Total debt ----
        debt_labels = [
            r"(?:total\s+)?long[- ]term\s+debt\s*[\$]?",
            r"total\s+(?:indebtedness|debt)\s*[\$]?",
            r"notes?\s+payable\s*[\$]?",
        ]
        for label in debt_labels:
            val = _find_dollar_near(text, label)
            if val is not None and val >= 0:
                result["total_debt"] = val
                break

        # ---- Cash & equivalents ----
        cash_labels = [
            r"cash\s+and\s+cash\s+equivalents\s*[\$]?",
            r"cash,?\s+cash\s+equivalents\s+and\s+(?:restricted\s+cash|short[- ]term)\s*[\$]?",
        ]
        for label in cash_labels:
            val = _find_dollar_near(text, label)
            if val is not None and val >= 0:
                result["cash_and_equiv"] = val
                break

        return result

    def _is_spac(self, text: str) -> bool:
        """Return True if the filing text contains SPAC indicators."""
        if not text:
            return False
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in _SPAC_KEYWORDS)

    def _extract_underwriters(self, text: str) -> list[str]:
        """Scan filing text for major investment bank names.

        Searches both exact-case and case-insensitive to handle ALL-CAPS prospectus covers.
        """
        if not text:
            return []
        found: list[str] = []
        text_lower = text.lower()
        for bank in _MAJOR_UNDERWRITERS:
            if bank.lower() in text_lower:
                if bank not in found:
                    found.append(bank)
        return found

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_atom_feed(self, xml_text: str, cutoff: Optional[date] = None) -> list[IPOFiling]:
        """Parse an EDGAR Atom feed and return a list of IPOFiling stubs.

        The Atom feed provides company name, CIK, form type, filing date, and
        accession number. Financial data is NOT available at this stage.
        """
        filings: list[IPOFiling] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Atom feed XML parse error", error=str(exc))
            return filings

        ns = {"atom": _ATOM_NS}

        for entry in root.findall("atom:entry", ns):
            # --- Filed date ---
            updated_el = entry.find("atom:updated", ns)
            filed_date: Optional[date] = None
            if updated_el is not None and updated_el.text:
                filed_date = _parse_date_str(updated_el.text[:10])
            if filed_date is None:
                filed_date = date.today()

            if cutoff and filed_date < cutoff:
                continue  # older than requested window

            # --- Company name and CIK from title / content ---
            title_el = entry.find("atom:title", ns)
            title = title_el.text.strip() if title_el is not None and title_el.text else ""

            # Title format: "S-1 - Company Name (0001234567) (Filing)"
            company_name = "Unknown"
            cik = "0"
            form_type = "S-1"

            title_m = re.match(
                r"^(S-1[^\-]*|S-11[^\-]*|424B4[^\-]*)[\s\-]+(.+?)\s*\((\d{10})\)",
                title,
            )
            if title_m:
                form_type = title_m.group(1).strip()
                company_name = title_m.group(2).strip()
                cik = title_m.group(3).lstrip("0") or "0"
            else:
                # Try alternate: just grab anything in parens as CIK
                cik_m = re.search(r"\((\d{7,10})\)", title)
                if cik_m:
                    cik = cik_m.group(1).lstrip("0") or "0"
                company_name = re.sub(r"\([\d]+\)", "", title).replace(form_type, "").strip(" -")

            # --- Accession number from filing index link ---
            accession_number = ""
            link_el = entry.find("atom:link", ns)
            if link_el is not None:
                href = link_el.get("href", "")
                acc_m = re.search(r"(\d{10}-\d{2}-\d{6})", href)
                if acc_m:
                    accession_number = acc_m.group(1)

            if not accession_number:
                # Fallback: use id element
                id_el = entry.find("atom:id", ns)
                if id_el is not None and id_el.text:
                    acc_m = re.search(r"(\d{10}-\d{2}-\d{6})", id_el.text)
                    if acc_m:
                        accession_number = acc_m.group(1)

            if not accession_number or not cik or cik == "0":
                continue

            filings.append(
                IPOFiling(
                    company_name=company_name,
                    cik=cik,
                    form_type=form_type,
                    filed_date=filed_date,
                    accession_number=accession_number,
                )
            )

        return filings

    async def _enrich_efts_hit(self, client: httpx.AsyncClient, hit: dict) -> Optional[IPOFiling]:
        """Convert an EFTS search hit into an IPOFiling stub."""
        src = hit.get("_source", {})
        acc = hit.get("_id", "").replace(":", "-")
        if not acc:
            return None

        entity_id = str(src.get("entity_id", "")).lstrip("0") or "0"
        company_name = src.get("display_names", [{}])[0].get("name", "Unknown") if src.get("display_names") else "Unknown"
        form_type = src.get("form_type", "S-1")
        filed_date_str = src.get("file_date", str(date.today()))
        filed_date = _parse_date_str(filed_date_str) or date.today()
        period_date_str = src.get("period_of_report", "")
        period_date = _parse_date_str(period_date_str) if period_date_str else None

        return IPOFiling(
            company_name=company_name,
            cik=entity_id,
            form_type=form_type,
            filed_date=filed_date,
            period_date=period_date,
            accession_number=acc,
        )

    async def _check_spac_filing(
        self, client: httpx.AsyncClient, filing: IPOFiling
    ) -> Optional[SPACFiling]:
        """Download filing text and return a SPACFiling if SPAC signals are present."""
        cik_clean = filing.cik.lstrip("0") or "0"
        acc_clean = filing.accession_number.replace("-", "")
        index_url = f"{EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{filing.accession_number}-index.json"

        filing_text = ""
        try:
            idx_resp = await client.get(index_url)
            idx_resp.raise_for_status()
            idx = idx_resp.json()
            docs = idx.get("documents", [])
            primary_doc: Optional[str] = None
            for doc in docs:
                if doc.get("filename", "").endswith((".htm", ".html", ".txt")):
                    primary_doc = doc["filename"]
                    break
            if primary_doc:
                doc_url = f"{EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{primary_doc}"
                doc_resp = await client.get(doc_url)
                doc_resp.raise_for_status()
                filing_text = doc_resp.text
        except Exception as exc:
            logger.debug("SPAC filing fetch failed", cik=filing.cik, error=str(exc))
            return None

        if not self._is_spac(filing_text):
            return None

        trust_amount = _extract_trust_amount(filing_text)
        sponsor = _extract_sponsor(filing_text)

        # Guess target industry from filing text keywords
        target_industry: Optional[str] = None
        industry_keywords = {
            "technology": ["software", "saas", "cloud", "artificial intelligence", "fintech"],
            "healthcare": ["pharmaceutical", "biotech", "medical device", "healthcare"],
            "energy": ["oil", "gas", "renewable", "clean energy", "power"],
            "consumer": ["retail", "consumer brand", "food", "beverage", "e-commerce"],
            "financial services": ["financial services", "insurance", "payments", "banking"],
        }
        text_lower = filing_text.lower()
        for ind, keywords in industry_keywords.items():
            if any(kw in text_lower for kw in keywords):
                target_industry = ind
                break

        # Determine deadline — search for "18 months" or "24 months"
        deadline_months: Optional[int] = None
        dl_m = re.search(r"(\d+)\s*months?\s+(?:to\s+)?(?:complete|consummate|close)", filing_text, re.IGNORECASE)
        if dl_m:
            deadline_months = int(dl_m.group(1))

        # Determine status — simplistic: if no 8-K with merger, assume searching
        status = "searching"

        return SPACFiling(
            company_name=filing.company_name,
            cik=filing.cik,
            filed_date=filing.filed_date,
            trust_amount=trust_amount,
            target_industry=target_industry,
            sponsor_name=sponsor,
            deadline_months=deadline_months,
            status=status,
        )


# ---------------------------------------------------------------------------
# SIC code → industry label mapping (abbreviated)
# ---------------------------------------------------------------------------

_SIC_INDUSTRY_MAP: dict[str, str] = {
    "7372": "Software",
    "7371": "Computer Programming Services",
    "7374": "Computer Processing & Data Preparation",
    "7389": "Services — Computer Integrated Systems",
    "8731": "Commercial Physical & Biological Research",
    "2836": "Pharmaceutical Preparations",
    "2835": "In Vitro & In Vivo Diagnostic Substances",
    "5912": "Drug Stores & Proprietary Stores",
    "6770": "Blank Checks (SPAC)",
    "6199": "Finance Services",
    "6211": "Security Brokers & Dealers",
    "7011": "Hotels & Motels",
    "5812": "Eating Places",
    "5411": "Grocery Stores",
    "5731": "Radio, TV & Consumer Electronics Stores",
    "4812": "Telephone Communications",
    "4813": "Telephone Communications (No Radio)",
    "4911": "Electric Services",
    "1311": "Crude Petroleum & Natural Gas",
    "3674": "Semiconductors",
    "3559": "Special Industry Machinery",
    "3825": "Instruments for Measuring",
    "6500": "Real Estate",
    "6512": "Operators of Apartment Buildings",
    "6552": "Land Subdividers & Developers",
}


def _sic_to_industry(sic_code: Optional[str]) -> Optional[str]:
    if not sic_code:
        return None
    return _SIC_INDUSTRY_MAP.get(sic_code.zfill(4))


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


async def recent_ipos(days_back: int = 30) -> list[IPOFiling]:
    """Return recent S-1 filings as IPOFiling stubs."""
    intel = IPOIntelligence()
    return await intel.get_recent_s1_filings(days_back=days_back, limit=50)


async def lockup_expirations(days_ahead: int = 90) -> list[LockupExpiry]:
    """Return lockup expirations within the next ``days_ahead`` days."""
    intel = IPOIntelligence()
    return await intel.get_lockup_expirations(days_ahead=days_ahead)


async def spac_tracker() -> list[SPACFiling]:
    """Return active SPAC filings."""
    intel = IPOIntelligence()
    return await intel.get_spac_tracker(limit=50)


async def ipo_calendar() -> IPOCalendar:
    """Return the IPO calendar for the current week."""
    intel = IPOIntelligence()
    return await intel.get_ipo_calendar()


async def search_ipos(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    industry: Optional[str] = None,
    min_revenue: Optional[float] = None,
) -> list[IPOFiling]:
    """Convenience wrapper for IPOIntelligence.search_ipos."""
    intel = IPOIntelligence()
    return await intel.search_ipos(
        start_date=start_date,
        end_date=end_date,
        industry=industry,
        min_revenue=min_revenue,
    )
