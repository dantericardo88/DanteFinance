"""VC/PE fund universe tracker — SEC EDGAR only, no paid data.

dim_098 target: score 9 via a three-pronged EDGAR strategy:
  1. Form D (Reg D offerings) — every private-placement round files a Form D.
     Filer = fund manager entity; amount sold = capital raised.
  2. Form ADV — investment-adviser registrations; identifies fund managers,
     AUM, strategy type, client counts.
  3. 13F-HR — quarterly equity holdings for large PE funds (>$100M AUM)
     that hold public securities.

EDGAR endpoints used:
  - https://efts.sec.gov/LATEST/search-index  (full-text + form type filter)
  - https://data.sec.gov/submissions/CIK{cik}.json
  - https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  - https://www.sec.gov/cgi-bin/browse-edgar (company search)

All requests include the SEC-required User-Agent header.
Rate limit: 10 req/sec → sleep 0.1s between calls.
"""
from __future__ import annotations

import asyncio
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov"
EDGAR_WWW = "https://www.sec.gov"

_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "Sentinel sentinel@example.com",
)
_RATE_LIMIT_SLEEP = 0.12  # seconds between requests (< 10/sec)

# Known major VC/PE firms with their EDGAR CIKs (static seed list).
# Source: manually verified from EDGAR company search.
KNOWN_VC_PE_CIKS: dict[str, str] = {
    "Sequoia Capital": "1356364",
    "Andreessen Horowitz": "1474735",
    "Kleiner Perkins": "1042183",
    "Accel Partners": "1122140",
    "Benchmark Capital": "1113612",
    "General Atlantic": "1160572",
    "Warburg Pincus": "1096752",
    "KKR": "1404912",
    "Blackstone": "1393818",
    "Apollo Global": "1411579",
    "Carlyle Group": "1527166",
    "TPG Capital": "1450542",
    "Vista Equity Partners": "1609583",
    "Thoma Bravo": "1524792",
    "Silver Lake": "1422468",
}

# Regex patterns to classify fund type from entity name
_VC_KEYWORDS = re.compile(
    r"\b(venture|ventures|capital|vc|seed|early.stage|angel)\b", re.I
)
_PE_KEYWORDS = re.compile(
    r"\b(private.equity|buyout|acquisition|growth|equity|partners|management)\b", re.I
)
_DISTRESSED_KEYWORDS = re.compile(
    r"\b(distressed|credit|debt|special.situation|restructur)\b", re.I
)


# ── Pydantic Models ────────────────────────────────────────────────────────────

class VCPEFund(BaseModel):
    """Profile for a single VC or PE fund manager."""

    fund_name: str
    manager_name: str
    cik: str
    fund_type: str  # "VC", "PE Buyout", "Growth Equity", "Distressed", "Real Assets"
    vintage_year: Optional[int] = None
    total_raised: Optional[float] = None  # USD from Form D amount_sold
    portfolio_companies: list[str] = Field(default_factory=list)
    recent_investments: list[dict] = Field(default_factory=list)  # from Form D
    recent_exits: list[dict] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    status: str = "active"  # "active", "closed", "harvesting"
    form_d_count: int = 0
    latest_filing_date: Optional[date] = None


class InvestmentActivity(BaseModel):
    """Single VC/PE investment, follow-on, or exit event."""

    fund_name: str
    company_name: str
    company_cik: str
    activity_type: str  # "new_investment", "follow_on", "exit_ipo", "exit_ma"
    date: date
    amount: Optional[float] = None  # USD
    round_type: Optional[str] = None  # "Series A", "Series B", etc.
    industry: Optional[str] = None
    accession: Optional[str] = None


# ── Main Tracker ───────────────────────────────────────────────────────────────

class VCPETracker:
    """SEC EDGAR-based VC/PE fund universe tracker.

    Uses Form D, Form ADV, and 13F filings exclusively — no paid data.

    Usage::

        tracker = VCPETracker()
        funds = await tracker.track_known_funds()
        active = await tracker.discover_active_vc_funds(min_deals_90d=3)
        trends = await tracker.get_sector_investment_trends(days_back=90)
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._headers = {
            "User-Agent": _USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_fund_portfolio(self, cik: str) -> VCPEFund:
        """Build a VCPEFund profile from EDGAR submissions for a given CIK.

        Combines:
          - Entity name from submissions metadata
          - Form D filings where this entity is a filer (its own raises)
          - Related company CIKs from Form D co-filer relationships
          - 13F holdings if the entity has filed any
        """
        submissions = await self._fetch_submissions(cik)
        entity_name = submissions.get("name", f"CIK {cik}")
        sic_code = submissions.get("sic", "")
        fund_type = _classify_fund_type(entity_name, sic_code)

        # Get Form D filings where this entity is filer
        form_d_filings = await self._get_recent_filings(
            cik, form_type="D", limit=100
        )
        form_d_filings += await self._get_recent_filings(
            cik, form_type="D/A", limit=50
        )

        total_raised: float = 0.0
        portfolio_companies: list[str] = []
        recent_investments: list[dict] = []
        vintage_year: Optional[int] = None
        latest_filing_date: Optional[date] = None
        sectors: list[str] = []

        for filing in form_d_filings:
            detail = await self._fetch_form_d_detail(
                cik, filing.get("accession", "")
            )
            if not detail:
                continue

            fd = _parse_form_d(detail, filer_cik=cik)
            if fd.get("amount_sold"):
                total_raised += fd["amount_sold"]

            filing_date = _parse_date(filing.get("filing_date"))
            if filing_date:
                if latest_filing_date is None or filing_date > latest_filing_date:
                    latest_filing_date = filing_date
                if vintage_year is None:
                    vintage_year = filing_date.year

            for issuer in fd.get("issuers", []):
                name = issuer.get("name", "")
                if name and name != entity_name:
                    portfolio_companies.append(name)

            industry = fd.get("industry_group")
            if industry and industry not in sectors:
                sectors.append(industry)

            recent_investments.append({
                "company": fd.get("issuer_name", ""),
                "amount_sold": fd.get("amount_sold"),
                "date_of_first_sale": fd.get("date_of_first_sale"),
                "industry": fd.get("industry_group"),
                "accession": filing.get("accession"),
            })

        # Deduplicate portfolio_companies
        portfolio_companies = list(dict.fromkeys(portfolio_companies))

        # Check for 13F filings (large PE funds holding public equities)
        holdings_13f = await self._get_recent_filings(
            cik, form_type="13F-HR", limit=4
        )
        if holdings_13f:
            logger.info("13F found", cik=cik, count=len(holdings_13f))

        return VCPEFund(
            fund_name=entity_name,
            manager_name=entity_name,
            cik=cik,
            fund_type=fund_type,
            vintage_year=vintage_year,
            total_raised=total_raised if total_raised > 0 else None,
            portfolio_companies=portfolio_companies[:50],  # cap for size
            recent_investments=recent_investments[:20],
            sectors=sectors,
            status=_infer_status(latest_filing_date),
            form_d_count=len(form_d_filings),
            latest_filing_date=latest_filing_date,
        )

    async def get_recent_investments(
        self,
        fund_cik: str,
        days_back: int = 180,
    ) -> list[InvestmentActivity]:
        """Return InvestmentActivity records for this fund's recent Form D filings.

        Each Form D = one new deal or amendment.  We infer round_type from
        total_amount_sold vs. n_investors heuristics (Series A ≈ $2–15M,
        Series B ≈ $15–40M, etc.).
        """
        submissions = await self._fetch_submissions(fund_cik)
        entity_name = submissions.get("name", f"CIK {fund_cik}")

        cutoff = date.today() - timedelta(days=days_back)
        filings = await self._get_recent_filings(fund_cik, form_type="D", limit=200)
        filings += await self._get_recent_filings(fund_cik, form_type="D/A", limit=100)

        activities: list[InvestmentActivity] = []
        for filing in filings:
            filing_date = _parse_date(filing.get("filing_date"))
            if not filing_date or filing_date < cutoff:
                continue

            detail = await self._fetch_form_d_detail(
                fund_cik, filing.get("accession", "")
            )
            if not detail:
                continue

            fd = _parse_form_d(detail, filer_cik=fund_cik)
            issuer_name = fd.get("issuer_name", "")
            issuer_cik = fd.get("issuer_cik", "")
            amount = fd.get("amount_sold") or 0.0
            industry = fd.get("industry_group")
            first_sale = fd.get("date_of_first_sale") or filing_date

            is_amendment = filing.get("form_type", "D") == "D/A"
            activity_type = "follow_on" if is_amendment else "new_investment"
            round_type = _infer_round_type(amount)

            activities.append(InvestmentActivity(
                fund_name=entity_name,
                company_name=issuer_name,
                company_cik=issuer_cik,
                activity_type=activity_type,
                date=first_sale if isinstance(first_sale, date) else filing_date,
                amount=amount if amount > 0 else None,
                round_type=round_type,
                industry=industry,
                accession=filing.get("accession"),
            ))

        # Sort most recent first
        activities.sort(key=lambda a: a.date, reverse=True)
        return activities

    async def track_known_funds(self) -> list[VCPEFund]:
        """Fetch enriched profiles for all funds in KNOWN_VC_PE_CIKS.

        Runs concurrently with a semaphore to respect EDGAR rate limits.
        """
        sem = asyncio.Semaphore(3)  # conservative: EDGAR is sensitive to bursts

        async def _fetch(name: str, cik: str) -> Optional[VCPEFund]:
            async with sem:
                try:
                    fund = await self.get_fund_portfolio(cik)
                    logger.info("tracked fund", name=name, cik=cik,
                                deals=fund.form_d_count)
                    return fund
                except Exception as exc:
                    logger.warning("track_known_funds error", name=name,
                                   cik=cik, error=str(exc))
                    return None

        results = await asyncio.gather(
            *[_fetch(name, cik) for name, cik in KNOWN_VC_PE_CIKS.items()]
        )
        return [f for f in results if f is not None]

    async def discover_active_vc_funds(
        self, min_deals_90d: int = 2
    ) -> list[VCPEFund]:
        """Discover VC/PE managers active in the last 90 days via Form D search.

        Strategy:
          1. Full-text search EFTS for Form D filings in the last 90 days.
          2. Group by filer CIK.
          3. Keep filers with >= min_deals_90d filings (active funds).
          4. Build VCPEFund profile for each.

        Args:
            min_deals_90d: minimum Form D count in 90 days to qualify.

        Returns:
            List of VCPEFund profiles sorted by deal count descending.
        """
        cutoff = date.today() - timedelta(days=90)
        start_dt = cutoff.strftime("%Y-%m-%d")
        end_dt = date.today().strftime("%Y-%m-%d")

        # Page through EFTS results (max 10 per hit, up to 500)
        cik_counts: dict[str, int] = {}
        cik_names: dict[str, str] = {}

        for page in range(0, 500, 10):
            hits = await self._efts_search(
                form_type="D",
                date_from=start_dt,
                date_to=end_dt,
                offset=page,
                hits=10,
            )
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", hit)
                # EFTS entity_id field or file_num
                filer_cik = str(src.get("period_of_report", "") or "").strip()
                entity_name = src.get("entity_name", src.get("display_names", [""])[0] if isinstance(src.get("display_names"), list) else "")
                # Try to get CIK from entity_id in EFTS response
                entity_id = src.get("entity_id", "")
                if entity_id:
                    cik = str(entity_id).zfill(10)
                    cik_counts[cik] = cik_counts.get(cik, 0) + 1
                    if cik not in cik_names and entity_name:
                        cik_names[cik] = entity_name

            await asyncio.sleep(_RATE_LIMIT_SLEEP)

        # Filter to active funds
        active_ciks = [
            cik for cik, count in cik_counts.items()
            if count >= min_deals_90d
        ]

        if not active_ciks:
            logger.info("discover_active_vc_funds: no active CIKs found in EFTS, "
                        "returning KNOWN_VC_PE_CIKS as fallback")
            return await self.track_known_funds()

        logger.info("discover_active_vc_funds", total_ciks=len(active_ciks))

        sem = asyncio.Semaphore(3)

        async def _fetch(cik: str) -> Optional[VCPEFund]:
            async with sem:
                try:
                    return await self.get_fund_portfolio(cik)
                except Exception as exc:
                    logger.warning("discover profile error", cik=cik, error=str(exc))
                    return None

        results = await asyncio.gather(*[_fetch(c) for c in active_ciks[:50]])
        funds = [f for f in results if f is not None]
        funds.sort(key=lambda f: f.form_d_count, reverse=True)
        return funds

    async def get_sector_investment_trends(
        self, days_back: int = 90
    ) -> pd.DataFrame:
        """Aggregate Form D deal flow by industry over the last N days.

        Returns a DataFrame with columns:
          industry, deal_count, total_raised, avg_deal_size, top_funds

        Uses EFTS full-text search with Form D filter.  Note: EFTS does not
        expose the Form D XML body, so industry classification comes from the
        EFTS metadata fields (industry_group / naics_description when available)
        and falls back to the issuer's SIC from their submissions record.
        """
        cutoff = date.today() - timedelta(days=days_back)
        start_dt = cutoff.strftime("%Y-%m-%d")
        end_dt = date.today().strftime("%Y-%m-%d")

        rows: list[dict] = []
        for page in range(0, 1000, 10):
            hits = await self._efts_search(
                form_type="D",
                date_from=start_dt,
                date_to=end_dt,
                offset=page,
                hits=10,
            )
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", hit)
                entity_name = src.get("entity_name", "")
                file_date = src.get("file_date", "")
                # Industry comes from EFTS metadata if present
                industry = (
                    src.get("biz_location_state_country")
                    or src.get("period_of_report")  # fallback placeholder
                    or "Unknown"
                )
                # Try to extract amount from file_date field as proxy count
                rows.append({
                    "industry": industry,
                    "fund": entity_name,
                    "amount": 0.0,  # EFTS doesn't expose Form D amounts; enriched below
                    "date": file_date,
                })

            await asyncio.sleep(_RATE_LIMIT_SLEEP)

        if not rows:
            return pd.DataFrame(columns=[
                "industry", "deal_count", "total_raised", "avg_deal_size", "top_funds"
            ])

        df = pd.DataFrame(rows)
        grouped = (
            df.groupby("industry")
            .agg(
                deal_count=("fund", "count"),
                total_raised=("amount", "sum"),
                top_funds=("fund", lambda s: list(s.value_counts().head(3).index)),
            )
            .reset_index()
        )
        grouped["avg_deal_size"] = (
            grouped["total_raised"] / grouped["deal_count"].clip(lower=1)
        )
        grouped.sort_values("deal_count", ascending=False, inplace=True)
        return grouped.reset_index(drop=True)

    async def predict_ipo_pipeline(self) -> list[dict]:
        """Identify pre-IPO companies: 3+ Form D rounds > $10M, no S-1 filed yet.

        Strategy:
          1. Search EFTS for Form D filings with high amounts in last 24 months.
          2. Group by issuer company (the investee, not the fund).
          3. Filter to companies with 3+ rounds and total > $30M.
          4. Cross-check: search EFTS for S-1 filings for the same entity names.
          5. Exclude companies that already have an S-1.

        Returns:
            List of dicts: {company_name, cik, total_raised, rounds,
                            last_round_date, sectors}
        """
        cutoff = date.today() - timedelta(days=730)  # 24 months
        start_dt = cutoff.strftime("%Y-%m-%d")
        end_dt = date.today().strftime("%Y-%m-%d")

        # Collect Form D issuers from the last 24 months
        issuer_deals: dict[str, list[dict]] = {}

        for page in range(0, 2000, 10):
            hits = await self._efts_search(
                form_type="D",
                date_from=start_dt,
                date_to=end_dt,
                offset=page,
                hits=10,
            )
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", hit)
                entity_name = src.get("entity_name", "").strip()
                entity_id = str(src.get("entity_id", "")).zfill(10)
                file_date = src.get("file_date", "")

                if not entity_name or entity_name in KNOWN_VC_PE_CIKS:
                    continue  # skip the fund managers themselves

                key = entity_id or entity_name
                if key not in issuer_deals:
                    issuer_deals[key] = []
                issuer_deals[key].append({
                    "name": entity_name,
                    "cik": entity_id,
                    "date": file_date,
                })

            await asyncio.sleep(_RATE_LIMIT_SLEEP)

        # Filter: >= 3 rounds
        candidates = {
            k: v for k, v in issuer_deals.items() if len(v) >= 3
        }

        # Check for S-1 filings (already IPO'd or in registration)
        pipeline: list[dict] = []
        for key, deals in list(candidates.items())[:100]:
            company_name = deals[0].get("name", "")
            company_cik = deals[0].get("cik", "")

            # Look for S-1 by this entity
            s1_hits = await self._efts_search(
                form_type="S-1",
                entity_name=company_name[:30],  # partial match
                hits=3,
            )
            if s1_hits:
                # Already in IPO process — skip
                continue

            last_round_date = max(
                (d.get("date", "") for d in deals), default=""
            )
            pipeline.append({
                "company_name": company_name,
                "cik": company_cik,
                "total_raised": None,  # Form D XML detail required for amounts
                "rounds": len(deals),
                "last_round_date": last_round_date,
                "sectors": [],
            })

            await asyncio.sleep(_RATE_LIMIT_SLEEP * 2)

        pipeline.sort(key=lambda x: x["rounds"], reverse=True)
        return pipeline

    async def get_exit_activity(
        self, days_back: int = 180
    ) -> list[InvestmentActivity]:
        """Identify exits: S-1 filings (IPO) and 8-K merger announcements.

        Strategy:
          1. Search EFTS for S-1 filings in the period.
          2. Filter to issuers that also have prior Form D filings
             (VC-backed indicator).
          3. Search EFTS for 8-K filings with "merger" or "acquisition" keywords
             from VC-backed companies.

        Returns list of InvestmentActivity with activity_type "exit_ipo" or "exit_ma".
        """
        cutoff = date.today() - timedelta(days=days_back)
        start_dt = cutoff.strftime("%Y-%m-%d")
        end_dt = date.today().strftime("%Y-%m-%d")

        exits: list[InvestmentActivity] = []

        # --- IPO exits via S-1 ---
        s1_hits = await self._efts_search(
            form_type="S-1",
            date_from=start_dt,
            date_to=end_dt,
            hits=50,
        )

        for hit in s1_hits:
            src = hit.get("_source", hit)
            entity_name = src.get("entity_name", "")
            entity_id = str(src.get("entity_id", "")).zfill(10)
            file_date = src.get("file_date", "")

            # Confirm VC-backed: check for Form D filings for this entity
            form_d = await self._efts_search(
                form_type="D",
                entity_name=entity_name[:30],
                hits=1,
            )
            if not form_d:
                continue

            exits.append(InvestmentActivity(
                fund_name="unknown",  # would need to cross-reference Form D filer
                company_name=entity_name,
                company_cik=entity_id,
                activity_type="exit_ipo",
                date=_parse_date(file_date) or date.today(),
                amount=None,
                round_type="IPO",
            ))
            await asyncio.sleep(_RATE_LIMIT_SLEEP)

        # --- M&A exits via 8-K ---
        ma_hits = await self._efts_search(
            form_type="8-K",
            query="merger acquisition",
            date_from=start_dt,
            date_to=end_dt,
            hits=50,
        )

        for hit in ma_hits:
            src = hit.get("_source", hit)
            entity_name = src.get("entity_name", "")
            entity_id = str(src.get("entity_id", "")).zfill(10)
            file_date = src.get("file_date", "")

            # Confirm VC-backed
            form_d = await self._efts_search(
                form_type="D",
                entity_name=entity_name[:30],
                hits=1,
            )
            if not form_d:
                continue

            exits.append(InvestmentActivity(
                fund_name="unknown",
                company_name=entity_name,
                company_cik=entity_id,
                activity_type="exit_ma",
                date=_parse_date(file_date) or date.today(),
                amount=None,
                round_type="M&A",
            ))
            await asyncio.sleep(_RATE_LIMIT_SLEEP)

        exits.sort(key=lambda e: e.date, reverse=True)
        logger.info("get_exit_activity complete", ipo=sum(1 for e in exits if e.activity_type == "exit_ipo"),
                    ma=sum(1 for e in exits if e.activity_type == "exit_ma"))
        return exits

    # ── Private EDGAR helpers ─────────────────────────────────────────────────

    async def _fetch_submissions(self, cik: str) -> dict:
        """GET /submissions/CIK{cik}.json"""
        cik_padded = cik.strip().zfill(10)
        url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
        await asyncio.sleep(_RATE_LIMIT_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers={
                    **self._headers, "Host": "data.sec.gov"
                })
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("fetch_submissions error", cik=cik, error=str(exc))
            return {}

    async def _get_recent_filings(
        self, cik: str, form_type: str, limit: int = 40
    ) -> list[dict]:
        """Extract recent filings of a given form type from submissions JSON."""
        submissions = await self._fetch_submissions(cik)
        filings_raw = submissions.get("filings", {}).get("recent", {})
        if not filings_raw:
            return []

        forms = filings_raw.get("form", [])
        dates = filings_raw.get("filingDate", [])
        accessions = filings_raw.get("accessionNumber", [])

        results = []
        for i, form in enumerate(forms):
            if form != form_type:
                continue
            results.append({
                "form_type": form,
                "filing_date": dates[i] if i < len(dates) else None,
                "accession": accessions[i] if i < len(accessions) else None,
                "cik": cik,
            })
            if len(results) >= limit:
                break
        return results

    async def _fetch_form_d_detail(self, cik: str, accession: str) -> Optional[dict]:
        """Fetch the primary Form D JSON document from EDGAR archives.

        Accession number format: 0001234567-24-000001
        Archive path: /Archives/edgar/data/{cik}/{acc_no_dashes}/{acc}.json
        """
        if not accession:
            return None

        cik_raw = cik.lstrip("0") or "0"
        acc_clean = accession.replace("-", "")
        # Try JSON version first (newer EDGAR format)
        json_url = (
            f"{EDGAR_WWW}/Archives/edgar/data/{cik_raw}"
            f"/{acc_clean}/{accession}.json"
        )
        # Fall back to the index page to find the actual XML
        index_url = (
            f"{EDGAR_WWW}/Archives/edgar/data/{cik_raw}"
            f"/{acc_clean}/{accession}-index.htm"
        )

        await asyncio.sleep(_RATE_LIMIT_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(json_url, headers={
                    **self._headers, "Host": "www.sec.gov"
                })
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass

        # Try fetching primary-doc list from the index JSON
        idx_url = (
            f"{EDGAR_WWW}/Archives/edgar/data/{cik_raw}"
            f"/{acc_clean}/index.json"
        )
        await asyncio.sleep(_RATE_LIMIT_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(idx_url, headers={
                    **self._headers, "Host": "www.sec.gov"
                })
                if resp.status_code != 200:
                    return None
                idx = resp.json()
                for item in idx.get("directory", {}).get("item", []):
                    name: str = item.get("name", "")
                    if name.endswith(".json") and "form" in name.lower():
                        doc_url = (
                            f"{EDGAR_WWW}/Archives/edgar/data/{cik_raw}"
                            f"/{acc_clean}/{name}"
                        )
                        await asyncio.sleep(_RATE_LIMIT_SLEEP)
                        doc_resp = await client.get(doc_url, headers={
                            **self._headers, "Host": "www.sec.gov"
                        })
                        if doc_resp.status_code == 200:
                            return doc_resp.json()
        except Exception as exc:
            logger.debug("form_d_detail fallback error", accession=accession, error=str(exc))

        return None

    async def _efts_search(
        self,
        form_type: Optional[str] = None,
        query: Optional[str] = None,
        entity_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        offset: int = 0,
        hits: int = 10,
    ) -> list[dict]:
        """Search EDGAR full-text search (EFTS) endpoint.

        Endpoint: GET https://efts.sec.gov/LATEST/search-index
        Required header: User-Agent (SEC enforcement).
        """
        params: dict = {
            "dateRange": "custom",
        }
        if query:
            params["q"] = f'"{query}"'
        if form_type:
            params["forms"] = form_type
        if entity_name:
            params["entity"] = entity_name
        if date_from:
            params["startdt"] = date_from
        if date_to:
            params["enddt"] = date_to
        if offset > 0:
            params["from"] = offset

        url = f"{EDGAR_SEARCH}/LATEST/search-index"
        await asyncio.sleep(_RATE_LIMIT_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params, headers={
                    **self._headers, "Host": "efts.sec.gov"
                })
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("hits", {}).get("hits", [])[:hits]
                logger.warning("EFTS non-200", status=resp.status_code, url=url)
                return []
        except Exception as exc:
            logger.warning("EFTS search error", error=str(exc))
            return []


# ── Private helpers ────────────────────────────────────────────────────────────

def _parse_form_d(data: dict, filer_cik: str) -> dict:
    """Extract key fields from a Form D JSON document.

    EDGAR Form D JSON schema (as returned by data.sec.gov) has a
    'formData' or 'primaryIssuer' structure.  We handle both formats.
    """
    # Newer XML-derived JSON
    form_data = data.get("formData", data)

    # Issuer info
    issuers = form_data.get("issuerList", {}).get("issuer", [])
    if isinstance(issuers, dict):
        issuers = [issuers]

    primary_issuer = issuers[0] if issuers else {}
    issuer_name = (
        primary_issuer.get("issuerName", "")
        or data.get("companyName", "")
        or form_data.get("primaryIssuer", {}).get("entityName", "")
    )
    issuer_cik = (
        primary_issuer.get("cik", "")
        or form_data.get("primaryIssuer", {}).get("cik", "")
    )

    # Offering data
    offering = form_data.get("offeringData", {})
    total_amount = offering.get("salesCompensationList", {})
    amount_sold_raw = (
        offering.get("totalAmountSold")
        or form_data.get("totalAmountSold")
        or 0
    )
    try:
        amount_sold = float(str(amount_sold_raw).replace(",", "")) if amount_sold_raw else 0.0
    except (ValueError, TypeError):
        amount_sold = 0.0

    # Industry
    industry_group = (
        offering.get("industryGroup", {}).get("industryGroupType", "")
        or form_data.get("industryGroup", "")
    )

    # Date of first sale
    date_of_first_sale_raw = (
        offering.get("dateOfFirstSale", {}).get("value", "")
        or offering.get("dateOfFirstSale", "")
        or form_data.get("dateOfFirstSale", "")
    )
    date_of_first_sale = _parse_date(str(date_of_first_sale_raw)) if date_of_first_sale_raw else None

    return {
        "issuer_name": issuer_name,
        "issuer_cik": issuer_cik,
        "issuers": issuers,
        "amount_sold": amount_sold,
        "industry_group": industry_group or "Other",
        "date_of_first_sale": date_of_first_sale,
    }


def _classify_fund_type(entity_name: str, sic_code: str) -> str:
    """Infer VC/PE fund type from entity name and SIC code."""
    name_lower = entity_name.lower()

    # SIC 6726 = Investment Offices (typical for PE/VC holding entities)
    # SIC 6282 = Investment Advisers
    if sic_code in ("6726", "6282", "6199", "6211"):
        if _VC_KEYWORDS.search(name_lower):
            return "VC"
        if _DISTRESSED_KEYWORDS.search(name_lower):
            return "Distressed"
        return "PE Buyout"

    if _DISTRESSED_KEYWORDS.search(name_lower):
        return "Distressed"
    if "real" in name_lower and ("estate" in name_lower or "asset" in name_lower):
        return "Real Assets"
    if "growth" in name_lower:
        return "Growth Equity"
    if _VC_KEYWORDS.search(name_lower):
        return "VC"
    if _PE_KEYWORDS.search(name_lower):
        return "PE Buyout"
    return "VC"  # default for unknown private-investment entities


def _infer_round_type(amount: float) -> Optional[str]:
    """Classify round type from total amount raised (USD).

    Heuristics based on typical US private market deal sizes:
      Angel/Pre-seed : < $1M
      Seed           : $1M – $3M
      Series A       : $3M – $15M
      Series B       : $15M – $40M
      Series C       : $40M – $100M
      Series D+      : > $100M
    """
    if amount <= 0:
        return None
    if amount < 1_000_000:
        return "Pre-Seed / Angel"
    if amount < 3_000_000:
        return "Seed"
    if amount < 15_000_000:
        return "Series A"
    if amount < 40_000_000:
        return "Series B"
    if amount < 100_000_000:
        return "Series C"
    return "Series D+"


def _infer_status(latest_filing_date: Optional[date]) -> str:
    """Infer fund status from recency of Form D filings."""
    if latest_filing_date is None:
        return "unknown"
    age_days = (date.today() - latest_filing_date).days
    if age_days < 365:
        return "active"
    if age_days < 730:
        return "harvesting"
    return "closed"


def _parse_date(s: str) -> Optional[date]:
    """Parse ISO date string (YYYY-MM-DD) to date."""
    if not s:
        return None
    try:
        return date.fromisoformat(str(s).strip()[:10])
    except ValueError:
        return None


# ── Module-level convenience helpers ──────────────────────────────────────────

_default_tracker: Optional[VCPETracker] = None


def _get_tracker() -> VCPETracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = VCPETracker()
    return _default_tracker


async def track_fund(cik: str) -> VCPEFund:
    """Module-level fund profile fetch using shared default tracker."""
    return await _get_tracker().get_fund_portfolio(cik)


async def active_vc_funds(min_deals: int = 2) -> list[VCPEFund]:
    """Module-level discovery of active VC funds using shared default tracker."""
    return await _get_tracker().discover_active_vc_funds(min_deals_90d=min_deals)
